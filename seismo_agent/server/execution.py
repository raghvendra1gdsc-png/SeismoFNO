"""
seismo_agent/server/execution.py — Safe Execution Manager & OpenSeesPy Concurrency Policy.

CRITICAL PROCESS SAFETY & CONCURRENCY CONSTRAINTS:
OpenSeesPy maintains a process-global native C++ singleton state. If multiple concurrent
threads invoke OpenSees operations simultaneously in the same process, native memory
corruption or segmentation faults will occur.

CONCURRENCY GUARANTEE: PROCESS-LOCAL SERIALIZATION
1. The system uses threading.Lock to enforce PROCESS-LOCAL SERIALIZATION of OpenSeesPy calls.
2. This is NOT a multi-process isolation mechanism. Each OS process maintains its own lock.
3. PRODUCTION DEPLOYMENT REQUIREMENT: The gateway MUST run as a single worker process:
       uvicorn seismo_agent.server.app:app --host 0.0.0.0 --port $PORT --workers 1
   Running multiple Uvicorn worker processes will bypass this Python lock and cause
   native OpenSees state conflicts.
4. Pure Python computation (EarthquakeTool, StructuralTool, OODDetectorTool, ReportTool)
   and PyTorch inference (InferenceTool) are safe for concurrent multithreading.
5. Physics simulations (PhysicsSimTool, SweepTool) MUST acquire the process-local OpenSeesLock.
6. ops.wipe() is strictly called before AND after every physics simulation.
7. Orchestrator requests are executed with explicit timeouts, separating LLM, tool, and
   wall-clock latency.
"""

import threading
import time
from typing import Callable, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

from seismo_agent.core.orchestrator import SeismoOrchestrator, OrchestrationResult
from seismo_agent.tools.physics_sim_tool import _OPENSEES_LOCK as OPENSEES_NATIVE_LOCK

# Thread pool for offloading synchronous orchestration outside the FastAPI async event loop
ORCHESTRATOR_THREAD_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="seismo_worker")


class GatewayExecutionError(Exception):
    """Raised when request execution encounters an unrecoverable failure."""
    pass


class GatewayTimeoutError(GatewayExecutionError):
    """Raised when request execution exceeds the allowed wall-clock timeout."""
    pass


def execute_orchestrator_safe(
    orchestrator: SeismoOrchestrator,
    user_query: str,
    event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    timeout_seconds: float = 120.0,
) -> OrchestrationResult:
    """
    Executes the SeismoOrchestrator inside a worker thread with an explicit timeout.
    Guarantees that synchronous tool execution never blocks FastAPI's async event loop.
    """
    future = ORCHESTRATOR_THREAD_POOL.submit(
        orchestrator.run,
        user_query,
        event_callback,
    )

    try:
        result = future.result(timeout=timeout_seconds)
        return result
    except FutureTimeoutError as e:
        raise GatewayTimeoutError(
            f"Execution timed out after {timeout_seconds}s. "
            f"The request exceeded the maximum allowed wall-clock limit."
        ) from e
    except Exception as e:
        raise GatewayExecutionError(f"Orchestration execution failed: {e}") from e
