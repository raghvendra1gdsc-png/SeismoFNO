"""
seismo_agent/server/streaming.py — WebSocket Streaming Protocol & Event Bridge.

Bridges synchronous SeismoOrchestrator progress events to asynchronous WebSockets
using an asyncio.Queue. Guarantees that the exact same SeismoOrchestrator engine
is used for both REST and streaming without duplicating the ReAct loop.
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional
from fastapi import WebSocket, WebSocketDisconnect

from seismo_agent.core.orchestrator import SeismoOrchestrator
from seismo_agent.server.execution import ORCHESTRATOR_THREAD_POOL
from seismo_agent.server.dependencies import generate_request_id

logger = logging.getLogger("seismo_agent.streaming")


async def handle_agent_stream_websocket(
    websocket: WebSocket,
    orchestrator: SeismoOrchestrator,
    default_timeout_seconds: float = 120.0,
):
    """
    Handle live agent execution and event streaming over WebSocket.
    """
    await websocket.accept()
    request_id = generate_request_id()

    try:
        # 1. Receive initial client message payload
        raw_msg = await websocket.receive_text()
        try:
            payload = json.loads(raw_msg)
            query = payload.get("message", "") if isinstance(payload, dict) else str(raw_msg)
        except Exception:
            query = str(raw_msg)

        query = query.strip()
        if not query:
            await websocket.send_json({
                "event": "error",
                "request_id": request_id,
                "error": "Query message cannot be empty.",
            })
            await websocket.close()
            return

        # 2. Setup Asyncio Event Queue to receive events from background worker thread
        loop = asyncio.get_running_loop()
        event_queue: asyncio.Queue = asyncio.Queue()

        def on_event(event_dict: Dict[str, Any]):
            # Inject request_id
            event_dict["request_id"] = request_id
            loop.call_soon_threadsafe(event_queue.put_nowait, event_dict)

        # 3. Offload orchestrator execution to worker thread pool
        execution_future = loop.run_in_executor(
            ORCHESTRATOR_THREAD_POOL,
            orchestrator.run,
            query,
            on_event,
        )

        # 4. Stream events to WebSocket until orchestration completes
        while True:
            # Check if orchestrator execution has completed or failed
            get_event_task = asyncio.create_task(event_queue.get())
            done, _ = await asyncio.wait(
                [get_event_task, execution_future],
                return_when=asyncio.FIRST_COMPLETED,
                timeout=default_timeout_seconds,
            )

            if not done:
                # Timeout occurred
                await websocket.send_json({
                    "event": "error",
                    "request_id": request_id,
                    "error": f"Streaming execution timed out after {default_timeout_seconds}s.",
                })
                break

            if get_event_task in done:
                event_data = get_event_task.result()
                await websocket.send_json(event_data)
                if event_data.get("event") in ("completed", "error"):
                    break
            elif execution_future in done:
                # Flush remaining events from queue
                while not event_queue.empty():
                    ev = event_queue.get_nowait()
                    await websocket.send_json(ev)
                break

    except WebSocketDisconnect:
        logger.info(f"Client disconnected from WebSocket stream: {request_id}")
    except Exception as e:
        logger.error(f"WebSocket streaming error ({request_id}): {e}", exc_info=True)
        try:
            await websocket.send_json({
                "event": "error",
                "request_id": request_id,
                "error": str(e),
            })
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
