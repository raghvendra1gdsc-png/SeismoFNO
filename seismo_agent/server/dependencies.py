"""
seismo_agent/server/dependencies.py — FastAPI Dependency Injection & Service Singletons.

Manages clean lifecycle access to the deterministic tool harness, Nemotron LLM client,
and SeismoOrchestrator. Avoids eager model loading or heavy GPU allocation at startup.
"""

import uuid
from typing import List
from functools import lru_cache

from seismo_agent.config import NEBIUS_API_KEY, NEBIUS_BASE_URL, NEBIUS_MODEL
from seismo_agent.core.llm_client import NemotronClient
from seismo_agent.core.orchestrator import SeismoOrchestrator
from seismo_agent.tools.base import BaseTool
from seismo_agent.tools import get_all_tools


def generate_request_id() -> str:
    """Generate a unique request tracking ID."""
    return f"req_{uuid.uuid4().hex[:12]}"


@lru_cache(maxsize=1)
def get_cached_tools() -> List[BaseTool]:
    """Returns the singleton list of all 8 deterministic engineering tools."""
    return get_all_tools()


def get_llm_client() -> NemotronClient:
    """
    Returns an instance of NemotronClient initialized from current environment configuration.
    """
    return NemotronClient(
        api_key=NEBIUS_API_KEY,
        base_url=NEBIUS_BASE_URL,
        model=NEBIUS_MODEL,
    )


def get_orchestrator() -> SeismoOrchestrator:
    """
    Returns a configured instance of SeismoOrchestrator using the shared tool registry.
    """
    tools = get_cached_tools()
    llm_client = get_llm_client()
    return SeismoOrchestrator(
        llm_client=llm_client,
        tools=tools,
    )
