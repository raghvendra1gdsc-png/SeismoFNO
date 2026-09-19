"""
seismo_agent/core/llm_client.py — Provider abstraction for Nebius OpenAI-compatible API.

Handles HTTP communication with NVIDIA Nemotron endpoints on Nebius Token Factory.
Protects API credentials, formats tool calling payloads, and returns structured LLM messages.
"""

import os
import json
import time
from typing import List, Dict, Any, Optional, Protocol, runtime_checkable
import requests

from seismo_agent.config import NEBIUS_API_KEY, NEBIUS_BASE_URL, NEBIUS_MODEL


class NebiusAPIError(Exception):
    """Base exception for Nebius LLM API errors."""
    def __init__(self, message: str, status_code: Optional[int] = None, response_body: Optional[str] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class AuthenticationError(NebiusAPIError):
    """Raised when Nebius API key is missing or invalid."""
    pass


class RateLimitError(NebiusAPIError):
    """Raised when Nebius API rate limit or quota is exceeded."""
    pass


@runtime_checkable
class LLMClientProtocol(Protocol):
    """Protocol interface for LLM client implementations (real or mocked)."""

    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> Dict[str, Any]:
        """Execute a chat completion and return the assistant message dict."""
        ...


class NemotronClient:
    """
    HTTP client for NVIDIA Nemotron inference hosted on Nebius Token Factory.
    Implements standard OpenAI-compatible `/chat/completions` protocol with function calling.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 35.0,
    ):
        self.api_key = api_key or os.environ.get("NEBIUS_API_KEY", NEBIUS_API_KEY)
        self.base_url = (base_url or os.environ.get("NEBIUS_BASE_URL", NEBIUS_BASE_URL)).rstrip("/")
        self.model = model or os.environ.get("NEBIUS_MODEL", NEBIUS_MODEL)
        self.timeout = timeout
        self._session = requests.Session()
        try:
            import certifi
            self._session.verify = certifi.where()
        except ImportError:
            pass

    def is_configured(self) -> bool:
        """Returns True if a non-empty API key is present."""
        return bool(self.api_key and len(self.api_key.strip()) > 0)

    DEFAULT_FALLBACKS = [
        "nvidia/nemotron-3-super-120b-a12b",
        "meta/llama-3.2-11b-vision-instruct",
        "nvidia/nemotron-3.5-lightning-30b-a3b",
        "openai/gpt-oss-20b",
    ]

    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> Dict[str, Any]:
        """
        Send chat completion request to NVIDIA / Nebius Nemotron endpoint.
        Implements automatic multi-model fallback if an individual upstream model is down.

        Returns:
            Assistant message dictionary containing `content` and optionally `tool_calls`.
        """
        if not self.is_configured():
            raise AuthenticationError(
                "NVIDIA_API_KEY is not configured. Please set the NVIDIA_API_KEY environment variable."
            )

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Build fallback list with primary model first
        candidates = [self.model]
        for fb in self.DEFAULT_FALLBACKS:
            if fb not in candidates:
                candidates.append(fb)

        last_exception = None

        for candidate_model in candidates:
            payload: Dict[str, Any] = {
                "model": candidate_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            if tools and len(tools) > 0:
                payload["tools"] = tools
                payload["tool_choice"] = tool_choice

            t0 = time.perf_counter()
            try:
                resp = self._session.post(url, json=payload, headers=headers, timeout=self.timeout)
            except requests.exceptions.Timeout as e:
                last_exception = NebiusAPIError(f"API request timed out for {candidate_model} after {self.timeout}s: {e}")
                continue
            except requests.exceptions.RequestException as e:
                last_exception = NebiusAPIError(f"Network error connecting to {self.base_url} ({candidate_model}): {e}")
                continue

            if resp.status_code in (401, 403):
                raise AuthenticationError(
                    f"NVIDIA API Authentication failed (status {resp.status_code}). Check NVIDIA_API_KEY.",
                    status_code=resp.status_code,
                    response_body=resp.text,
                )
            elif resp.status_code == 429:
                raise RateLimitError(
                    f"NVIDIA API rate limit exceeded (status 429).",
                    status_code=resp.status_code,
                    response_body=resp.text,
                )
            elif resp.status_code in (500, 502, 503, 504, 404):
                # Upstream server error or model unavailable — try next candidate
                last_exception = NebiusAPIError(
                    f"NVIDIA API error (status {resp.status_code}) for model {candidate_model}: {resp.text}",
                    status_code=resp.status_code,
                    response_body=resp.text,
                )
                continue
            elif resp.status_code != 200:
                raise NebiusAPIError(
                    f"NVIDIA API error (status {resp.status_code}): {resp.text}",
                    status_code=resp.status_code,
                    response_body=resp.text,
                )

            try:
                data = resp.json()
            except Exception as e:
                last_exception = NebiusAPIError(f"Failed to decode JSON from API response ({candidate_model}): {e}\nBody: {resp.text[:500]}")
                continue

            choices = data.get("choices", [])
            if not choices:
                last_exception = NebiusAPIError(f"API response contained empty 'choices' list ({candidate_model}).", response_body=resp.text)
                continue

            message = choices[0].get("message", {})
            # Attach latency telemetry to metadata
            message["_telemetry"] = {
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                "usage": data.get("usage", {}),
                "model": data.get("model", candidate_model),
            }
            # Update self.model to the working candidate if it succeeded
            self.model = candidate_model
            return message

        # If all candidates failed, raise the last recorded exception
        if last_exception:
            raise last_exception
        raise NebiusAPIError("All candidate LLM models failed.")

    def test_connection(self) -> Dict[str, Any]:
        """Smoke test to verify connectivity and authentication with Nebius."""
        if not self.is_configured():
            return {"status": "skipped", "message": "NEBIUS_API_KEY not configured."}

        messages = [
            {"role": "system", "content": "Respond with the single word 'OK'."},
            {"role": "user", "content": "Ping."},
        ]
        msg = self.chat_completion(messages, max_tokens=10)
        return {
            "status": "success",
            "model": self.model,
            "response": msg.get("content", "").strip(),
            "latency_ms": msg.get("_telemetry", {}).get("latency_ms", 0.0),
        }
