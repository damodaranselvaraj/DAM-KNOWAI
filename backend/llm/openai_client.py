"""
GPT-4o client — streaming and non-streaming completions.

Features
────────
• Async streaming via Server-Sent Events (token-by-token).
• Non-streaming for batch / background use.
• Exponential-backoff retry on rate-limit (429) and transient 5xx errors.
• Token usage tracking returned with every response.
• Structured JSON logging for every call (latency, tokens, model).

Configuration:
    OPENAI_API_KEY         via backend/config.py / .env — required.
    OPENAI_CHAT_MODEL      via backend/config.py / .env (default: gpt-4o).
    OPENAI_MAX_TOKENS      via backend/config.py / .env (default: 8191).

    temperature, max_retries, and timeout are NOT backed by settings/env
    vars — they are plain constructor defaults on OpenAIClient
    (temperature=0.2, max_retries=3, timeout=60.0s). Pass them explicitly
    to OpenAIClient(...) to override; there is no LLM_TEMPERATURE,
    LLM_MAX_RETRIES, or LLM_TIMEOUT_SECONDS setting.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import AsyncIterator, Dict, List, Optional

import openai
from openai import AsyncOpenAI, OpenAI

from backend.config import settings

logger = logging.getLogger(__name__)

# ─── Token-usage dataclass ────────────────────────────────────────────────────

class TokenUsage:
    """Carries prompt / completion / total token counts from a completion."""

    __slots__ = ("prompt_tokens", "completion_tokens", "total_tokens")

    def __init__(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
    ) -> None:
        self.prompt_tokens     = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens      = total_tokens

    def to_dict(self) -> Dict[str, int]:
        return {
            "prompt_tokens":     self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens":      self.total_tokens,
        }

    def __repr__(self) -> str:
        return (
            f"TokenUsage(prompt={self.prompt_tokens}, "
            f"completion={self.completion_tokens}, "
            f"total={self.total_tokens})"
        )


# ─── Completion result ────────────────────────────────────────────────────────

class CompletionResult:
    """Returned by ``complete()`` (non-streaming path)."""

    __slots__ = ("text", "usage", "model", "latency_ms", "finish_reason")

    def __init__(
        self,
        text: str,
        usage: TokenUsage,
        model: str,
        latency_ms: float,
        finish_reason: str = "stop",
    ) -> None:
        self.text          = text
        self.usage         = usage
        self.model         = model
        self.latency_ms    = latency_ms
        self.finish_reason = finish_reason


# ─── OpenAI client ────────────────────────────────────────────────────────────

class OpenAIClient:
    """
    Thin, production-ready wrapper around the OpenAI Python SDK.

    Two execution paths:
    • ``complete(messages)``               — blocking, returns full text.
    • ``stream(messages)`` (async gen)     — yields text deltas token by token.

    Both paths share the same retry / logging logic.

    Usage (non-streaming)::

        client = OpenAIClient()
        result = client.complete(messages)
        print(result.text, result.usage)

    Usage (streaming inside an async route)::

        client = OpenAIClient()
        async for token in client.stream(messages):
            yield f"data: {token}\\n\\n"
    """

    def __init__(
        self,
        model:       Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens:  Optional[int] = None,
        max_retries: int = 3,
        timeout:     float = 60.0,
    ) -> None:
        self.model       = model       or settings.openai_chat_model
        self.temperature = temperature if temperature is not None else 0.2
        self.max_tokens  = max_tokens  or settings.openai_max_tokens
        self.max_retries = max_retries
        self.timeout     = timeout

        # Sync client for non-streaming; async client for streaming
        self._sync  = OpenAI(
            api_key=settings.openai_api_key,
            timeout=timeout,
            max_retries=0,   # we handle retries ourselves
        )
        self._async = AsyncOpenAI(
            api_key=settings.openai_api_key,
            timeout=timeout,
            max_retries=0,
        )

    # ── Retry helpers ─────────────────────────────────────────────────────────

    def _backoff(self, attempt: int) -> float:
        """Seconds to wait before attempt N (1-indexed). Capped at 30s."""
        return min(30.0, 2 ** attempt)

    def _is_retryable(self, exc: Exception) -> bool:
        """True for rate-limit (429) and transient server errors (5xx)."""
        if isinstance(exc, openai.RateLimitError):
            return True
        if isinstance(exc, openai.APIStatusError):
            return exc.status_code is not None and exc.status_code >= 500
        if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
            return True
        return False

    # ── Non-streaming completion ───────────────────────────────────────────────

    def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens:  Optional[int]   = None,
    ) -> CompletionResult:
        """
        Send *messages* to GPT-4o and return the full completion.

        Retries up to ``max_retries`` times on retryable errors with
        exponential backoff.

        Parameters
        ----------
        messages:
            OpenAI-format messages list ``[{"role": ..., "content": ...}]``.
        temperature:
            Override instance temperature for this call.
        max_tokens:
            Override instance max_tokens for this call.

        Returns
        -------
        CompletionResult
            ``.text`` = full assistant reply string.
        """
        temp = temperature if temperature is not None else self.temperature
        mtok = max_tokens  if max_tokens  is not None else self.max_tokens
        last_exc: Optional[Exception] = None
        t0 = time.perf_counter()

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._sync.chat.completions.create(
                    model=self.model,
                    messages=messages,      # type: ignore[arg-type]
                    temperature=temp,
                    max_tokens=mtok,
                    stream=False,
                )
                latency_ms = (time.perf_counter() - t0) * 1_000
                choice     = response.choices[0]
                usage_obj  = response.usage

                usage = TokenUsage(
                    prompt_tokens=usage_obj.prompt_tokens     if usage_obj else 0,
                    completion_tokens=usage_obj.completion_tokens if usage_obj else 0,
                    total_tokens=usage_obj.total_tokens       if usage_obj else 0,
                )

                logger.info(
                    json.dumps({
                        "event":           "llm_complete",
                        "model":           self.model,
                        "latency_ms":      round(latency_ms, 1),
                        "prompt_tokens":   usage.prompt_tokens,
                        "completion_tokens": usage.completion_tokens,
                        "total_tokens":    usage.total_tokens,
                        "finish_reason":   choice.finish_reason,
                    })
                )

                return CompletionResult(
                    text=choice.message.content or "",
                    usage=usage,
                    model=response.model,
                    latency_ms=latency_ms,
                    finish_reason=choice.finish_reason or "stop",
                )

            except Exception as exc:
                if self._is_retryable(exc) and attempt < self.max_retries:
                    wait = self._backoff(attempt)
                    logger.warning(
                        json.dumps({
                            "event":   "llm_retry",
                            "attempt": attempt,
                            "wait_s":  wait,
                            "error":   str(exc),
                        })
                    )
                    time.sleep(wait)
                    last_exc = exc
                else:
                    raise

        raise RuntimeError(
            f"LLM completion failed after {self.max_retries} attempts. "
            f"Last error: {last_exc}"
        )

    # ── Async streaming completion ────────────────────────────────────────────

    async def stream(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens:  Optional[int]   = None,
    ) -> AsyncIterator[str]:
        """
        Stream GPT-4o tokens as an async generator.

        Each yielded value is a *text delta* string (one or more characters).
        Yields an empty string for keep-alive / role chunks.

        After the stream completes, a final ``[DONE]`` sentinel is NOT
        yielded here — the SSE route wraps this generator and appends it.

        Example::

            async for token in client.stream(messages):
                # token is a text fragment, e.g. "The ", "claims ", "process "
                yield f"data: {json.dumps({'token': token})}\\n\\n"
        """
        temp = temperature if temperature is not None else self.temperature
        mtok = max_tokens  if max_tokens  is not None else self.max_tokens
        last_exc: Optional[Exception] = None
        t0 = time.perf_counter()

        for attempt in range(1, self.max_retries + 1):
            try:
                stream_resp = await self._async.chat.completions.create(
                    model=self.model,
                    messages=messages,      # type: ignore[arg-type]
                    temperature=temp,
                    max_tokens=mtok,
                    stream=True,
                )

                completion_tokens = 0
                async for chunk in stream_resp:
                    delta = chunk.choices[0].delta if chunk.choices else None
                    if delta and delta.content:
                        completion_tokens += 1    # approx 1 token per chunk
                        yield delta.content

                latency_ms = (time.perf_counter() - t0) * 1_000
                logger.info(
                    json.dumps({
                        "event":              "llm_stream_complete",
                        "model":              self.model,
                        "latency_ms":         round(latency_ms, 1),
                        "approx_tokens_out":  completion_tokens,
                    })
                )
                return   # success — exit retry loop

            except Exception as exc:
                if self._is_retryable(exc) and attempt < self.max_retries:
                    wait = self._backoff(attempt)
                    logger.warning(
                        json.dumps({
                            "event":   "llm_stream_retry",
                            "attempt": attempt,
                            "wait_s":  wait,
                            "error":   str(exc),
                        })
                    )
                    await asyncio.sleep(wait)
                    last_exc = exc
                else:
                    raise

        raise RuntimeError(
            f"LLM streaming failed after {self.max_retries} attempts. "
            f"Last error: {last_exc}"
        )
