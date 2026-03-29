"""
Token usage tracker for LLM engine calls.

Monkey-patches engine.client.chat.completions.create to intercept
response.usage without modifying any existing engine code.
"""

import threading
from typing import Any, Dict, Optional


class TokenTracker:
    """Lightweight, thread-safe token usage tracker.

    Usage::

        tracker = TokenTracker()
        engine = get_engine("online-qwen2.5-coder-32b-instruct")
        tracker.wrap_engine(engine)

        snap = tracker.snapshot()
        # ... do LLM work ...
        delta = tracker.diff(snap)
        print(delta)  # {'prompt_tokens': ..., 'completion_tokens': ..., ...}
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.call_count: int = 0
        self._wrapped_engines: list = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def wrap_engine(self, engine: Any) -> Any:
        """Monkey-patch *engine.client.chat.completions.create* so every
        API call automatically records token usage reported by the server.

        Safe to call on the same engine more than once (idempotent).
        Returns the engine for convenience chaining.
        """
        client = getattr(engine, "client", None)
        if client is None:
            return engine

        completions = getattr(getattr(client, "chat", None), "completions", None)
        if completions is None:
            return engine

        if getattr(completions.create, "_token_tracker_wrapped", False):
            return engine

        original_create = completions.create
        tracker = self

        def _tracked_create(*args: Any, **kwargs: Any) -> Any:
            response = original_create(*args, **kwargs)
            usage = getattr(response, "usage", None)
            if usage is not None:
                with tracker._lock:
                    tracker.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
                    tracker.completion_tokens += getattr(usage, "completion_tokens", 0) or 0
                    tracker.call_count += 1
            else:
                with tracker._lock:
                    tracker.call_count += 1
            return response

        _tracked_create._token_tracker_wrapped = True  # type: ignore[attr-defined]
        completions.create = _tracked_create
        self._wrapped_engines.append(engine)
        return engine

    def snapshot(self) -> Dict[str, int]:
        """Return a frozen copy of cumulative counters."""
        with self._lock:
            return {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.prompt_tokens + self.completion_tokens,
                "call_count": self.call_count,
            }

    def diff(self, prev: Dict[str, int]) -> Dict[str, int]:
        """Compute the delta between the current state and *prev* snapshot."""
        cur = self.snapshot()
        return {
            "prompt_tokens": cur["prompt_tokens"] - prev["prompt_tokens"],
            "completion_tokens": cur["completion_tokens"] - prev["completion_tokens"],
            "total_tokens": cur["total_tokens"] - prev["total_tokens"],
            "call_count": cur["call_count"] - prev["call_count"],
        }

    def reset(self) -> None:
        """Zero all counters."""
        with self._lock:
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.call_count = 0
