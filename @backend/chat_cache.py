"""
Normalized query → ChatResponse payload cache (LRU).

Skips retrieval, embeddings, LLM, verification, and chunk persistence on repeated questions.
"""
import os
import time
from collections import OrderedDict
from threading import Lock
from typing import Any, Dict, Optional


def normalize_cache_key(message: str) -> str:
    """Stable key: lowercase, collapsed whitespace."""
    return " ".join((message or "").strip().lower().split())


def cache_disabled() -> bool:
    return os.getenv("CHAT_CACHE_DISABLED", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _max_entries() -> int:
    raw = os.getenv("CHAT_CACHE_MAX_ENTRIES", "256").strip()
    try:
        n = int(raw)
    except ValueError:
        return 256
    return max(8, min(10_000, n))


def _ttl_seconds() -> float:
    raw = os.getenv("CHAT_CACHE_TTL_SECONDS", "0").strip()
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 0.0


_lock = Lock()
_lru: OrderedDict[str, Dict[str, Any]] = OrderedDict()


def get_cached_response(cache_key: str) -> Optional[Dict[str, Any]]:
    """Return ChatResponse.model_dump() dict if hit and not expired; else None."""
    if not cache_key or cache_disabled():
        return None
    ttl = _ttl_seconds()
    with _lock:
        if cache_key not in _lru:
            return None
        entry = _lru[cache_key]
        _lru.move_to_end(cache_key)
        if ttl > 0 and time.time() - entry["stored_at"] > ttl:
            del _lru[cache_key]
            return None
        return entry["response"]


def set_cached_response(cache_key: str, response: Dict[str, Any]) -> None:
    """Store serialized ChatResponse; LRU evict when over capacity."""
    if not cache_key or cache_disabled():
        return
    cap = _max_entries()
    with _lock:
        if cache_key in _lru:
            del _lru[cache_key]
        _lru[cache_key] = {"stored_at": time.time(), "response": response}
        _lru.move_to_end(cache_key)
        while len(_lru) > cap:
            _lru.popitem(last=False)
