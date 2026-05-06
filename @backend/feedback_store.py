"""Append-only JSONL log for positive user feedback (query, answer, helpful, timestamp).

Rejected verification drafts are never written here; callers must not submit them."""
import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

_LOG_PATH = Path(__file__).resolve().parent / "user_feedback.jsonl"
_lock = Lock()


def append_feedback(query: str, answer: str, helpful: bool) -> None:
    record = {
        "query": query,
        "answer": answer,
        "helpful": helpful,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with _lock:
        with _LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(line)
