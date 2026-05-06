"""
Verified corpus layer: extracted chunks from VERIFIED answers only.

Persisted separately from IRS_DATASET (data.py). Hybrid retrieval (rag.TaxRAGPipeline)
searches IRS first, then this layer, with official chunks prioritized over verified.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

_LOG_PATH = Path(__file__).resolve().parent / "verified_dataset.jsonl"
_lock = Lock()

# In-memory mirror of the JSONL file (reloaded on import, updated on append).
VERIFIED_DATASET: List[Dict[str, Any]] = []


def get_verified_records_snapshot() -> List[Dict[str, Any]]:
    """Thread-safe shallow copy of current records (for embedding / retrieval sync)."""
    with _lock:
        return [dict(r) for r in VERIFIED_DATASET]


def _load_verified_dataset_from_disk() -> None:
    global VERIFIED_DATASET
    VERIFIED_DATASET = []
    if not _LOG_PATH.exists():
        return
    with _LOG_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                VERIFIED_DATASET.append(json.loads(line))
            except json.JSONDecodeError:
                continue


def append_verified_chunks(
    query: str,
    chunks: List[str],
    sources: Optional[List[str]] = None,
) -> None:
    """
    Append each chunk as one record. Same batch shares timestamp and source list.
    """
    if not chunks:
        return
    ts = datetime.now(timezone.utc).isoformat()
    src = list(sources) if sources else []
    q = (query or "").strip()

    with _lock:
        with _LOG_PATH.open("a", encoding="utf-8") as f:
            for content in chunks:
                text = (content or "").strip()
                if not text:
                    continue
                rec: Dict[str, Any] = {
                    "content": text,
                    "query": q,
                    "sources": src,
                    "timestamp": ts,
                    "topic": "verified_extract",
                    "form": "",
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                VERIFIED_DATASET.append(rec)


_load_verified_dataset_from_disk()
