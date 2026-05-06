"""
RAG Pipeline for Tax Assistant
Handles embedding, vector search, and context retrieval
"""
import os
import re
import threading
from typing import Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from verified_dataset import get_verified_records_snapshot


_embedding_init_lock = threading.Lock()
_shared_embedding_model: Optional["LocalSentenceEmbeddingModel"] = None


class LocalSentenceEmbeddingModel:
    """
    Local embeddings via sentence-transformers (default: all-MiniLM-L6-v2).
    L2-normalized vectors for cosine similarity in the existing index.
    """

    def __init__(self, model_name: str):
        self._model = SentenceTransformer(model_name)
        self.model_name = model_name

    def encode(self, text: str) -> np.ndarray:
        payload = text if text and text.strip() else " "
        v = self._model.encode(
            payload,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(v, dtype=np.float64)

    def encode_batch(self, texts: List[str]) -> List[np.ndarray]:
        if not texts:
            return []
        inputs = [t if t and str(t).strip() else " " for t in texts]
        mat = self._model.encode(
            inputs,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=64,
        )
        return [np.asarray(mat[i], dtype=np.float64) for i in range(len(inputs))]


def _get_shared_embedding_model() -> "LocalSentenceEmbeddingModel":
    """Initialize once per process; reused by TaxRAGPipeline and all encode calls."""
    global _shared_embedding_model
    if _shared_embedding_model is not None:
        return _shared_embedding_model
    with _embedding_init_lock:
        if _shared_embedding_model is None:
            name = (os.getenv("LOCAL_EMBEDDING_MODEL") or "all-MiniLM-L6-v2").strip()
            _shared_embedding_model = LocalSentenceEmbeddingModel(name)
    return _shared_embedding_model


def extract_keywords(text: str) -> List[str]:
    """Extract key tax-related keywords from query"""
    text_lower = text.lower()

    form_patterns = [r"\b(1040|1040\s*x|1099|w2|schedule\s+\w+|form\s+\d+)\b"]

    tax_concepts = [
        "deduction", "bracket", "filing", "refund", "income", "tax", "credit",
        "deductible", "mortgage", "charitable", "business",
        "self-employed", "schedule c", "itemized", "standard", "eitc",
        "roth", "ira", "estate", "penalty", "estimated", "quarterly",
    ]

    keywords = []

    for pattern in form_patterns:
        matches = re.findall(pattern, text_lower)
        keywords.extend(matches)

    for concept in tax_concepts:
        if concept in text_lower:
            keywords.append(concept)

    number_matches = re.findall(r"\$?\s*(\d+(?:[,.]\d+)?)\s*%", text_lower)
    for num in number_matches:
        keywords.append(num)

    return keywords


class TaxRAGPipeline:
    """
    RAG Pipeline for Tax Assistant

    Features:
    - Query processing and keyword extraction
    - Local sentence-transformers embeddings (default all-MiniLM-L6-v2)
    - In-memory vector index over IRS_DATASET (official) plus synced VERIFIED_DATASET
    - Similarity search (top 3); official chunks tie-break over verified at equal score
    - Context builder with source metadata
    """

    def __init__(self):
        self.embedding_model = _get_shared_embedding_model()
        self.index = self._build_empty_index()
        self._verified_synced_count: int = -1
        self._verified_index: List[Dict] = []
        self.top_k = 3
        self.similarity_threshold = 0.3

    def _build_empty_index(self) -> List[Dict]:
        """Build index from IRS dataset"""
        from data import IRS_DATASET

        contents = [record["content"] for record in IRS_DATASET]
        embeddings = self.embedding_model.encode_batch(contents)
        if len(embeddings) != len(IRS_DATASET):
            raise RuntimeError(
                f"Embedding batch size mismatch: got {len(embeddings)}, "
                f"expected {len(IRS_DATASET)}"
            )

        index = []
        for i, record in enumerate(IRS_DATASET):
            index.append(
                {
                    "id": i,
                    "content": record["content"],
                    "source": record["source"],
                    "form": record["form"],
                    "topic": record["topic"],
                    "embedding": embeddings[i],
                    "is_official": True,
                }
            )

        return index

    def _sync_verified_index(self) -> None:
        """Rebuild verified-vector rows when VERIFIED_DATASET size changes."""
        snapshot = get_verified_records_snapshot()
        n = len(snapshot)
        if n == self._verified_synced_count:
            return
        if n == 0:
            self._verified_index = []
            self._verified_synced_count = 0
            return
        contents = [r["content"] for r in snapshot]
        embeddings = self.embedding_model.encode_batch(contents)
        if len(embeddings) != n:
            raise RuntimeError(
                f"Verified embedding batch mismatch: got {len(embeddings)}, expected {n}"
            )
        self._verified_index = []
        for i, rec in enumerate(snapshot):
            sources_list = rec.get("sources") or []
            primary = (
                sources_list[0]
                if sources_list
                else "internal://verified-dataset"
            )
            self._verified_index.append(
                {
                    "id": f"v{i}",
                    "content": rec["content"],
                    "source": primary,
                    "form": rec.get("form", ""),
                    "topic": rec.get("topic", "verified_extract"),
                    "embedding": embeddings[i],
                    "is_official": False,
                }
            )
        self._verified_synced_count = n

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(v1, v2) / (norm1 * norm2))

    def retrieve(self, query: str) -> List[Dict]:
        """
        Retrieve top-k over official IRS chunks first, then verified cache.
        At equal cosine similarity, official (IRS) rows rank above verified.
        """
        self._sync_verified_index()
        query_embedding = self.embedding_model.encode(query)

        if query_embedding is None or len(query_embedding) == 0:
            return []

        similarities = []
        for chunk in self.index:
            if chunk["embedding"] is not None:
                sim = self._cosine_similarity(query_embedding, chunk["embedding"])
                similarities.append(
                    {
                        "chunk": chunk,
                        "similarity": sim,
                        "is_official": chunk.get("is_official", True),
                    }
                )
        for chunk in self._verified_index:
            if chunk["embedding"] is not None:
                sim = self._cosine_similarity(query_embedding, chunk["embedding"])
                similarities.append(
                    {
                        "chunk": chunk,
                        "similarity": sim,
                        "is_official": chunk.get("is_official", False),
                    }
                )

        similarities.sort(
            key=lambda x: (-x["similarity"], 0 if x["is_official"] else 1)
        )

        def _row(result):
            ch = result["chunk"]
            return {
                "content": ch["content"],
                "source": ch["source"],
                "form": ch["form"],
                "topic": ch["topic"],
                "similarity": result["similarity"],
            }

        top_results = []
        for result in similarities:
            if result["similarity"] >= self.similarity_threshold:
                top_results.append(_row(result))
                if len(top_results) >= self.top_k:
                    break

        if len(top_results) == 0:
            for result in similarities[: self.top_k]:
                top_results.append(_row(result))

        return top_results

    def build_context(self, chunks: List[Dict]) -> str:
        """
        Build context string from retrieved chunks.
        """
        if not chunks:
            return ""

        context_parts = []
        for chunk in chunks:
            context_text = "## Source: " + chunk["source"] + "\n"
            context_text += "Form: " + chunk["form"] + "\n"
            context_text += "Topic: " + chunk["topic"] + "\n\n"
            context_text += chunk["content"] + "\n---"
            context_parts.append(context_text)

        return "\n\n".join(context_parts)

    def get_all_sources(self) -> List[str]:
        """Get all unique source identifiers (IRS + verified primary source)."""
        self._sync_verified_index()
        sources = set()
        for chunk in self.index:
            sources.add(chunk["source"])
        for chunk in self._verified_index:
            sources.add(chunk["source"])
        return list(sources)
