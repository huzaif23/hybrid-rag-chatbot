"""
Tax Assistant API - FastAPI Backend
RAG retrieval + xAI Grok generation grounded on retrieved context only.
"""
import logging
import mimetypes
import re
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from chat_cache import get_cached_response, normalize_cache_key, set_cached_response
from feedback_store import append_feedback
from models import ChatRequest, ChatResponse, FeedbackRequest
from grok_tax import (
    extract_verified_answer_chunks,
    generate_answer_from_retrieved_context,
    generate_fallback_answer_without_rag,
    verify_assistant_answer,
)
from rag import TaxRAGPipeline, extract_keywords
from retrieval_confidence import assess_retrieval_confidence
from verified_dataset import append_verified_chunks

logger = logging.getLogger(__name__)
rag_pipeline = TaxRAGPipeline()

# User-facing copy when verification returns REJECTED (never persist as training/feedback data).
VERIFICATION_REJECTED_MESSAGE_FALLBACK = (
    "This response did not pass verification. Please rephrase your "
    "question or ask about a specific IRS form or topic."
)
VERIFICATION_REJECTED_MESSAGE_RAG = (
    "This response did not pass verification against the retrieved IRS "
    "excerpts. Please try a more specific question."
)
_VERIFICATION_REJECTION_ANSWERS = frozenset(
    {VERIFICATION_REJECTED_MESSAGE_FALLBACK, VERIFICATION_REJECTED_MESSAGE_RAG}
)


def _is_verification_rejection_placeholder(answer: str) -> bool:
    """True if the shown answer is the post-rejection stub (discarded draft must not be stored)."""
    return (answer or "").strip() in _VERIFICATION_REJECTION_ANSWERS


app = FastAPI(
    title="Tax Assistant API",
    description="AI-powered tax assistant: RAG over IRS chunks + Grok (xAI)",
    version="1.0.0",
)

STATIC_DIR = Path(__file__).resolve().parent / "static"

TAX_KEYWORDS = [
    "tax", "deduction", "filing", "1040", "bracket", "form", "irs",
    "wage", "income", "credit", "refund", "payment", "amended", "return",
    "schedule", "expense", "mortgage", "charitable", "self-employed",
    "schedule c", "itemized", "standard", "eic", "eitc", "roth",
    "ira", "estate", "gift", "penalty", "deadline", "extension",
    "business", "farming", "llc", "corporation", "s corporation",
    "partnership", "head of household", "married", "dependent",
]


def _log_chat_response(
    query: str, resp: ChatResponse, *, cache_hit: bool = False
) -> None:
    qprev = (query or "").replace("\n", " ")[:160]
    logger.info(
        "chat_response answer_source=%s is_tax_topic=%s verification_failed=%s "
        "cache_hit=%s query_preview=%r",
        resp.answer_source,
        resp.is_tax_topic,
        resp.verification_failed,
        cache_hit,
        qprev,
    )


def _cache_and_return(resp: ChatResponse, cache_key: str, query: str) -> ChatResponse:
    _log_chat_response(query, resp)
    if cache_key:
        set_cached_response(cache_key, resp.model_dump())
    return resp


def is_tax_related(query: str) -> bool:
    """Check if query is tax-related using keyword matching and common patterns."""
    query_lower = query.lower()

    for keyword in TAX_KEYWORDS:
        if keyword in query_lower:
            return True

    tax_patterns = [
        r"\b(filing|file|income|tax|deduction|refund|w\d+|1099|1040|x\d+)\b",
        r"\b(w2|wag)\b",
        r"\b(charitable|mortgage|business|expense|schedule)\b",
    ]

    for pattern in tax_patterns:
        if re.search(pattern, query_lower):
            return True

    return False


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Embedding -> vector similarity (top 3) -> context string -> Grok (context-only).
    Repeated normalized questions return from LRU cache (no LLM / retrieval).
    """
    query = request.message
    cache_key = normalize_cache_key(query)
    if cache_key:
        cached = get_cached_response(cache_key)
        if cached is not None:
            hit = ChatResponse(**cached)
            _log_chat_response(query, hit, cache_hit=True)
            return hit

    if not is_tax_related(query):
        return _cache_and_return(
            ChatResponse(
                answer="I only answer tax-related questions based on IRS data. I can help with questions about taxes, deductions, credits, filing requirements, forms, and related topics.",
                sources=[],
                confidence=0.0,
                is_tax_topic=False,
                answer_source="non_tax",
            ),
            cache_key,
            query,
        )

    try:
        extract_keywords(query)

        chunks = rag_pipeline.retrieve(query)
        if not chunks:
            return _cache_and_return(
                ChatResponse(
                    answer="I could not find specific information about your question in my IRS data sources. I only answer questions about federal taxes, forms, and deductions.",
                    sources=[],
                    confidence=0.0,
                    is_tax_topic=True,
                    answer_source="no_chunks",
                ),
                cache_key,
                query,
            )

        confidence, sufficient_for_strict_rag = assess_retrieval_confidence(chunks)
        if not sufficient_for_strict_rag:
            answer = await generate_fallback_answer_without_rag(query)
            verdict = await verify_assistant_answer(
                query, answer, "", unsourced_fallback=True
            )
            if verdict != "VERIFIED":
                # Discard draft answer; do not return or persist it (IRS index / feedback log).
                return _cache_and_return(
                    ChatResponse(
                        answer=VERIFICATION_REJECTED_MESSAGE_FALLBACK,
                        sources=[],
                        confidence=confidence,
                        is_tax_topic=True,
                        verification_failed=True,
                        answer_source="fallback_verification_rejected",
                    ),
                    cache_key,
                    query,
                )
            extracted_chunks = await extract_verified_answer_chunks(query, answer)
            if extracted_chunks:
                append_verified_chunks(query, extracted_chunks, sources=[])
            return _cache_and_return(
                ChatResponse(
                    answer=answer,
                    sources=[],
                    confidence=confidence,
                    is_tax_topic=True,
                    extracted_chunks=extracted_chunks,
                    answer_source="fallback",
                ),
                cache_key,
                query,
            )

        context = rag_pipeline.build_context(chunks)
        sources: List[str] = list(dict.fromkeys(chunk["source"] for chunk in chunks))

        answer = await generate_answer_from_retrieved_context(query, context)
        if not answer:
            return _cache_and_return(
                ChatResponse(
                    answer="The model returned an empty response. Please try again.",
                    sources=sources,
                    confidence=confidence,
                    is_tax_topic=True,
                    answer_source="rag_model_empty",
                ),
                cache_key,
                query,
            )

        verdict = await verify_assistant_answer(query, answer, context)
        if verdict != "VERIFIED":
            # Discard draft answer; do not return or persist it.
            return _cache_and_return(
                ChatResponse(
                    answer=VERIFICATION_REJECTED_MESSAGE_RAG,
                    sources=[],
                    confidence=confidence,
                    is_tax_topic=True,
                    verification_failed=True,
                    answer_source="rag_verification_rejected",
                ),
                cache_key,
                query,
            )

        extracted_chunks = await extract_verified_answer_chunks(query, answer)
        if extracted_chunks:
            append_verified_chunks(query, extracted_chunks, sources=sources)
        return _cache_and_return(
            ChatResponse(
                answer=answer,
                sources=sources,
                confidence=confidence,
                is_tax_topic=True,
                extracted_chunks=extracted_chunks,
                answer_source="rag",
            ),
            cache_key,
            query,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        import traceback

        error_msg = str(e)
        print(f"Error processing query: {error_msg}")
        traceback.print_exc()
        err_resp = ChatResponse(
            answer=f"I encountered an error: {error_msg}. Please try rephrasing your question.",
            sources=[],
            confidence=0.0,
            is_tax_topic=True,
            answer_source="error",
        )
        _log_chat_response(query, err_resp)
        return err_resp


@app.get("/")
async def root():
    index = STATIC_DIR / "index.html"
    if index.is_file():
        return FileResponse(index)
    return {
        "service": "Tax Assistant API",
        "version": "1.0.0",
        "description": "AI-powered tax assistant using IRS data and RAG",
        "endpoints": {
            "POST /api/chat": "Send a tax-related question and get an answer with sources",
            "POST /api/feedback": "Submit helpful feedback for fallback answers",
        },
        "requirements": {
            "Only tax-related questions",
            "Based on retrieved IRS chunks + Grok (xAI)",
            "Top 3 relevant chunks retrieved",
        },
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Tax Assistant API"}


@app.get("/sources")
async def get_all_sources():
    return {"sources": rag_pipeline.get_all_sources()}


@app.post("/api/feedback")
async def submit_feedback(body: FeedbackRequest):
    """
    Persist feedback only when the user marked the answer as helpful (positive signal).
    "Not helpful" must not be stored (no processing).
    Verification-rejection placeholders must not be stored (discarded-answer path).
    """
    if not body.helpful:
        return {"ok": True, "stored": False}
    if _is_verification_rejection_placeholder(body.answer):
        return {"ok": True, "stored": False}
    append_feedback(body.query.strip(), body.answer, True)
    return {"ok": True, "stored": True}


@app.get("/{filename:path}")
async def spa_public_files(filename: str):
    """Serve Vite dist files (e.g. /assets/*, /zyvzen-logo.png) from static/."""
    if filename.startswith("api/"):
        raise HTTPException(status_code=404)
    for part in filename.split("/"):
        if part == "..":
            raise HTTPException(status_code=404)
    fp = (STATIC_DIR / filename).resolve()
    try:
        fp.relative_to(STATIC_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=404) from None
    if not fp.is_file():
        raise HTTPException(status_code=404)
    media_type, _ = mimetypes.guess_type(str(fp))
    return FileResponse(fp, media_type=media_type)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
