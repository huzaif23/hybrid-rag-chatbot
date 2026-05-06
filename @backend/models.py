"""
Data models for the Tax Assistant API
"""
from pydantic import BaseModel, Field
from typing import List


class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    message: str = Field(..., description="User's tax-related question")


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    answer: str = Field(..., description="AI-generated answer")
    sources: List[str] = Field(default_factory=list, description="List of source URLs")
    confidence: float = Field(
        default=0.0,
        description="0–100 from best chunk cosine similarity; strict RAG gated by RETRIEVAL_RAG_MIN_SIMILARITY",
    )
    is_tax_topic: bool = Field(default=True, description="Whether query is tax-related")
    verification_failed: bool = Field(
        default=False,
        description="True if a generated answer was discarded after REJECTED verification",
    )
    extracted_chunks: List[str] = Field(
        default_factory=list,
        description="1–3 factual snippets derived from the answer when verification passed; empty otherwise",
    )
    answer_source: str = Field(
        default="unknown",
        description=(
            "How the answer was produced: "
            "non_tax | no_chunks | fallback | fallback_verification_rejected | "
            "rag | rag_verification_rejected | rag_model_empty | error | unknown"
        ),
    )


class FeedbackRequest(BaseModel):
    """Feedback for a prior reply. Only helpful=True is persisted by the API."""
    query: str = Field(..., min_length=1, description="User question as sent to /chat")
    answer: str = Field(..., description="Assistant answer shown to the user")
    helpful: bool = Field(
        ...,
        description="Must be True to store; False is ignored (not processed)",
    )
