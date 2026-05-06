"""
Chat completions for RAG-grounded answers.

Supports:
- **xAI Grok** — key from https://console.x.ai (`XAI_API_KEY`, base https://api.x.ai/v1)
- **Groq** (OpenAI-compatible) — key from https://console.groq.com starts with `gsk_`
  (`GROQ_API_KEY`, or `XAI_API_KEY` if the value is a Groq key)
"""
import json
import os
import re
from pathlib import Path
from typing import List, Literal, Optional, Tuple

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

DEFAULT_XAI_BASE = "https://api.x.ai/v1"
GROQ_BASE = "https://api.groq.com/openai/v1"
DEFAULT_XAI_MODEL = "grok-3-mini"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
MAX_COMPLETION_TOKENS = 512
FALLBACK_MAX_COMPLETION_TOKENS = 256
VERIFICATION_MAX_COMPLETION_TOKENS = 16
CHUNK_EXTRACTION_MAX_COMPLETION_TOKENS = 600

FALLBACK_DISCLAIMER = (
    "This answer is AI-generated and not directly from official sources"
)

FALLBACK_SYSTEM_PROMPT = """You are a US tax information assistant in FALLBACK mode.

No IRS document excerpts are attached to this request. Answer from general knowledge only.

Requirements:
- Be very short: a few sentences or a small bullet list—no long essays.
- Be clear and directly address the user's question.
- Do not claim the answer is quoted from or verified against IRS materials.
- Do not invent IRS URLs, form citations, or a "Sources" section.
- If you cannot answer briefly and safely, say so in one or two sentences and suggest IRS.gov or a qualified tax professional.
- Do not add your own disclaimer about AI or official sources (the app adds one)."""


RAG_SYSTEM_PROMPT = """You are a US Tax Assistant.

Rules:
- Only answer using the provided IRS context
- Do NOT use prior knowledge
- If the answer is not in context, say exactly:
  "I could not find this in IRS sources"
- Be concise and step-by-step
- Always end with a "## Sources" section listing only IRS URLs that appear in the CONTEXT blocks (no URLs not present in CONTEXT)"""


VERIFICATION_SYSTEM_PROMPT = """You are an independent verification judge for a US tax assistant.

You receive:
1) USER QUESTION
2) ASSISTANT ANSWER (the exact text that may be shown to the user)
3) CONTEXT — IRS-related excerpts the assistant was required to ground on in sourced mode.
   If CONTEXT is exactly (none) on its own, there were no excerpts (unsourced fallback mode).

Rules (strict — default to REJECTED when in doubt):
- When CONTEXT is not (none): every substantive factual claim in the ASSISTANT ANSWER must be directly and fully supported by CONTEXT. Reject contradictions, speculation, invented details, partial answers, hedged or uncertain conclusions presented as firm guidance, and anything beyond CONTEXT.
- When CONTEXT is (none): reject partial or uncertain answers; reject specific dollar amounts, thresholds, or form procedures unless clearly generic and cautiously framed (not as official fact).
- Reject internal inconsistencies or overclaims relative to the evidence.

Mandatory output (nothing else):
- Exactly one line, one word only: VERIFIED or REJECTED
- No punctuation, no explanation, no markdown."""


FALLBACK_VERIFICATION_SYSTEM_PROMPT = """You verify a short UNSOURCED assistant reply.

No IRS excerpts were attached (CONTEXT is (none)). The model answered from general knowledge only because retrieval did not confidently match the indexed IRS material.

This is NOT the same as a sourced RAG answer: do not require every figure to appear in CONTEXT.

VERIFIED when:
- The reply is coherent, on-topic, and not internally contradictory.
- It does not claim to quote IRS publications, verified IRS data, or official IRS determinations.
- It does not invent IRS.gov URLs or pretend the text came from this assistant's IRS excerpt index.
- If it gives fees, dollar amounts, state filing rules, or dates, it either (a) frames them as general/approximate knowledge with reasonable caution, or (b) tells the user to confirm with the official state agency / current fee schedule — not as an infallible official figure from this system.

REJECTED when:
- It contradicts itself or impersonates an IRS source.
- It states a specific government fee, tax, or legal outcome as an unquestionable fact with no caveat when nothing was sourced (high-confidence false precision).

Output exactly one line, one word only: VERIFIED or REJECTED
No punctuation, no explanation, no markdown."""


CHUNK_EXTRACTION_SYSTEM_PROMPT = """You extract reusable factual snippets from a verified US tax assistant answer.

You receive the USER QUESTION and the ASSISTANT ANSWER (already verified as acceptable).

Task:
- Output ONLY a JSON array containing 1 to 3 strings. No markdown code fences, no keys, no commentary before or after the array.
- Each string must be factual, concise, and reusable on its own (suitable for a retrieval index card).
- Neutral declarative style only: no "you", "we", "I", no greetings, no "here is" / "in summary", no rhetorical questions.
- Remove conversational filler; keep substance (rules, thresholds, steps, definitions) that appear in the answer.
- Omit standalone "## Sources" URL blocks and any AI-source disclaimer lines from the chunks; do not invent URLs or facts.

If the answer is very short, a single-string array is fine. Never output more than 3 strings."""


def _normalize_key(value: Optional[str]) -> str:
    if not value:
        return ""
    return value.strip().strip('"').strip("'")


def _resolve_xai_base_url() -> str:
    raw = (os.getenv("XAI_BASE_URL") or DEFAULT_XAI_BASE).strip().rstrip("/")
    if raw.endswith("/v1"):
        return raw
    return raw + "/v1"


def _build_client() -> Tuple[Optional[AsyncOpenAI], str]:
    """
    Returns (client, provider) where provider is 'xai' or 'groq'.
    """
    groq_key = _normalize_key(os.getenv("GROQ_API_KEY"))
    xai_key = _normalize_key(os.getenv("XAI_API_KEY"))

    if groq_key:
        return AsyncOpenAI(api_key=groq_key, base_url=GROQ_BASE), "groq"
    if not xai_key:
        return None, ""
    if xai_key.startswith("gsk_"):
        return AsyncOpenAI(api_key=xai_key, base_url=GROQ_BASE), "groq"
    return AsyncOpenAI(api_key=xai_key, base_url=_resolve_xai_base_url()), "xai"


def _resolve_chat_model(provider: str) -> str:
    if provider == "groq":
        return (
            os.getenv("GROQ_MODEL") or os.getenv("GROK_MODEL") or DEFAULT_GROQ_MODEL
        ).strip()
    return (os.getenv("GROK_MODEL") or os.getenv("XAI_MODEL") or DEFAULT_XAI_MODEL).strip()


def _parse_verification_verdict(raw: str) -> Literal["VERIFIED", "REJECTED"]:
    """Fail closed: anything other than a lone VERIFIED token is REJECTED."""
    text = (raw or "").strip()
    if not text:
        return "REJECTED"
    first = text.splitlines()[0].strip().upper()
    token = first.split()[0] if first else ""
    if token == "VERIFIED":
        return "VERIFIED"
    return "REJECTED"


def _build_user_message(retrieved_context: str, user_query: str) -> str:
    return (
        "CONTEXT:\n"
        f"{retrieved_context}\n\n"
        "QUESTION:\n"
        f"{user_query}"
    )


async def generate_answer_from_retrieved_context(
    user_query: str,
    retrieved_context: str,
) -> str:
    """
    Call Grok (xAI) or Groq-hosted chat model with strict RAG-only instructions.
    """
    client, provider = _build_client()
    if client is None:
        raise RuntimeError(
            "Set XAI_API_KEY (https://console.x.ai for Grok) or "
            "GROQ_API_KEY (https://console.groq.com for Groq; keys start with gsk_)."
        )

    model = _resolve_chat_model(provider)

    user_content = _build_user_message(retrieved_context, user_query)

    response = await client.chat.completions.create(
        model=model,
        temperature=0.2,
        max_completion_tokens=MAX_COMPLETION_TOKENS,
        messages=[
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )

    choice = response.choices[0].message
    return (choice.content or "").strip()


async def generate_fallback_answer_without_rag(user_query: str) -> str:
    """
    Low-retrieval-confidence path: LLM answer without RAG context.
    Reply is kept short; a fixed disclaimer is always appended.
    """
    client, provider = _build_client()
    if client is None:
        raise RuntimeError(
            "Set XAI_API_KEY (https://console.x.ai for Grok) or "
            "GROQ_API_KEY (https://console.groq.com for Groq; keys start with gsk_)."
        )

    model = _resolve_chat_model(provider)

    response = await client.chat.completions.create(
        model=model,
        temperature=0.3,
        max_completion_tokens=FALLBACK_MAX_COMPLETION_TOKENS,
        messages=[
            {"role": "system", "content": FALLBACK_SYSTEM_PROMPT},
            {"role": "user", "content": user_query.strip()},
        ],
    )

    text = (response.choices[0].message.content or "").strip()
    if not text:
        text = (
            "I could not generate a brief answer. Please try rephrasing your question."
        )
    return f"{text}\n\n{FALLBACK_DISCLAIMER}"


def _build_verification_user_message(
    user_query: str, assistant_answer: str, retrieved_context: str
) -> str:
    ctx = retrieved_context.strip() if retrieved_context.strip() else "(none)"
    return (
        "USER QUESTION:\n"
        f"{user_query.strip()}\n\n"
        "ASSISTANT ANSWER:\n"
        f"{assistant_answer}\n\n"
        "CONTEXT:\n"
        f"{ctx}"
    )


async def verify_assistant_answer(
    user_query: str,
    assistant_answer: str,
    retrieved_context: str,
    *,
    unsourced_fallback: bool = False,
) -> Literal["VERIFIED", "REJECTED"]:
    """
    Second LLM pass: grounding check for RAG, or lighter check for unsourced fallback replies.
    """
    client, provider = _build_client()
    if client is None:
        raise RuntimeError(
            "Set XAI_API_KEY (https://console.x.ai for Grok) or "
            "GROQ_API_KEY (https://console.groq.com for Groq; keys start with gsk_)."
        )

    model = _resolve_chat_model(provider)
    user_content = _build_verification_user_message(
        user_query, assistant_answer, retrieved_context
    )
    system = (
        FALLBACK_VERIFICATION_SYSTEM_PROMPT
        if unsourced_fallback
        else VERIFICATION_SYSTEM_PROMPT
    )

    response = await client.chat.completions.create(
        model=model,
        temperature=0,
        max_completion_tokens=VERIFICATION_MAX_COMPLETION_TOKENS,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
    )

    raw = (response.choices[0].message.content or "").strip()
    return _parse_verification_verdict(raw)


def _text_for_chunk_extraction(verified_answer: str) -> str:
    """Strip meta tail (disclaimer, Sources block) before chunk extraction."""
    text = (verified_answer or "").strip()
    if FALLBACK_DISCLAIMER in text:
        text = text[: text.find(FALLBACK_DISCLAIMER)].strip()
    low = text.lower()
    marker = "## sources"
    pos = low.rfind(marker)
    if pos != -1:
        text = text[:pos].strip()
    return text.strip()


def _fallback_chunks_from_text(body: str) -> List[str]:
    """Deterministic 1–3 chunks if JSON extraction fails."""
    body = (body or "").strip()
    if not body:
        return []
    parts = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    chunks = [p for p in parts if len(p) >= 24][:3]
    if chunks:
        return chunks
    cap = 900
    one = body if len(body) <= cap else body[: cap].rsplit(" ", 1)[0].strip() + "…"
    return [one]


def _parse_chunk_json_array(raw: str) -> List[str]:
    text = (raw or "").strip()
    if not text:
        return []
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"\s*```\s*$", "", text).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    out: List[str] = []
    for item in data:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
    return out[:3]


async def extract_verified_answer_chunks(
    user_query: str, verified_answer: str
) -> List[str]:
    """
    After VERIFIED: convert the shown answer into 1–3 clean, factual, reusable chunks.
    On model/parse failure, uses a simple paragraph split fallback (still 1–3 items).
    """
    body = _text_for_chunk_extraction(verified_answer)
    if not body:
        return []

    client, provider = _build_client()
    if client is None:
        return _fallback_chunks_from_text(body)

    model = _resolve_chat_model(provider)
    user_content = (
        "USER QUESTION:\n"
        f"{user_query.strip()}\n\n"
        "ASSISTANT ANSWER:\n"
        f"{verified_answer.strip()}"
    )

    try:
        response = await client.chat.completions.create(
            model=model,
            temperature=0.2,
            max_completion_tokens=CHUNK_EXTRACTION_MAX_COMPLETION_TOKENS,
            messages=[
                {"role": "system", "content": CHUNK_EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        raw = (response.choices[0].message.content or "").strip()
        parsed = _parse_chunk_json_array(raw)
        if parsed:
            return parsed
    except Exception:
        pass

    return _fallback_chunks_from_text(body)
