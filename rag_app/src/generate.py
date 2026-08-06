"""
generate.py — LLM answer generation via OpenRouter with citations.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "google/gemma-4-31b-it:free"
NOT_AVAILABLE_MSG = "The information requested is not available in the supplied documents."


def _build_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[{i}] Source: {chunk['doc_name']}, Page {chunk['page_number']}\n"
            f"{chunk['text']}"
        )
    return "\n\n".join(parts)


def _format_citations(chunks: list[dict]) -> str:
    lines = ["\n--- Citations ---"]
    for i, chunk in enumerate(chunks, start=1):
        lines.append(
            f"\n[{i}] Document: {chunk['doc_name']}\n"
            f"    Page: {chunk['page_number']}\n"
            f"    Retrieved text: \"{chunk['text'][:300]}{'...' if len(chunk['text']) > 300 else ''}\""
        )
    return "\n".join(lines)


def generate_answer(query: str, chunks: list[dict]) -> str:
    if not chunks:
        return NOT_AVAILABLE_MSG

    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY is not set in the environment.")

    client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)

    context = _build_context(chunks)
    system_prompt = (
        "You are a legal research assistant. Answer the user's question using ONLY "
        "the context provided below. Do not use any external knowledge. "
        "If the context does not contain enough information to answer, say so explicitly. "
        "Be concise and factual."
    )
    user_prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=512,
    )

    answer = response.choices[0].message.content.strip()
    citations = _format_citations(chunks)
    return f"{answer}\n{citations}"
