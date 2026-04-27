"""Pipeline state — LangGraph state definition."""

from __future__ import annotations

from typing import TypedDict

from rag.core.types import QueryResult


class PipelineState(TypedDict, total=False):
    """State flowing through the RAG LangGraph pipeline."""

    # Input
    query: str

    # After rewrite node
    rewritten_query: str

    # After retrieve node
    retrieved_chunks: list[QueryResult]

    # After review_retrieval node
    # Chunks confirmed as relevant to the question (always set by review_retrieval)
    confirmed_chunks: list[QueryResult]
    retrieval_review_passed: bool
    retrieval_review_feedback: str
    retrieval_review_attempts: int

    # After generate node
    answer: str

    # After review_generation node
    generation_review_passed: bool
    generation_review_feedback: str
    generation_review_attempts: int

    # After evaluate node
    eval_faithfulness: int        # 0–5
    eval_answer_relevance: int    # 0–5
    eval_context_precision: int   # 0–5
    eval_reasoning: dict          # per-metric reasoning strings
