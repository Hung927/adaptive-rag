"""Layer 2: Foundation types — TypedDict definitions for all I/O."""

from typing import NotRequired, TypedDict


class IngestResult(TypedDict):
    """Single document ingestion result."""

    status: str
    source_file: str
    total_chunks: int
    error: NotRequired[str]


class QueryResult(TypedDict):
    """Single retrieval result (one chunk)."""

    chunk_id: str
    text: str
    source_file: str
    page_number: NotRequired[int]
    chunk_index: int
    similarity: float


class DocumentInfo(TypedDict):
    """Document summary."""

    document_id: str
    source_file: str
    total_chunks: int


class EvalScores(TypedDict):
    """LLM-as-judge evaluation scores (0–5 each)."""

    faithfulness: int
    answer_relevance: int
    context_precision: int
    reasoning: NotRequired[dict]


class GenerationResult(TypedDict):
    """RAG pipeline generation result."""

    answer: str
    sources: list[QueryResult]
    query: str
    rewritten_query: NotRequired[str]
    review_passed: NotRequired[bool]
    eval_scores: NotRequired[EvalScores]
