"""Retrieval quality evaluation using RAGAS metrics."""

from __future__ import annotations

import os

import pytest

from eval.helpers import load_goldens, write_csv


@pytest.fixture(scope="module")
def goldens():
    path = os.getenv("EVAL_GOLDENS_PATH", "eval/datasets/goldens.json")
    limit = int(os.getenv("EVAL_LIMIT", "3"))
    return load_goldens(path, limit)


@pytest.fixture(scope="module")
def rag_components():
    """Initialize RAG components for evaluation."""
    from rag.core.config import load_settings
    from rag.retrieval.store import QdrantStore

    settings = load_settings()
    store = QdrantStore(settings)
    return settings, store


def test_retrieval_quality(goldens, rag_components):
    """Evaluate retrieval quality: context precision and recall."""
    from ragas import evaluate
    from ragas.metrics import context_precision, context_recall
    from datasets import Dataset

    settings, store = rag_components
    top_k = int(os.getenv("EVAL_TOP_K", "5"))

    questions = []
    ground_truths = []
    contexts_list = []

    for golden in goldens:
        query = golden["question"]
        results = store.query(query_text=query, top_k=top_k)
        retrieved_contexts = [r["text"] for r in results]

        questions.append(query)
        ground_truths.append(golden["ground_truth"])
        contexts_list.append(retrieved_contexts)

    dataset = Dataset.from_dict(
        {
            "question": questions,
            "ground_truth": ground_truths,
            "contexts": contexts_list,
        }
    )

    result = evaluate(dataset, metrics=[context_precision, context_recall])
    df = result.to_pandas()

    rows = df.to_dict("records")
    csv_path = write_csv(rows, "retrieval_eval")
    print(f"\nRetrieval eval results saved to: {csv_path}")

    precision_threshold = float(os.getenv("EVAL_THRESHOLD_PRECISION", "0.5"))
    recall_threshold = float(os.getenv("EVAL_THRESHOLD_RECALL", "0.5"))

    avg_precision = df["context_precision"].mean()
    avg_recall = df["context_recall"].mean()

    print(f"Avg context_precision: {avg_precision:.3f} (threshold: {precision_threshold})")
    print(f"Avg context_recall: {avg_recall:.3f} (threshold: {recall_threshold})")

    assert avg_precision >= precision_threshold, (
        f"Context precision {avg_precision:.3f} < {precision_threshold}"
    )
    assert avg_recall >= recall_threshold, (
        f"Context recall {avg_recall:.3f} < {recall_threshold}"
    )
