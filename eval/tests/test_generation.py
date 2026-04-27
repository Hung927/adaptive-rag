"""Generation quality evaluation using RAGAS metrics."""

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
    from rag.core.config import load_settings
    from rag.pipeline.builder import run_pipeline
    from rag.retrieval.store import QdrantStore

    settings = load_settings()
    store = QdrantStore(settings)
    return settings, store


def test_generation_quality(goldens, rag_components):
    """Evaluate generation quality: faithfulness and answer relevancy."""
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, faithfulness
    from datasets import Dataset

    settings, store = rag_components

    questions = []
    answers = []
    ground_truths = []
    contexts_list = []

    for golden in goldens:
        from rag.pipeline.builder import run_pipeline

        result = run_pipeline(settings, store, golden["question"])

        questions.append(golden["question"])
        answers.append(result["answer"])
        ground_truths.append(golden["ground_truth"])
        contexts_list.append([s["text"] for s in result["sources"]])

    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "ground_truth": ground_truths,
            "contexts": contexts_list,
        }
    )

    result = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
    df = result.to_pandas()

    rows = df.to_dict("records")
    csv_path = write_csv(rows, "generation_eval")
    print(f"\nGeneration eval results saved to: {csv_path}")

    faith_threshold = float(os.getenv("EVAL_THRESHOLD_FAITHFULNESS", "0.5"))
    relevancy_threshold = float(os.getenv("EVAL_THRESHOLD_RELEVANCY", "0.5"))

    avg_faith = df["faithfulness"].mean()
    avg_relevancy = df["answer_relevancy"].mean()

    print(f"Avg faithfulness: {avg_faith:.3f} (threshold: {faith_threshold})")
    print(f"Avg answer_relevancy: {avg_relevancy:.3f} (threshold: {relevancy_threshold})")

    assert avg_faith >= faith_threshold, (
        f"Faithfulness {avg_faith:.3f} < {faith_threshold}"
    )
    assert avg_relevancy >= relevancy_threshold, (
        f"Answer relevancy {avg_relevancy:.3f} < {relevancy_threshold}"
    )
