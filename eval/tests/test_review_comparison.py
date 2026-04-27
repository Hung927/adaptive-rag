"""Review comparison evaluation — compare 4 pipeline combinations.

Combinations:
  A: no review              (enable_review_retrieval=False, enable_review_generation=False)
  B: retrieval review only  (enable_review_retrieval=True,  enable_review_generation=False)
  C: generation review only (enable_review_retrieval=False, enable_review_generation=True)
  D: both reviews           (enable_review_retrieval=True,  enable_review_generation=True)

Metrics per combination:
  - faithfulness
  - answer_relevancy
  - context_precision
  - context_recall
  - retrieval_review_triggered  (rate of review_retrieval failing → retry)
  - generation_review_triggered (rate of review_generation failing → retry)
"""

from __future__ import annotations

import logging
import os
import time

import pytest

from eval.helpers import load_goldens, write_csv

logger = logging.getLogger(__name__)

COMBINATIONS = [
    {"name": "A_no_review",       "enable_review_retrieval": False, "enable_review_generation": False},
    {"name": "B_ret_review_only", "enable_review_retrieval": True,  "enable_review_generation": False},
    {"name": "C_gen_review_only", "enable_review_retrieval": False, "enable_review_generation": True},
    {"name": "D_both_reviews",    "enable_review_retrieval": True,  "enable_review_generation": True},
]


@pytest.fixture(scope="module")
def goldens():
    path = os.getenv("EVAL_GOLDENS_PATH", "eval/datasets/goldens.json")
    limit = int(os.getenv("EVAL_LIMIT", "0"))  # 0 = no limit, run all
    return load_goldens(path, limit)


@pytest.fixture(scope="module")
def rag_components():
    from rag.core.config import load_settings
    from rag.retrieval.store import QdrantStore

    settings = load_settings()
    store = QdrantStore(settings)
    return settings, store


@pytest.fixture(scope="module")
def ragas_azure_config(request):
    return request.getfixturevalue("ragas_azure_config")


def _run_combination(goldens, settings, store, combo: dict) -> tuple[dict, dict]:
    """Run one pipeline combination over all goldens."""
    from rag.pipeline.builder import run_pipeline

    top_k = int(os.getenv("EVAL_TOP_K", "5"))
    name = combo["name"]

    logger.info("=" * 80)
    logger.info("[%s] 開始執行", name)
    logger.info("[%s] enable_review_retrieval : %s", name, combo["enable_review_retrieval"])
    logger.info("[%s] enable_review_generation: %s", name, combo["enable_review_generation"])

    questions, answers, ground_truths, contexts_list = [], [], [], []
    latencies = []
    ret_review_triggered = 0
    gen_review_triggered = 0

    for idx, golden in enumerate(goldens):
        q = golden["question"]
        gt = golden["ground_truth"]

        logger.info("-" * 60)
        logger.info("[%s] 題目 %d / %d", name, idx + 1, len(goldens))
        logger.info("[%s]   question    : %s", name, q)
        logger.info("[%s]   ground_truth: %s", name, gt)

        t0 = time.time()
        result = run_pipeline(
            settings, store,
            query=q,
            top_k=top_k,
            enable_review_retrieval=combo["enable_review_retrieval"],
            enable_review_generation=combo["enable_review_generation"],
        )
        elapsed = time.time() - t0
        latencies.append(elapsed)

        answer = result["answer"]
        sources = result["sources"]
        contexts = [s["text"] for s in sources]

        logger.info("[%s]   answer      : %s", name, answer[:300].replace("\n", " "))
        logger.info("[%s]   sources 數量: %d 筆", name, len(sources))
        for i, s in enumerate(sources):
            logger.info(
                "[%s]   source[%d] %s p%s | %s",
                name, i,
                s["source_file"], s.get("page_number", "-"),
                s["text"][:100].replace("\n", " "),
            )
        logger.info("[%s]   latency     : %.2fs", name, elapsed)

        # Track review trigger counts
        if combo["enable_review_retrieval"] and result.get("review_passed") is False:
            ret_review_triggered += 1
            logger.info("[%s]   review_retrieval triggered retry", name)
        if combo["enable_review_generation"] and result.get("review_passed") is False:
            gen_review_triggered += 1
            logger.info("[%s]   review_generation triggered retry", name)

        questions.append(q)
        answers.append(answer)
        ground_truths.append(gt)
        contexts_list.append(contexts)

    ragas_data = {
        "question": questions,
        "answer": answers,
        "ground_truth": ground_truths,
        "contexts": contexts_list,
    }

    summary = {
        "combination": name,
        "latencies": latencies,
        "avg_latency_s": round(sum(latencies) / len(latencies), 2),
        "ret_review_triggered": ret_review_triggered,
        "gen_review_triggered": gen_review_triggered,
    }

    logger.info("[%s] 完成，avg_latency=%.2fs, ret_triggered=%d, gen_triggered=%d",
                name, summary["avg_latency_s"], ret_review_triggered, gen_review_triggered)

    return ragas_data, summary


def test_review_comparison(goldens, rag_components, ragas_azure_config, eval_log_path):
    """Run all 4 combinations and output a comparison CSV."""
    from ragas import evaluate
    from ragas.metrics import answer_correctness, answer_relevancy, context_precision, context_recall, faithfulness
    from datasets import Dataset

    settings, store = rag_components
    ragas_llm = ragas_azure_config["llm"]
    ragas_embeddings = ragas_azure_config["embeddings"]
    ragas_run_config = ragas_azure_config["run_config"]

    logger.info("=" * 80)
    logger.info("開始 Review Comparison 評估，共 %d 題，%d 種組合", len(goldens), len(COMBINATIONS))
    logger.info("Log 檔案：%s", eval_log_path)

    all_rows = []       # per-question rows
    summary_rows = []   # average rows

    for combo in COMBINATIONS:
        ragas_data, summary = _run_combination(goldens, settings, store, combo)
        name = combo["name"]

        # RAGAS evaluation
        logger.info("[%s] 開始 RAGAS 評估...", name)
        dataset = Dataset.from_dict(ragas_data)
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, answer_correctness, context_precision, context_recall],
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            run_config=ragas_run_config,
        )
        df = result.to_pandas()

        # Log & collect per-question rows
        for i, row_data in df.iterrows():
            faith  = round(row_data.get("faithfulness", 0), 4)
            relev  = round(row_data.get("answer_relevancy", 0), 4)
            corr   = round(row_data.get("answer_correctness", 0), 4)
            prec   = round(row_data.get("context_precision", 0), 4)
            rec    = round(row_data.get("context_recall", 0), 4)
            logger.info(
                "[%s] 題目 %d RAGAS | faith=%.4f | relevancy=%.4f | correctness=%.4f | precision=%.4f | recall=%.4f",
                name, i + 1, faith, relev, corr, prec, rec,
            )
            all_rows.append({
                "row_type":               "per_question",
                "combination":            name,
                "enable_review_retrieval":  combo["enable_review_retrieval"],
                "enable_review_generation": combo["enable_review_generation"],
                "question_no":            i + 1,
                "question":               ragas_data["question"][i],
                "faithfulness":           faith,
                "answer_relevancy":       relev,
                "answer_correctness":     corr,
                "context_precision":      prec,
                "context_recall":         rec,
                "latency_s":              round(summary["latencies"][i], 2),
                "ret_review_triggered":   "",
                "gen_review_triggered":   "",
            })

        avg_faith       = round(df["faithfulness"].mean(), 4)
        avg_relevancy   = round(df["answer_relevancy"].mean(), 4)
        avg_correctness = round(df["answer_correctness"].mean(), 4)
        avg_precision   = round(df["context_precision"].mean(), 4)
        avg_recall      = round(df["context_recall"].mean(), 4)

        logger.info(
            "[%s] 平均 | faith=%.4f | relevancy=%.4f | correctness=%.4f | precision=%.4f | recall=%.4f | latency=%.2fs",
            name, avg_faith, avg_relevancy, avg_correctness, avg_precision, avg_recall, summary["avg_latency_s"],
        )

        avg_row = {
            "row_type":               "average",
            "combination":            name,
            "enable_review_retrieval":  combo["enable_review_retrieval"],
            "enable_review_generation": combo["enable_review_generation"],
            "question_no":            "AVG",
            "question":               "",
            "faithfulness":           avg_faith,
            "answer_relevancy":       avg_relevancy,
            "answer_correctness":     avg_correctness,
            "context_precision":      avg_precision,
            "context_recall":         avg_recall,
            "latency_s":              summary["avg_latency_s"],
            "ret_review_triggered":   summary["ret_review_triggered"],
            "gen_review_triggered":   summary["gen_review_triggered"],
        }
        all_rows.append(avg_row)
        summary_rows.append(avg_row)

    csv_path = write_csv(all_rows, "review_comparison")

    logger.info("=" * 80)
    logger.info("評估完成 — CSV 儲存至：%s", csv_path)
    logger.info("%-25s %7s %10s %12s %10s %8s %9s", "組合", "faith", "relevancy", "correctness", "precision", "recall", "latency")
    logger.info("-" * 90)
    for r in summary_rows:
        logger.info(
            "%-25s %7.4f %10.4f %12.4f %10.4f %8.4f %8.2fs",
            r["combination"], r["faithfulness"], r["answer_relevancy"],
            r["answer_correctness"], r["context_precision"], r["context_recall"], r["latency_s"],
        )
    logger.info("Log 檔案：%s", eval_log_path)

    assert summary_rows, "No results generated"
