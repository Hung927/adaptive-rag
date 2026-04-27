"""Review Retrieval node — Step 1: are retrieved chunks sufficient to answer the question?"""

from __future__ import annotations

import json
import logging
import re

from openai import AzureOpenAI

from rag.core.llm import chat_completion
from rag.pipeline.state import PipelineState

logger = logging.getLogger(__name__)

REVIEW_RETRIEVAL_PROMPT = """你是一個 RAG 檢索品質審查員。

逐一檢查每個 chunk，找出其中包含足以直接回答問題的具體資訊的 chunk。
判斷這些 chunk 合在一起是否足以完整回答問題。

請以 JSON 格式回覆：
{
    "confirmed_chunk_ids": [0, 2, ...],
    "passed": true/false,
    "feedback": "說明為何參考資料不足（passed=false 時填寫）"
}

注意：confirmed_chunk_ids 只放能直接回答問題的 chunk，不論 passed 為何都必須填入。"""


def review_retrieval_node(
    state: PipelineState,
    client: AzureOpenAI,
    deployment: str,
) -> dict:
    """Step 1 review: check if retrieved chunks are sufficient to answer the question.

    Always outputs confirmed_chunks (relevant chunks) regardless of pass/fail.
    - passed=True  → confirmed_chunks go to generate
    - passed=False → confirmed_chunks are carried over during re-retrieval
    """
    query = state["query"]
    chunks = state.get("retrieved_chunks", [])
    attempts = state.get("retrieval_review_attempts", 0)

    numbered_chunks = "\n---\n".join(
        f"[Chunk {i}]\n{c['text']}" for i, c in enumerate(chunks)
    )

    logger.info("=" * 60)
    logger.info("[review_retrieval] 輸入")
    logger.info("[review_retrieval]   問題: %s", query)
    for i, c in enumerate(chunks):
        logger.info(
            "[review_retrieval]   Chunk %d | %s p%s | 相似度: %.3f | %s",
            i, c["source_file"], c.get("page_number", "-"),
            c["similarity"], c["text"][:80].replace("\n", " "),
        )

    messages = [
        {"role": "system", "content": REVIEW_RETRIEVAL_PROMPT},
        {
            "role": "user",
            "content": f"問題：{query}\n\n參考資料：\n{numbered_chunks}",
        },
    ]

    response = chat_completion(client, messages, deployment, temperature=0.0, json_mode=True)

    try:
        cleaned = re.sub(r"^```(?:json)?\s*", "", response.strip())
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
        result = json.loads(cleaned)
        passed = result.get("passed", True)
        confirmed_ids = result.get("confirmed_chunk_ids", list(range(len(chunks))))
        feedback = result.get("feedback", "")
    except json.JSONDecodeError:
        logger.warning("[review_retrieval] 回傳非 JSON，視為通過 | raw=%s", response[:300])
        passed = True
        confirmed_ids = list(range(len(chunks)))
        feedback = ""

    confirmed_chunks = [chunks[i] for i in confirmed_ids if i < len(chunks)]

    logger.info("[review_retrieval] 輸出")
    logger.info("[review_retrieval]   passed        : %s", passed)
    logger.info("[review_retrieval]   confirmed_ids : %s", confirmed_ids)
    logger.info(
        "[review_retrieval]   confirmed_chunks: %d 筆 / 共 %d 筆",
        len(confirmed_chunks), len(chunks),
    )
    if not passed:
        logger.info("[review_retrieval]   feedback      : %s", feedback)
    logger.info("=" * 60)

    return {
        "retrieval_review_passed": passed,
        "retrieval_review_feedback": feedback,
        "retrieval_review_attempts": attempts + 1,
        "confirmed_chunks": confirmed_chunks,
    }
