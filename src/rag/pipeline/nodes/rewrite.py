"""Rewrite node — query rewriting for better retrieval (Pattern 1: Reflection)."""

from __future__ import annotations

import logging

from openai import AzureOpenAI

from rag.core.llm import chat_completion
from rag.pipeline.state import PipelineState

logger = logging.getLogger(__name__)

REWRITE_PROMPT = """你是一個查詢改寫助手。你的任務是改寫使用者的查詢，使其更適合向量搜尋。

規則：
1. 保留原始查詢的核心意圖
2. 展開縮寫和專業術語
3. 補充可能的同義詞或相關概念
4. 輸出改寫後的查詢（只輸出查詢本身，不要加說明）"""

REWRITE_WITH_FEEDBACK_PROMPT = """你是一個查詢改寫助手。上一次的向量搜尋結果不夠好，請根據失敗原因重新改寫查詢。

規則：
1. 保留原始查詢的核心意圖
2. 針對失敗原因調整查詢方向
3. 展開縮寫和專業術語，補充同義詞或相關概念
4. 輸出改寫後的查詢（只輸出查詢本身，不要加說明）"""


def rewrite_node(
    state: PipelineState,
    client: AzureOpenAI,
    deployment: str,
) -> dict:
    """Rewrite the user query for better retrieval.

    On first attempt: no rewrite (skipped, handled by builder entry point).
    On retry due to retrieval failure: rewrite with review feedback.
    """
    query = state["query"]
    feedback = state.get("retrieval_review_feedback", "")

    if feedback:
        # Retry path: rewrite guided by review feedback
        messages = [
            {"role": "system", "content": REWRITE_WITH_FEEDBACK_PROMPT},
            {
                "role": "user",
                "content": f"原始查詢：{query}\n\n上次檢索失敗原因：{feedback}",
            },
        ]
    else:
        # Should not normally be reached (first attempt skips rewrite)
        messages = [
            {"role": "system", "content": REWRITE_PROMPT},
            {"role": "user", "content": query},
        ]

    rewritten = chat_completion(client, messages, deployment, temperature=0.1)

    logger.info("[rewrite] 原始問題: %s", query)
    logger.info("[rewrite] 失敗原因: %s", feedback or "（無）")
    logger.info("[rewrite] 改寫結果: %s", rewritten.strip())

    return {"rewritten_query": rewritten.strip()}
