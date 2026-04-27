"""Evaluate node — LLM-as-judge scoring of the final RAG output."""

from __future__ import annotations

import json
import logging
import re

from openai import AzureOpenAI

from rag.core.llm import chat_completion
from rag.pipeline.state import PipelineState

logger = logging.getLogger(__name__)

EVALUATE_PROMPT = """你是一個 RAG 系統的評估專家。
請針對以下三個面向，對 RAG 系統的回答進行評分（每項 0–5 分，整數）：

1. **faithfulness（忠實性）**
   - 5：回答所有內容都有參考資料支撐，沒有捏造
   - 3：大部分正確，少量推論或細節不明確
   - 0：包含與參考資料矛盾或完全捏造的內容

2. **answer_relevance（回答相關性）**
   - 5：完整、直接地回答了問題
   - 3：有回答但不完整或稍微偏題
   - 0：完全沒有回答問題

3. **context_precision（上下文精準性）**
   - 5：所有參考資料都與問題高度相關
   - 3：部分資料相關，部分無關
   - 0：參考資料與問題完全無關

請以 JSON 格式回覆，不要加任何說明：
{
    "faithfulness": <0-5>,
    "answer_relevance": <0-5>,
    "context_precision": <0-5>,
    "reasoning": {
        "faithfulness": "一句話說明評分理由",
        "answer_relevance": "一句話說明評分理由",
        "context_precision": "一句話說明評分理由"
    }
}"""


def evaluate_node(
    state: PipelineState,
    client: AzureOpenAI,
    deployment: str,
) -> dict:
    """Evaluate the final RAG output using LLM-as-judge."""
    query = state["query"]
    answer = state.get("answer", "")
    chunks = state.get("confirmed_chunks") or state.get("retrieved_chunks", [])

    numbered_chunks = "\n---\n".join(
        f"[Chunk {i}]\n{c['text']}" for i, c in enumerate(chunks)
    )

    logger.info("=" * 60)
    logger.info("[evaluate] 輸入")
    logger.info("[evaluate]   問題: %s", query)
    logger.info("[evaluate]   回答: %s", answer[:200].replace("\n", " "))
    logger.info("[evaluate]   Chunks 數量: %d", len(chunks))

    messages = [
        {"role": "system", "content": EVALUATE_PROMPT},
        {
            "role": "user",
            "content": (
                f"問題：{query}\n\n"
                f"參考資料：\n{numbered_chunks}\n\n"
                f"回答：{answer}"
            ),
        },
    ]

    response = chat_completion(client, messages, deployment, temperature=0.0, json_mode=True)

    try:
        # Some proxies wrap the JSON in markdown code fences or extra text;
        # try to extract the first {...} block if direct parse fails.
        raw = response.strip()
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                result = json.loads(match.group())
            else:
                raise

        faithfulness = int(result.get("faithfulness", 0))
        answer_relevance = int(result.get("answer_relevance", 0))
        context_precision = int(result.get("context_precision", 0))
        reasoning = result.get("reasoning", {})
    except (json.JSONDecodeError, ValueError):
        logger.warning("[evaluate] 回傳非預期格式，原始內容: %r", response[:300])
        faithfulness = 0
        answer_relevance = 0
        context_precision = 0
        reasoning = {}

    logger.info("[evaluate] 輸出")
    logger.info("[evaluate]   faithfulness    : %d/5", faithfulness)
    logger.info("[evaluate]   answer_relevance: %d/5", answer_relevance)
    logger.info("[evaluate]   context_precision: %d/5", context_precision)
    if reasoning:
        logger.info("[evaluate]   reasoning: %s", json.dumps(reasoning, ensure_ascii=False))
    logger.info("=" * 60)

    return {
        "eval_faithfulness": faithfulness,
        "eval_answer_relevance": answer_relevance,
        "eval_context_precision": context_precision,
        "eval_reasoning": reasoning,
    }
