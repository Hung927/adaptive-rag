"""Review Generation node — Step 2: did the LLM answer correctly using the confirmed chunks?"""

from __future__ import annotations

import json
import logging

from openai import AzureOpenAI

from rag.core.llm import chat_completion
from rag.pipeline.state import PipelineState

logger = logging.getLogger(__name__)

REVIEW_GENERATION_PROMPT = """你是一個 RAG 回答品質審查員。

參考資料已確認與問題相關。你的任務是判斷「回答」的事實內容是否正確。

【參考資料格式說明】
參考資料可能是 markdown 格式，包含 table 結構與 <br> 換行標籤。
請將 <br> 視為換行符號，仔細解讀 table 中每一行的內容再進行判斷。

【評估重點：事實正確性】
1. 回答中的具體數字、名稱、條件是否與參考資料一致？
2. 回答是否編造了參考資料中不存在的內容？

【不列入評估的項目】
- 回答的格式、語氣、用詞
- 有沒有標注來源或參考資料編號
- 回答是否夠詳細或夠簡潔

【判斷標準】
- passed=true：回答的事實內容正確，沒有與參考資料矛盾或編造的內容
- passed=false：回答包含與參考資料不符的事實，或編造了參考資料中沒有的內容

請以 JSON 格式回覆：
{
    "passed": true/false,
    "feedback": "具體說明哪個事實有誤，以及正確的內容為何（passed=false 時填寫）"
}

只輸出 JSON，不要加其他說明。"""


def review_generation_node(
    state: PipelineState,
    client: AzureOpenAI,
    deployment: str,
) -> dict:
    """Step 2 review: check if the generated answer correctly uses the confirmed chunks."""
    query = state["query"]
    answer = state.get("answer", "")
    chunks = state.get("confirmed_chunks") or state.get("retrieved_chunks", [])
    attempts = state.get("generation_review_attempts", 0)

    numbered_chunks = "\n---\n".join(
        f"[Chunk {i}]\n{c['text']}" for i, c in enumerate(chunks)
    )

    logger.info("=" * 60)
    logger.info("[review_generation] 輸入")
    logger.info("[review_generation]   問題: %s", query)
    logger.info("[review_generation]   回答: %s", answer[:200].replace("\n", " "))
    for i, c in enumerate(chunks):
        logger.info(
            "[review_generation]   Chunk %d | %s p%s | %s",
            i, c["source_file"], c.get("page_number", "-"),
            c["text"][:80].replace("\n", " "),
        )

    messages = [
        {"role": "system", "content": REVIEW_GENERATION_PROMPT},
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
        result = json.loads(response)
        passed = result.get("passed", True)
        feedback = result.get("feedback", "")
    except json.JSONDecodeError:
        logger.warning("[review_generation] 回傳非 JSON，視為通過")
        passed = True
        feedback = ""

    logger.info("[review_generation] 輸出")
    logger.info("[review_generation]   passed  : %s", passed)
    if not passed:
        logger.info("[review_generation]   feedback: %s", feedback)
    logger.info("=" * 60)

    return {
        "generation_review_passed": passed,
        "generation_review_feedback": feedback,
        "generation_review_attempts": attempts + 1,
    }
