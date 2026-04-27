"""Generation layer — build prompt from context and call LLM."""

from __future__ import annotations

from openai import AzureOpenAI

from rag.core.llm import chat_completion
from rag.core.types import QueryResult


SYSTEM_PROMPT = """你是一個 RAG 問答助手。根據提供的參考資料回答使用者的問題。

規則：
1. 只根據參考資料中的內容回答，不要編造資訊
2. 如果參考資料不足以回答問題，請明確說明
3. 回答時引用來源文件名稱
4. 使用繁體中文回答"""


def build_context(results: list[QueryResult]) -> str:
    """Format retrieval results into context string."""
    if not results:
        return "（沒有找到相關參考資料）"

    parts: list[str] = []
    for i, r in enumerate(results, 1):
        source = r["source_file"]
        page = r.get("page_number")
        page_str = f" (第{page}頁)" if page else ""
        parts.append(f"[{i}] 來源: {source}{page_str}\n{r['text']}")

    return "\n\n---\n\n".join(parts)


def generate_answer(
    client: AzureOpenAI,
    deployment: str,
    query: str,
    context: str,
    system_prompt: str = SYSTEM_PROMPT,
) -> str:
    """Generate an answer from context using LLM."""
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"參考資料：\n{context}\n\n問題：{query}",
        },
    ]
    return chat_completion(client, messages, deployment)
