"""Generate node — produce answer from confirmed chunks."""

from __future__ import annotations

import logging

from openai import AzureOpenAI

from rag.generation.generator import SYSTEM_PROMPT, build_context, generate_answer
from rag.pipeline.state import PipelineState

logger = logging.getLogger(__name__)


def generate_node(
    state: PipelineState,
    client: AzureOpenAI,
    deployment: str,
) -> dict:
    """Generate answer from confirmed_chunks (relevant chunks identified by review_retrieval)."""
    chunks = state.get("confirmed_chunks") or state.get("retrieved_chunks", [])
    query = state["query"]
    context = build_context(chunks)

    # Include generation review feedback if retrying
    system_extra = ""
    feedback = state.get("generation_review_feedback")
    if feedback:
        system_extra = f"\n\n上次回答的改善建議：{feedback}\n請根據建議改善你的回答。"

    logger.info("=" * 60)
    logger.info("[generate] 輸入")
    logger.info("[generate]   問題: %s", query)
    for i, c in enumerate(chunks):
        logger.info(
            "[generate]   Chunk %d | %s p%s | %s",
            i, c["source_file"], c.get("page_number", "-"),
            c["text"][:80].replace("\n", " "),
        )
    if feedback:
        logger.info("[generate]   feedback: %s", feedback)

    answer = generate_answer(
        client=client,
        deployment=deployment,
        query=query,
        context=context,
        system_prompt=SYSTEM_PROMPT + system_extra,
    )

    logger.info("[generate] 輸出")
    logger.info("[generate]   回答: %s", answer[:200].replace("\n", " "))
    logger.info("=" * 60)

    return {"answer": answer}
