"""Pipeline builder — assemble LangGraph workflow."""

from __future__ import annotations

import logging
from functools import partial
from typing import Any, Callable

from langgraph.graph import END, StateGraph

from rag.core.config import Settings
from rag.core.llm import create_chat_client
from rag.core.tracing import PipelineTracer
from rag.core.types import EvalScores, GenerationResult
from rag.pipeline.nodes.evaluate import evaluate_node
from rag.pipeline.nodes.generate import generate_node
from rag.pipeline.nodes.retrieve import retrieve_node
from rag.pipeline.nodes.review_generation import review_generation_node
from rag.pipeline.nodes.review_retrieval import review_retrieval_node
from rag.pipeline.nodes.rewrite import rewrite_node
from rag.pipeline.state import PipelineState
from rag.retrieval.store import QdrantStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------


def _route_retrieval_review(state: PipelineState, max_retries: int) -> str:
    if state.get("retrieval_review_passed", True):
        return "generate"
    if state.get("retrieval_review_attempts", 0) >= max_retries:
        return "generate"
    return "rewrite"


def _route_generation_review(state: PipelineState, max_retries: int, enable_evaluate: bool) -> str:
    next_node = "evaluate" if enable_evaluate else "end"
    if state.get("generation_review_passed", True):
        return next_node
    if state.get("generation_review_attempts", 0) >= max_retries:
        return next_node
    return "generate"


# ---------------------------------------------------------------------------
# Langfuse node wrapper
# ---------------------------------------------------------------------------

_INPUT_KEYS: dict[str, list[str]] = {
    "retrieve": ["query", "rewritten_query"],
    "rewrite": ["query", "retrieval_review_feedback"],
    "review_retrieval": ["query", "retrieved_chunks"],
    "generate": ["query", "confirmed_chunks", "retrieved_chunks", "generation_review_feedback"],
    "review_generation": ["query", "answer", "confirmed_chunks", "retrieved_chunks"],
    "evaluate": ["query", "answer", "confirmed_chunks", "retrieved_chunks"],
}


def _wrap_with_tracing(
    node_fn: Callable[[PipelineState], dict],
    node_name: str,
    tracer: PipelineTracer,
    trace_ref: list[Any],  # mutable container so the closure can read the trace
) -> Callable[[PipelineState], dict]:
    """Return a wrapped node that records input/output to Langfuse."""

    def _wrapped(state: PipelineState) -> dict:
        trace = trace_ref[0]

        # Build input snapshot from relevant state keys
        input_keys = _INPUT_KEYS.get(node_name, [])
        input_snapshot: dict[str, Any] = {}
        for k in input_keys:
            v = state.get(k)  # type: ignore[call-overload]
            if v is not None:
                input_snapshot[k] = v

        with tracer.span(trace, node_name, input=input_snapshot) as span:
            result = node_fn(state)
            span.update(output=result)
            return result

    return _wrapped


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_pipeline(
    settings: Settings,
    store: QdrantStore,
    top_k: int = 5,
    enable_review_retrieval: bool = True,
    enable_review_generation: bool = True,
    enable_evaluate: bool = True,
    tracer: PipelineTracer | None = None,
    trace_ref: list[Any] | None = None,
) -> StateGraph:
    """Build the RAG LangGraph pipeline.

    Possible flows (all flags on):
      retrieve → review_retrieval → generate → review_generation → evaluate → END

    Flags:
      enable_review_retrieval : wrap retrieve with retrieval quality review
      enable_review_generation: wrap generate with generation quality review
      enable_evaluate         : run LLM-as-judge evaluation as the final node
    """
    client = create_chat_client(settings)
    chat_deployment = settings.azure_openai.chat_deployment
    max_retries = settings.pipeline.max_review_retries

    graph = StateGraph(PipelineState)

    # ------------------------------------------------------------------
    # Helper: optionally wrap a node with Langfuse tracing
    # ------------------------------------------------------------------
    def _node(name: str, fn: Callable) -> Callable:
        if tracer is not None and trace_ref is not None:
            return _wrap_with_tracing(fn, name, tracer, trace_ref)
        return fn

    # ------------------------------------------------------------------
    # Core nodes
    # ------------------------------------------------------------------
    graph.add_node("retrieve", _node("retrieve", partial(retrieve_node, store=store, top_k=top_k)))
    graph.add_node("generate", _node("generate", partial(generate_node, client=client, deployment=chat_deployment)))

    graph.set_entry_point("retrieve")

    # ------------------------------------------------------------------
    # Optional: retrieval review + rewrite loop
    # ------------------------------------------------------------------
    if enable_review_retrieval:
        graph.add_node("rewrite", _node("rewrite", partial(rewrite_node, client=client, deployment=chat_deployment)))
        graph.add_node("review_retrieval", _node("review_retrieval", partial(review_retrieval_node, client=client, deployment=chat_deployment)))
        graph.add_edge("retrieve", "review_retrieval")
        graph.add_edge("rewrite", "retrieve")
        graph.add_conditional_edges(
            "review_retrieval",
            partial(_route_retrieval_review, max_retries=max_retries),
            {"generate": "generate", "rewrite": "rewrite"},
        )
    else:
        graph.add_edge("retrieve", "generate")

    # ------------------------------------------------------------------
    # Optional: generation review loop
    # ------------------------------------------------------------------
    if enable_review_generation:
        graph.add_node("review_generation", _node("review_generation", partial(review_generation_node, client=client, deployment=chat_deployment)))
        graph.add_edge("generate", "review_generation")

        if enable_evaluate:
            graph.add_node("evaluate", _node("evaluate", partial(evaluate_node, client=client, deployment=chat_deployment)))
            graph.add_conditional_edges(
                "review_generation",
                partial(_route_generation_review, max_retries=max_retries, enable_evaluate=True),
                {"evaluate": "evaluate", "generate": "generate", "end": END},
            )
            graph.add_edge("evaluate", END)
        else:
            graph.add_conditional_edges(
                "review_generation",
                partial(_route_generation_review, max_retries=max_retries, enable_evaluate=False),
                {"end": END, "generate": "generate"},
            )
    else:
        if enable_evaluate:
            graph.add_node("evaluate", _node("evaluate", partial(evaluate_node, client=client, deployment=chat_deployment)))
            graph.add_edge("generate", "evaluate")
            graph.add_edge("evaluate", END)
        else:
            graph.add_edge("generate", END)

    return graph


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------


def run_pipeline(
    settings: Settings,
    store: QdrantStore,
    query: str,
    top_k: int = 5,
) -> GenerationResult:
    """Run the full RAG pipeline and return a GenerationResult.

    Pipeline feature flags are read from settings.pipeline:
      ENABLE_REVIEW_RETRIEVAL  (default: true)
      ENABLE_REVIEW_GENERATION (default: true)
      ENABLE_EVALUATE          (default: true)
    """
    pipeline_cfg = settings.pipeline
    tracer = PipelineTracer(settings.langfuse)
    trace_ref: list[Any] = [None]

    with tracer.trace("rag-pipeline", query=query) as trace:
        trace_ref[0] = trace

        graph = build_pipeline(
            settings,
            store,
            top_k=top_k,
            enable_review_retrieval=pipeline_cfg.enable_review_retrieval,
            enable_review_generation=pipeline_cfg.enable_review_generation,
            enable_evaluate=pipeline_cfg.enable_evaluate,
            tracer=tracer,
            trace_ref=trace_ref,
        )
        app = graph.compile()
        result = app.invoke({"query": query})

        # Build eval_scores if the evaluate node ran
        eval_scores: EvalScores | None = None
        if pipeline_cfg.enable_evaluate and "eval_faithfulness" in result:
            eval_scores = EvalScores(
                faithfulness=result["eval_faithfulness"],
                answer_relevance=result["eval_answer_relevance"],
                context_precision=result["eval_context_precision"],
                reasoning=result.get("eval_reasoning", {}),
            )

        generation_result = GenerationResult(
            answer=result.get("answer", ""),
            sources=result.get("confirmed_chunks") or result.get("retrieved_chunks", []),
            query=query,
            rewritten_query=result.get("rewritten_query"),
            review_passed=result.get("generation_review_passed"),
        )
        if eval_scores is not None:
            generation_result["eval_scores"] = eval_scores

        # Update the top-level trace with final output
        trace.update(
            output={
                "answer": generation_result["answer"][:500],
                "eval_scores": eval_scores,
            }
        )

    return generation_result
