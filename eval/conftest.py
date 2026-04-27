"""Eval pytest fixtures."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv()


def _setup_eval_logging() -> Path:
    """Set up logging to both console and timestamped log file."""
    log_dir = Path("eval/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"eval_{ts}.log"

    fmt = "%(asctime)s %(name)-40s %(levelname)-8s %(message)s"

    # Root logger
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    # Console handler
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(fmt))
    root.addHandler(console)

    # File handler
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(fmt))
    root.addHandler(file_handler)

    logging.info("=" * 80)
    logging.info("Eval session started — log: %s", log_path)
    logging.info("=" * 80)

    return log_path


# Set up logging once at module load
_LOG_PATH = _setup_eval_logging()


@pytest.fixture(autouse=True)
def _eval_env():
    """Ensure eval-specific env vars are set."""
    os.environ.setdefault("EVAL_LIMIT", "3")
    yield


@pytest.fixture(scope="session")
def eval_log_path():
    """Return the current eval log file path (for reference in tests)."""
    return _LOG_PATH


@pytest.fixture(scope="session")
def ragas_azure_config():
    """Return RAGAS-compatible Azure OpenAI LLM and embeddings via LangChain."""
    from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
    from ragas import RunConfig

    llm = AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
        temperature=0,
    )

    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
    )

    run_config = RunConfig(max_retries=3, max_wait=60)

    return {"llm": llm, "embeddings": embeddings, "run_config": run_config}
