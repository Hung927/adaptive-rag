"""Layer 1: Configuration — environment variable management."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass
class AzureOpenAISettings:
    """Azure OpenAI connection settings."""

    api_key: str
    endpoint: str
    api_version: str = "2024-06-01"
    chat_deployment: str = "gpt-4o"
    embedding_deployment: str = "text-embedding-3-small"


@dataclass
class QdrantSettings:
    """Qdrant connection settings."""

    url: str = "http://localhost:6333"
    collection: str = "documents"


@dataclass
class ChunkingSettings:
    """Document chunking settings."""

    chunk_size: int = 500
    chunk_overlap: int = 50


@dataclass
class PipelineSettings:
    """RAG pipeline settings."""

    max_review_retries: int = 2
    enable_review_retrieval: bool = True
    enable_review_generation: bool = True
    enable_evaluate: bool = True


@dataclass
class LangfuseSettings:
    """Langfuse observability settings."""

    secret_key: str = ""
    public_key: str = ""
    host: str = "https://cloud.langfuse.com"
    enabled: bool = False


@dataclass
class Settings:
    """Application configuration."""

    azure_openai: AzureOpenAISettings
    qdrant: QdrantSettings
    chunking: ChunkingSettings
    pipeline: PipelineSettings
    langfuse: LangfuseSettings = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.langfuse is None:
            self.langfuse = LangfuseSettings()


def _parse_int(name: str, default: str) -> int:
    raw = os.getenv(name, default)
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{name} must be a valid integer, got: {raw!r}")


def load_settings() -> Settings:
    """Load settings from environment variables and .env file."""
    load_dotenv()

    api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("Missing required: AZURE_OPENAI_API_KEY")

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    if not endpoint:
        raise ValueError("Missing required: AZURE_OPENAI_ENDPOINT")

    return Settings(
        azure_openai=AzureOpenAISettings(
            api_key=api_key,
            endpoint=endpoint,
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01"),
            chat_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o"),
            embedding_deployment=os.getenv(
                "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"
            ),
        ),
        qdrant=QdrantSettings(
            url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            collection=os.getenv("QDRANT_COLLECTION", "documents"),
        ),
        chunking=ChunkingSettings(
            chunk_size=_parse_int("CHUNK_SIZE", "500"),
            chunk_overlap=_parse_int("CHUNK_OVERLAP", "50"),
        ),
        pipeline=PipelineSettings(
            max_review_retries=_parse_int("MAX_REVIEW_RETRIES", "2"),
            enable_review_retrieval=os.getenv("ENABLE_REVIEW_RETRIEVAL", "true").lower() != "false",
            enable_review_generation=os.getenv("ENABLE_REVIEW_GENERATION", "true").lower() != "false",
            enable_evaluate=os.getenv("ENABLE_EVALUATE", "true").lower() != "false",
        ),
        langfuse=LangfuseSettings(
            secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            enabled=bool(os.getenv("LANGFUSE_SECRET_KEY")),
        ),
    )
