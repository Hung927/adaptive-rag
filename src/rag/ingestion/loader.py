"""Document loaders — file → pages of text."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class DocumentLoader(Protocol):
    """Document loader interface."""

    def load(self, file_path: Path) -> list[dict]:
        """Load document into pages.

        Returns:
            [{"text": str, "page_number": int | None}, ...]
        """
        ...


class PyMuPDFLoader:
    """PDF loader using pymupdf4llm (outputs Markdown)."""

    def load(self, file_path: Path) -> list[dict]:
        import pymupdf4llm

        md_pages = pymupdf4llm.to_markdown(str(file_path), page_chunks=True)
        return [
            {
                "text": page["text"],
                "page_number": page["metadata"]["page_number"],
            }
            for page in md_pages
        ]


class PythonDocxLoader:
    """DOCX loader with heading style detection."""

    _STYLE_TO_LEVEL: dict[str, int] = {
        "Heading 1": 1,
        "Heading 2": 2,
        "Heading 3": 3,
    }

    def load(self, file_path: Path) -> list[dict]:
        from docx import Document

        doc = Document(str(file_path))
        lines: list[str] = []

        for p in doc.paragraphs:
            text = p.text.strip()
            if not text:
                continue
            style_name = p.style.name if p.style else ""
            if style_name.startswith("toc"):
                continue
            level = self._STYLE_TO_LEVEL.get(style_name)
            if level:
                lines.append(f"\n{'#' * level} {text}\n")
            else:
                lines.append(text)

        return [{"text": "\n".join(lines), "page_number": None}]


class TextLoader:
    """Plain text / markdown loader."""

    def load(self, file_path: Path) -> list[dict]:
        content = file_path.read_text(encoding="utf-8")
        return [{"text": content, "page_number": None}]


# Registry — add new loaders by extending this dict
LOADERS: dict[str, DocumentLoader] = {
    ".pdf": PyMuPDFLoader(),
    ".docx": PythonDocxLoader(),
    ".txt": TextLoader(),
    ".md": TextLoader(),
}


def get_loader(file_path: Path) -> DocumentLoader:
    """Get loader by file extension."""
    suffix = file_path.suffix.lower()
    if suffix not in LOADERS:
        raise ValueError(f"Unsupported file type: {suffix}")
    return LOADERS[suffix]
