"""Text cleaner — normalize markdown/HTML chunk text before storage."""

from __future__ import annotations

import re


def clean_chunk_text(text: str) -> str:
    """Clean markdown and HTML formatting from chunk text.

    Applied at ingest time so all downstream consumers (generate,
    review nodes, RAGAS) receive plain readable text.

    Steps:
    1. Replace <br> / <br/> with newline
    2. Convert markdown tables to plain key-value text
    3. Strip markdown bold/italic markers
    4. Strip markdown heading markers (keep heading text)
    5. Collapse excessive blank lines
    """
    text = _convert_tables(text)   # must run before _replace_br to detect table rows correctly
    text = _replace_br(text)       # handle remaining <br> outside tables
    text = _strip_markdown_markers(text)
    text = _collapse_blank_lines(text)
    return text.strip()


def _replace_br(text: str) -> str:
    """Replace HTML <br> tags with newline."""
    return re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)


def _convert_tables(text: str) -> str:
    """Convert markdown tables to plain indented text.

    Input:
        |**項目**|**內容**|
        |---|---|
        |體檢規定|0~50歲\n3,450萬新臺幣\n51~60歲\n2,700萬新臺幣|

    Output:
        項目: 內容
        體檢規定:
          0~50歲
          3,450萬新臺幣
          51~60歲
          2,700萬新臺幣
    """
    lines = text.split("\n")
    result: list[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Detect table row: starts and ends with |
        if _is_table_row(line):
            table_lines: list[str] = []
            while i < len(lines) and (_is_table_row(lines[i]) or _is_separator_row(lines[i])):
                table_lines.append(lines[i])
                i += 1
            result.extend(_table_to_plain(table_lines))
        else:
            result.append(line)
            i += 1

    return "\n".join(result)


def _is_table_row(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|") and not _is_separator_row(line)


def _is_separator_row(line: str) -> bool:
    stripped = line.strip()
    return bool(re.match(r"^\|[-:|\s]+\|$", stripped))


def _table_to_plain(table_lines: list[str]) -> list[str]:
    """Convert a block of markdown table lines to plain text."""
    rows: list[list[str]] = []
    for line in table_lines:
        if _is_separator_row(line):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        # Expand <br> within cells before stripping bold markers
        cells = [re.sub(r"<br\s*/?>", "\n", c, flags=re.IGNORECASE) for c in cells]
        cells = [re.sub(r"\*\*(.+?)\*\*", r"\1", c, flags=re.DOTALL) for c in cells]
        rows.append(cells)

    if not rows:
        return []

    plain: list[str] = []

    # If first row looks like a header (2 columns: key, value pattern)
    if len(rows) >= 2 and len(rows[0]) == 2:
        # header row
        plain.append(f"{rows[0][0]}: {rows[0][1]}")
        for row in rows[1:]:
            if len(row) == 2:
                key = row[0]
                # value may contain embedded newlines from <br> expansion
                val_lines = [v.strip() for v in row[1].split("\n") if v.strip()]
                if len(val_lines) == 1:
                    plain.append(f"  {key}: {val_lines[0]}")
                else:
                    plain.append(f"  {key}:")
                    for v in val_lines:
                        plain.append(f"    {v}")
            else:
                plain.append("  " + " | ".join(row))
    else:
        # Generic: just flatten each row
        for row in rows:
            plain.append("  " + " | ".join(row))

    return plain


def _strip_markdown_markers(text: str) -> str:
    """Remove markdown formatting markers, keep content."""
    # Bold **text** → text
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    # Italic *text* or _text_ → text
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"_(.+?)_", r"\1", text)
    # Heading ## Title → Title
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    return text


def _collapse_blank_lines(text: str) -> str:
    """Collapse 3+ consecutive blank lines into 2."""
    return re.sub(r"\n{3,}", "\n\n", text)
