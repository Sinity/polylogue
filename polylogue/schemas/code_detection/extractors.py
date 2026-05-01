"""Code-block extraction helpers built on the canonical detection workflow."""

from __future__ import annotations

import re

from polylogue.core.json import JSONDocument, json_document
from polylogue.schemas.code_detection.runtime import detect_language


def extract_code_block_from_dict(content_block: JSONDocument) -> JSONDocument | None:
    """Extract and enrich a code block from a content-block dict."""
    block_type = content_block.get("type")
    text_value = content_block.get("text", "")
    if not isinstance(text_value, str):
        return None
    text = text_value

    fence_match = re.match(r"```(\w*)\n(.*?)\n```", text, re.DOTALL)
    if fence_match:
        declared_lang = fence_match.group(1) or None
        code = fence_match.group(2)
        detected_lang = detect_language(code, declared_lang)
        return json_document(
            {
                "type": "code",
                "language": detected_lang,
                "text": code,
                "declared_language": declared_lang,
            }
        )

    if block_type == "text" and len(text) > 20:
        detected_lang = detect_language(text)
        if detected_lang:
            return json_document(
                {
                    "type": "code",
                    "language": detected_lang,
                    "text": text,
                    "declared_language": None,
                }
            )
    return None


def extract_code_block(text: str) -> str:
    """Extract code from markdown or code-like text."""
    if not text:
        return ""

    thinking_match = re.search(r"<thinking>(.*?)</thinking>", text, re.DOTALL)
    if thinking_match:
        return thinking_match.group(1).strip()

    fence_match = re.match(r"```(?:\w*)\n(.*?)\n```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1)

    lines = text.split("\n")
    indented_lines: list[str] = []
    in_block = False
    first_line_indented = lines[0].startswith("    ") if lines else False
    blank_line_before_indent = False
    for index, line in enumerate(lines):
        if line.startswith("    "):
            if not in_block and index > 0 and not blank_line_before_indent:
                blank_line_before_indent = lines[index - 1].strip() == ""
            in_block = True
            indented_lines.append(line[4:])
        elif in_block and line.strip() == "":
            indented_lines.append("")
        elif in_block and not line.startswith("    "):
            break

    if indented_lines:
        while indented_lines and not indented_lines[-1].strip():
            indented_lines.pop()
        if indented_lines and (first_line_indented or blank_line_before_indent):
            return "\n".join(indented_lines)

    if len(text) >= 10 and detect_language(text):
        return text
    return ""


__all__ = ["extract_code_block", "extract_code_block_from_dict"]
