"""Content enrichment utilities for post-import processing."""

from __future__ import annotations

from typing import Any

from polylogue.core.code_detection import detect_language


def enrich_content_blocks(content_blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Enrich content blocks with additional metadata.

    Currently adds:
    - Language detection for code blocks
    - Code extraction from fenced blocks in text

    Args:
        content_blocks: List of content block dicts from provider_meta

    Returns:
        Enriched content blocks (may add new blocks, modify existing)

    Examples:
        >>> blocks = [{"type": "text", "text": "```python\\nprint('hi')\\n```"}]
        >>> enriched = enrich_content_blocks(blocks)
        >>> enriched[0]['type']
        'code'
        >>> enriched[0]['language']
        'python'
    """
    if not content_blocks:
        return []

    enriched: list[dict[str, Any]] = []

    for block in content_blocks:
        block_type = block.get("type")
        text = block.get("text", "")

        # Extract code blocks from fenced text
        if block_type == "text" and "```" in text:
            # Split on fenced code blocks
            import re

            parts = re.split(r"(```\w*\n.*?\n```)", text, flags=re.DOTALL)

            for part in parts:
                if not part.strip():
                    continue

                # Check if this part is a fenced code block
                fence_match = re.match(r"```(\w*)\n(.*?)\n```", part, re.DOTALL)
                if fence_match:
                    declared_lang = fence_match.group(1) or None
                    code = fence_match.group(2)
                    detected_lang = detect_language(code, declared_lang)

                    enriched.append({
                        "type": "code",
                        "language": detected_lang,
                        "text": code,
                        "declared_language": declared_lang,
                    })
                else:
                    # Plain text between code blocks
                    enriched.append({"type": "text", "text": part})

        # Detect language for existing code blocks without language
        elif block_type == "code":
            if not block.get("language") and block.get("text"):
                detected_lang = detect_language(block["text"])
                enriched.append({**block, "language": detected_lang})
            else:
                enriched.append(block)

        # Pass through other block types unchanged
        else:
            enriched.append(block)

    return enriched


def enrich_message_metadata(provider_meta: dict[str, Any] | None) -> dict[str, Any] | None:
    """Enrich message provider_meta with content analysis.

    Args:
        provider_meta: Message provider_meta dict

    Returns:
        Enriched provider_meta with processed content_blocks

    Examples:
        >>> meta = {"content_blocks": [{"type": "text", "text": "```python\\ncode\\n```"}]}
        >>> enriched = enrich_message_metadata(meta)
        >>> enriched['content_blocks'][0]['type']
        'code'
    """
    if not provider_meta or "content_blocks" not in provider_meta:
        return provider_meta

    enriched_blocks = enrich_content_blocks(provider_meta["content_blocks"])

    return {**provider_meta, "content_blocks": enriched_blocks}
