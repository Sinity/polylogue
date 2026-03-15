"""Inject x-polylogue-semantic-role annotations into baseline provider schemas.

One-shot script. Run once to annotate, commit the updated schemas, then
this script can remain as a re-annotation utility.

Usage: python -m devtools.inject_semantic_annotations [--dry-run]
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path
from typing import Any


SCHEMAS_DIR = Path(__file__).resolve().parent.parent / "polylogue" / "schemas" / "providers"

# ---------------------------------------------------------------------------
# Annotation map: provider → list of (json_path_segments, semantic_role)
#
# Paths use a mini-DSL:
#   "properties.X"         → schema["properties"]["X"]
#   "items"                → schema["items"]
#   "additionalProperties" → schema["additionalProperties"]
#   "anyOf:props"          → find the anyOf variant that has "properties"
#   "anyOf:array"          → find the anyOf variant with type="array"
# ---------------------------------------------------------------------------

ANNOTATION_MAP: dict[str, list[tuple[list[str], str]]] = {
    "chatgpt": [
        # mapping.*.message(anyOf→props).author.role → message_role
        (
            [
                "properties.mapping",
                "additionalProperties",
                "properties.message",
                "anyOf:props",
                "properties.author",
                "properties.role",
            ],
            "message_role",
        ),
        # mapping.*.message(anyOf→props).content.parts → message_body (on items)
        (
            [
                "properties.mapping",
                "additionalProperties",
                "properties.message",
                "anyOf:props",
                "properties.content",
                "properties.parts",
                "items",
            ],
            "message_body",
        ),
        # mapping.*.message(anyOf→props).create_time → message_timestamp
        (
            [
                "properties.mapping",
                "additionalProperties",
                "properties.message",
                "anyOf:props",
                "properties.create_time",
            ],
            "message_timestamp",
        ),
        # title → conversation_title
        (["properties.title"], "conversation_title"),
    ],
    "claude-ai": [
        (["properties.chat_messages", "items", "properties.sender"], "message_role"),
        (["properties.chat_messages", "items", "properties.text"], "message_body"),
        (["properties.chat_messages", "items", "properties.created_at"], "message_timestamp"),
        (["properties.name"], "conversation_title"),
    ],
    "claude-code": [
        # top-level type field IS the role for claude-code
        (["properties.type"], "message_role"),
        # message.content → anyOf → array variant → items → find text property
        (
            [
                "properties.message",
                "properties.content",
                "anyOf:array",
                "items",
                "properties.text",
            ],
            "message_body",
        ),
        (["properties.timestamp"], "message_timestamp"),
    ],
    "codex": [
        (["properties.role"], "message_role"),
        (["properties.content", "items", "properties.text"], "message_body"),
        (["properties.timestamp"], "message_timestamp"),
    ],
    "gemini": [
        (
            [
                "properties.chunkedPrompt",
                "properties.chunks",
                "items",
                "properties.role",
            ],
            "message_role",
        ),
        (
            [
                "properties.chunkedPrompt",
                "properties.chunks",
                "items",
                "properties.text",
            ],
            "message_body",
        ),
    ],
}


def _navigate(schema: dict[str, Any], path_segments: list[str]) -> dict[str, Any] | None:
    """Navigate a schema using path segments, returning the target node."""
    node = schema
    for segment in path_segments:
        if not isinstance(node, dict):
            return None

        if segment.startswith("properties."):
            key = segment[len("properties."):]
            node = node.get("properties", {}).get(key)
        elif segment == "items":
            node = node.get("items")
        elif segment == "additionalProperties":
            node = node.get("additionalProperties")
        elif segment == "anyOf:props":
            # Find the anyOf variant that has "properties"
            variants = node.get("anyOf", [])
            found = None
            for v in variants:
                if "properties" in v:
                    found = v
                    break
            node = found
        elif segment == "anyOf:array":
            # Find the anyOf variant with type="array"
            variants = node.get("anyOf", [])
            found = None
            for v in variants:
                if v.get("type") == "array":
                    found = v
                    break
            node = found
        else:
            node = node.get(segment)

        if node is None:
            return None

    return node if isinstance(node, dict) else None


def inject_annotations(provider: str, schema: dict[str, Any], *, dry_run: bool = False) -> int:
    """Inject semantic role annotations into a schema. Returns count of annotations added."""
    annotations = ANNOTATION_MAP.get(provider, [])
    count = 0

    for path_segments, role in annotations:
        target = _navigate(schema, path_segments)
        if target is None:
            print(f"  WARNING: path not found for {provider}: {' → '.join(path_segments)}")
            continue

        if target.get("x-polylogue-semantic-role") == role:
            print(f"  SKIP (already set): {provider}.{' → '.join(path_segments)} = {role}")
            continue

        if not dry_run:
            target["x-polylogue-semantic-role"] = role
        print(f"  {'DRY-RUN: ' if dry_run else ''}SET: {provider}.{' → '.join(path_segments)} = {role}")
        count += 1

    return count


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Show what would be changed without modifying files")
    args = parser.parse_args(argv)

    total = 0
    for provider in ANNOTATION_MAP:
        schema_path = SCHEMAS_DIR / f"{provider}.schema.json.gz"
        if not schema_path.exists():
            print(f"SKIP: {schema_path} not found")
            continue

        print(f"\n--- {provider} ---")
        with gzip.open(schema_path, "rt") as f:
            schema = json.load(f)

        count = inject_annotations(provider, schema, dry_run=args.dry_run)
        total += count

        if count > 0 and not args.dry_run:
            with gzip.open(schema_path, "wt") as f:
                json.dump(schema, f, indent=2, ensure_ascii=False)
            print(f"  Written {count} annotations to {schema_path.name}")

    print(f"\nTotal annotations {'would be ' if args.dry_run else ''}injected: {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
