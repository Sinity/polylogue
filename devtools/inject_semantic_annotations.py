"""Inject x-polylogue-semantic-role annotations into baseline provider schemas.

One-shot script. Run once to annotate, commit the updated schemas, then
this script can remain as a re-annotation utility.

Usage: devtools inject-semantic-annotations [--dry-run]
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

from polylogue.core.json import JSONDocument, is_json_document, json_document
from polylogue.schemas.registry import SchemaRegistry

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
        (["properties.title"], "conversation_title"),
        (["properties.create_time"], "message_timestamp"),
        (["properties.mapping"], "message_container"),
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
    ],
    "claude-ai": [
        (["properties.name"], "conversation_title"),
        (["properties.created_at"], "message_timestamp"),
        (["properties.chat_messages"], "message_container"),
        (["properties.chat_messages", "items", "properties.sender"], "message_role"),
        (["properties.chat_messages", "items", "properties.text"], "message_body"),
    ],
    "claude-code": [
        (["properties.type"], "message_role"),
        (
            [
                "properties.message",
                "properties.content",
                "anyOf:array",
                "items",
                "properties.content",
            ],
            "message_body",
        ),
        (["properties.timestamp"], "message_timestamp"),
        (["properties.gitBranch"], "conversation_title"),
    ],
    "codex": [
        (["properties.timestamp"], "message_timestamp"),
        (["properties.payload"], "message_container"),
        (["properties.payload", "properties.role"], "message_role"),
        (
            [
                "properties.payload",
                "properties.summary",
                "anyOf:array",
                "items",
                "properties.text",
            ],
            "message_body",
        ),
        (["properties.payload", "properties.name"], "conversation_title"),
    ],
    "gemini": [
        (
            [
                "properties.chunkedPrompt",
                "properties.chunks",
            ],
            "message_container",
        ),
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
        (
            [
                "properties.chunkedPrompt",
                "properties.chunks",
                "items",
                "properties.createTime",
            ],
            "message_timestamp",
        ),
        (["properties.runSettings", "properties.thinkingLevel"], "conversation_title"),
    ],
}


def _navigate(schema: JSONDocument, path_segments: list[str]) -> JSONDocument | None:
    """Navigate a schema using path segments, returning the target node."""
    node: object = schema
    for segment in path_segments:
        if not is_json_document(node):
            return None

        mapping = node

        if segment.startswith("properties."):
            key = segment[len("properties.") :]
            properties = mapping.get("properties")
            if not is_json_document(properties):
                return None
            node = properties.get(key)
        elif segment == "items":
            node = mapping.get("items")
        elif segment == "additionalProperties":
            node = mapping.get("additionalProperties")
        elif segment == "anyOf:props":
            # Find the anyOf variant that has "properties"
            variants = mapping.get("anyOf", [])
            found = None
            if not isinstance(variants, list):
                return None
            for variant in variants:
                if is_json_document(variant) and "properties" in variant:
                    found = variant
                    break
            node = found
        elif segment == "anyOf:array":
            # Find the anyOf variant with type="array"
            variants = mapping.get("anyOf", [])
            found = None
            if not isinstance(variants, list):
                return None
            for variant in variants:
                if is_json_document(variant) and variant.get("type") == "array":
                    found = variant
                    break
            node = found
        else:
            node = mapping.get(segment)

        if node is None:
            return None

    return node if is_json_document(node) else None


def inject_annotations(provider: str, schema: JSONDocument, *, dry_run: bool = False) -> int:
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

    registry = SchemaRegistry(storage_root=SCHEMAS_DIR)
    total = 0
    for provider in ANNOTATION_MAP:
        package = registry.get_package(provider, version="default")
        if package is None:
            print(f"SKIP: no bundled schema package found for {provider}")
            continue
        element = package.element(package.default_element_kind)
        if element is None or element.schema_file is None:
            print(f"SKIP: no default element schema found for {provider}")
            continue
        schema_path = SCHEMAS_DIR / provider / "versions" / package.version / "elements" / element.schema_file
        if not schema_path.exists():
            print(f"SKIP: {schema_path} not found")
            continue

        print(f"\n--- {provider} ---")
        with gzip.open(schema_path, "rt") as f:
            schema = json_document(json.load(f))

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
