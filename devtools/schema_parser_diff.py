"""Static schema-vs-parser diff: which observed wire keys does nothing read?

The schema packages under ``polylogue/schemas/providers`` record what each
provider actually emits, with per-key observation counts. The parsers decide
what becomes archive content. Nothing compared the two, so a field could be
present in every record for months and be silently invisible -- which is how
1,172,890 Claude Code sidecar records and 105,123 structured diffs went
unread (polylogue-2qx.3, polylogue-cgfy).

This produces the candidate list and ranks it by how much data actually
carries each key, so a batch of parser work can be scoped by evidence rather
than by guess.

DELIBERATELY APPROXIMATE ON THE PARSER SIDE. A key counts as "referenced" if
its name appears as a string constant anywhere in the provider's parser
modules -- a subscript, a ``.get()``, a comparison, a set membership. That
over-counts (a name may appear without being read into archive content) and
can under-count (a key read through a variable or a shared helper is missed).
So the output is a TRIAGE QUEUE, not a verdict: every row needs a human
disposition of read / deliberately-dropped-with-reason / to-acquire. Treating
a row as proof of a defect without reading the parser is exactly the mistake
this tool exists to make cheap to avoid.
"""

from __future__ import annotations

import argparse
import ast
import gzip
import json
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_ROOT = REPO_ROOT / "polylogue" / "schemas" / "providers"
PARSER_ROOT = REPO_ROOT / "polylogue" / "sources" / "parsers"

#: Provider schema directory -> the parser modules that own its wire format.
#: A provider whose parsing is spread over several modules lists all of them;
#: a key referenced in any of them counts as referenced.
PROVIDER_PARSERS: dict[str, tuple[str, ...]] = {
    "claude-code": ("claude/code_parser.py", "claude/common.py", "claude/history.py", "claude/orchestration.py"),
    "claude-ai": ("claude/ai_parser.py", "claude/common.py"),
    "codex": ("codex.py",),
    "chatgpt": ("chatgpt.py",),
    "gemini-cli": ("local_agent.py",),
    "gemini": ("drive.py", "drive_support.py", "drive_support_blocks.py", "drive_support_text.py"),
    "hermes": ("hermes_state.py", "hermes_spans.py", "hermes_lifecycle.py", "hermes_verification.py"),
    "antigravity": ("antigravity.py",),
    "browser-capture": ("browser_capture.py",),
}


@dataclass(frozen=True)
class ObservedKey:
    """One key the provider emits, with the evidence for how common it is."""

    path: str
    name: str
    documents: int
    encountered: int

    @property
    def coverage(self) -> float:
        return (self.encountered / self.documents) if self.documents else 0.0


def _schema_documents(provider_dir: Path) -> list[dict[str, object]]:
    return [
        json.loads(gzip.decompress(path.read_bytes()).decode("utf-8"))
        for path in sorted(provider_dir.glob("versions/*/elements/*.schema.json.gz"))
    ]


def observed_keys(schema: object, *, prefix: str = "") -> list[ObservedKey]:
    """Every named property in a schema, with its observation counts."""
    if not isinstance(schema, dict):
        return []
    found: list[ObservedKey] = []
    properties = schema.get("properties")
    if isinstance(properties, dict):
        for name, child in properties.items():
            path = f"{prefix}.{name}" if prefix else name
            distribution = child.get("x-polylogue-observed-distribution") if isinstance(child, dict) else None
            documents = encountered = 0
            if isinstance(distribution, dict):
                documents = int(distribution.get("documents") or 0)
                encountered = int(distribution.get("encountered_documents") or 0)
            found.append(ObservedKey(path=path, name=name, documents=documents, encountered=encountered))
            found.extend(observed_keys(child, prefix=path))
    for keyword in ("anyOf", "oneOf", "allOf"):
        branches = schema.get(keyword)
        if isinstance(branches, list):
            for branch in branches:
                found.extend(observed_keys(branch, prefix=prefix))
    items = schema.get("items")
    if isinstance(items, dict):
        found.extend(observed_keys(items, prefix=f"{prefix}[]"))
    additional = schema.get("additionalProperties")
    if isinstance(additional, dict):
        found.extend(observed_keys(additional, prefix=f"{prefix}.*"))
    return found


def referenced_strings(paths: list[Path]) -> set[str]:
    """Every string constant appearing anywhere in the given modules."""
    names: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                names.add(node.value)
    return names


def diff_provider(provider: str, *, min_encountered: int) -> list[ObservedKey]:
    provider_dir = SCHEMA_ROOT / provider
    if not provider_dir.exists():
        return []
    seen: dict[str, ObservedKey] = {}
    for schema in _schema_documents(provider_dir):
        for key in observed_keys(schema):
            existing = seen.get(key.path)
            if existing is None or key.encountered > existing.encountered:
                seen[key.path] = key
    referenced = referenced_strings([PARSER_ROOT / module for module in PROVIDER_PARSERS.get(provider, ())])
    unread = [key for key in seen.values() if key.name not in referenced and key.encountered >= min_encountered]
    return sorted(unread, key=lambda key: (-key.encountered, key.path))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", action="append", help="limit to these providers (repeatable)")
    parser.add_argument("--min-encountered", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=20, help="rows shown per provider")
    parser.add_argument("--json", dest="as_json", action="store_true")
    args = parser.parse_args(argv)

    providers = args.provider or sorted(PROVIDER_PARSERS)
    report: dict[str, list[ObservedKey]] = {
        provider: diff_provider(provider, min_encountered=args.min_encountered) for provider in providers
    }

    if args.as_json:
        payload = {
            provider: [
                {
                    "path": key.path,
                    "encountered_documents": key.encountered,
                    "documents": key.documents,
                    "coverage": round(key.coverage, 4),
                }
                for key in keys
            ]
            for provider, keys in report.items()
        }
        print(json.dumps(payload, indent=2))
        return 0

    for provider, keys in report.items():
        if not keys:
            continue
        total = sum(key.encountered for key in keys)
        print(f"\n=== {provider}: {len(keys)} unreferenced keys over {args.min_encountered} docs ({total:,} records)")
        for key in keys[: args.limit]:
            print(f"   {key.encountered:>9,}  {key.coverage:>6.1%}  {key.path}")
        if len(keys) > args.limit:
            print(f"   ... {len(keys) - args.limit} more")
    print("\nEvery row needs triage: read / deliberately-dropped-with-reason / to-acquire.")
    print("Parser-side matching is name-based and approximate -- read the parser before acting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
