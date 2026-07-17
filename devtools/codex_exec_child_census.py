"""Census Codex code-mode transport envelopes and lowered child actions.

Run against one or more Codex session JSON/JSONL files or directories. The
report compares the historical outer-transport-only projection with the
current parser's typed child projection. Text containing ``exit_code`` is
reported only as a diagnostic; it never contributes to structured outcomes.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import TextIO

from polylogue.core.enums import BlockType
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedSession
from polylogue.sources.parsers.codex import parse_stream

_CHILD_KIND = "codex.functions_exec_child"
_CHILD_ID_MARKER = "::polylogue-child::"
_TRANSPORT_TOOL_NAMES = frozenset({"exec", "functions.exec"})
_SUPPORTED_SUFFIXES = (
    ".json",
    ".jsonl",
    ".ndjson",
    ".json.gz",
    ".jsonl.gz",
    ".ndjson.gz",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="Codex session files or directories to scan.")
    parser.add_argument("--output", type=Path, help="Write the JSON report to this path instead of stdout.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when any discovered file cannot be decoded or parsed.",
    )
    return parser


def _is_supported(path: Path) -> bool:
    lowered = path.name.lower()
    return any(lowered.endswith(suffix) for suffix in _SUPPORTED_SUFFIXES)


def _discover(paths: Sequence[Path]) -> list[Path]:
    discovered: set[Path] = set()
    for raw_path in paths:
        path = raw_path.expanduser().resolve()
        if path.is_file():
            if _is_supported(path):
                discovered.add(path)
            continue
        if path.is_dir():
            discovered.update(
                candidate for candidate in path.rglob("*") if candidate.is_file() and _is_supported(candidate)
            )
    return sorted(discovered, key=lambda candidate: candidate.as_posix())


def _open_text(path: Path) -> TextIO:
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, mode="rt", encoding="utf-8")
    return path.open(mode="r", encoding="utf-8")


def _records(path: Path) -> Iterator[object]:
    lowered = path.name.lower()
    with _open_text(path) as handle:
        if lowered.endswith((".jsonl", ".ndjson", ".jsonl.gz", ".ndjson.gz")):
            for line_number, line in enumerate(handle, start=1):
                candidate = line.strip()
                if not candidate:
                    continue
                try:
                    yield json.loads(candidate)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at line {line_number}: {exc.msg}") from exc
            return
        value = json.load(handle)
    if isinstance(value, list):
        yield from value
    else:
        yield value


def _block_type(block: ParsedContentBlock) -> str:
    return block.type.value


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _child_provenance(block: ParsedContentBlock) -> Mapping[str, object] | None:
    provenance = _mapping(block.tool_input).get("_polylogue")
    if not isinstance(provenance, Mapping) or provenance.get("kind") != _CHILD_KIND:
        return None
    return provenance


def _has_nonempty_string(mapping: Mapping[str, object], *keys: str) -> bool:
    return any(isinstance(mapping.get(key), str) and bool(str(mapping[key]).strip()) for key in keys)


def _has_path(mapping: Mapping[str, object]) -> bool:
    if _has_nonempty_string(mapping, "path", "file_path"):
        return True
    paths = mapping.get("paths") or mapping.get("file_paths")
    return isinstance(paths, list) and any(isinstance(value, str) and value.strip() for value in paths)


def _has_byte_count(mapping: Mapping[str, object]) -> bool:
    value = mapping.get("byte_count")
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _result_has_path(block: ParsedContentBlock) -> bool:
    metadata = _mapping(block.metadata)
    return _has_path(metadata)


def _result_has_byte_count(block: ParsedContentBlock) -> bool:
    return _has_byte_count(_mapping(block.metadata))


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _session_census(session: ParsedSession) -> dict[str, object]:
    child_uses: list[ParsedContentBlock] = []
    child_results: list[ParsedContentBlock] = []
    transport_uses: list[ParsedContentBlock] = []
    all_results: list[ParsedContentBlock] = []
    registry_counts: Counter[str] = Counter()
    parse_state_counts: Counter[str] = Counter()

    for message in session.messages:
        for block in message.blocks:
            block_type = _block_type(block)
            if block_type == BlockType.TOOL_USE.value:
                provenance = _child_provenance(block)
                if provenance is not None:
                    child_uses.append(block)
                    registry_counts[str(provenance.get("registry_type") or "unknown")] += 1
                    parse_state_counts[str(provenance.get("parse_state") or "unknown")] += 1
                elif isinstance(block.tool_name, str) and block.tool_name.lower() in _TRANSPORT_TOOL_NAMES:
                    transport_uses.append(block)
            elif block_type == BlockType.TOOL_RESULT.value:
                all_results.append(block)
                if isinstance(block.tool_id, str) and _CHILD_ID_MARKER in block.tool_id:
                    child_results.append(block)

    transport_ids = Counter(
        block.tool_id for block in transport_uses if isinstance(block.tool_id, str) and block.tool_id
    )
    transport_results = [
        block
        for block in all_results
        if isinstance(block.tool_id, str) and block.tool_id in transport_ids and _CHILD_ID_MARKER not in block.tool_id
    ]

    results_by_id: dict[str, list[ParsedContentBlock]] = defaultdict(list)
    for result in child_results:
        if isinstance(result.tool_id, str) and result.tool_id:
            results_by_id[result.tool_id].append(result)
    result_offsets: Counter[str] = Counter()
    paired_results: list[ParsedContentBlock] = []
    for child in child_uses:
        if not isinstance(child.tool_id, str) or not child.tool_id:
            continue
        offset = result_offsets[child.tool_id]
        candidates = results_by_id.get(child.tool_id, [])
        if offset < len(candidates):
            paired_results.append(candidates[offset])
            result_offsets[child.tool_id] += 1

    child_inputs = [_mapping(block.tool_input) for block in child_uses]
    structured_outcomes = [
        block for block in child_results if block.is_error is not None or block.exit_code is not None
    ]
    return {
        "transport_actions": len(transport_uses),
        "transport_results": len(transport_results),
        "transport_result_texts_with_exit_code_token": sum(
            1 for block in transport_results if isinstance(block.text, str) and "exit_code" in block.text
        ),
        "typed_child_actions": len(child_uses),
        "child_results": len(child_results),
        "paired_child_results": len(paired_results),
        "unpaired_child_actions": len(child_uses) - len(paired_results),
        "orphan_child_results": len(child_results) - len(paired_results),
        "child_actions_with_command": sum(1 for value in child_inputs if _has_nonempty_string(value, "command")),
        "child_actions_with_path": sum(1 for value in child_inputs if _has_path(value)),
        "child_actions_with_byte_count": sum(1 for value in child_inputs if _has_byte_count(value)),
        "child_results_with_structured_outcome": len(structured_outcomes),
        "child_results_with_exit_code": sum(1 for block in child_results if block.exit_code is not None),
        "child_results_with_path": sum(1 for block in child_results if _result_has_path(block)),
        "child_results_with_byte_count": sum(1 for block in child_results if _result_has_byte_count(block)),
        "child_results_with_unknown_outcome": len(child_results) - len(structured_outcomes),
        "child_result_texts_with_exit_code_token": sum(
            1 for block in child_results if isinstance(block.text, str) and "exit_code" in block.text
        ),
        "children_by_registry_type": dict(sorted(registry_counts.items())),
        "children_by_parse_state": dict(sorted(parse_state_counts.items())),
    }


def _counter_value(mapping: Mapping[str, object], key: str) -> int:
    value = mapping.get(key, 0)
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"counter value for {key} is not an integer")
    return value


def _add_counts(total: dict[str, object], current: Mapping[str, object]) -> None:
    for key, value in current.items():
        if isinstance(value, int) and not isinstance(value, bool):
            total[key] = _counter_value(total, key) + value
        elif isinstance(value, Mapping):
            target = total.setdefault(key, {})
            if not isinstance(target, dict):
                raise TypeError(f"counter shape changed for {key}")
            for nested_key, nested_value in value.items():
                if not isinstance(nested_value, int) or isinstance(nested_value, bool):
                    raise TypeError(f"nested counter value for {nested_key} is not an integer")
                nested_name = str(nested_key)
                target[nested_name] = _counter_value(target, nested_name) + nested_value


def build_report(paths: Sequence[Path]) -> dict[str, object]:
    files = _discover(paths)
    aggregate: dict[str, object] = {}
    errors: list[dict[str, str]] = []
    parsed_sessions = 0
    for path in files:
        try:
            session = parse_stream(_records(path), fallback_id=path.stem)
        except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
            errors.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
            continue
        parsed_sessions += 1
        _add_counts(aggregate, _session_census(session))

    for key in ("children_by_registry_type", "children_by_parse_state"):
        nested = aggregate.get(key)
        if isinstance(nested, dict):
            aggregate[key] = dict(sorted(nested.items()))

    transport_actions = _counter_value(aggregate, "transport_actions")
    typed_children = _counter_value(aggregate, "typed_child_actions")
    child_results = _counter_value(aggregate, "child_results")
    paired_results = _counter_value(aggregate, "paired_child_results")
    structured_outcomes = _counter_value(aggregate, "child_results_with_structured_outcome")
    path_count = _counter_value(aggregate, "child_actions_with_path")

    return {
        "schema_version": 1,
        "files_discovered": len(files),
        "sessions_parsed": parsed_sessions,
        "parse_errors": errors,
        "outer_transport_only_baseline": {
            "method": "counterfactual projection retaining only transport blocks (the pre-polylogue-j2zz behavior)",
            "transport_actions": transport_actions,
            "typed_child_actions": 0,
            "child_actions_with_path": 0,
            "child_results_with_structured_outcome": 0,
        },
        "lowered": aggregate,
        "coverage": {
            "typed_children_per_transport": _ratio(typed_children, transport_actions),
            "paired_result_coverage": _ratio(paired_results, typed_children),
            "structured_outcome_coverage": _ratio(structured_outcomes, child_results),
            "path_coverage": _ratio(path_count, typed_children),
        },
        "methodology": {
            "pairing": (
                "Nth child use to Nth child result for the same deterministic child tool_id, in transcript order"
            ),
            "structural_outcomes": "Only parser-populated is_error/exit_code fields count; text tokens do not",
            "exit_code_token_counts": "Diagnostic only; never used to promote an outcome",
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_report(args.paths)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        sys.stdout.write(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 1 if args.strict and bool(report["parse_errors"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
