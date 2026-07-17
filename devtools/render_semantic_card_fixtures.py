"""Regenerate or verify semantic-card golden fixture expectations.

The input side of each case is hand-authored. ``expected_contract`` is the
compact independent oracle reviewers can audit; ``expected_cards`` freezes card
contracts. Cases that exercise prose/notices also carry ``expected_entries`` so
the complete ordered transcript behavior is reviewable without duplicating it
for every card-only fixture.
"""

from __future__ import annotations

import argparse
import base64
import copy
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from polylogue.core.json import JSONDocument
from polylogue.rendering.semantic_card_models import (
    LineageDescriptor,
    SemanticTranscript,
    card_json_documents,
)
from polylogue.rendering.semantic_cards import build_semantic_transcript

DEFAULT_ROOT = Path("tests/data/semantic_cards/cases")


def _load_text_value(value: object, *, case_path: Path) -> object:
    if isinstance(value, list):
        return [_load_text_value(item, case_path=case_path) for item in value]
    if not isinstance(value, Mapping):
        return value
    if set(value) == {"text_fixture"} and isinstance(value.get("text_fixture"), str):
        return (case_path.parent / cast(str, value["text_fixture"])).read_text(encoding="utf-8")
    if set(value) == {"text_bytes_base64"} and isinstance(value.get("text_bytes_base64"), str):
        return base64.b64decode(cast(str, value["text_bytes_base64"]), validate=True)
    return {str(key): _load_text_value(item, case_path=case_path) for key, item in value.items()}


def _materialize_messages(value: object, *, case_path: Path) -> list[dict[str, object]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{case_path}: messages must be a JSON array")
    rendered = _load_text_value(copy.deepcopy(value), case_path=case_path)
    if not isinstance(rendered, list) or not all(isinstance(item, dict) for item in rendered):
        raise ValueError(f"{case_path}: messages must contain JSON objects")
    return cast(list[dict[str, object]], rendered)


def _lineage(value: object) -> dict[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("lineage must be a JSON object")
    return cast(dict[str, object], value)


def build_case_transcript(case: Mapping[str, object], *, case_path: Path) -> SemanticTranscript:
    """Build the full pure transcript for one hand-authored fixture case."""

    session_id = str(case.get("session_id", ""))
    if not session_id:
        raise ValueError(f"{case_path}: session_id is required")
    lineage_data = _lineage(case.get("lineage"))
    lineage = LineageDescriptor(**lineage_data) if lineage_data is not None else None  # type: ignore[arg-type]
    return build_semantic_transcript(
        _materialize_messages(case.get("messages"), case_path=case_path),
        session_id=session_id,
        lineage=lineage,
        provider_family=str(case.get("provider_family", "unknown")),
    )


def render_case(case: Mapping[str, object], *, case_path: Path) -> list[JSONDocument]:
    return card_json_documents(build_case_transcript(case, case_path=case_path).cards)


def render_case_entries(case: Mapping[str, object], *, case_path: Path) -> list[JSONDocument]:
    document = build_case_transcript(case, case_path=case_path).to_document()
    entries = document.get("entries")
    if not isinstance(entries, list):
        raise ValueError(f"{case_path}: renderer did not emit transcript entries")
    return [cast(JSONDocument, entry) for entry in entries]


def _assert_compact_contract(
    case: Mapping[str, object],
    actual: list[JSONDocument],
    actual_entries: list[JSONDocument],
    *,
    case_path: Path,
) -> list[str]:
    contract = case.get("expected_contract")
    if not isinstance(contract, Mapping):
        return [f"{case_path}: expected_contract must be an object"]
    errors: list[str] = []
    expected_count = contract.get("card_count")
    if expected_count != len(actual):
        errors.append(f"{case_path}: card_count expected {expected_count!r}, got {len(actual)}")
    expected_kinds = contract.get("kinds")
    kinds = [doc.get("kind") for doc in actual]
    if expected_kinds != kinds:
        errors.append(f"{case_path}: kinds expected {expected_kinds!r}, got {kinds!r}")
    expected_outcomes = contract.get("outcomes")
    outcomes: list[object] = []
    for doc in actual:
        outcome = doc.get("outcome")
        outcomes.append(outcome.get("state") if isinstance(outcome, dict) else None)
    if expected_outcomes != outcomes:
        errors.append(f"{case_path}: outcomes expected {expected_outcomes!r}, got {outcomes!r}")
    if "entry_types" in contract:
        entry_types = [entry.get("entry_type") for entry in actual_entries]
        if contract.get("entry_types") != entry_types:
            errors.append(f"{case_path}: entry_types expected {contract.get('entry_types')!r}, got {entry_types!r}")
    if "notice_counts" in contract:
        notice_counts = [
            cast(Mapping[str, object], entry.get("notice")).get("count")
            for entry in actual_entries
            if entry.get("entry_type") == "notice" and isinstance(entry.get("notice"), Mapping)
        ]
        if contract.get("notice_counts") != notice_counts:
            errors.append(
                f"{case_path}: notice_counts expected {contract.get('notice_counts')!r}, got {notice_counts!r}"
            )
    return errors


def _case_files(root: Path, explicit: Sequence[str]) -> list[Path]:
    if explicit:
        return [Path(item) for item in explicit]
    return sorted(root.glob("*.json"))


def _load_case(path: Path) -> dict[str, object]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path}: root must be an object")
    return cast(dict[str, object], loaded)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--fixture", action="append", default=[], help="Explicit fixture path; repeatable")
    args = parser.parse_args(argv)

    paths = _case_files(args.root, args.fixture)
    if not paths:
        print("semantic-card fixtures: no cases found", file=sys.stderr)
        return 2

    failures: list[str] = []
    for path in paths:
        try:
            case = _load_case(path)
            transcript = build_case_transcript(case, case_path=path)
            actual = card_json_documents(transcript.cards)
            transcript_document = transcript.to_document()
            raw_entries = transcript_document.get("entries")
            if not isinstance(raw_entries, list):
                raise ValueError(f"{path}: renderer did not emit transcript entries")
            actual_entries = [cast(JSONDocument, entry) for entry in raw_entries]
            failures.extend(_assert_compact_contract(case, actual, actual_entries, case_path=path))
            if args.write:
                case["expected_cards"] = actual
                if case.get("freeze_entries") is True or "expected_entries" in case:
                    case["expected_entries"] = actual_entries
                path.write_text(json.dumps(case, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            else:
                expected = case.get("expected_cards")
                if expected != actual:
                    failures.append(f"{path}: full expected_cards contract differs from renderer output")
                if "expected_entries" in case and case.get("expected_entries") != actual_entries:
                    failures.append(f"{path}: full expected_entries contract differs from renderer output")
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            failures.append(f"{path}: {exc}")

    if failures:
        print("semantic-card fixtures: FAILED", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    verb = "wrote" if args.write else "verified"
    print(f"semantic-card fixtures: {verb} {len(paths)} case(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
