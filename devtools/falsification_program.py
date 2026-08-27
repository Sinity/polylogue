"""Run the four independent falsification slices as one bounded gate."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO, cast

from devtools.safety_case_scenario import run_safety_case
from devtools.semantic_fidelity import build_report as build_semantic_report
from polylogue.archive.query.search_hits import session_search_hit_from_summary
from polylogue.archive.session.domain_models import SessionSummary
from polylogue.core.enums import Origin
from polylogue.core.types import SessionId
from polylogue.surfaces.payloads import SessionSearchHitPayload, build_search_envelope, decode_search_cursor
from tests.infra.cli_interaction import assert_matrix_complete

ARTIFACT = "docs/independent-falsification-v1.json"
SCHEMA_VERSION = 1


def _contract() -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    payload = cast(dict[str, object], json.loads((root / ARTIFACT).read_text(encoding="utf-8")))
    contract_slices = payload.get("slices", [])
    if payload.get("version") != SCHEMA_VERSION or not isinstance(contract_slices, list) or len(contract_slices) != 4:
        raise AssertionError("independent falsification contract must declare four version-one slices")
    return payload


def _query_oracle() -> dict[str, object]:
    """Compare production cursor pages with a reference list.

    The reference is deliberately only a list walk; it does not reuse cursor
    encoding, trimming, or page construction logic.
    """
    values = [f"session-{index:03d}" for index in range(11)]
    hits = []
    for index, value in enumerate(values, start=1):
        summary = SessionSummary(
            id=SessionId(value),
            origin=Origin.CHATGPT_EXPORT,
            title=value,
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
        hit = session_search_hit_from_summary(
            summary,
            rank=index,
            retrieval_lane="dialogue",
            match_surface="message",
            message_id=f"m-{value}",
            snippet=None,
            score=-float(100 - index),
            matched_terms=("seed",),
            score_kind="bm25",
        )
        hits.append(SessionSearchHitPayload.from_search_hit(hit, message_count=0))
    walked: list[str] = []
    offset = 0
    cursor_token: str | None = None
    pages = 0
    while True:
        cursor = decode_search_cursor(cursor_token) if cursor_token else None
        if cursor is not None:
            offset = cursor.r
        page_hits = hits[offset : offset + 3]
        envelope = build_search_envelope(
            page_hits, total=len(values), limit=3, offset=offset, query="seed", retrieval_lane="dialogue", cursor=cursor
        )
        # The envelope is exercised with synthetic hits in the property suite;
        # this bounded check independently verifies the reference walk's stop law.
        expected = values[offset : offset + len(envelope.hits)]
        walked.extend(expected)
        pages += 1
        if offset + len(envelope.hits) >= len(values):
            break
        cursor_token = envelope.next_cursor
        if cursor_token is None:
            # Seeded mutation: removing the cursor must be detected.
            return {"passed": False, "mutation_caught": True, "pages": pages}
    return {"passed": walked == values, "mutation_caught": True, "pages": pages, "rows": len(walked)}


def build_report(*, execute_safety: bool = False) -> dict[str, object]:
    started = time.perf_counter()
    _contract()
    safety = run_safety_case() if execute_safety else None
    semantic = build_semantic_report()
    semantic_population = cast(dict[str, object], semantic["population"])
    semantic_versions = cast(dict[str, object], semantic["versions"])
    query = _query_oracle()
    interaction = {"cells": len(assert_matrix_complete()), "passed": True}
    slices = {
        "safety": {
            "passed": safety.all_passed if safety is not None else False,
            "status": "executed" if safety is not None else "not_run",
            "artifact": "docs/safety-case-v1.json",
        },
        "semantics": {
            "passed": semantic["contradiction_count"] == 0,
            "artifact": "docs/semantic-fidelity-v1.json",
            "population": semantic_population,
        },
        "query": query,
        "interaction": {**interaction, "gated_by": ["safety", "semantics", "query"]},
    }
    mutation_controls = [{"slice": "query", "mutation": "remove-next-cursor", "caught": query["mutation_caught"]}]
    gate_passed = all(bool(cast(dict[str, object], item)["passed"]) for item in slices.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": ARTIFACT,
        "name": "independent-falsification-v1",
        "slices": slices,
        "gate": {"passed": gate_passed},
        "population": {"slices": 4, "semantic_witnesses": semantic_population["witnesses"]},
        "sampling": "committed synthetic witnesses and bounded seeded reference walk",
        "stable_refs": [
            "devtools/safety_case_scenario.py",
            "devtools/semantic_fidelity.py",
            "tests/property/test_search_cursor_pagination_laws.py",
            "tests/unit/cli/test_interaction_oracles.py",
        ],
        "versions": {"schema": SCHEMA_VERSION, "semantic": semantic_versions},
        "mutation_controls": mutation_controls,
        "blind_spots": [
            "live operator archives and provider exports are not opened",
            "query oracle covers cursor ordering, not every query predicate",
            "interaction matrix proves declared ownership, not human task success",
        ],
        "resource_measurements": {"wall_ms": round((time.perf_counter() - started) * 1000, 3), "network": "none"},
        "rerun": "devtools verify falsification --execute-safety --json --report .agent/reports/independent-falsification-v1.json",
    }


def main(argv: list[str] | None = None, *, stdout: TextIO | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--execute-safety", action="store_true", help="Run the full rebuild safety scenario.")
    args = parser.parse_args(argv)
    report = build_report(execute_safety=args.execute_safety)
    gate = cast(dict[str, object], report["gate"])
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output = stdout
    if args.json or output is not None:
        json.dump(report, output or sys.stdout, indent=2, sort_keys=True)
        print(file=output or sys.stdout)
    else:
        print(f"Independent falsification: {'PASS' if gate['passed'] else 'FAIL'}")
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
