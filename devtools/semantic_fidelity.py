"""Bounded, privacy-safe semantic contradiction and construct-flow census."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import TextIO

from polylogue.sources.dispatch import detect_provider_evidence, parse_payload, require_positive_conversational_evidence
from polylogue.sources.origin_specs import ORIGIN_SPECS, lowering_fingerprint
from tests.infra.origin_capability_matrix import load_manifest, load_witness_fixture

ARTIFACT_PATH = "docs/semantic-fidelity-v1.json"
SCHEMA_VERSION = 1

_CONSTRUCTS = (
    ("session_identity", "preserved", "polylogue/storage/sqlite/archive_tiers/write.py:363-382"),
    ("session_title", "normalized", "polylogue/sources/parsers/base_models.py:424-500"),
    ("messages", "preserved", "polylogue/sources/parsers/base_models.py:225-330"),
    ("blocks", "queryable", "polylogue/storage/sqlite/archive_tiers/archive_tiers_specs.py:266-313"),
    ("tool_use_and_result", "queryable", "polylogue/storage/sqlite/archive_tiers/archive_tiers_specs.py:293-299"),
    ("usage", "provenance-marked", "polylogue/sources/parsers/base_models.py:260-278"),
    ("timestamps", "provenance-marked", "polylogue/sources/parsers/base_models.py:432-445"),
    ("lineage", "queryable", "polylogue/storage/sqlite/archive_tiers/write.py:505-563"),
    ("attachments", "preserved", "polylogue/sources/parsers/base_models.py:330-397"),
    ("session_events", "preserved", "polylogue/sources/parsers/base_models.py:412-424"),
    ("instructions_and_context", "rendered", "polylogue/sources/parsers/base_models.py:470-500"),
    ("unknown_opaque_fields", "intentionally unsupported", "polylogue/sources/origin_specs.py:1253-1263"),
)


def _shape(value: object) -> str:
    if isinstance(value, list):
        return "list[object]"
    if isinstance(value, dict):
        return "object{" + ",".join(sorted(str(key) for key in value)[:12]) + "}"
    return type(value).__name__


def _mutated_empty(payload: object) -> object:
    """Drop the conversation-bearing construct while retaining its envelope."""
    if isinstance(payload, list):
        return []
    if isinstance(payload, dict):
        # Empty all fields through the parser's real shape gate.  Keeping an
        # identity-only envelope is deliberate: a parser must not manufacture
        # a conversational session from an otherwise valid source container.
        result: dict[object, object] = {}
        for key, value in payload.items():
            if isinstance(value, list):
                result[key] = []
            elif isinstance(value, dict):
                result[key] = {}
            else:
                result[key] = "" if isinstance(value, str) else None
        return result
    return payload


def _representative_refs(rows: list[dict[str, object]], limit: int = 20) -> list[object]:
    refs: list[object] = []
    for row in rows:
        witnesses = row.get("witnesses")
        if not isinstance(witnesses, list):
            continue
        for witness in witnesses:
            if isinstance(witness, dict):
                refs.append(witness["fixture_ref"])
    return refs[:limit]


def build_report() -> dict[str, object]:
    started = time.perf_counter()
    manifest = load_manifest()
    rows: list[dict[str, object]] = []
    shapes: Counter[str] = Counter()
    contradictions: list[dict[str, object]] = []
    mutation_receipts: list[dict[str, object]] = []
    population = 0
    parsed_sessions = 0
    parsed_messages = 0

    for entry in manifest.entries:
        spec = next(spec for spec in ORIGIN_SPECS if spec.origin.value == entry.origin.value)
        if entry.unsupported is not None:
            rows.append(
                {
                    "origin": entry.origin.value,
                    "lifecycle": spec.lifecycle,
                    "status": "unsupported",
                    "classification": entry.unsupported.reason,
                    "authority": entry.unsupported.detail,
                }
            )
            continue
        witness_rows = []
        for witness in entry.witnesses:
            payload = load_witness_fixture(witness)
            population += 1
            shapes[_shape(payload)] += 1
            detected, evidence = detect_provider_evidence(payload, witness.fixture_path)
            sessions = parse_payload(
                witness.parser_claims[0].provider, payload, witness.fallback_id, source_path=witness.fixture_path
            )
            accepted = require_positive_conversational_evidence(
                sessions, provider=witness.parser_claims[0].provider, source_path=witness.fixture_path
            )
            parsed_sessions += len(accepted)
            parsed_messages += sum(len(session.messages) for session in accepted)
            detector_matches = (
                detected is witness.parser_claims[0].provider
                if witness.route == "detected"
                else detected is not None and bool(evidence.strip())
            )
            if not detector_matches or not accepted:
                contradictions.append(
                    {
                        "class": "production_route_rejected_positive",
                        "origin": entry.origin.value,
                        "evidence_ref": witness.fixture_path,
                    }
                )
            witness_rows.append(
                {
                    "fixture_ref": witness.fixture_path,
                    "route": witness.route,
                    "provider_wire": witness.parser_claims[0].provider.value,
                    "detector_evidence": evidence,
                    "sessions": len(accepted),
                    "messages": sum(len(session.messages) for session in accepted),
                }
            )

            mutated = _mutated_empty(payload)
            mutated_sessions = parse_payload(
                witness.parser_claims[0].provider, mutated, "mutation-dropped-construct", source_path=None
            )
            mutation_rejected = not require_positive_conversational_evidence(
                mutated_sessions, provider=witness.parser_claims[0].provider, source_path=None
            )
            mutation_receipts.append(
                {
                    "kind": "dropped_construct",
                    "origin": entry.origin.value,
                    "seed_ref": witness.fixture_path,
                    "caught": mutation_rejected,
                }
            )
            if not mutation_rejected:
                contradictions.append(
                    {
                        "class": "dropped_construct_accepted",
                        "origin": entry.origin.value,
                        "evidence_ref": witness.fixture_path,
                    }
                )

        rows.append(
            {
                "origin": entry.origin.value,
                "lifecycle": spec.lifecycle,
                "status": "covered",
                "parser_refs": list(spec.parser_paths),
                "witnesses": witness_rows,
                "construct_flow": [
                    {"construct": name, "classification": classification, "owner_ref": owner}
                    for name, classification, owner in _CONSTRUCTS
                ],
            }
        )

    for case in manifest.collisions:
        detected, evidence = detect_provider_evidence(case.payload)
        caught = detected is case.expected_provider and bool(evidence.strip())
        mutation_receipts.append({"kind": "seeded_disagreement", "case": case.name, "caught": caught})
        if not caught:
            contradictions.append({"class": "detector_precedence_disagreement", "evidence_ref": case.name})

    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": ARTIFACT_PATH,
        "method": "OriginSpec -> origin capability manifest -> production detector/parser -> content gate",
        "bounds": {"max_witnesses": population, "max_seconds": 60, "implementation": "bounded-single-process"},
        "population": {
            "witnesses": population,
            "accepted_sessions": parsed_sessions,
            "accepted_messages": parsed_messages,
        },
        "denominator": {
            "origin_specs": len(ORIGIN_SPECS),
            "executable_origin_specs": sum(s.lifecycle == "executable" for s in ORIGIN_SPECS),
            "witnesses": population,
            "negative_families": {
                "empty": len(manifest.empty),
                "partial": len(manifest.partial),
                "malformed": len(manifest.malformed),
            },
        },
        "strata": {
            "raw_shapes": dict(sorted(shapes.items())),
            "routes": {
                "detected": sum(w.route == "detected" for e in manifest.entries for w in e.witnesses),
                "source-hint": sum(w.route == "source-hint" for e in manifest.entries for w in e.witnesses),
            },
        },
        "contradictions": contradictions,
        "contradiction_count": len(contradictions),
        "confidence": "high for the committed witness population; not a claim about unobserved provider exports",
        "representative_refs": _representative_refs(rows),
        "construct_flow": rows,
        "mutation_controls": mutation_receipts,
        "versions": {"schema": SCHEMA_VERSION, "lowering_fingerprint": lowering_fingerprint()},
        "blind_spots": [
            "live operator archives are not read",
            "opaque fields are shape-counted, not semantically interpreted",
            "fixture coverage is one or two witnesses per executable mode",
            "storage read/render parity is outside this bounded parser census",
        ],
        "resource_measurements": {
            "wall_ms": elapsed_ms,
            "memory": "not measured",
            "network": "none",
            "privacy": "fixture paths and stable case names only; no payload text",
        },
        "rerun": "devtools verify semantic-fidelity --json --report .agent/reports/semantic-fidelity-v1.json",
    }


def main(argv: list[str] | None = None, *, stdout: TextIO | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, help="Write the privacy-safe JSON report to this path.")
    parser.add_argument("--json", action="store_true", help="Emit the report as JSON.")
    args = parser.parse_args(argv)
    output = stdout if stdout is not None else sys.stdout
    report = build_report()
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json or stdout is not None:
        json.dump(report, output, indent=2, sort_keys=True)
        print(file=output)
    else:
        population = report["population"]
        assert isinstance(population, dict)
        print(f"Semantic fidelity: {population['witnesses']} witnesses, {report['contradiction_count']} contradictions")
    return 1 if report["contradiction_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
