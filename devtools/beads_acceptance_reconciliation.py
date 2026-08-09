"""Reconcile guarded acceptance contracts against a read-only Beads export.

This module deliberately operates on JSONL files. It never invokes ``bd`` and
never selects an authority for a changed record. The generated import wave is
made from the live record, with only the two contract fields replaced, so
``bd import --allow-stale`` cannot overwrite unrelated live work.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from devtools import beads_acceptance_contracts as _contracts
from polylogue.core.json import JSONDecodeError
from polylogue.core.json import dumps as json_dumps
from polylogue.core.json import loads as json_loads

source_digest = _contracts.source_digest
validate = _contracts.validate


def render(contract: dict[str, Any]) -> str:
    """Expose the merged validator renderer for synthetic reconciliation fixtures."""
    return _contracts.render(contract)


_CONTRACT_KEY = "acceptance_contract_v1"
_CONTRACT_FIELDS = frozenset({"acceptance_criteria", "metadata"})
_REPORT_CATEGORIES = (
    "master_only",
    "live_only",
    "master_newer",
    "live_newer",
    "same_timestamp_different",
    "contract_refused",
)


class ReconciliationError(ValueError):
    """Raised when a file-level reconciliation input is unsafe to consume."""


def _metadata_object(value: object) -> dict[str, Any] | None:
    if value is None:
        return {}
    if isinstance(value, dict):
        return copy.deepcopy(value)
    if not isinstance(value, str):
        return None
    try:
        decoded = json_loads(value)
    except JSONDecodeError:
        return None
    return copy.deepcopy(decoded) if isinstance(decoded, dict) else None


def _contract(issue: Mapping[str, Any]) -> dict[str, Any] | None:
    metadata = _metadata_object(issue.get("metadata"))
    if metadata is None:
        return None
    value = metadata.get(_CONTRACT_KEY)
    return copy.deepcopy(value) if isinstance(value, dict) else None


def load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    """Load a Beads JSONL export, rejecting malformed or duplicate IDs."""
    rows: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json_loads(line)
        except JSONDecodeError as exc:
            raise ReconciliationError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise ReconciliationError(f"{path}:{line_number}: expected a JSON object")
        bead_id = value.get("id")
        if not isinstance(bead_id, str) or not bead_id:
            raise ReconciliationError(f"{path}:{line_number}: missing string Bead id")
        if bead_id in rows:
            raise ReconciliationError(f"{path}:{line_number}: duplicate Bead id {bead_id}")
        rows[bead_id] = value
    return rows


def _canonical_rows(rows: Mapping[str, Mapping[str, Any]], ids: Iterable[str]) -> str:
    return "\n".join(json_dumps(rows[bead_id], sort_keys=True) for bead_id in sorted(ids)) + "\n"


def equality_digest(rows: Mapping[str, Mapping[str, Any]], ids: Iterable[str] | None = None) -> str:
    """Hash complete canonical rows, ordered by Bead ID."""
    selected = rows.keys() if ids is None else ids
    return hashlib.sha256(_canonical_rows(rows, selected).encode("utf-8")).hexdigest()


def non_contract_equality_digest(rows: Mapping[str, Mapping[str, Any]], ids: Iterable[str]) -> str:
    """Hash rows after removing only the two contract fields."""
    scrubbed: dict[str, dict[str, Any]] = {}
    for bead_id in ids:
        row = dict(copy.deepcopy(rows[bead_id]))
        row.pop("acceptance_criteria", None)
        raw_metadata = row.get("metadata")
        metadata = _metadata_object(raw_metadata)
        if metadata is not None:
            metadata.pop(_CONTRACT_KEY, None)
            if not metadata:
                # The contract is the only allowed metadata mutation. Remove
                # the now-empty container from both sides of the projection;
                # all non-contract metadata keys remain visible and checked.
                row.pop("metadata", None)
            elif isinstance(raw_metadata, str):
                row["metadata"] = json_dumps(metadata)
            else:
                row["metadata"] = metadata
        scrubbed[bead_id] = row
    return equality_digest(scrubbed)


def _classify_timestamp(master: Mapping[str, Any], live: Mapping[str, Any]) -> str | None:
    master_timestamp = master.get("updated_at")
    live_timestamp = live.get("updated_at")
    if not isinstance(master_timestamp, str) or not isinstance(live_timestamp, str):
        return None
    if master_timestamp == live_timestamp:
        return (
            "same_timestamp_same"
            if source_digest(dict(master)) == source_digest(dict(live))
            else "same_timestamp_different"
        )
    if master_timestamp > live_timestamp:
        return "master_newer"
    return "live_newer"


def _guarded_row(
    *,
    master: Mapping[str, Any],
    live: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    metadata = _metadata_object(live.get("metadata"))
    if metadata is None:
        raise ReconciliationError(f"{live.get('id')}: live metadata is not a JSON object")
    row = copy.deepcopy(dict(live))
    metadata[_CONTRACT_KEY] = copy.deepcopy(dict(contract))
    row["metadata"] = json_dumps(metadata) if isinstance(live.get("metadata"), str) else metadata
    row["acceptance_criteria"] = master.get("acceptance_criteria")
    changed = {key for key in set(row) | set(live) if key not in _CONTRACT_FIELDS and row.get(key) != live.get(key)}
    if changed:
        raise ReconciliationError(f"{live.get('id')}: guarded wave changed non-contract fields {sorted(changed)}")
    return row


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json_dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def reconcile(repository: Path, live_export: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return a report and a minimal guarded import wave.

    A contract is eligible only when the live source digest equals the digest
    carried by the canonical contract. Timestamp ordering is reported but is
    never used as an authority choice. A malformed live metadata document is
    an explicit contract refusal.
    """
    master = load_jsonl(repository)
    live = load_jsonl(live_export)
    master_contracts = {
        bead_id: contract for bead_id, row in master.items() if (contract := _contract(row)) is not None
    }
    invalid_contracts = {bead_id: errors for bead_id in master_contracts if (errors := validate(master[bead_id]))}
    if invalid_contracts:
        details = "; ".join(f"{bead_id}: {', '.join(errors)}" for bead_id, errors in sorted(invalid_contracts.items()))
        raise ReconciliationError(f"canonical contract validation failed: {details}")

    master_ids = set(master)
    live_ids = set(live)
    report: dict[str, Any] = {
        "repository": str(repository),
        "live_export": str(live_export),
        "counts": dict.fromkeys(_REPORT_CATEGORIES, 0),
        "ids": {category: [] for category in _REPORT_CATEGORIES},
        "contract_denominator": len(master_contracts),
        "contract_present_denominator": len(set(master_contracts) & live_ids),
        "contract_guarded_count": 0,
        "contract_refused_denominator": 0,
        "contract_refused_reasons": {},
        "contract_deferred_denominator": 0,
        "contract_deferred_reasons": {},
        "already_guarded_ids": [],
        "targeted_ids": [],
    }
    for report_category, ids in (
        ("master_only", master_ids - live_ids),
        ("live_only", live_ids - master_ids),
    ):
        report["ids"][report_category] = sorted(ids)
        report["counts"][report_category] = len(ids)

    wave: list[dict[str, Any]] = []
    refused_reasons: dict[str, list[str]] = {}
    for bead_id in sorted(master_ids & live_ids):
        master_row = master[bead_id]
        live_row = live[bead_id]
        timestamp_category = _classify_timestamp(master_row, live_row)
        if timestamp_category in {"master_newer", "live_newer", "same_timestamp_different"}:
            report["ids"][timestamp_category].append(bead_id)
            report["counts"][timestamp_category] += 1

        contract = master_contracts.get(bead_id)
        if contract is None:
            continue
        report["contract_refused_denominator"] += 1
        expected_digest = contract["source_digest"]
        actual_digest = source_digest(live_row)
        reasons: list[str] = []
        if actual_digest != expected_digest:
            reasons.append(f"source digest mismatch: expected contract {expected_digest}, live {actual_digest}")
        if _metadata_object(live_row.get("metadata")) is None:
            reasons.append("live metadata is not a JSON object")
        if timestamp_category is None:
            reasons.append("updated_at must be a string on both repository and live records")
        if reasons:
            report["ids"]["contract_refused"].append(bead_id)
            report["counts"]["contract_refused"] += 1
            refused_reasons[bead_id] = reasons
            continue
        if timestamp_category == "live_newer":
            report["contract_deferred_denominator"] += 1
            report["contract_deferred_reasons"][bead_id] = (
                "live-newer record is excluded from the targeted wave; coordinator adjudication is required"
            )
            continue
        candidate = _guarded_row(master=master_row, live=live_row, contract=contract)
        current_contract = _contract(live_row)
        if (
            live_row.get("acceptance_criteria") == master_row.get("acceptance_criteria")
            and current_contract == contract
        ):
            report["already_guarded_ids"].append(bead_id)
            continue
        wave.append(candidate)
        report["targeted_ids"].append(bead_id)

    report["ids"]["contract_refused"] = sorted(report["ids"]["contract_refused"])
    report["contract_refused_reasons"] = {bead_id: refused_reasons[bead_id] for bead_id in sorted(refused_reasons)}
    report["contract_deferred_reasons"] = {
        bead_id: report["contract_deferred_reasons"][bead_id] for bead_id in sorted(report["contract_deferred_reasons"])
    }
    report["contract_guarded_count"] = len(report["targeted_ids"]) + len(report["already_guarded_ids"])
    report["live_equality_digest"] = equality_digest(live)
    report["targeted_non_contract_equality_digest"] = non_contract_equality_digest(live, report["targeted_ids"])
    report["targeted_wave_equality_digest"] = equality_digest({row["id"]: row for row in wave}, report["targeted_ids"])
    report["targeted_wave_non_contract_equality_digest"] = non_contract_equality_digest(
        {row["id"]: row for row in wave}, report["targeted_ids"]
    )
    return report, wave


def verify_post_import(*, before: Path, after: Path, wave: Path) -> dict[str, Any]:
    """Verify that a targeted import changed only the guarded contract fields."""
    before_rows = load_jsonl(before)
    after_rows = load_jsonl(after)
    wave_rows = load_jsonl(wave)
    target_ids = set(wave_rows)
    if not target_ids <= set(before_rows):
        raise ReconciliationError("targeted wave contains records absent from the before export")
    if not target_ids <= set(after_rows):
        raise ReconciliationError("targeted wave contains records absent from the after export")
    unchanged_ids = set(before_rows) - target_ids
    added_ids = sorted(set(after_rows) - set(before_rows))
    removed_ids = sorted(set(before_rows) - set(after_rows))
    if added_ids or removed_ids:
        raise ReconciliationError(
            f"post-import export changed the record universe: added={added_ids}, removed={removed_ids}"
        )
    changed_outside_wave = sorted(
        bead_id for bead_id in unchanged_ids if before_rows[bead_id] != after_rows.get(bead_id)
    )
    if changed_outside_wave:
        raise ReconciliationError(f"post-import export changed records outside targeted wave: {changed_outside_wave}")
    before_non_contract = non_contract_equality_digest(before_rows, target_ids)
    after_non_contract = non_contract_equality_digest(after_rows, target_ids)
    expected_wave = equality_digest(wave_rows)
    actual_target = equality_digest(after_rows, target_ids)
    if before_non_contract != after_non_contract:
        raise ReconciliationError("post-import targeted records changed non-contract fields")
    if expected_wave != actual_target:
        raise ReconciliationError("post-import targeted records differ from the guarded wave")
    return {
        "ok": True,
        "targeted_ids": sorted(target_ids),
        "unchanged_outside_wave": len(unchanged_ids),
        "targeted_non_contract_equality_digest": after_non_contract,
        "targeted_wave_equality_digest": actual_target,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reconcile repository acceptance contracts with a read-only live Beads export."
    )
    parser.add_argument("--repository", type=Path, help="canonical repository JSONL")
    parser.add_argument("--live", type=Path, help="read-only live Beads export")
    parser.add_argument("--wave", type=Path, help="write the guarded targeted import JSONL")
    parser.add_argument("--report", type=Path, help="write the reconciliation report JSON")
    parser.add_argument("--verify-before", type=Path, help="before export for post-import verification")
    parser.add_argument("--verify-after", type=Path, help="after export for post-import verification")
    parser.add_argument("--verify-wave", type=Path, help="guarded wave for post-import verification")
    parser.add_argument("--json", action="store_true", help="emit the report or verification result as JSON")
    args = parser.parse_args(argv)
    try:
        if args.verify_before or args.verify_after or args.verify_wave:
            if not all((args.verify_before, args.verify_after, args.verify_wave)):
                raise ReconciliationError(
                    "post-import verification requires --verify-before, --verify-after, and --verify-wave"
                )
            result = verify_post_import(
                before=args.verify_before,
                after=args.verify_after,
                wave=args.verify_wave,
            )
        else:
            if not args.repository or not args.live:
                raise ReconciliationError("reconciliation requires --repository and --live")
            result, wave = reconcile(args.repository, args.live)
            if args.wave:
                _write_jsonl(args.wave, wave)
            if args.report:
                args.report.write_text(json_dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except ReconciliationError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json_dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
