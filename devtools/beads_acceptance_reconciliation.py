"""Reconcile guarded acceptance contracts against a read-only Beads export.

This module deliberately operates on JSONL files. It never invokes ``bd`` and
never selects an authority for a changed record. The generated import wave is
made from the live record, with only the two contract fields replaced, so
``bd import --allow-stale`` cannot overwrite unrelated live work.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import re
import sys
from collections.abc import Iterable, Mapping
from decimal import Decimal
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
_BEADS_TIMESTAMP = re.compile(
    r"^(?P<second>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})"
    r"(?:\.(?P<fraction>\d+))?(?P<timezone>Z|[+-]\d{2}:\d{2})$"
)
_REPORT_VERSION = 2


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


def _parse_beads_timestamp(value: object) -> tuple[dt.datetime, Decimal]:
    if not isinstance(value, str):
        raise ValueError("updated_at must be a string on both repository and live records")
    match = _BEADS_TIMESTAMP.fullmatch(value)
    if match is None:
        raise ValueError("updated_at must be a valid canonical Beads timestamp")
    try:
        parsed = dt.datetime.fromisoformat(f"{match['second']}{match['timezone'].replace('Z', '+00:00')}")
    except ValueError as exc:
        raise ValueError("updated_at must be a valid canonical Beads timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError("updated_at must include a timezone")
    fraction = Decimal(f"0.{match['fraction'] or '0'}")
    return parsed.astimezone(dt.UTC).replace(microsecond=0), fraction


def _classify_timestamp(master: Mapping[str, Any], live: Mapping[str, Any]) -> str:
    master_timestamp = _parse_beads_timestamp(master.get("updated_at"))
    live_timestamp = _parse_beads_timestamp(live.get("updated_at"))
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


def report_digest(report: Mapping[str, Any]) -> str:
    """Hash the exact reconciliation report payload."""
    return hashlib.sha256(json_dumps(dict(report), sort_keys=True).encode("utf-8")).hexdigest()


def _load_report(path: Path) -> dict[str, Any]:
    try:
        value = json_loads(path.read_text(encoding="utf-8"))
    except (OSError, JSONDecodeError) as exc:
        raise ReconciliationError(f"{path}: invalid reconciliation report: {exc}") from exc
    if not isinstance(value, dict):
        raise ReconciliationError(f"{path}: reconciliation report must be an object")
    return value


def _validate_report_and_wave(
    *,
    repository: Path,
    before: Path,
    wave: Path,
    report_path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    master = load_jsonl(repository)
    before_rows = load_jsonl(before)
    wave_rows = load_jsonl(wave)
    report = _load_report(report_path)
    required_ids = _contracts.load_manifest(_contracts._DEFAULT_MANIFEST)
    registry_errors = _contracts.validate_route_registry(required_ids)
    if registry_errors:
        raise ReconciliationError("canonical route registry validation failed: " + "; ".join(registry_errors))
    if report.get("report_version") != _REPORT_VERSION:
        raise ReconciliationError("reconciliation report version is stale or unsupported")
    if report.get("manifest_digest") != _contracts._EXPECTED_MANIFEST_DIGEST:
        raise ReconciliationError("reconciliation report manifest digest does not match this exact head")
    if report.get("route_registry_digest") != _contracts.route_registry_digest():
        raise ReconciliationError("reconciliation report route registry digest is stale")
    if report.get("contract_denominator") != len(required_ids):
        raise ReconciliationError("reconciliation report contract denominator is stale")
    if report.get("repository_population_digest") != equality_digest(master):
        raise ReconciliationError("canonical repository population digest does not match the reconciliation report")
    if report.get("live_population_digest") != equality_digest(before_rows):
        raise ReconciliationError("before population digest does not match the reconciliation report")
    target_ids = report.get("targeted_ids")
    if (
        not isinstance(target_ids, list)
        or target_ids != sorted(set(target_ids))
        or any(not isinstance(i, str) for i in target_ids)
    ):
        raise ReconciliationError("reconciliation report targeted_ids must be sorted and unique")
    wave_ids = list(wave_rows)
    if wave_ids != target_ids:
        raise ReconciliationError("targeted wave order or population does not match the reconciliation report")
    if set(target_ids) - set(before_rows) or set(target_ids) - set(master):
        raise ReconciliationError("targeted wave contains records absent from the bound populations")
    expected_wave_digest = equality_digest(wave_rows, target_ids)
    if report.get("targeted_wave_equality_digest") != expected_wave_digest:
        raise ReconciliationError("targeted wave digest does not match the reconciliation report")
    row_digests = report.get("targeted_wave_row_digests")
    if not isinstance(row_digests, dict) or set(row_digests) != set(target_ids):
        raise ReconciliationError("reconciliation report does not carry every targeted wave row digest")
    for bead_id in target_ids:
        if row_digests.get(bead_id) != equality_digest({bead_id: wave_rows[bead_id]}, [bead_id]):
            raise ReconciliationError(f"targeted wave row digest mismatch for {bead_id}")
        canonical = master[bead_id]
        contract = _contract(canonical)
        if contract is None or validate(canonical):
            raise ReconciliationError(f"canonical contract revalidation failed for {bead_id}")
        if source_digest(before_rows[bead_id]) != contract.get("source_digest"):
            raise ReconciliationError(f"stale source digest refuses targeted row {bead_id}")
        expected = _guarded_row(master=canonical, live=before_rows[bead_id], contract=contract)
        if wave_rows[bead_id] != expected:
            raise ReconciliationError(f"targeted wave row is not the canonical guarded row for {bead_id}")
    return master, before_rows, wave_rows, report


def reconcile(
    repository: Path,
    live_export: Path,
    *,
    manifest: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return a report and a minimal guarded import wave.

    A contract is eligible only when the live source digest equals the digest
    carried by the canonical contract. Timestamp ordering is reported but is
    never used as an authority choice. A malformed live metadata document is
    an explicit contract refusal.
    """
    try:
        required_ids = _contracts.load_manifest(manifest or _contracts._DEFAULT_MANIFEST)
    except SystemExit as exc:
        raise ReconciliationError(str(exc)) from exc
    master = load_jsonl(repository)
    live = load_jsonl(live_export)
    missing_manifest_ids = sorted(set(required_ids) - set(master))
    if missing_manifest_ids:
        raise ReconciliationError(
            "canonical acceptance-contract manifest is incomplete: "
            f"missing {len(missing_manifest_ids)} IDs {missing_manifest_ids}"
        )
    master_contracts: dict[str, dict[str, Any]] = {}
    invalid_contracts: dict[str, list[str]] = {}
    for bead_id in required_ids:
        contract = _contract(master[bead_id])
        if contract is None:
            invalid_contracts[bead_id] = ["missing metadata.acceptance_contract_v1"]
            continue
        master_contracts[bead_id] = contract
        errors = validate(master[bead_id])
        if errors:
            invalid_contracts[bead_id] = errors
    if invalid_contracts:
        details = "; ".join(f"{bead_id}: {', '.join(errors)}" for bead_id, errors in sorted(invalid_contracts.items()))
        raise ReconciliationError(f"canonical contract validation failed: {details}")

    master_ids = set(master)
    live_ids = set(live)
    report: dict[str, Any] = {
        "report_version": _REPORT_VERSION,
        "repository": str(repository),
        "live_export": str(live_export),
        "counts": dict.fromkeys(_REPORT_CATEGORIES, 0),
        "ids": {category: [] for category in _REPORT_CATEGORIES},
        "contract_denominator": len(required_ids),
        "contract_present_denominator": len(set(required_ids) & live_ids),
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
        timestamp_error: str | None = None
        try:
            timestamp_category = _classify_timestamp(master_row, live_row)
        except ValueError as exc:
            timestamp_category = None
            timestamp_error = str(exc)
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
        if timestamp_error is not None:
            reasons.append(timestamp_error)
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
    report["manifest_digest"] = _contracts._EXPECTED_MANIFEST_DIGEST
    report["route_registry_digest"] = _contracts.route_registry_digest()
    report["repository_population_digest"] = equality_digest(master)
    report["live_equality_digest"] = equality_digest(live)
    report["live_population_digest"] = report["live_equality_digest"]
    report["targeted_non_contract_equality_digest"] = non_contract_equality_digest(live, report["targeted_ids"])
    report["targeted_wave_equality_digest"] = equality_digest({row["id"]: row for row in wave}, report["targeted_ids"])
    report["targeted_wave_non_contract_equality_digest"] = non_contract_equality_digest(
        {row["id"]: row for row in wave}, report["targeted_ids"]
    )
    report["targeted_wave_row_digests"] = {row["id"]: equality_digest({row["id"]: row}, [row["id"]]) for row in wave}
    return report, wave


def verify_post_import(*, repository: Path, before: Path, after: Path, wave: Path, report: Path) -> dict[str, Any]:
    """Verify that a targeted import changed only the guarded contract fields."""
    _, before_rows, wave_rows, reconciliation = _validate_report_and_wave(
        repository=repository,
        before=before,
        wave=wave,
        report_path=report,
    )
    after_rows = load_jsonl(after)
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
    expected_wave = equality_digest(wave_rows, reconciliation["targeted_ids"])
    actual_target = equality_digest(after_rows, target_ids)
    if before_non_contract != after_non_contract:
        raise ReconciliationError("post-import targeted records changed non-contract fields")
    if expected_wave != actual_target:
        raise ReconciliationError("post-import targeted records differ from the guarded wave")
    before_population_digest = equality_digest(before_rows)
    after_population_digest = equality_digest(after_rows)
    if reconciliation.get("live_population_digest") != before_population_digest:
        raise ReconciliationError("full before population digest changed after reconciliation")
    return {
        "ok": True,
        "report_digest": report_digest(reconciliation),
        "reconciliation_wave_digest": reconciliation["targeted_wave_equality_digest"],
        "before_population_digest": before_population_digest,
        "after_population_digest": after_population_digest,
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
    parser.add_argument(
        "--manifest", type=Path, default=_contracts._DEFAULT_MANIFEST, help="ratcheted contract ID manifest"
    )
    parser.add_argument("--wave", type=Path, help="write the guarded targeted import JSONL")
    parser.add_argument("--report", type=Path, help="write the reconciliation report JSON")
    parser.add_argument("--verify-before", type=Path, help="before export for post-import verification")
    parser.add_argument("--verify-after", type=Path, help="after export for post-import verification")
    parser.add_argument("--verify-wave", type=Path, help="guarded wave for post-import verification")
    parser.add_argument("--verify-repository", type=Path, help="canonical repository bound to the report")
    parser.add_argument("--verify-report", type=Path, help="exact reconciliation report bound to the wave")
    parser.add_argument("--json", action="store_true", help="emit the report or verification result as JSON")
    args = parser.parse_args(argv)
    try:
        if any(
            (
                args.verify_before,
                args.verify_after,
                args.verify_wave,
                args.verify_repository,
                args.verify_report,
            )
        ):
            if not all(
                (args.verify_before, args.verify_after, args.verify_wave, args.verify_repository, args.verify_report)
            ):
                raise ReconciliationError(
                    "post-import verification requires --verify-repository, --verify-report, --verify-before, "
                    "--verify-after, and --verify-wave"
                )
            result = verify_post_import(
                repository=args.verify_repository,
                before=args.verify_before,
                after=args.verify_after,
                wave=args.verify_wave,
                report=args.verify_report,
            )
        else:
            if not args.repository or not args.live:
                raise ReconciliationError("reconciliation requires --repository and --live")
            result, wave = reconcile(args.repository, args.live, manifest=args.manifest)
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
