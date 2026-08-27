"""Evidence-backed classification of failures recorded by ``VerifyRun``.

This is deliberately a derived, append-only view.  It never retries a check
and it never changes the verifier's exit status.  The receipt remains the
source of truth for what happened; this module adds stable identity,
cross-run comparison, and policy diagnostics.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from devtools.verify_runs import VERIFY_HISTORY_PATH

LEDGER_PATH = Path(".cache/verify/failure-ledger.jsonl")
LEDGER_SCHEMA_VERSION = 1
CLASSIFICATIONS = (
    "deterministic_regression",
    "same_head_variance",
    "timeout_resource",
    "environment_contamination",
    "infrastructure_failure",
    "expected_transition",
    "unknown",
)


def validate_disposition(record: Mapping[str, Any]) -> None:
    """Fail closed for quarantine/baseline metadata instead of hiding reds."""
    if record.get("disposition") not in {"open", "closed", "quarantined", "baseline"}:
        raise ValueError("unknown verification failure disposition")
    if record.get("disposition") in {"quarantined", "baseline"}:
        required = ("authority", "owner_bead", "scope", "expiry")
        missing = [key for key in required if not record.get(key)]
        if missing:
            raise ValueError(f"{record.get('disposition')} requires: {', '.join(missing)}")
        try:
            expiry = datetime.fromisoformat(str(record["expiry"]).replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("verification exception expiry must be an ISO timestamp") from exc
        if expiry <= datetime.now(UTC):
            raise ValueError("verification exception has expired")


def environment_fingerprint(*, root: Path | None = None, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Return identity useful for separating product failures from poisoned runs."""
    root = root or Path.cwd()
    environ = os.environ if env is None else env
    executable = Path(sys.executable).resolve()
    return {
        "checkout_root": str(root.absolute()),
        "python_executable": str(executable),
        "python_environment": str(Path(environ.get("VIRTUAL_ENV", executable.parent)).absolute()),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "testmon_environment": environ.get("POLYLOGUE_TESTMON_ENVIRONMENT"),
        "harness": environ.get("POLYLOGUE_VERIFY_HARNESS", "devtools"),
    }


def _stable_id(check_id: str) -> str:
    return "vfr-" + hashlib.sha256(check_id.encode("utf-8")).hexdigest()[:24]


def _same_head_variance(check_id: str, rows: Iterable[Mapping[str, Any]]) -> bool:
    outcomes = {str(row.get("outcome")) for row in rows if row.get("check_id") == check_id}
    return "failed" in outcomes and "passed" in outcomes


def _environment_contaminated(payload: Mapping[str, Any], step: Mapping[str, Any]) -> bool:
    diagnosis = str(payload.get("diagnosis") or step.get("diagnosis") or "")
    if "checkout" in diagnosis and "mismatch" in diagnosis:
        return True
    fingerprint = payload.get("environment_fingerprint")
    if isinstance(fingerprint, Mapping):
        checkout = str(fingerprint.get("checkout_root", ""))
        executable = str(fingerprint.get("python_executable", ""))
        if checkout and executable and checkout not in executable and ".venv" in executable:
            return True
    return any(
        token in diagnosis.lower() for token in ("mixed_checkout", "environment_contamination", "worktree_poison")
    )


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def classify_failure(payload: Mapping[str, Any], check_id: str, *, history: Iterable[Mapping[str, Any]] = ()) -> str:
    """Classify using evidence precedence; absence of evidence stays unknown."""
    steps = _as_list(payload.get("steps"))
    step = next(
        (
            item
            for item in steps
            if isinstance(item, Mapping) and (item.get("name") == check_id or item.get("step_id") == check_id)
        ),
        {},
    )
    diagnosis = str(payload.get("diagnosis") or step.get("diagnosis") or "").lower()
    if _environment_contaminated(payload, step):
        return "environment_contamination"
    if any(token in diagnosis for token in ("timeout", "resource", "oom", "killed")) or payload.get("exit_code") in (
        124,
        137,
    ):
        return "timeout_resource"
    if any(token in diagnosis for token in ("missing_executable", "subprocess", "runner_exception", "infrastructure")):
        return "infrastructure_failure"
    rows = (*tuple(history), {"check_id": check_id, "outcome": "failed", "git_head": payload.get("git_head")})
    observations: list[dict[str, Any]] = []
    for historic in rows:
        historic_steps = _as_list(historic.get("steps"))
        for historic_step in historic_steps:
            if not isinstance(historic_step, Mapping):
                continue
            name = str(historic_step.get("name") or historic_step.get("step_id") or "verification")
            statistics = _as_mapping(historic_step.get("statistics"))
            outcomes = _as_mapping(statistics.get("outcomes"))
            if name == check_id:
                observations.extend(
                    {"check_id": name, "outcome": "failed", "git_head": historic.get("git_head")}
                    for _ in range(int(outcomes.get("failed", 0) or 0))
                )
                observations.extend(
                    {"check_id": name, "outcome": "passed", "git_head": historic.get("git_head")}
                    for _ in range(int(outcomes.get("passed", 0) or 0))
                )
                if not outcomes:
                    observations.append(
                        {
                            "check_id": name,
                            "outcome": "failed" if historic_step.get("exit") else "passed",
                            "git_head": historic.get("git_head"),
                        }
                    )
    rows = (*rows, *observations)
    if _same_head_variance(check_id, rows):
        return "same_head_variance"
    heads = {row.get("git_head") for row in rows if row.get("check_id") == check_id and row.get("outcome") == "failed"}
    if len({head for head in heads if head}) >= 2:
        return "deterministic_regression"
    if diagnosis in {"expected_transition", "baseline", "known_red"}:
        return "expected_transition"
    return "unknown"


def _failed_checks(payload: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    result: list[tuple[str, Mapping[str, Any]]] = []
    steps = _as_list(payload.get("steps"))
    for step in steps:
        if not isinstance(step, Mapping) or step.get("exit") in (None, 0):
            continue
        name = str(step.get("name") or step.get("step_id") or "verification")
        statistics = _as_mapping(step.get("statistics"))
        outcomes = _as_mapping(statistics.get("outcomes"))
        failed_count = int(outcomes.get("failed", 0) or 0)
        if failed_count:
            for index in range(failed_count):
                result.append((f"{name}#failed-{index + 1}", step))
        else:
            result.append((name, step))
    return result


def ledger_records(payload: Mapping[str, Any], *, history: Iterable[Mapping[str, Any]] = ()) -> list[dict[str, Any]]:
    """Project one terminal receipt into complete machine-readable records."""
    if payload.get("status") == "running":
        return []
    history_rows = tuple(history)
    observed_at = str(payload.get("finished_at") or payload.get("started_at") or datetime.now(UTC).isoformat())
    fingerprint = payload.get("environment_fingerprint")
    if not isinstance(fingerprint, Mapping):
        fingerprint = environment_fingerprint()
    records: list[dict[str, Any]] = []
    for check_id, step in _failed_checks(payload):
        base_id = check_id.split("#", 1)[0]
        classification = classify_failure(payload, base_id, history=history_rows)
        records.append(
            {
                "schema_version": LEDGER_SCHEMA_VERSION,
                "failure_id": _stable_id(base_id),
                "check_id": base_id,
                "run_id": payload.get("run_id"),
                "git_head": payload.get("git_head"),
                "git_dirty": payload.get("git_dirty"),
                "environment_fingerprint": dict(fingerprint),
                "harness_fingerprint": {"argv": payload.get("argv"), "tier": payload.get("tier")},
                "dependency_fingerprint": payload.get("testmon_selection"),
                "first_seen": observed_at,
                "last_seen": observed_at,
                "outcome": "failed",
                "runtime_s": step.get("duration_s", payload.get("duration_s")),
                "resource_data": payload.get("workload_receipt"),
                "artifact_refs": [step.get("artifact_dir"), payload.get("artifact_dir")],
                "classification": classification,
                "classification_confidence": "high" if classification != "unknown" else "low",
                "owner_bead": None,
                "authority": None,
                "scope": {"tier": payload.get("tier"), "check": base_id},
                "disposition": "open",
                "expiry": None,
                "evidence": {
                    "diagnosis": step.get("diagnosis") or payload.get("diagnosis"),
                    "exit_code": payload.get("exit_code"),
                },
            }
        )
    return records


def append_failure_ledger(records: Iterable[Mapping[str, Any]], *, path: Path = LEDGER_PATH) -> None:
    """Append records atomically enough for the single-writer verifier route."""
    rows = tuple(records)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_failure_ledger(path: Path = LEDGER_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def read_verify_history(path: Path = VERIFY_HISTORY_PATH) -> list[dict[str, Any]]:
    """Read compact VerifyRun history, ignoring torn or foreign lines."""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def policy_diagnostics(records: Iterable[Mapping[str, Any]], *, now: datetime | None = None) -> dict[str, Any]:
    """Return annotations; open regressions remain policy reds."""
    now = now or datetime.now(UTC)
    rows = tuple(records)
    expired: list[str] = []
    unexplained: list[str] = []
    open_regressions: list[str] = []
    for row in rows:
        failure_id = str(row.get("failure_id"))
        if row.get("classification") == "unknown":
            unexplained.append(failure_id)
        if row.get("classification") == "deterministic_regression" and row.get("disposition") != "closed":
            open_regressions.append(failure_id)
        expiry = row.get("expiry")
        if expiry:
            try:
                if datetime.fromisoformat(str(expiry).replace("Z", "+00:00")) < now:
                    expired.append(failure_id)
            except ValueError:
                expired.append(failure_id)
    invalid_exceptions: list[str] = []
    for row in rows:
        if row.get("disposition") in {"quarantined", "baseline"}:
            try:
                validate_disposition(row)
            except ValueError:
                invalid_exceptions.append(str(row.get("failure_id")))
    return {
        "ledger_matches": len(rows),
        "unexplained_reds": unexplained,
        "open_regressions": open_regressions,
        "expired_exceptions": expired,
        "invalid_exceptions": invalid_exceptions,
        "policy_red": bool(open_regressions or expired or invalid_exceptions),
    }


__all__ = [
    "CLASSIFICATIONS",
    "LEDGER_PATH",
    "append_failure_ledger",
    "classify_failure",
    "environment_fingerprint",
    "ledger_records",
    "policy_diagnostics",
    "read_failure_ledger",
    "read_verify_history",
    "validate_disposition",
]
