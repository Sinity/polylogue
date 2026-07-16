#!/usr/bin/env python3
"""Validate, render, run, and inspect shared-worktree implementation waves."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
from collections import defaultdict
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[4]
SCHEMA = HERE / "completion.schema.json"
DEFAULT_RUNS_ROOT = ROOT / ".local/testsuite-diet/runs"
MAX_CONCURRENCY = 4
JOB_FIELDS = {
    "id",
    "wave",
    "model",
    "effort",
    "mission",
    "required_reads",
    "write_files",
    "avoid_files",
    "acceptance",
    "focused_tests",
}
ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
MODEL_RE = re.compile(r"^gpt-5\.6-(?:terra|luna)$")
GLOB_CHARS = frozenset("*?[]{}")

PROMPT_PREFIX = """POLYLOGUE SHARED-WORKTREE TESTSUITE-DIET WORKER v1

You are one narrowly scoped implementation worker in a coordinator-owned editing wave.
Work directly in the named shared checkout. Other workers may edit disjoint files concurrently.
The coordinator owns architecture, integration, broad verification, git, Beads, commits, and publication.
"""

WORKER_RULES = """Worker rules:
1. Work directly in the shared checkout.
2. Read only named context plus immediate dependencies needed to understand it.
3. Edit only named write files; ignore concurrent changes elsewhere. If another file is needed, block.
4. Do not use git, Beads, broad formatters, generated-surface sweeps, or broad tests.
5. Implement the behavioral result. Do not delete tests/helpers unless the mission explicitly says it is a certified deletion job and names the certification receipt.
6. Run only named devtools test selectors and cheap local checks; never bypass the existing test lock.
7. Stop with a structured blocker if a design decision is unresolved.
8. Return the structured receipt: changed files, behavioral result, production dependencies, exact checks/outputs, actual and proposed deletions, sensitivity state, residual risks, recommended integration checks, and blocker state.

Anti-vacuity: name the production dependency exercised and the implementation mutation/removal that makes the strengthened test fail. Toy replicas, self-authored completeness registries, implementation-spelling assertions, and shadow production algorithms are not acceptable proof.
"""


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_hash(path: Path) -> str | None:
    return _sha256_bytes(path.read_bytes()) if path.is_file() else None


def _tree_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    if path.is_file():
        return _file_hash(path)
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(child.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _git_dirty_state(root: Path) -> dict[str, str | None]:
    completed = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise ValueError(f"execution root is not a readable git checkout: {root}")
    records = completed.stdout.split(b"\0")
    state: dict[str, str | None] = {}
    index = 0
    while index < len(records):
        raw = records[index]
        index += 1
        if not raw:
            continue
        entry = raw.decode("utf-8", errors="surrogateescape")
        status = entry[:2]
        path = entry[3:]
        state[path] = f"{status}:{_file_hash(root / path)}"
        if ("R" in status or "C" in status) and index < len(records) and records[index]:
            source = records[index].decode("utf-8", errors="surrogateescape")
            state[source] = f"{status}:source"
            index += 1
    return state


def _path_list(
    job: Mapping[str, Any], field: str, *, root: Path, require_existing: bool
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    value = job.get(field)
    if not isinstance(value, list):
        return [], [f"{job.get('id', '<unknown>')}.{field} must be a list"]
    result: list[str] = []
    for index, raw in enumerate(value):
        label = f"{job.get('id', '<unknown>')}.{field}[{index}]"
        if not isinstance(raw, str) or not raw:
            errors.append(f"{label} must be a non-empty string")
            continue
        path = Path(raw)
        if path.is_absolute() or ".." in path.parts or any(char in raw for char in GLOB_CHARS):
            errors.append(f"{label} must be an exact repository-relative file: {raw!r}")
            continue
        resolved = root / path
        if resolved.exists() and not resolved.is_file():
            errors.append(f"{label} names a directory, not an exact file: {raw}")
            continue
        if require_existing and not resolved.is_file():
            errors.append(f"{label} does not exist: {raw}")
        result.append(path.as_posix())
    if len(result) != len(set(result)):
        errors.append(f"{job.get('id', '<unknown>')}.{field} contains duplicates")
    return result, errors


def validate_manifest(value: object, *, root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    if not isinstance(value, list) or not value:
        return [], ["manifest must be a non-empty JSON array of jobs"]
    jobs: list[dict[str, Any]] = []
    errors: list[str] = []
    ids: set[str] = set()
    writes_by_wave: dict[int, dict[str, str]] = defaultdict(dict)
    writes_across_waves: dict[str, list[int]] = defaultdict(list)
    required_across_waves: list[tuple[str, int, str]] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, dict):
            errors.append(f"job[{index}] must be an object")
            continue
        job_id = raw.get("id")
        label = str(job_id or f"job[{index}]")
        extra = set(raw) - JOB_FIELDS
        missing = JOB_FIELDS - set(raw)
        if extra:
            errors.append(f"{label} has non-operational fields: {', '.join(sorted(extra))}")
        if missing:
            errors.append(f"{label} is missing fields: {', '.join(sorted(missing))}")
        if not isinstance(job_id, str) or not ID_RE.fullmatch(job_id):
            errors.append(f"job[{index}].id must match {ID_RE.pattern}")
        elif job_id in ids:
            errors.append(f"duplicate job id: {job_id}")
        else:
            ids.add(job_id)
        wave = raw.get("wave")
        if not isinstance(wave, int) or isinstance(wave, bool) or wave < 1:
            errors.append(f"{label}.wave must be a positive integer")
            wave = -1
        model = raw.get("model")
        if not isinstance(model, str) or not MODEL_RE.fullmatch(model):
            errors.append(f"{label}.model must explicitly select worker model gpt-5.6-terra or -luna")
        if raw.get("effort") != "high":
            errors.append(f"{label}.effort must be 'high'")
        for field in ("mission", "acceptance"):
            if not isinstance(raw.get(field), str) or not str(raw[field]).strip():
                errors.append(f"{label}.{field} must be non-empty text")
        required_reads, path_errors = _path_list(raw, "required_reads", root=root, require_existing=False)
        errors.extend(path_errors)
        write_files, path_errors = _path_list(raw, "write_files", root=root, require_existing=False)
        errors.extend(path_errors)
        avoid_files, path_errors = _path_list(raw, "avoid_files", root=root, require_existing=False)
        errors.extend(path_errors)
        overlap = set(write_files) & set(avoid_files)
        if overlap:
            errors.append(f"{label} writes avoided files: {', '.join(sorted(overlap))}")
        if wave > 0:
            required_across_waves.extend((label, wave, path) for path in required_reads)
            for path in write_files:
                owner = writes_by_wave[wave].get(path)
                if owner is not None:
                    errors.append(f"wave {wave} write collision: {path} owned by {owner} and {label}")
                writes_by_wave[wave][path] = label
                writes_across_waves[path].append(wave)
        focused = raw.get("focused_tests")
        if not isinstance(focused, list) or not focused:
            errors.append(f"{label}.focused_tests must be a non-empty list")
        else:
            for command in focused:
                if not isinstance(command, str) or not command.startswith("devtools test "):
                    errors.append(f"{label}.focused_tests permits only named 'devtools test ...' commands")
                elif any(token in command for token in ("\n", ";", "&&", "||", "`", "$(")):
                    errors.append(f"{label}.focused_tests contains shell control syntax: {command!r}")
                elif command.strip() in {"devtools test tests", "devtools test tests/unit"}:
                    errors.append(f"{label}.focused_tests contains a broad selector: {command}")
        normalized = dict(raw)
        normalized["required_reads"] = required_reads
        normalized["write_files"] = write_files
        normalized["avoid_files"] = avoid_files
        jobs.append(normalized)
    for label, wave, path in required_across_waves:
        if (root / path).is_file():
            continue
        producer_waves = writes_across_waves.get(path, [])
        if not producer_waves or min(producer_waves) >= wave:
            errors.append(f"{label}.required_reads does not exist and has no earlier-wave producer: {path}")
    for label, wave, path in required_across_waves:
        owner = writes_by_wave.get(wave, {}).get(path)
        if owner is not None and owner != label:
            errors.append(f"wave {wave} read/write collision: {path} is read by {label} and written by {owner}")
    return jobs, errors


def render_prompt(job: Mapping[str, Any]) -> str:
    contract = json.dumps(job, indent=2, sort_keys=True)
    reads = "\n".join(f"- {path}" for path in job["required_reads"]) or "- none"
    writes = "\n".join(f"- {path}" for path in job["write_files"]) or "- none (read-only calibration)"
    avoids = "\n".join(f"- {path}" for path in job["avoid_files"]) or "- none"
    checks = "\n".join(f"- {command}" for command in job["focused_tests"])
    return (
        PROMPT_PREFIX
        + "\n"
        + WORKER_RULES
        + f"\nMission:\n{job['mission']}\n\n"
        + f"Acceptance:\n{job['acceptance']}\n\n"
        + f"Required reads:\n{reads}\n\nWrite files:\n{writes}\n\nAvoid files:\n{avoids}\n\n"
        + f"Permitted focused tests:\n{checks}\n\n"
        + "BEGIN JOB CONTRACT\n"
        + contract
        + "\nEND JOB CONTRACT\n"
    )


def _run_id() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{os.getpid()}"


def prepare_run(manifest_path: Path, *, root: Path, runs_root: Path, run_id: str) -> tuple[Path, list[dict[str, Any]]]:
    jobs, errors = validate_manifest(_json(manifest_path), root=root)
    if errors:
        raise ValueError("\n".join(errors))
    run_dir = runs_root / run_id
    if run_dir.exists():
        raise FileExistsError(f"run already exists: {run_dir}")
    (run_dir / "prompts").mkdir(parents=True)
    (run_dir / "logs").mkdir()
    (run_dir / "final").mkdir()
    (run_dir / "attested").mkdir()
    shutil.copy2(SCHEMA, run_dir / SCHEMA.name)
    _write_json(run_dir / "manifest.json", jobs)
    for job in jobs:
        prompt = render_prompt(job)
        (run_dir / "prompts" / f"{job['id']}.prompt").write_text(prompt, encoding="utf-8")
    record = {
        "schema_version": 1,
        "run_id": run_id,
        "root": str(root),
        "manifest_source": str(manifest_path),
        "state": "rendered",
        "max_concurrency": None,
        "jobs": {
            job["id"]: {"state": "rendered", "wave": job["wave"], "model": job["model"], "effort": job["effort"]}
            for job in jobs
        },
    }
    _write_json(run_dir / "run.json", record)
    return run_dir, jobs


def _launcher_path(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    sinnix_root = Path(os.environ.get("SINNIX_ROOT", "/realm/project/sinnix"))
    path = sinnix_root / "dots/_ai/skills/agent-orchestration/scripts/run_agent_prompt.sh"
    if not path.is_file():
        raise FileNotFoundError(f"attested launcher not found: {path}")
    return path


def _receipt_errors(receipt: object, *, job: Mapping[str, Any], actual_changed: set[str]) -> list[str]:
    if not isinstance(receipt, dict):
        return ["completion receipt is not a JSON object"]
    required = {
        "changed_files",
        "behavioral_result",
        "production_dependencies",
        "checks",
        "deleted_tests_helpers",
        "proposed_deletions",
        "sensitivity",
        "residual_risks",
        "recommended_integration_checks",
        "blocker",
    }
    errors: list[str] = []
    if set(receipt) != required:
        errors.append(f"completion keys differ from schema: {sorted(set(receipt) ^ required)}")
        return errors
    changed = receipt["changed_files"]
    if not isinstance(changed, list) or not all(isinstance(item, str) for item in changed):
        errors.append("changed_files must be a string list")
        changed_set: set[str] = set()
    else:
        changed_set = set(changed)
    assigned = set(job["write_files"])
    outside = changed_set - assigned
    if outside:
        errors.append(f"receipt reports unassigned files: {sorted(outside)}")
    blocker = receipt["blocker"]
    blocked = isinstance(blocker, dict) and blocker.get("blocked") is True
    if not isinstance(blocker, dict) or set(blocker) != {"blocked", "reason", "decision_needed"}:
        errors.append("blocker must contain blocked, reason, and decision_needed")
    elif (
        not isinstance(blocker.get("blocked"), bool)
        or not isinstance(blocker.get("reason"), str)
        or not isinstance(blocker.get("decision_needed"), str)
    ):
        errors.append("blocker fields have invalid types")
    if blocked:
        if changed_set:
            errors.append(f"blocked receipt claims changed files: {sorted(changed_set)}")
        if actual_changed:
            errors.append(f"blocked job changed assigned files: {sorted(actual_changed)}")
    else:
        if changed_set != actual_changed:
            errors.append(
                f"receipt/actual assigned-file delta differs: receipt={sorted(changed_set)} actual={sorted(actual_changed)}"
            )
        if assigned and not actual_changed:
            errors.append("prose-only completion: implementation job changed no assigned files")
        checks = receipt["checks"]
        commands = (
            {item.get("command") for item in checks if isinstance(item, dict)} if isinstance(checks, list) else set()
        )
        missing_checks = set(job["focused_tests"]) - commands
        if missing_checks:
            errors.append(f"named focused checks missing from receipt: {sorted(missing_checks)}")
        if isinstance(checks, list):
            for index, item in enumerate(checks):
                if not isinstance(item, dict) or set(item) != {"command", "exit_code", "output"}:
                    errors.append(f"checks[{index}] must contain command, exit_code, and output")
                    continue
                if (
                    not isinstance(item.get("command"), str)
                    or not isinstance(item.get("exit_code"), int)
                    or isinstance(item.get("exit_code"), bool)
                    or not isinstance(item.get("output"), str)
                ):
                    errors.append(f"checks[{index}] fields have invalid types")
            failed = [item.get("command") for item in checks if isinstance(item, dict) and item.get("exit_code") != 0]
            if failed:
                errors.append(f"receipt contains failed checks: {failed}")
    for field in ("behavioral_result",):
        if not isinstance(receipt[field], str):
            errors.append(f"{field} must be text")
    for field in (
        "production_dependencies",
        "checks",
        "deleted_tests_helpers",
        "proposed_deletions",
        "residual_risks",
        "recommended_integration_checks",
    ):
        if not isinstance(receipt[field], list):
            errors.append(f"{field} must be a list")
    for field in (
        "changed_files",
        "production_dependencies",
        "deleted_tests_helpers",
        "proposed_deletions",
        "residual_risks",
        "recommended_integration_checks",
    ):
        value = receipt[field]
        if isinstance(value, list) and not all(isinstance(item, str) for item in value):
            errors.append(f"{field} must contain only strings")
    sensitivity = receipt["sensitivity"]
    sensitivity_fields = {"executed", "witness", "mutation", "result", "artifact"}
    if not isinstance(sensitivity, dict) or set(sensitivity) != sensitivity_fields:
        errors.append("sensitivity must contain executed, witness, mutation, result, and artifact")
    elif not isinstance(sensitivity.get("executed"), bool) or not all(
        isinstance(sensitivity.get(field), str) for field in sensitivity_fields - {"executed"}
    ):
        errors.append("sensitivity fields have invalid types")
    return errors


def _attestation_errors(path: Path, *, job: Mapping[str, Any], job_id: str, prompt_path: Path, root: Path) -> list[str]:
    if not path.is_file():
        return [f"attested launcher manifest missing: {path}"]
    try:
        value = _json(path)
    except (OSError, json.JSONDecodeError) as error:
        return [f"invalid launcher manifest: {error}"]
    expected = {
        "job_id": job_id,
        "lifecycle": "completed",
        "model": job["model"],
        "effort": job["effort"],
        "worktree": str(root.resolve()),
    }
    errors = [
        f"attestation {key}: expected {wanted!r}, got {value.get(key)!r}"
        for key, wanted in expected.items()
        if value.get(key) != wanted
    ]
    prompt = value.get("prompt")
    prompt_hash = _sha256_bytes(prompt_path.read_bytes())
    if not isinstance(prompt, Mapping) or prompt.get("sha256") != prompt_hash:
        errors.append("attestation prompt hash does not match rendered prompt")
    return errors


def _execute_job(
    job: Mapping[str, Any],
    *,
    root: Path,
    run_dir: Path,
    run_id: str,
    launcher: Path,
) -> dict[str, Any]:
    job_name = str(job["id"])
    job_id = f"testsuite-diet-{run_id}-{job_name}"
    prompt = run_dir / "prompts" / f"{job_name}.prompt"
    log = run_dir / "logs" / f"{job_name}.log"
    final = run_dir / "final" / f"{job_name}.json"
    attested = run_dir / "attested" / f"{job_id}.json"
    before = {path: _file_hash(root / path) for path in job["write_files"]}
    command = [
        str(launcher),
        "--agent",
        "codex",
        "--workdir",
        str(root),
        "--prompt-file",
        str(prompt),
        "--log-file",
        str(log),
        "--last-file",
        str(final),
        "--schema-file",
        str(run_dir / SCHEMA.name),
        "--model",
        str(job["model"]),
        "--reasoning-effort",
        str(job["effort"]),
        "--job-id",
        job_id,
        "--job-state-dir",
        str(run_dir / "attested"),
        "--job-role",
        "shared-worktree-testsuite-diet-worker",
        "--work-item",
        job_name,
    ]
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    after = {path: _file_hash(root / path) for path in job["write_files"]}
    actual_changed = {path for path in before if before[path] != after[path]}
    delta = {"before": before, "after": after, "changed": sorted(actual_changed)}
    _write_json(run_dir / "final" / f"{job_name}.file-delta.json", delta)
    errors: list[str] = []
    receipt: object = None
    if completed.returncode != 0:
        errors.append(f"launcher exited {completed.returncode}")
    if final.is_file():
        try:
            receipt = _json(final)
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"invalid completion receipt: {error}")
    else:
        errors.append("completion receipt missing")
    if receipt is not None:
        errors.extend(_receipt_errors(receipt, job=job, actual_changed=actual_changed))
    errors.extend(_attestation_errors(attested, job=job, job_id=job_id, prompt_path=prompt, root=root))
    blocked = (
        isinstance(receipt, Mapping)
        and isinstance(receipt.get("blocker"), Mapping)
        and receipt["blocker"].get("blocked") is True
    )
    state = "invalid" if errors else ("blocked" if blocked else "completed")
    return {
        "state": state,
        "wave": job["wave"],
        "model": job["model"],
        "effort": job["effort"],
        "job_id": job_id,
        "launcher_exit": completed.returncode,
        "errors": errors,
        "artifacts": {
            "prompt": str(prompt),
            "log": str(log),
            "final": str(final),
            "attested": str(attested),
            "file_delta": str(run_dir / "final" / f"{job_name}.file-delta.json"),
        },
    }


def execute_run(
    run_dir: Path,
    jobs: list[dict[str, Any]],
    *,
    root: Path,
    launcher: Path,
    max_concurrency: int,
    dry_run: bool,
) -> int:
    if max_concurrency < 1 or max_concurrency > MAX_CONCURRENCY:
        raise ValueError(f"max concurrency must be 1..{MAX_CONCURRENCY}")
    record_path = run_dir / "run.json"
    record = _json(record_path)
    record["state"] = "dry-run" if dry_run else "running"
    record["max_concurrency"] = max_concurrency
    _write_json(record_path, record)
    if dry_run:
        for job in jobs:
            record["jobs"][job["id"]]["state"] = "dry-run"
        _write_json(record_path, record)
        return 0
    initial_dirty = _git_dirty_state(root)
    if initial_dirty:
        record["state"] = "invalid"
        record["errors"] = [f"execution checkout must be clean before a shared-worktree run: {sorted(initial_dirty)}"]
        _write_json(record_path, record)
        return 1
    control_root = root / ".agent/scratch/testsuite_diet"
    control_hash = _tree_hash(control_root)

    lock = threading.Lock()
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for job in jobs:
        grouped[int(job["wave"])].append(job)

    def skip_later_waves(failed_wave: int) -> None:
        for later_wave in sorted(item for item in grouped if item > failed_wave):
            for later_job in grouped[later_wave]:
                record["jobs"][later_job["id"]] = {
                    "state": "skipped-upstream-wave",
                    "wave": later_wave,
                    "model": later_job["model"],
                    "effort": later_job["effort"],
                    "errors": [f"wave {failed_wave} did not complete cleanly"],
                }

    for wave in sorted(grouped):
        before_wave = _git_dirty_state(root)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrency) as pool:
            futures = {
                pool.submit(
                    _execute_job,
                    job,
                    root=root,
                    run_dir=run_dir,
                    run_id=str(record["run_id"]),
                    launcher=launcher,
                ): job
                for job in grouped[wave]
            }
            for future in concurrent.futures.as_completed(futures):
                job = futures[future]
                try:
                    result = future.result()
                except Exception as error:  # preserve remaining worker receipts
                    result = {
                        "state": "invalid",
                        "wave": wave,
                        "model": job["model"],
                        "effort": job["effort"],
                        "errors": [repr(error)],
                    }
                with lock:
                    record["jobs"][job["id"]] = result
                    _write_json(record_path, record)
        after_wave = _git_dirty_state(root)
        if _tree_hash(control_root) != control_hash:
            record.setdefault("errors", []).append(
                f"wave {wave} modified the ignored Testsuite Diet control tree: {control_root}"
            )
            record["state"] = "invalid"
            skip_later_waves(wave)
            _write_json(record_path, record)
            break
        changed_this_wave = {
            path for path in set(before_wave) | set(after_wave) if before_wave.get(path) != after_wave.get(path)
        }
        allowed_this_wave = {path for job in grouped[wave] for path in job["write_files"]}
        outside = changed_this_wave - allowed_this_wave
        if outside:
            record.setdefault("errors", []).append(
                f"wave {wave} changed paths outside its assignment union: {sorted(outside)}"
            )
            record["state"] = "invalid"
            skip_later_waves(wave)
            _write_json(record_path, record)
            break
        wave_states = {record["jobs"][job["id"]]["state"] for job in grouped[wave]}
        if wave_states - {"completed"}:
            skip_later_waves(wave)
            _write_json(record_path, record)
            break
    states = {item["state"] for item in record["jobs"].values()}
    if record.get("state") == "invalid" or "invalid" in states:
        record["state"] = "invalid"
    elif "blocked" in states or "skipped-upstream-wave" in states:
        record["state"] = "completed-with-blockers"
    else:
        record["state"] = "completed"
    _write_json(record_path, record)
    return 1 if record["state"] == "invalid" else 0


def _print_status(run_dir: Path, *, as_json: bool) -> int:
    path = run_dir / "run.json"
    if not path.is_file():
        raise FileNotFoundError(f"run record not found: {path}")
    record = _json(path)
    if as_json:
        print(json.dumps(record, indent=2, sort_keys=True))
    else:
        print(f"run {record['run_id']}: {record['state']}")
        for job_id, item in sorted(record["jobs"].items(), key=lambda pair: (pair[1]["wave"], pair[0])):
            print(f"wave={item['wave']}\t{job_id}\t{item['model']}\t{item['state']}")
            for error in item.get("errors", []):
                print(f"  error: {error}")
    return 1 if record["state"] == "invalid" else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("manifest", type=Path)
    render = subparsers.add_parser("render")
    render.add_argument("manifest", type=Path)
    render.add_argument("--run-id", default=None)
    run = subparsers.add_parser("run")
    run.add_argument("manifest", type=Path)
    run.add_argument("--run-id", default=None)
    run.add_argument("--launcher", type=Path)
    run.add_argument("--max-concurrency", type=int, default=MAX_CONCURRENCY)
    run.add_argument("--dry-run", action="store_true")
    status = subparsers.add_parser("status")
    status.add_argument("run")
    status.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    runs_root = args.runs_root.resolve()

    if args.command == "validate":
        _, errors = validate_manifest(_json(args.manifest), root=root)
        if errors:
            for error in errors:
                print(f"ERROR: {error}")
            return 1
        print(f"valid: {args.manifest}")
        return 0
    if args.command in {"render", "run"}:
        run_id = args.run_id or _run_id()
        try:
            run_dir, jobs = prepare_run(args.manifest.resolve(), root=root, runs_root=runs_root, run_id=run_id)
        except (ValueError, FileExistsError) as error:
            print(f"ERROR: {error}")
            return 1
        print(run_dir)
        if args.command == "render":
            return 0
        return execute_run(
            run_dir,
            jobs,
            root=root,
            launcher=_launcher_path(args.launcher),
            max_concurrency=args.max_concurrency,
            dry_run=args.dry_run,
        )
    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = runs_root / run_dir
    return _print_status(run_dir, as_json=args.json)


if __name__ == "__main__":
    raise SystemExit(main())
