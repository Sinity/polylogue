#!/usr/bin/env python3
"""Verify the frozen Incident 14:32 oracle against independent source materials."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

_ABSOLUTE_PATH_RE = re.compile(r"(?:^|[\s\"'])/(?:home|Users|mnt|var|tmp)/[^\s\"']+")
_SECRET_RE = re.compile(r"(?i)(?:api[_-]?key|token|password|secret)\s*[:=]\s*[A-Za-z0-9_\-]{12,}")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            rows.append(value)
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def derive(root: Path) -> dict[str, Any]:
    transcript = load_jsonl(root / "materials" / "transcript.jsonl")
    anti_grep = load_jsonl(root / "materials" / "anti-grep.jsonl")
    lineage = load_jsonl(root / "materials" / "lineage.jsonl")
    terminal = load_jsonl(root / "materials" / "terminal.jsonl")
    git_rows = load_jsonl(root / "materials" / "git.jsonl")
    bead_rows = load_jsonl(root / "materials" / "beads.jsonl")
    source_health = load_jsonl(root / "materials" / "source-health.jsonl")
    assertions = load_jsonl(root / "materials" / "assertions.jsonl")
    context = load_json(root / "materials" / "context-delivery.json")
    parser_v1 = load_json(root / "parser" / "v1.json")
    parser_v2 = load_json(root / "parser" / "v2.json")

    by_id = {str(row["record_id"]): row for row in transcript}
    claim = by_id["m-003"]
    failed = by_id["r-001"]
    recovery = by_id["r-003"]

    text_hits = sum(str(row.get("text", "")).casefold().count("error") for row in anti_grep)
    failed_actions = sum(
        1
        for row in anti_grep
        if row.get("type") == "tool_result" and (row.get("is_error") is True or int(row.get("exit_code", 0)) != 0)
    )

    physical_tokens = sum(int(row["input_tokens"]) for row in lineage)
    logical_tokens = 0
    copied_tokens = 0
    fresh_subagent_unique = False
    for row in lineage:
        kind = row["kind"]
        if kind == "physical":
            logical_tokens += int(row["input_tokens"])
        elif kind == "prefix_sharing_fork":
            logical_tokens += int(row["unique_input_tokens"])
            copied_tokens += int(row["input_tokens"]) - int(row["unique_input_tokens"])
        elif kind == "fresh_subagent":
            logical_tokens += int(row["unique_input_tokens"])
            fresh_subagent_unique = int(row["unique_input_tokens"]) == int(row["input_tokens"])

    browser = next(row for row in source_health if row["source_id"] == "browser")
    rejected = next(row for row in assertions if row["assertion_id"] == "assert-001")
    accepted = next(row for row in assertions if row["assertion_id"] == "assert-002")
    final_bead = bead_rows[-1]
    commit = next(row for row in git_rows if row["kind"] == "commit")

    terminal_fail = next(row for row in terminal if int(row["exit_code"]) != 0)
    terminal_pass = next(row for row in terminal if row["command"].startswith("pytest") and int(row["exit_code"]) == 0)
    transcript_terminal_agreement = int(failed["exit_code"]) == int(terminal_fail["exit_code"]) and int(
        recovery["exit_code"]
    ) == int(terminal_pass["exit_code"])

    return {
        "claim_record_id": claim["record_id"],
        "claim_text": claim["text"],
        "claim_time": claim["ts"],
        "failed_result_record_id": failed["record_id"],
        "failed_exit_code": int(failed["exit_code"]),
        "failed_result_precedes_claim": parse_time(failed["ts"]) < parse_time(claim["ts"]),
        "recovery_result_record_id": recovery["record_id"],
        "recovery_exit_code": int(recovery["exit_code"]),
        "recovery_follows_claim": parse_time(recovery["ts"]) > parse_time(claim["ts"]),
        "claim_verdict": (
            "contradicted_at_claim_time_then_repaired"
            if int(failed["exit_code"]) != 0
            and parse_time(failed["ts"]) < parse_time(claim["ts"]) < parse_time(recovery["ts"])
            and int(recovery["exit_code"]) == 0
            else "unsupported"
        ),
        "anti_grep_error_text_hits": text_hits,
        "anti_grep_failed_actions": failed_actions,
        "physical_input_tokens": physical_tokens,
        "logical_unique_input_tokens": logical_tokens,
        "copied_prefix_tokens": copied_tokens,
        "fresh_subagent_tokens_count_as_unique": fresh_subagent_unique,
        "browser_interval_claimability": (
            "unknown_due_to_unobserved_source" if browser["state"] == "unobserved" else "claimable"
        ),
        "parser_current_semantics": parser_v2["semantics_version"]
        if parser_v2["promoted"]
        else parser_v1["semantics_version"],
        "stable_object_id_survives_replay": (
            parser_v2["stable_object_id"]
            if parser_v1["stable_object_id"] == parser_v2["stable_object_id"]
            and parser_v2["supersedes_interpretation_event_id"] == parser_v1["interpretation_event_id"]
            else "mismatch"
        ),
        "rejected_assertion_not_context_eligible": (
            rejected["assertion_id"]
            if rejected["state"] == "rejected" and rejected["context_policy"]["inject"] is False
            else "mismatch"
        ),
        "accepted_lesson_context_eligible": (
            accepted["assertion_id"]
            if accepted["state"] == "accepted" and accepted["context_policy"]["inject"] is True
            else "mismatch"
        ),
        "context_delivery_id": context["delivery_id"],
        "bead_final_status": final_bead["status"],
        "verification_commit": commit["commit"],
        "transcript_terminal_agreement": transcript_terminal_agreement,
    }


def verify(root: Path) -> dict[str, Any]:
    manifest = yaml.safe_load((root / "manifest.yaml").read_text(encoding="utf-8"))
    oracle = load_json(root / "oracle.json")
    problems: list[str] = []

    if manifest.get("schema") != "legibility.incident-manifest/v1":
        problems.append("unsupported manifest schema")
    if oracle.get("independent_of_product_output") is not True:
        problems.append("oracle is not declared independent of product output")

    hash_results: list[dict[str, Any]] = []
    for item in manifest.get("files", []):
        rel = item["path"]
        path = root / rel
        actual = sha256(path) if path.exists() else None
        ok = actual == item["sha256"]
        hash_results.append({"path": rel, "ok": ok, "expected": item["sha256"], "actual": actual})
        if not ok:
            problems.append(f"material hash mismatch: {rel}")

    try:
        derived = derive(root)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        derived = {}
        problems.append(f"could not derive incident facts: {exc}")

    comparisons: list[dict[str, Any]] = []
    expected_facts = oracle.get("facts", {})
    for key, expected in expected_facts.items():
        actual = derived.get(key)
        ok = actual == expected
        comparisons.append({"fact": key, "ok": ok, "expected": expected, "actual": actual})
        if not ok:
            problems.append(f"oracle mismatch for {key}: expected {expected!r}, got {actual!r}")

    if derived.get("transcript_terminal_agreement") is not True:
        problems.append("transcript and independent terminal outcomes disagree")

    scrub_findings: list[dict[str, str]] = []
    for item in manifest.get("files", []):
        path = root / item["path"]
        if path.suffix.lower() not in {".json", ".jsonl", ".yaml", ".yml", ".md"}:
            continue
        text = path.read_text(encoding="utf-8")
        for regex, kind in ((_ABSOLUTE_PATH_RE, "absolute_path"), (_SECRET_RE, "credential_like")):
            match = regex.search(text)
            if match:
                scrub_findings.append({"path": item["path"], "kind": kind, "match": match.group(0)})
                problems.append(f"public-safety scrub failed for {item['path']}: {kind}")

    return {
        "schema": "legibility.incident-verification/v1",
        "ok": not problems,
        "root": ".",
        "hashes": hash_results,
        "derived": derived,
        "oracle_comparisons": comparisons,
        "public_safety_findings": scrub_findings,
        "non_claims": oracle.get("non_claims", []),
        "problems": problems,
    }


def render_markdown(report: dict[str, Any]) -> str:
    passed = sum(1 for row in report["oracle_comparisons"] if row["ok"])
    total = len(report["oracle_comparisons"])
    lines = [
        "# Incident 14:32 verification receipt",
        "",
        f"Status: **{'passed' if report['ok'] else 'failed'}**",
        "",
        f"- Material hashes: {sum(1 for row in report['hashes'] if row['ok'])}/{len(report['hashes'])}",
        f"- Oracle facts: {passed}/{total}",
        f"- Public-safety findings: {len(report['public_safety_findings'])}",
        "",
        "## Core verdict",
        "",
        f"`{report['derived'].get('claim_verdict', 'unavailable')}`",
        "",
        "The first focused test exited 1 before the assistant claimed success. A later edit and rerun exited 0. The later recovery does not retroactively make the earlier claim true.",
        "",
        "## Independent controls",
        "",
        f"- Anti-grep text hits: {report['derived'].get('anti_grep_error_text_hits')}.",
        f"- Anti-grep failed actions: {report['derived'].get('anti_grep_failed_actions')}.",
        f"- Physical input tokens: {report['derived'].get('physical_input_tokens')}.",
        f"- Logical unique input tokens: {report['derived'].get('logical_unique_input_tokens')}.",
        f"- Copied-prefix tokens: {report['derived'].get('copied_prefix_tokens')}.",
        f"- Browser interval: {report['derived'].get('browser_interval_claimability')}.",
        f"- Promoted parser semantics: {report['derived'].get('parser_current_semantics')}.",
        "",
        "## What this corpus does not prove",
        "",
        *[f"- {item}" for item in report["non_claims"]],
        "",
        "## Problems",
        "",
        *([f"- {item}" for item in report["problems"]] or ["- none"]),
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    out_dir = (args.out_dir or root / "output").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report = verify(root)
    (out_dir / "verification.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "verification.md").write_text(render_markdown(report), encoding="utf-8")
    report_path = out_dir / "verification.json"
    try:
        public_report_path = report_path.relative_to(root).as_posix()
    except ValueError:
        public_report_path = str(report_path)
    print(json.dumps({"ok": report["ok"], "report": public_report_path}, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
