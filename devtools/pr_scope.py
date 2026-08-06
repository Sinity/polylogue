"""Structured PR scope carrier used by CI and the merge boundary.

The carrier is an embedded JSON comment, not a convention inferred from PR
prose. It binds the declared Bead scope to a PR head, a canonical digest of the
assigned Bead records, whole-Bead dispositions, and concrete evidence refs.
Bead acceptance text is deliberately opaque here: it is hashed as part of the
record snapshot, never interpreted by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib import request

_CARRIER_PREFIX = "polylogue-pr-scope:v1"
_CARRIER_START = f"<!-- {_CARRIER_PREFIX}"
_CARRIER_END = "-->"
_VERSION = 1
_BEADS_PATH = Path(".beads/issues.jsonl")
_DISPOSITIONS = frozenset({"satisfied", "partial", "deferred", "superseded"})
_RESIDUAL_DISPOSITIONS = frozenset({"partial", "deferred", "superseded"})
_EVIDENCE_KINDS = frozenset({"command", "commit", "diff", "receipt", "review", "test"})


@dataclass(frozen=True, slots=True)
class ScopeVerdict:
    ok: bool
    reasons: list[str] = field(default_factory=list)
    scope_digest: str | None = None
    beads_digest: str | None = None
    assigned_beads: list[str] = field(default_factory=list)


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def load_bead_records(path: Path = _BEADS_PATH) -> dict[str, dict[str, Any]]:
    """Load the repository's committed Bead records without invoking ``bd``.

    The carrier needs the exact Bead snapshot that CI checked out. Reading the
    JSONL avoids mutating shared Dolt state from a worktree and keeps this
    validation independent of Beads' CLI synchronization hooks.
    """
    records: dict[str, dict[str, Any]] = {}
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, dict) or raw.get("_type") != "issue" or not isinstance(raw.get("id"), str):
            continue
        bead_id = raw["id"]
        if bead_id in records:
            raise ValueError(f"duplicate Bead id {bead_id!r} on line {line_no}")
        records[bead_id] = raw
    return records


def canonical_beads_digest(records: dict[str, dict[str, Any]], bead_ids: list[str]) -> str:
    """Digest whole canonical records for the declared IDs, sorted by Bead ID."""
    missing = [bead_id for bead_id in bead_ids if bead_id not in records]
    if missing:
        raise ValueError(f"assigned Bead record(s) missing: {', '.join(missing)}")
    return _digest({"version": _VERSION, "records": [records[bead_id] for bead_id in sorted(bead_ids)]})


def carrier_digest(carrier: dict[str, Any]) -> str:
    """Digest a carrier excluding its self-referential digest field."""
    payload = dict(carrier)
    payload.pop("scope_digest", None)
    return _digest(payload)


def extract_carrier(body: str) -> tuple[dict[str, Any] | None, list[str]]:
    starts = [index for index in range(len(body)) if body.startswith(_CARRIER_START, index)]
    if not starts:
        return None, ["PR body is missing the structured pr-scope carrier"]
    if len(starts) != 1:
        return None, ["PR body contains more than one structured pr-scope carrier"]
    start = starts[0] + len(_CARRIER_START)
    end = body.find(_CARRIER_END, start)
    if end < 0:
        return None, ["structured pr-scope carrier is missing its closing comment"]
    payload = body[start:end].strip()
    try:
        carrier = json.loads(payload)
    except json.JSONDecodeError as exc:
        return None, [f"structured pr-scope carrier is not valid JSON: {exc.msg}"]
    if not isinstance(carrier, dict):
        return None, ["structured pr-scope carrier must be a JSON object"]
    return carrier, []


def validate_carrier(
    carrier: dict[str, Any],
    *,
    head_sha: str,
    is_draft: bool,
    beads_path: Path = _BEADS_PATH,
) -> ScopeVerdict:
    reasons: list[str] = []
    assigned = carrier.get("assigned_beads")
    if not isinstance(assigned, list) or not assigned or not all(isinstance(item, str) and item for item in assigned):
        reasons.append("assigned_beads must be a non-empty list of Bead IDs")
        assigned_ids: list[str] = []
    else:
        assigned_ids = assigned
        if len(set(assigned_ids)) != len(assigned_ids):
            reasons.append("assigned_beads contains duplicate Bead IDs")

    if carrier.get("version") != _VERSION:
        reasons.append(f"carrier version must be {_VERSION}")
    if carrier.get("head_sha") != head_sha:
        reasons.append("carrier head_sha does not match the PR head SHA")
    if is_draft:
        reasons.append("PR is draft; publish a non-draft PR before validation")

    expected_scope_digest = carrier_digest(carrier)
    scope_digest = carrier.get("scope_digest")
    if not isinstance(scope_digest, str) or scope_digest != expected_scope_digest:
        reasons.append("carrier scope_digest does not match its canonical content")

    records: dict[str, dict[str, Any]] = {}
    if assigned_ids:
        try:
            records = load_bead_records(beads_path)
            expected_beads_digest = canonical_beads_digest(records, assigned_ids)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            expected_beads_digest = None
            reasons.append(f"cannot resolve assigned Bead records: {exc}")
        if expected_beads_digest is not None and carrier.get("beads_digest") != expected_beads_digest:
            reasons.append("carrier beads_digest is stale for the canonical assigned Bead records")
    else:
        expected_beads_digest = None

    dispositions = carrier.get("dispositions")
    by_bead: dict[str, dict[str, Any]] = {}
    if not isinstance(dispositions, list):
        reasons.append("dispositions must be a list with one entry per assigned Bead")
    else:
        for entry in dispositions:
            if not isinstance(entry, dict) or not isinstance(entry.get("bead_id"), str):
                reasons.append("each disposition must be an object with a bead_id")
                continue
            bead_id = entry["bead_id"]
            if bead_id in by_bead:
                reasons.append(f"duplicate disposition for assigned Bead {bead_id}")
            by_bead[bead_id] = entry
        missing = sorted(set(assigned_ids) - set(by_bead))
        extra = sorted(set(by_bead) - set(assigned_ids))
        if missing:
            reasons.append(f"missing whole-Bead disposition(s): {', '.join(missing)}")
        if extra:
            reasons.append(f"disposition(s) reference unassigned Bead(s): {', '.join(extra)}")

        for bead_id, entry in by_bead.items():
            disposition = entry.get("disposition")
            if disposition not in _DISPOSITIONS:
                reasons.append(f"{bead_id}: unknown whole-Bead disposition {disposition!r}")
            evidence = entry.get("evidence")
            if not isinstance(evidence, list) or not evidence:
                reasons.append(f"{bead_id}: disposition needs at least one typed evidence reference")
            else:
                for ref in evidence:
                    if (
                        not isinstance(ref, dict)
                        or ref.get("kind") not in _EVIDENCE_KINDS
                        or not isinstance(ref.get("ref"), str)
                        or not ref["ref"].strip()
                    ):
                        reasons.append(f"{bead_id}: evidence references need a known kind and non-empty ref")
                        break
            successors = entry.get("successors", [])
            if not isinstance(successors, list) or not all(isinstance(item, str) and item for item in successors):
                reasons.append(f"{bead_id}: successors must be a list of Bead IDs")
                continue
            if len(set(successors)) != len(successors):
                reasons.append(f"{bead_id}: successors contains duplicate Bead IDs")
            if disposition in _RESIDUAL_DISPOSITIONS and not successors:
                reasons.append(f"{bead_id}: {disposition} disposition requires a named successor Bead")
            if disposition == "satisfied" and successors:
                reasons.append(f"{bead_id}: satisfied disposition cannot carry residual successors")
            for successor in successors:
                record = records.get(successor)
                if record is None:
                    reasons.append(f"{bead_id}: successor {successor} is unknown")
                elif record.get("status") == "closed":
                    reasons.append(f"{bead_id}: successor {successor} is closed")
                if successor == bead_id:
                    reasons.append(f"{bead_id}: cannot name itself as a successor")

    return ScopeVerdict(
        ok=not reasons,
        reasons=reasons,
        scope_digest=scope_digest if isinstance(scope_digest, str) else None,
        beads_digest=carrier.get("beads_digest") if isinstance(carrier.get("beads_digest"), str) else None,
        assigned_beads=assigned_ids,
    )


def validate_pr_body(
    body: str,
    *,
    head_sha: str,
    is_draft: bool,
    beads_path: Path = _BEADS_PATH,
) -> ScopeVerdict:
    carrier, reasons = extract_carrier(body)
    if carrier is None:
        return ScopeVerdict(ok=False, reasons=reasons)
    return validate_carrier(carrier, head_sha=head_sha, is_draft=is_draft, beads_path=beads_path)


def render_carrier(carrier: dict[str, Any]) -> str:
    return f"{_CARRIER_START}\n{json.dumps(carrier, indent=2, ensure_ascii=False, sort_keys=True)}\n{_CARRIER_END}"


def build_carrier(input_payload: dict[str, Any], *, head_sha: str, beads_path: Path = _BEADS_PATH) -> dict[str, Any]:
    carrier = dict(input_payload)
    carrier["version"] = _VERSION
    carrier["head_sha"] = head_sha
    carrier.pop("beads_digest", None)
    carrier.pop("scope_digest", None)
    assigned = carrier.get("assigned_beads")
    if not isinstance(assigned, list) or not all(isinstance(item, str) for item in assigned):
        raise ValueError("input assigned_beads must be a list of Bead IDs")
    carrier["beads_digest"] = canonical_beads_digest(load_bead_records(beads_path), assigned)
    carrier["scope_digest"] = carrier_digest(carrier)
    return carrier


def _git_head_sha() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    return result.stdout.strip()


def _github_repo_slug() -> str:
    remote = subprocess.run(
        ["git", "remote", "get-url", "origin"], capture_output=True, text=True, check=True
    ).stdout.strip()
    match = re.search(r"github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?$", remote)
    if match is None:
        raise ValueError(f"cannot derive a GitHub repository from origin remote {remote!r}")
    return f"{match.group(1)}/{match.group(2)}"


def _pr_body_from_github_api(pr: int) -> tuple[str, str, bool]:
    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    api_request = request.Request(f"https://api.github.com/repos/{_github_repo_slug()}/pulls/{pr}", headers=headers)
    with request.urlopen(api_request, timeout=30) as response:
        payload = json.loads(response.read())
    return payload.get("body") or "", payload["head"]["sha"], bool(payload.get("draft"))


def _pr_body(pr: int) -> tuple[str, str, bool]:
    """Read the published PR through ``gh``, or GitHub's public API in CI."""
    try:
        result = subprocess.run(
            ["gh", "pr", "view", str(pr), "--json", "body,headRefOid,isDraft"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return _pr_body_from_github_api(pr)
    payload = json.loads(result.stdout)
    return payload.get("body") or "", payload["headRefOid"], bool(payload.get("isDraft"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)

    render = sub.add_parser("render", help="render a carrier from a JSON scope input")
    render.add_argument("--input", required=True, type=Path, help="JSON with assigned_beads and dispositions")
    render.add_argument("--head-sha", default=None, help="PR head SHA (default: current git HEAD)")
    render.add_argument("--beads-path", type=Path, default=_BEADS_PATH)

    check = sub.add_parser("check", help="validate a PR's embedded carrier")
    check_source = check.add_mutually_exclusive_group(required=True)
    check_source.add_argument("--pr", type=int, help="GitHub PR number to inspect")
    check_source.add_argument("--body-file", type=Path, help="PR body file for local validation")
    check.add_argument("--head-sha", help="required with --body-file")
    check.add_argument("--beads-path", type=Path, default=_BEADS_PATH)
    check.add_argument("--json", action="store_true", dest="as_json")

    args = parser.parse_args(argv)
    if args.action == "render":
        try:
            payload = json.loads(args.input.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("input must be a JSON object")
            carrier = build_carrier(payload, head_sha=args.head_sha or _git_head_sha(), beads_path=args.beads_path)
        except (OSError, ValueError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
            print(f"REFUSING to render pr-scope carrier: {exc}", file=sys.stderr)
            return 2
        print(render_carrier(carrier))
        return 0

    try:
        if args.pr is not None:
            body, head_sha, is_draft = _pr_body(args.pr)
        else:
            if not args.head_sha:
                raise ValueError("--head-sha is required with --body-file")
            body = args.body_file.read_text(encoding="utf-8")
            head_sha = args.head_sha
            is_draft = False
        verdict = validate_pr_body(body, head_sha=head_sha, is_draft=is_draft, beads_path=args.beads_path)
    except (OSError, ValueError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
        print(f"REFUSING to check pr-scope carrier: {exc}", file=sys.stderr)
        return 2

    if args.as_json:
        print(json.dumps(asdict(verdict), indent=2))
    elif verdict.ok:
        print(f"pr-scope OK @ {head_sha[:8]}: {', '.join(verdict.assigned_beads)}")
    else:
        print(f"pr-scope BLOCK @ {head_sha[:8]}:")
        for reason in verdict.reasons:
            print(f"  - {reason}")
    return 0 if verdict.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
