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
import tempfile
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any
from urllib import error, parse, request

_CARRIER_PREFIX = "polylogue-pr-scope:v1"
_CARRIER_START = f"<!-- {_CARRIER_PREFIX}"
_CARRIER_END = "-->"
_VERSION = 1
_BEADS_PATH = Path(".beads/issues.jsonl")
_GITHUB_API_URL = "https://api.github.com"
_REPOSITORY_PATTERN = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
_CARRIER_KEYS = frozenset({"version", "head_sha", "assigned_beads", "beads_digest", "dispositions", "scope_digest"})
_DISPOSITION_KEYS = frozenset({"bead_id", "disposition", "evidence", "successors"})
_EVIDENCE_KEYS = frozenset({"kind", "ref"})


class ScopeDisposition(StrEnum):
    SATISFIED = "satisfied"
    PARTIAL = "partial"
    DEFERRED = "deferred"
    SUPERSEDED = "superseded"


class EvidenceKind(StrEnum):
    COMMAND = "command"
    COMMIT = "commit"
    DIFF = "diff"
    RECEIPT = "receipt"
    REVIEW = "review"
    TEST = "test"


_DISPOSITIONS = frozenset(item.value for item in ScopeDisposition)
_RESIDUAL_DISPOSITIONS = frozenset(
    {ScopeDisposition.PARTIAL.value, ScopeDisposition.DEFERRED.value, ScopeDisposition.SUPERSEDED.value}
)
_EVIDENCE_KINDS = frozenset(item.value for item in EvidenceKind)


@dataclass(frozen=True, slots=True)
class ScopeVerdict:
    ok: bool
    reasons: list[str] = field(default_factory=list)
    scope_digest: str | None = None
    beads_digest: str | None = None
    assigned_beads: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class PullRequestMetadata:
    body: str
    head_sha: str
    base_sha: str
    is_draft: bool


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


def _validate_keys(
    value: dict[str, Any],
    *,
    label: str,
    allowed: frozenset[str],
    required: frozenset[str],
    reasons: list[str],
) -> None:
    unknown = sorted(set(value) - allowed)
    missing = sorted(required - set(value))
    if unknown:
        reasons.append(f"{label} has unknown field(s): {', '.join(unknown)}")
    if missing:
        reasons.append(f"{label} is missing required field(s): {', '.join(missing)}")


def validate_carrier(
    carrier: dict[str, Any],
    *,
    head_sha: str,
    is_draft: bool,
    beads_path: Path = _BEADS_PATH,
) -> ScopeVerdict:
    reasons: list[str] = []
    _validate_keys(
        carrier,
        label="carrier",
        allowed=_CARRIER_KEYS,
        required=_CARRIER_KEYS,
        reasons=reasons,
    )
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
            _validate_keys(
                entry,
                label=f"disposition for {entry['bead_id']}",
                allowed=_DISPOSITION_KEYS,
                required=_DISPOSITION_KEYS,
                reasons=reasons,
            )
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
                    if isinstance(ref, dict):
                        _validate_keys(
                            ref,
                            label=f"evidence for {bead_id}",
                            allowed=_EVIDENCE_KEYS,
                            required=_EVIDENCE_KEYS,
                            reasons=reasons,
                        )
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
    verdict = validate_carrier(carrier, head_sha=head_sha, is_draft=False, beads_path=beads_path)
    if not verdict.ok:
        raise ValueError("invalid scope input: " + "; ".join(verdict.reasons))
    return carrier


def _git_head_sha() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    return result.stdout.strip()


def _repository_from_remote(remote: str) -> str | None:
    value = remote.strip()
    if value.startswith("git@github.com:"):
        value = value.removeprefix("git@github.com:")
    elif "github.com/" in value:
        value = value.split("github.com/", 1)[1]
    else:
        return None
    value = value.removesuffix(".git").strip("/")
    return value if _REPOSITORY_PATTERN.fullmatch(value) else None


def resolve_repository(explicit: str | None = None) -> str:
    candidates = [
        explicit,
        os.environ.get("GITHUB_REPOSITORY"),
        (
            f"{os.environ['CIRCLE_PROJECT_USERNAME']}/{os.environ['CIRCLE_PROJECT_REPONAME']}"
            if os.environ.get("CIRCLE_PROJECT_USERNAME") and os.environ.get("CIRCLE_PROJECT_REPONAME")
            else None
        ),
    ]
    for candidate in candidates:
        if candidate and _REPOSITORY_PATTERN.fullmatch(candidate):
            return candidate
    remote = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        capture_output=True,
        text=True,
        check=False,
    )
    if remote.returncode == 0:
        repository = _repository_from_remote(remote.stdout)
        if repository:
            return repository
    raise ValueError("cannot resolve GitHub repository; pass --repo OWNER/REPO")


def _github_request_bytes(
    path: str,
    *,
    accept: str = "application/vnd.github+json",
    missing_ok: bool = False,
) -> bytes | None:
    api_url = os.environ.get("GITHUB_API_URL", _GITHUB_API_URL).rstrip("/")
    url = f"{api_url}/{path.lstrip('/')}"
    headers = {
        "Accept": accept,
        "User-Agent": "polylogue-pr-scope",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    api_request = request.Request(url, headers=headers)
    try:
        with request.urlopen(api_request, timeout=30) as response:
            payload = response.read()
            if not isinstance(payload, bytes):
                raise RuntimeError(f"GitHub API returned non-bytes content for {path}")
            return payload
    except error.HTTPError as exc:
        if missing_ok and exc.code == 404:
            return None
        detail = exc.read().decode(errors="replace")[:300]
        raise RuntimeError(f"GitHub API returned HTTP {exc.code} for {path}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"GitHub API request failed for {path}: {exc.reason}") from exc


def _pr_metadata_from_payload(payload: object) -> PullRequestMetadata:
    if not isinstance(payload, dict):
        raise ValueError("GitHub PR response must be an object")
    head = payload.get("head")
    base = payload.get("base")
    if not isinstance(head, dict) or not isinstance(head.get("sha"), str):
        raise ValueError("GitHub PR response is missing head.sha")
    if not isinstance(base, dict) or not isinstance(base.get("sha"), str):
        raise ValueError("GitHub PR response is missing base.sha")
    return PullRequestMetadata(
        body=payload.get("body") or "",
        head_sha=head["sha"],
        base_sha=base["sha"],
        is_draft=bool(payload.get("draft")),
    )


def fetch_pr_metadata(pr: int, *, repository: str) -> PullRequestMetadata:
    raw = _github_request_bytes(f"repos/{repository}/pulls/{pr}")
    if raw is None:
        raise RuntimeError(f"GitHub API returned no metadata for PR #{pr}")
    return _pr_metadata_from_payload(json.loads(raw))


def fetch_pr_for_head(*, repository: str, head_sha: str) -> tuple[int, PullRequestMetadata]:
    raw = _github_request_bytes(f"repos/{repository}/commits/{head_sha}/pulls")
    if raw is None:
        raise RuntimeError(f"GitHub API returned no PR metadata for head {head_sha[:8]}")
    payload = json.loads(raw)
    if not isinstance(payload, list):
        raise ValueError("GitHub commit-pulls response must be a list")
    candidates: list[tuple[int, PullRequestMetadata]] = []
    for item in payload:
        if not isinstance(item, dict) or item.get("state") != "open" or not isinstance(item.get("number"), int):
            continue
        metadata = _pr_metadata_from_payload(item)
        if metadata.head_sha == head_sha:
            candidates.append((item["number"], metadata))
    if len(candidates) != 1:
        raise ValueError(f"expected one open PR for head {head_sha[:8]}, found {len(candidates)}")
    return candidates[0]


def fetch_base_validator_source(*, repository: str, base_sha: str) -> bytes | None:
    path = f"repos/{repository}/contents/devtools/pr_scope.py?ref={parse.quote(base_sha, safe='')}"
    return _github_request_bytes(path, accept="application/vnd.github.raw+json", missing_ok=True)


def _emit_verdict(verdict: ScopeVerdict, *, head_sha: str, as_json: bool) -> int:
    if as_json:
        print(json.dumps(asdict(verdict), indent=2))
    elif verdict.ok:
        print(f"pr-scope OK @ {head_sha[:8]}: {', '.join(verdict.assigned_beads)}")
    else:
        print(f"pr-scope BLOCK @ {head_sha[:8]}:")
        for reason in verdict.reasons:
            print(f"  - {reason}")
    return 0 if verdict.ok else 1


def _run_validator_source(
    source: bytes,
    *,
    metadata: PullRequestMetadata,
    beads_path: Path,
) -> int:
    with tempfile.TemporaryDirectory(prefix="polylogue-pr-scope-base-") as temporary:
        root = Path(temporary)
        validator_path = root / "pr_scope.py"
        body_path = root / "pr-body.md"
        validator_path.write_bytes(source)
        body_path.write_text(metadata.body, encoding="utf-8")
        result = subprocess.run(
            [
                sys.executable,
                str(validator_path),
                "check",
                "--body-file",
                str(body_path),
                "--head-sha",
                metadata.head_sha,
                "--beads-path",
                str(beads_path.resolve()),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    return result.returncode


def check_ci_metadata(
    metadata: PullRequestMetadata,
    *,
    repository: str,
    beads_path: Path,
    checkout_head_sha: str,
    expected_head_sha: str | None,
) -> int:
    if checkout_head_sha != metadata.head_sha:
        print(
            f"REFUSING CI pr-scope check: checkout HEAD {checkout_head_sha[:8]} does not match "
            f"PR head {metadata.head_sha[:8]}",
            file=sys.stderr,
        )
        return 2
    if expected_head_sha and expected_head_sha != metadata.head_sha:
        print(
            f"REFUSING CI pr-scope check: CI head {expected_head_sha[:8]} does not match "
            f"PR head {metadata.head_sha[:8]}",
            file=sys.stderr,
        )
        return 2

    base_source = fetch_base_validator_source(repository=repository, base_sha=metadata.base_sha)
    if base_source is not None:
        print(f"pr-scope CI authority: base revision {metadata.base_sha[:8]}")
        return _run_validator_source(base_source, metadata=metadata, beads_path=beads_path)

    print(
        f"pr-scope CI bootstrap: base revision {metadata.base_sha[:8]} has no validator; "
        "using the checked-out validator for this first landing",
        file=sys.stderr,
    )
    verdict = validate_pr_body(
        metadata.body,
        head_sha=metadata.head_sha,
        is_draft=metadata.is_draft,
        beads_path=beads_path,
    )
    return _emit_verdict(verdict, head_sha=metadata.head_sha, as_json=False)


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
    check.add_argument("--repo", help="GitHub OWNER/REPO (default: CI metadata or origin remote)")
    check.add_argument("--head-sha", help="required with --body-file")
    check.add_argument("--beads-path", type=Path, default=_BEADS_PATH)
    check.add_argument("--json", action="store_true", dest="as_json")

    check_ci = sub.add_parser("check-ci", help="validate with the PR base revision's authoritative checker")
    check_ci.add_argument("--pr", type=int, help="GitHub PR number (default: resolve from --expected-head-sha)")
    check_ci.add_argument("--repo", required=True, help="GitHub OWNER/REPO from CI metadata")
    check_ci.add_argument("--expected-head-sha", default=os.environ.get("CIRCLE_SHA1"))
    check_ci.add_argument("--beads-path", type=Path, default=_BEADS_PATH)

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

    if args.action == "check-ci":
        try:
            repository = resolve_repository(args.repo)
            if args.pr is not None:
                pr_number = args.pr
                metadata = fetch_pr_metadata(args.pr, repository=repository)
            else:
                if not args.expected_head_sha:
                    raise ValueError("--pr or --expected-head-sha is required")
                pr_number, metadata = fetch_pr_for_head(repository=repository, head_sha=args.expected_head_sha)
                print(f"pr-scope CI metadata: resolved PR #{pr_number} from head {args.expected_head_sha[:8]}")
            checkout_head_sha = _git_head_sha()
            return check_ci_metadata(
                metadata,
                repository=repository,
                beads_path=args.beads_path,
                checkout_head_sha=checkout_head_sha,
                expected_head_sha=args.expected_head_sha,
            )
        except (OSError, ValueError, json.JSONDecodeError, RuntimeError, subprocess.SubprocessError) as exc:
            print(f"REFUSING CI pr-scope check: {exc}", file=sys.stderr)
            return 2

    try:
        if args.pr is not None:
            repository = resolve_repository(args.repo)
            metadata = fetch_pr_metadata(args.pr, repository=repository)
            body = metadata.body
            head_sha = metadata.head_sha
            is_draft = metadata.is_draft
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

    return _emit_verdict(verdict, head_sha=head_sha, as_json=args.as_json)


if __name__ == "__main__":
    raise SystemExit(main())
