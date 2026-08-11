"""Structured PR scope carrier used by CI and the merge boundary.

The embedded v2 JSON comment is stable intent, not a mutable commit receipt.
The merge boundary binds that intent to the current PR head and canonical Bead
snapshot in its exact-head attestation. Bead acceptance text remains opaque:
the checker consumes typed identifiers, dispositions, evidence refs, and graph
links without interpreting prose.
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

_CARRIER_PREFIX = "polylogue-pr-scope"
_CARRIER_START_PATTERN = re.compile(r"<!--\s+polylogue-pr-scope:v(?P<version>[1-9][0-9]*)\b")
_CARRIER_END = "-->"
_V1 = 1
_VERSION = 2
_BEADS_PATH = Path(".beads/issues.jsonl")
_GITHUB_API_URL = "https://api.github.com"
_REPOSITORY_PATTERN = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
_GIT_OBJECT_PATTERN = re.compile(r"[0-9a-fA-F]{7,64}")
_V1_CARRIER_KEYS = frozenset({"version", "head_sha", "assigned_beads", "beads_digest", "dispositions", "scope_digest"})
_V2_CARRIER_KEYS = frozenset(
    {"version", "scope_kind", "assigned_beads", "mutated_beads", "dispositions", "scope_digest"}
)
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


class ScopeKind(StrEnum):
    BEAD = "bead"
    SELF_CONTAINED = "self_contained"


_DISPOSITIONS = frozenset(item.value for item in ScopeDisposition)
_RESIDUAL_DISPOSITIONS = frozenset(
    {ScopeDisposition.PARTIAL.value, ScopeDisposition.DEFERRED.value, ScopeDisposition.SUPERSEDED.value}
)
_EVIDENCE_KINDS = frozenset(item.value for item in EvidenceKind)
_SCOPE_KINDS = frozenset(item.value for item in ScopeKind)
_SUCCESSOR_LINK_TYPES = frozenset({"blocks", "discovered-from", "relates-to", "supersedes"})


@dataclass(frozen=True, slots=True)
class ScopeVerdict:
    ok: bool
    reasons: list[str] = field(default_factory=list)
    scope_digest: str | None = None
    beads_digest: str | None = None
    assigned_beads: list[str] = field(default_factory=list)
    mutated_beads: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class PullRequestMetadata:
    body: str
    head_sha: str
    base_sha: str
    is_draft: bool
    author_login: str | None = None
    author_type: str | None = None
    changed_files: tuple[str, ...] = ()


class NoOpenPullRequestError(ValueError):
    """CI checked a commit that has no open pull request to validate."""


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _parse_bead_records(lines: object, *, line_label: str) -> dict[str, dict[str, Any]]:
    """Parse issue records from a Beads JSONL snapshot."""
    if not isinstance(lines, list) or not all(isinstance(line, str) for line in lines):
        raise ValueError("Bead JSONL snapshot must contain text lines")
    records: dict[str, dict[str, Any]] = {}
    for line_no, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, dict) or raw.get("_type") != "issue" or not isinstance(raw.get("id"), str):
            continue
        bead_id = raw["id"]
        if bead_id in records:
            raise ValueError(f"duplicate Bead id {bead_id!r} on {line_label} {line_no}")
        records[bead_id] = raw
    return records


def load_bead_records(path: Path = _BEADS_PATH) -> dict[str, dict[str, Any]]:
    """Load the repository's committed Bead records without invoking ``bd``.

    The carrier needs the exact Bead snapshot that CI checked out. Reading the
    JSONL avoids mutating shared Dolt state from a worktree and keeps this
    validation independent of Beads' CLI synchronization hooks.
    """
    return _parse_bead_records(path.read_text(encoding="utf-8").splitlines(), line_label="line")


def canonical_beads_digest(
    records: dict[str, dict[str, Any]], bead_ids: list[str], *, carrier_version: int = _V1
) -> str:
    """Digest whole canonical records for the declared IDs, sorted by Bead ID."""
    missing = [bead_id for bead_id in bead_ids if bead_id not in records]
    if missing:
        raise ValueError(f"assigned Bead record(s) missing: {', '.join(missing)}")
    return _digest({"version": carrier_version, "records": [records[bead_id] for bead_id in sorted(bead_ids)]})


def _successor_is_linked(source_id: str, successor_id: str, records: dict[str, dict[str, Any]]) -> bool:
    """Require a durable Beads relationship between a source and its successor."""
    for record_id, target_id in ((source_id, successor_id), (successor_id, source_id)):
        record = records.get(record_id)
        dependencies = record.get("dependencies") if record is not None else None
        if not isinstance(dependencies, list):
            continue
        for dependency in dependencies:
            if not isinstance(dependency, dict):
                continue
            if dependency.get("depends_on_id") == target_id and dependency.get("type") in _SUCCESSOR_LINK_TYPES:
                return True
    return False


def carrier_digest(carrier: dict[str, Any]) -> str:
    """Digest a carrier excluding its self-referential digest field."""
    payload = dict(carrier)
    payload.pop("scope_digest", None)
    return _digest(payload)


def extract_carrier(body: str) -> tuple[dict[str, Any] | None, list[str]]:
    starts = list(_CARRIER_START_PATTERN.finditer(body))
    if not starts:
        return None, ["PR body is missing the structured pr-scope carrier"]
    if len(starts) != 1:
        return None, ["PR body contains more than one structured pr-scope carrier"]
    marker = starts[0]
    start = marker.end()
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
    carrier_version = carrier.get("version")
    if type(carrier_version) is not int or carrier_version != int(marker.group("version")):
        return None, ["structured pr-scope carrier version does not match its comment marker"]
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


def _bead_ids(value: object, *, label: str, allow_empty: bool, reasons: list[str]) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        reasons.append(f"{label} must be a list of Bead IDs")
        return []
    if not value and not allow_empty:
        reasons.append(f"{label} must be a non-empty list of Bead IDs")
    if len(set(value)) != len(value):
        reasons.append(f"{label} contains duplicate Bead IDs")
    return value


def _bead_records_at(base_sha: str) -> dict[str, dict[str, Any]]:
    if not _GIT_OBJECT_PATTERN.fullmatch(base_sha):
        raise ValueError("base revision must be a hexadecimal Git object name")
    result = subprocess.run(
        ["git", "show", f"{base_sha}:.beads/issues.jsonl"], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise ValueError(f"cannot read .beads/issues.jsonl at base revision {base_sha[:8]}")
    return _parse_bead_records(result.stdout.splitlines(), line_label="base line")


def _ensure_local_commit(revision: str) -> None:
    """Ensure a Git commit reported by GitHub exists in this checkout.

    Feature worktrees and shallow CI checkouts can have a current PR head while
    lacking a target-branch tip that advanced after the checkout was created.
    Mutation scope is defined from the merge base, so resolve that authority
    explicitly instead of requiring an operator-side ``git fetch`` first.
    """
    present = subprocess.run(
        ["git", "cat-file", "-e", f"{revision}^{{commit}}"],
        capture_output=True,
        check=False,
    )
    if present.returncode == 0:
        return
    fetched = subprocess.run(
        ["git", "fetch", "--no-tags", "--quiet", "origin", revision],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    if fetched.returncode != 0:
        detail = fetched.stderr.strip()
        suffix = f": {detail}" if detail else ""
        raise ValueError(f"cannot fetch base revision {revision[:8]} from origin{suffix}")
    present = subprocess.run(
        ["git", "cat-file", "-e", f"{revision}^{{commit}}"],
        capture_output=True,
        check=False,
    )
    if present.returncode != 0:
        raise ValueError(f"fetched base revision {revision[:8]} is not a local commit")


def _merge_base(base_sha: str) -> str:
    """Resolve the PR merge base, recovering complete history in shallow CI clones."""
    result = subprocess.run(["git", "merge-base", base_sha, "HEAD"], capture_output=True, text=True, check=False)
    if result.returncode == 0 and _GIT_OBJECT_PATTERN.fullmatch(result.stdout.strip()):
        return result.stdout.strip()

    shallow = subprocess.run(
        ["git", "rev-parse", "--is-shallow-repository"], capture_output=True, text=True, check=False
    )
    if shallow.returncode == 0 and shallow.stdout.strip() == "true":
        fetched = subprocess.run(
            ["git", "fetch", "--no-tags", "--quiet", "--unshallow", "origin"],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        if fetched.returncode != 0:
            detail = fetched.stderr.strip()
            suffix = f": {detail}" if detail else ""
            raise ValueError(f"cannot deepen shallow history from origin{suffix}")
        result = subprocess.run(["git", "merge-base", base_sha, "HEAD"], capture_output=True, text=True, check=False)
        if result.returncode == 0 and _GIT_OBJECT_PATTERN.fullmatch(result.stdout.strip()):
            return result.stdout.strip()

    raise ValueError(f"cannot resolve PR merge base from base revision {base_sha[:8]}")


def changed_bead_ids(*, base_sha: str, beads_path: Path = _BEADS_PATH) -> list[str]:
    """Return the complete Bead-record mutation set from the PR merge base."""
    if not _GIT_OBJECT_PATTERN.fullmatch(base_sha):
        raise ValueError("base revision must be a hexadecimal Git object name")
    _ensure_local_commit(base_sha)
    before = _bead_records_at(_merge_base(base_sha))
    after = load_bead_records(beads_path)
    return sorted(bead_id for bead_id in set(before) | set(after) if before.get(bead_id) != after.get(bead_id))


def _validate_dispositions(
    dispositions: object,
    *,
    assigned_ids: list[str],
    records: dict[str, dict[str, Any]],
    reasons: list[str],
) -> None:
    by_bead: dict[str, dict[str, Any]] = {}
    if not isinstance(dispositions, list):
        reasons.append("dispositions must be a list with one entry per assigned Bead")
        return
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
            elif not _successor_is_linked(bead_id, successor, records):
                reasons.append(f"{bead_id}: successor {successor} has no durable Beads relationship")
            if successor == bead_id:
                reasons.append(f"{bead_id}: cannot name itself as a successor")


def validate_carrier(
    carrier: dict[str, Any],
    *,
    head_sha: str,
    is_draft: bool,
    beads_path: Path = _BEADS_PATH,
    base_sha: str | None = None,
) -> ScopeVerdict:
    reasons: list[str] = []
    version = carrier.get("version")
    if type(version) is not int or version not in {_V1, _VERSION}:
        return ScopeVerdict(ok=False, reasons=[f"carrier version must be {_V1} or {_VERSION}"])
    is_v1 = version == _V1
    _validate_keys(
        carrier,
        label="carrier",
        allowed=_V1_CARRIER_KEYS if is_v1 else _V2_CARRIER_KEYS,
        required=_V1_CARRIER_KEYS if is_v1 else _V2_CARRIER_KEYS,
        reasons=reasons,
    )
    assigned_ids = _bead_ids(
        carrier.get("assigned_beads"), label="assigned_beads", allow_empty=not is_v1, reasons=reasons
    )
    mutated_ids: list[str] = []
    if is_v1:
        if carrier.get("head_sha") != head_sha:
            reasons.append("carrier head_sha does not match the PR head SHA")
    else:
        scope_kind = carrier.get("scope_kind")
        if scope_kind not in _SCOPE_KINDS:
            reasons.append(f"scope_kind must be one of: {', '.join(sorted(_SCOPE_KINDS))}")
        mutated_ids = _bead_ids(carrier.get("mutated_beads"), label="mutated_beads", allow_empty=True, reasons=reasons)
        if scope_kind == ScopeKind.BEAD.value and not assigned_ids:
            reasons.append("bead scope requires at least one assigned Bead")
        if scope_kind == ScopeKind.SELF_CONTAINED.value and (
            assigned_ids or mutated_ids or carrier.get("dispositions") != []
        ):
            reasons.append("self_contained scope cannot declare or mutate Beads")
    if is_draft:
        reasons.append("PR is draft; publish a non-draft PR before validation")

    expected_scope_digest = carrier_digest(carrier)
    scope_digest = carrier.get("scope_digest")
    if not isinstance(scope_digest, str) or scope_digest != expected_scope_digest:
        reasons.append("carrier scope_digest does not match its canonical content")

    records: dict[str, dict[str, Any]] = {}
    try:
        records = load_bead_records(beads_path)
        missing_assigned = [bead_id for bead_id in assigned_ids if bead_id not in records]
        if missing_assigned:
            reasons.append(f"assigned Bead record(s) missing: {', '.join(missing_assigned)}")
        bound_ids = list(assigned_ids)
        if not is_v1:
            bound_ids.extend(mutated_ids)
            dispositions = carrier.get("dispositions")
            for entry in dispositions if isinstance(dispositions, list) else []:
                if isinstance(entry, dict) and isinstance(entry.get("successors"), list):
                    bound_ids.extend(item for item in entry["successors"] if isinstance(item, str))
        bound_ids = sorted({bead_id for bead_id in bound_ids if bead_id in records})
        expected_beads_digest = canonical_beads_digest(records, bound_ids, carrier_version=int(version))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        expected_beads_digest = None
        reasons.append(f"cannot resolve declared Bead records: {exc}")
    if is_v1 and expected_beads_digest is not None and carrier.get("beads_digest") != expected_beads_digest:
        reasons.append("carrier beads_digest is stale for the canonical assigned Bead records")
    actual_mutations: list[str] = []
    if base_sha is not None:
        try:
            actual_mutations = changed_bead_ids(base_sha=base_sha, beads_path=beads_path)
        except (OSError, ValueError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
            reasons.append(f"cannot resolve Bead mutation scope: {exc}")
        else:
            if is_v1 and actual_mutations:
                reasons.append("legacy v1 carrier cannot omit Bead mutations; render a v2 carrier")
            elif not is_v1 and actual_mutations != sorted(mutated_ids):
                reasons.append("mutated_beads does not match the complete Bead mutation set")

    disposition_records = records
    dispositions = carrier.get("dispositions")
    disposition_entries = dispositions if isinstance(dispositions, list) else []
    successor_ids = {
        successor
        for entry in disposition_entries
        if isinstance(entry, dict)
        for successor in entry.get("successors", [])
        if isinstance(entry.get("successors"), list) and isinstance(successor, str)
    }
    if base_sha is not None and successor_ids:
        try:
            disposition_records = _bead_records_at(base_sha)
            for bead_id in actual_mutations:
                if bead_id in records:
                    disposition_records[bead_id] = records[bead_id]
                else:
                    disposition_records.pop(bead_id, None)
        except (OSError, ValueError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
            reasons.append(f"cannot resolve prospective Bead state: {exc}")

    _validate_dispositions(
        dispositions,
        assigned_ids=assigned_ids,
        records=disposition_records,
        reasons=reasons,
    )
    return ScopeVerdict(
        ok=not reasons,
        reasons=reasons,
        scope_digest=scope_digest if isinstance(scope_digest, str) else None,
        beads_digest=expected_beads_digest,
        assigned_beads=assigned_ids,
        mutated_beads=mutated_ids,
    )


def validate_pr_body(
    body: str,
    *,
    head_sha: str,
    is_draft: bool,
    beads_path: Path = _BEADS_PATH,
    base_sha: str | None = None,
) -> ScopeVerdict:
    carrier, reasons = extract_carrier(body)
    if carrier is None:
        return ScopeVerdict(ok=False, reasons=reasons)
    return validate_carrier(carrier, head_sha=head_sha, is_draft=is_draft, beads_path=beads_path, base_sha=base_sha)


def render_carrier(carrier: dict[str, Any]) -> str:
    version = carrier.get("version")
    return f"<!-- {_CARRIER_PREFIX}:v{version}\n{json.dumps(carrier, indent=2, ensure_ascii=False, sort_keys=True)}\n{_CARRIER_END}"


def build_carrier(input_payload: dict[str, Any], *, head_sha: str, beads_path: Path = _BEADS_PATH) -> dict[str, Any]:
    carrier = dict(input_payload)
    version = carrier.get("version", _VERSION if "scope_kind" in carrier else _V1)
    if type(version) is not int or version not in {_V1, _VERSION}:
        raise ValueError(f"input version must be {_V1} or {_VERSION}")
    carrier["version"] = version
    carrier.pop("scope_digest", None)
    if version == _V1:
        carrier["head_sha"] = head_sha
        carrier.pop("beads_digest", None)
        assigned = carrier.get("assigned_beads")
        if not isinstance(assigned, list) or not all(isinstance(item, str) for item in assigned):
            raise ValueError("input assigned_beads must be a list of Bead IDs")
        carrier["beads_digest"] = canonical_beads_digest(load_bead_records(beads_path), assigned, carrier_version=_V1)
    else:
        carrier.pop("head_sha", None)
        carrier.pop("beads_digest", None)
    carrier["scope_digest"] = carrier_digest(carrier)
    verdict = validate_carrier(carrier, head_sha=head_sha, is_draft=False, beads_path=beads_path)
    if not verdict.ok:
        raise ValueError("invalid scope input: " + "; ".join(verdict.reasons))
    return carrier


def attestation_payload(scope: ScopeVerdict, *, head_sha: str, base_sha: str | None) -> dict[str, object]:
    """Bind stable body intent to the mutable state observed at a merge boundary."""
    payload: dict[str, object] = {
        "version": _VERSION,
        "head_sha": head_sha,
        "base_sha": base_sha,
        "scope_digest": scope.scope_digest,
        "beads_digest": scope.beads_digest,
        "assigned_beads": scope.assigned_beads,
        "mutated_beads": scope.mutated_beads,
    }
    payload["attestation_digest"] = _digest(payload)
    return payload


def _git_head_sha() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    return result.stdout.strip()


def _beads_snapshot_matches_head(beads_path: Path) -> bool:
    """Return whether ``beads_path`` is exactly the blob committed at HEAD."""
    root_result = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=False)
    if root_result.returncode != 0:
        return False
    root = Path(root_result.stdout.strip()).resolve()
    try:
        relative = beads_path.resolve().relative_to(root)
    except ValueError:
        return False
    blob = subprocess.run(["git", "show", f"HEAD:{relative.as_posix()}"], capture_output=True, check=False)
    if blob.returncode != 0:
        return False
    try:
        return beads_path.read_bytes() == blob.stdout
    except OSError:
        return False


def _require_checkout_authority(*, head_sha: str, beads_path: Path) -> None:
    """Require validation inputs to be the exact committed PR revision."""
    if _git_head_sha() != head_sha:
        raise ValueError("current checkout HEAD does not match the fetched PR head")
    if not _beads_snapshot_matches_head(beads_path):
        raise ValueError("Beads snapshot does not match the committed PR head")


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
    user = payload.get("user")
    author_login = user.get("login") if isinstance(user, dict) and isinstance(user.get("login"), str) else None
    author_type = user.get("type") if isinstance(user, dict) and isinstance(user.get("type"), str) else None
    files = payload.get("changed_files")
    changed_files = tuple(files) if isinstance(files, list) and all(isinstance(item, str) for item in files) else ()
    return PullRequestMetadata(
        body=payload.get("body") or "",
        head_sha=head["sha"],
        base_sha=base["sha"],
        is_draft=bool(payload.get("draft")),
        author_login=author_login,
        author_type=author_type,
        changed_files=changed_files,
    )


def fetch_pr_files(pr: int, *, repository: str) -> tuple[str, ...]:
    """Fetch the complete changed-file list used by automated-scope policy."""
    raw = _github_request_bytes(f"repos/{repository}/pulls/{pr}/files?per_page=100")
    if raw is None:
        raise RuntimeError(f"GitHub API returned no file list for PR #{pr}")
    payload = json.loads(raw)
    if not isinstance(payload, list):
        raise ValueError("GitHub PR files response must be a list")
    names = tuple(
        item["filename"] for item in payload if isinstance(item, dict) and isinstance(item.get("filename"), str)
    )
    if len(names) != len(payload):
        raise ValueError("GitHub PR files response contains an invalid filename")
    if len(names) >= 100:
        raise ValueError("automated dependency scope refuses PRs with 100 or more changed files")
    return names


def fetch_pr_metadata(pr: int, *, repository: str) -> PullRequestMetadata:
    raw = _github_request_bytes(f"repos/{repository}/pulls/{pr}")
    if raw is None:
        raise RuntimeError(f"GitHub API returned no metadata for PR #{pr}")
    metadata = _pr_metadata_from_payload(json.loads(raw))
    if metadata.author_login == "dependabot[bot]" and metadata.author_type == "Bot":
        metadata = PullRequestMetadata(
            **{**asdict(metadata), "changed_files": fetch_pr_files(pr, repository=repository)}
        )
    return metadata


def _automated_dependency_scope_allowed(metadata: PullRequestMetadata) -> bool:
    """Recognize only GitHub's bot identity and an exact dependency file set."""
    return automated_dependency_scope_allowed(
        author_login=metadata.author_login,
        author_type=metadata.author_type,
        changed_files=metadata.changed_files,
    )


def automated_dependency_scope_allowed(
    *,
    author_login: str | None,
    author_type: str | None,
    author_is_bot: bool | None = None,
    changed_files: tuple[str, ...],
) -> bool:
    """Recognize the typed dependency-only scope across REST and gh schemas."""
    allowed = {
        ".github/workflows/codeql.yml",
        "pyproject.toml",
        "uv.lock",
    }
    return (
        (
            (author_login == "dependabot[bot]" and author_type == "Bot")
            or (author_login == "app/dependabot" and author_is_bot is True)
        )
        and bool(changed_files)
        and set(changed_files).issubset(allowed)
    )


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
            if metadata.author_login == "dependabot[bot]" and metadata.author_type == "Bot":
                metadata = PullRequestMetadata(
                    **{**asdict(metadata), "changed_files": fetch_pr_files(item["number"], repository=repository)}
                )
            candidates.append((item["number"], metadata))
    if not candidates:
        raise NoOpenPullRequestError(f"no open PR found for head {head_sha[:8]}")
    if len(candidates) != 1:
        raise ValueError(f"expected one open PR for head {head_sha[:8]}, found {len(candidates)}")
    return candidates[0]


def fetch_base_validator_source(*, repository: str, base_sha: str) -> bytes | None:
    local = subprocess.run(
        ["git", "show", f"{base_sha}:devtools/pr_scope.py"],
        capture_output=True,
        check=False,
    )
    if local.returncode == 0 and local.stdout:
        return local.stdout
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
        help_result = subprocess.run(
            [sys.executable, str(validator_path), "check", "--help"], capture_output=True, text=True, check=False
        )
        if help_result.returncode != 0:
            raise RuntimeError("base-revision pr-scope validator cannot report its check interface")
        argv = [
            sys.executable,
            str(validator_path),
            "check",
            "--body-file",
            str(body_path),
            "--head-sha",
            metadata.head_sha,
            "--beads-path",
            str(beads_path.resolve()),
        ]
        if "--base-sha" in help_result.stdout:
            argv.extend(["--base-sha", metadata.base_sha])
        result = subprocess.run(
            argv,
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
    if metadata.is_draft:
        print("REFUSING CI pr-scope check: PR is draft; publish it before validation", file=sys.stderr)
        return 2

    if _automated_dependency_scope_allowed(metadata):
        print(
            "pr-scope CI OK: typed automated-dependency disposition "
            f"(no Bead carrier; files={','.join(metadata.changed_files)})"
        )
        return 0

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
        base_sha=metadata.base_sha,
    )
    return _emit_verdict(verdict, head_sha=metadata.head_sha, as_json=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)

    render = sub.add_parser("render", help="render a carrier from a JSON scope input")
    render.add_argument("--input", required=True, type=Path, help="JSON with assigned_beads and dispositions")
    render.add_argument("--head-sha", default=None, help="v1 transition head SHA (default: current git HEAD)")
    render.add_argument("--beads-path", type=Path, default=_BEADS_PATH)

    check = sub.add_parser("check", help="validate a PR's embedded carrier")
    check_source = check.add_mutually_exclusive_group(required=True)
    check_source.add_argument("--pr", type=int, help="GitHub PR number to inspect")
    check_source.add_argument("--body-file", type=Path, help="PR body file for local validation")
    check.add_argument("--repo", help="GitHub OWNER/REPO (default: CI metadata or origin remote)")
    check.add_argument("--head-sha", help="required with --body-file")
    check.add_argument("--base-sha", help="base SHA used to validate the complete Bead mutation scope")
    check.add_argument("--beads-path", type=Path, default=_BEADS_PATH)
    check.add_argument("--json", action="store_true", dest="as_json")

    check_ci = sub.add_parser("check-ci", help="validate with the PR base revision's authoritative checker")
    check_ci.add_argument("--pr", type=int, help="GitHub PR number (default: resolve from --expected-head-sha)")
    check_ci.add_argument("--repo", required=True, help="GitHub OWNER/REPO from CI metadata")
    check_ci.add_argument("--expected-head-sha", default=os.environ.get("CIRCLE_SHA1"))
    check_ci.add_argument("--beads-path", type=Path, default=_BEADS_PATH)

    sync = sub.add_parser("sync", help="report the current mutable attestation for a stable PR carrier")
    sync.add_argument("--pr", type=int, required=True, help="GitHub PR number to inspect")
    sync.add_argument("--repo", help="GitHub OWNER/REPO (default: CI metadata or origin remote)")
    sync.add_argument("--beads-path", type=Path, default=_BEADS_PATH)

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
        except NoOpenPullRequestError as exc:
            print(f"pr-scope CI skip: {exc}", file=sys.stderr)
            return 0
        except (OSError, ValueError, json.JSONDecodeError, RuntimeError, subprocess.SubprocessError) as exc:
            print(f"REFUSING CI pr-scope check: {exc}", file=sys.stderr)
            return 2

    if args.action == "sync":
        try:
            repository = resolve_repository(args.repo)
            metadata = fetch_pr_metadata(args.pr, repository=repository)
            _require_checkout_authority(head_sha=metadata.head_sha, beads_path=args.beads_path)
            verdict = validate_pr_body(
                metadata.body,
                head_sha=metadata.head_sha,
                is_draft=metadata.is_draft,
                beads_path=args.beads_path,
                base_sha=metadata.base_sha,
            )
            if not verdict.ok:
                return _emit_verdict(verdict, head_sha=metadata.head_sha, as_json=False)
            print(
                json.dumps(
                    attestation_payload(verdict, head_sha=metadata.head_sha, base_sha=metadata.base_sha), indent=2
                )
            )
            return 0
        except (OSError, ValueError, json.JSONDecodeError, RuntimeError, subprocess.SubprocessError) as exc:
            print(f"REFUSING to sync pr-scope attestation: {exc}", file=sys.stderr)
            return 2

    try:
        if args.pr is not None:
            repository = resolve_repository(args.repo)
            metadata = fetch_pr_metadata(args.pr, repository=repository)
            body = metadata.body
            head_sha = metadata.head_sha
            is_draft = metadata.is_draft
            base_sha = metadata.base_sha
            _require_checkout_authority(head_sha=head_sha, beads_path=args.beads_path)
        else:
            if not args.head_sha:
                raise ValueError("--head-sha is required with --body-file")
            body = args.body_file.read_text(encoding="utf-8")
            head_sha = args.head_sha
            is_draft = False
            base_sha = args.base_sha
            extracted_carrier, _carrier_reasons = extract_carrier(body)
            if extracted_carrier is not None and extracted_carrier.get("version") == _VERSION and base_sha is None:
                raise ValueError("--base-sha is required with --body-file for a v2 carrier")
            if extracted_carrier is not None and extracted_carrier.get("version") == _VERSION:
                _require_checkout_authority(head_sha=head_sha, beads_path=args.beads_path)
        verdict = validate_pr_body(
            body,
            head_sha=head_sha,
            is_draft=is_draft,
            beads_path=args.beads_path,
            base_sha=base_sha,
        )
    except (OSError, ValueError, json.JSONDecodeError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"REFUSING to check pr-scope carrier: {exc}", file=sys.stderr)
        return 2

    return _emit_verdict(verdict, head_sha=head_sha, as_json=args.as_json)


if __name__ == "__main__":
    raise SystemExit(main())
