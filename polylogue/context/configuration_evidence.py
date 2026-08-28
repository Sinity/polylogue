"""Versioned agent-configuration evidence and exact context resolution.

This module is deliberately storage-independent.  Acquisition adapters can
persist the immutable records in the source tier, while readers can resolve a
historical context without consulting the current checkout.
"""

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from polylogue.core.refs import ExecutionContextRef

ArtifactKind = Literal["instruction", "skill", "hook", "setting", "mcp_profile", "tool_profile"]
Validity = Literal["valid", "unknown", "invalid"]


@dataclass(frozen=True, slots=True)
class ConfigurationArtifactVersion:
    """One observed immutable revision of a setup artifact."""

    artifact_id: str
    kind: ArtifactKind
    path: str
    content_hash: str
    owner: str
    repository: str | None
    observed_from_ms: int
    observed_until_ms: int | None = None
    validity: Validity = "valid"
    source_revision: str | None = None

    def __post_init__(self) -> None:
        if not self.artifact_id.strip() or not self.path.strip() or not self.owner.strip():
            raise ValueError("configuration artifact identity, path, and owner are required")
        if len(self.content_hash) != 64 or any(c not in "0123456789abcdef" for c in self.content_hash):
            raise ValueError("configuration artifact content_hash must be a lowercase SHA-256 hex digest")
        if self.observed_from_ms < 0 or (
            self.observed_until_ms is not None and self.observed_until_ms <= self.observed_from_ms
        ):
            raise ValueError("configuration artifact validity interval is invalid")


@dataclass(frozen=True, slots=True)
class ConfigurationObservation:
    """The complete setup evidence available at one observation boundary."""

    artifacts: tuple[ConfigurationArtifactVersion, ...] = ()
    unknown_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len({artifact.artifact_id for artifact in self.artifacts}) != len(self.artifacts):
            raise ValueError("configuration observation contains duplicate artifact ids")


@dataclass(frozen=True, slots=True)
class ContextResolution:
    """Resolution result that distinguishes exact, incomplete, and ambiguous evidence."""

    context: ExecutionContextRef | None
    status: Literal["exact", "partial", "gap", "overlap"]
    artifacts: tuple[ConfigurationArtifactVersion, ...]
    missing_paths: tuple[str, ...] = ()
    overlapping_paths: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class StructuralInvocation:
    """A structural skill/tool/hook use joined to its declaring revision."""

    name: str
    kind: ArtifactKind
    observed_at_ms: int
    declaration: ConfigurationArtifactVersion | None


@dataclass(frozen=True, slots=True)
class EfficacyComparison:
    """An honest cohort comparison; it carries no causal or line-attention claim."""

    cohort: str
    compared_cohort: str
    outcome: str
    confounds: tuple[str, ...]
    coverage: str
    judgment_authority: str


def artifact_from_bytes(
    *,
    kind: ArtifactKind,
    path: str,
    payload: bytes,
    owner: str,
    repository: str | None,
    observed_from_ms: int,
    observed_until_ms: int | None = None,
    validity: Validity = "valid",
    source_revision: str | None = None,
) -> ConfigurationArtifactVersion:
    """Create a content-addressed artifact revision from exact acquired bytes."""

    digest = hashlib.sha256(payload).hexdigest()
    artifact_id = f"config:{kind}:{path}:{digest}"
    return ConfigurationArtifactVersion(
        artifact_id=artifact_id,
        kind=kind,
        path=path,
        content_hash=digest,
        owner=owner,
        repository=repository,
        observed_from_ms=observed_from_ms,
        observed_until_ms=observed_until_ms,
        validity=validity,
        source_revision=source_revision,
    )


def capture_path(
    path: Path, *, kind: ArtifactKind, owner: str, repository: str | None, observed_from_ms: int
) -> ConfigurationArtifactVersion:
    """Capture one live file; missing files are reported to the caller."""

    return artifact_from_bytes(
        kind=kind,
        path=str(path),
        payload=path.read_bytes(),
        owner=owner,
        repository=repository,
        observed_from_ms=observed_from_ms,
    )


def resolve_context(
    artifacts: Sequence[ConfigurationArtifactVersion], *, at_ms: int, expected_paths: Sequence[str] = ()
) -> ContextResolution:
    """Resolve only revisions whose recorded interval contains ``at_ms``.

    Multiple matching revisions are an overlap, and an absent expected path is
    a gap.  Neither condition is filled from current files.
    """

    by_path: dict[str, list[ConfigurationArtifactVersion]] = {}
    for artifact in artifacts:
        if artifact.observed_from_ms <= at_ms and (
            artifact.observed_until_ms is None or at_ms < artifact.observed_until_ms
        ):
            by_path.setdefault(artifact.path, []).append(artifact)
    overlaps = tuple(sorted(path for path, rows in by_path.items() if len(rows) > 1))
    missing = tuple(sorted(set(expected_paths) - by_path.keys()))
    selected = tuple(sorted((rows[0] for path, rows in by_path.items() if len(rows) == 1), key=lambda item: item.path))
    if overlaps:
        return ContextResolution(None, "overlap", selected, missing, overlaps)
    fields = {f"artifact:{item.path}": item.content_hash for item in selected}
    unknown = tuple(f"missing:{path}" for path in missing)
    if not selected and missing:
        return ContextResolution(None, "gap", selected, missing)
    context = ExecutionContextRef.from_observation(fields, unknown_fields=unknown)
    return ContextResolution(context, "partial" if missing else "exact", selected, missing)


def join_invocations(
    invocations: Sequence[tuple[str, ArtifactKind, int]], artifacts: Sequence[ConfigurationArtifactVersion]
) -> tuple[StructuralInvocation, ...]:
    """Join structural invocations to the unique declaration active at that time."""

    result: list[StructuralInvocation] = []
    for name, kind, observed_at_ms in invocations:
        matches = [
            artifact
            for artifact in artifacts
            if artifact.kind == kind
            and artifact.path == name
            and artifact.observed_from_ms <= observed_at_ms
            and (artifact.observed_until_ms is None or observed_at_ms < artifact.observed_until_ms)
        ]
        result.append(StructuralInvocation(name, kind, observed_at_ms, matches[0] if len(matches) == 1 else None))
    return tuple(result)


def git_artifact_history(
    repository: Path, path: str, *, owner: str, kind: ArtifactKind, limit: int = 100
) -> tuple[ConfigurationArtifactVersion, ...]:
    """Read authoritative committed revisions without treating the worktree as history."""

    result = subprocess.run(
        ["git", "log", f"--max-count={limit}", "--format=%H", "--", path],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        return ()
    revisions: list[ConfigurationArtifactVersion] = []
    commits = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    for commit in reversed(commits):
        show = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=repository, capture_output=True, check=False)
        if show.returncode == 0:
            timestamp = subprocess.run(
                ["git", "show", "-s", "--format=%ct", commit],
                cwd=repository,
                capture_output=True,
                text=True,
                check=False,
            )
            if timestamp.returncode != 0 or not timestamp.stdout.strip().isdigit():
                continue
            observed_from_ms = int(timestamp.stdout.strip()) * 1000
            revisions.append(
                artifact_from_bytes(
                    kind=kind,
                    path=path,
                    payload=show.stdout,
                    owner=owner,
                    repository=str(repository),
                    observed_from_ms=observed_from_ms,
                    source_revision=commit,
                )
            )
    return tuple(
        replace(
            revision,
            observed_until_ms=next(
                (
                    later.observed_from_ms
                    for later in revisions[index + 1 :]
                    if later.observed_from_ms > revision.observed_from_ms
                ),
                None,
            ),
        )
        for index, revision in enumerate(revisions)
    )


def compare_cohorts(
    *, cohort: str, compared_cohort: str, outcome: str, confounds: Sequence[str], coverage: str, judgment_authority: str
) -> EfficacyComparison:
    """Construct a comparison whose limitations are mandatory data, not prose."""

    if not confounds or not coverage.strip() or not judgment_authority.strip():
        raise ValueError("efficacy comparisons require confounds, coverage, and judgment authority")
    return EfficacyComparison(cohort, compared_cohort, outcome, tuple(confounds), coverage, judgment_authority)


__all__ = [
    "ConfigurationArtifactVersion",
    "ConfigurationObservation",
    "ContextResolution",
    "StructuralInvocation",
    "EfficacyComparison",
    "artifact_from_bytes",
    "capture_path",
    "resolve_context",
    "join_invocations",
    "git_artifact_history",
    "compare_cohorts",
]
