"""Declared, recorded schema-source frontier.

Schema generation reads operator-owned source roots that keep moving: a live
``~/.gemini/tmp`` gains a directory between two runs eleven minutes apart, and
a provider export root can be reorganized wholesale. A digest measured over
those live roots is therefore not a baseline -- it is a reading. Two different
runs legitimately disagree, and nothing distinguishes "the corpus grew" from
"the declared root silently stopped resolving and the sample set narrowed".

This module makes the frontier an explicit, recorded artifact:

* the **declaration** names one or more roots per schema subject, with
  exclusion patterns and their reasons, and is operator-owned state (private
  absolute paths never enter the repository);
* the **baseline** records, per declared root, every admitted member with its
  byte count and content digest, plus one aggregate digest over the whole
  declaration. It is read from the file, so it is conserved by construction:
  repeated reads return the same digest whatever the live tree does;
* the **check** re-inventories the live roots through the production admission
  route (:func:`inventory_schema_sources`) and reports every divergence as a
  finding. A deleted, moved, emptied or mutated root is red, not a quietly
  smaller sample set.

Freezing the bytes physically (a snapshot tree under the archive state root)
would also conserve the digest, at the cost of duplicating tens of gigabytes of
private operator material on a wear-limited host and creating a second copy
that itself needs retention and review. The recorded manifest conserves the
same thing for a few megabytes of digests, and it keeps generation reading the
one authoritative copy.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from fnmatch import fnmatch
from pathlib import Path
from typing import Final

from polylogue.core.hashing import hash_file, hash_payload
from polylogue.core.schema_subjects import inference_exclusion_reason
from polylogue.schemas.source_inference import (
    SchemaSourceInput,
    SourceInferenceError,
    inventory_schema_sources,
)

FRONTIER_SCHEMA: Final[str] = "polylogue.schema-source-frontier.v1"
FRONTIER_FILENAME: Final[str] = "frontier.json"
FRONTIER_ENV_VAR: Final[str] = "POLYLOGUE_SCHEMA_FRONTIER"


class SchemaFrontierError(RuntimeError):
    """The declared frontier could not be read or is not well formed."""


@dataclass(frozen=True, slots=True)
class FrontierExclusion:
    """One declared exclusion pattern under a root, with its reason and owner."""

    pattern: str
    reason: str
    owner: str | None = None

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {"pattern": self.pattern, "reason": self.reason}
        if self.owner is not None:
            payload["owner"] = self.owner
        return payload


@dataclass(frozen=True, slots=True)
class FrontierRoot:
    """One declared source root for a schema subject."""

    path: Path
    scope: str
    exclusions: tuple[FrontierExclusion, ...] = ()
    admit: tuple[str, ...] = ()
    #: ``frozen`` (the default) is an export root whose members never change:
    #: any divergence from the baseline is an error. ``append`` is a live
    #: harness root that legitimately gains files and grows existing ones
    #: while a session runs; there, growth is a notice and only a member that
    #: disappeared, shrank or changed under a constant size is an error.
    mutability: str = "frozen"
    zero_material_reason: str | None = None

    def __post_init__(self) -> None:
        if self.mutability not in {"frozen", "append"}:
            raise SchemaFrontierError(f"declared root {self.path} has an unknown mutability: {self.mutability!r}")

    @property
    def key(self) -> str:
        return str(self.path)

    def admits(self, relative: str) -> bool:
        """Whether a member is inside the declared denominator of this root.

        ``admit`` is for a shared drop root that is not a provider's own export
        tree: without it every member of the root would be attributed to the
        subject. An empty ``admit`` admits everything the production route
        inventories, minus the declared exclusions.
        """
        if self.admit and not any(fnmatch(relative, pattern) for pattern in self.admit):
            return False
        return self.excluded_by(relative) is None

    def excluded_by(self, relative: str) -> FrontierExclusion | None:
        for exclusion in self.exclusions:
            if fnmatch(relative, exclusion.pattern):
                return exclusion
        return None

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {"path": str(self.path), "scope": self.scope}
        if self.mutability != "frozen":
            payload["mutability"] = self.mutability
        if self.admit:
            payload["admit"] = list(self.admit)
        if self.exclusions:
            payload["exclusions"] = [item.to_payload() for item in self.exclusions]
        if self.zero_material_reason is not None:
            payload["zero_material_reason"] = self.zero_material_reason
        return payload


@dataclass(frozen=True, slots=True)
class FrontierSubject:
    """Every declared root for one schema subject."""

    subject: str
    roots: tuple[FrontierRoot, ...]

    def to_payload(self) -> dict[str, object]:
        return {"subject": self.subject, "roots": [root.to_payload() for root in self.roots]}


@dataclass(frozen=True, slots=True)
class FrontierMember:
    """One admitted source member of a declared root."""

    relative: str
    byte_count: int
    sha256: str

    def to_payload(self) -> dict[str, object]:
        return {"relative": self.relative, "byte_count": self.byte_count, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class RootBaseline:
    """Recorded admitted membership of one declared root."""

    subject: str
    root: str
    member_count: int
    byte_count: int
    members: tuple[FrontierMember, ...]

    @property
    def digest(self) -> str:
        return hash_payload(
            {
                "subject": self.subject,
                "root": self.root,
                "members": [member.to_payload() for member in self.members],
            }
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "subject": self.subject,
            "root": self.root,
            "member_count": self.member_count,
            "byte_count": self.byte_count,
            "digest": self.digest,
            "members": [member.to_payload() for member in self.members],
        }


@dataclass(frozen=True, slots=True)
class SchemaFrontier:
    """A declared frontier and, once recorded, its conserved baseline."""

    subjects: tuple[FrontierSubject, ...]
    baselines: tuple[RootBaseline, ...] = ()
    recorded_at: str | None = None
    notes: tuple[str, ...] = ()

    @property
    def declaration_digest(self) -> str:
        return hash_payload({"subjects": [subject.to_payload() for subject in self.subjects]})

    @property
    def baseline_digest(self) -> str | None:
        """Aggregate digest over the declaration and every recorded root.

        ``None`` until the frontier has been recorded. Read back from the file
        it is a pure function of stored bytes, so repeated reads conserve it.
        """
        if not self.baselines:
            return None
        return hash_payload(
            {
                "declaration": self.declaration_digest,
                "roots": [
                    {"subject": item.subject, "root": item.root, "digest": item.digest}
                    for item in sorted(self.baselines, key=lambda item: (item.subject, item.root))
                ],
            }
        )

    def subject(self, token: str) -> FrontierSubject | None:
        return next((item for item in self.subjects if item.subject == token), None)

    def baseline_for(self, subject: str, root: str) -> RootBaseline | None:
        return next((item for item in self.baselines if item.subject == subject and item.root == root), None)

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": FRONTIER_SCHEMA,
            "declaration_digest": self.declaration_digest,
            "subjects": [subject.to_payload() for subject in self.subjects],
        }
        if self.notes:
            payload["notes"] = list(self.notes)
        if self.baselines:
            payload["recorded_at"] = self.recorded_at
            payload["baseline_digest"] = self.baseline_digest
            payload["baselines"] = [
                item.to_payload() for item in sorted(self.baselines, key=lambda item: (item.subject, item.root))
            ]
        return payload


@dataclass(frozen=True, slots=True)
class FrontierFinding:
    """One divergence between the declared baseline and the live roots."""

    kind: str
    subject: str
    root: str
    detail: str
    member: str | None = None
    #: ``error`` fails the check. ``notice`` records expected drift on a root
    #: declared ``append``: the baseline no longer describes the live tree, but
    #: nothing has narrowed.
    severity: str = "error"

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "kind": self.kind,
            "subject": self.subject,
            "root": self.root,
            "detail": self.detail,
            "severity": self.severity,
        }
        if self.member is not None:
            payload["member"] = self.member
        return payload


@dataclass(frozen=True, slots=True)
class FrontierCheck:
    """The outcome of checking a recorded frontier against the live roots."""

    baseline_digest: str | None
    declaration_digest: str
    findings: tuple[FrontierFinding, ...]
    checked_roots: int
    checked_members: int
    content_verified: bool

    @property
    def errors(self) -> tuple[FrontierFinding, ...]:
        return tuple(item for item in self.findings if item.severity == "error")

    @property
    def notices(self) -> tuple[FrontierFinding, ...]:
        return tuple(item for item in self.findings if item.severity != "error")

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_payload(self) -> dict[str, object]:
        return {
            "kind": "polylogue.schema-source-frontier-check",
            "ok": self.ok,
            "declaration_digest": self.declaration_digest,
            "baseline_digest": self.baseline_digest,
            "checked_roots": self.checked_roots,
            "checked_members": self.checked_members,
            "content_verified": self.content_verified,
            "findings": [finding.to_payload() for finding in self.findings],
        }


def frontier_path(explicit: Path | None = None) -> Path:
    """Resolve the declared frontier location.

    Precedence: an explicit path, then ``POLYLOGUE_SCHEMA_FRONTIER``, then
    ``<archive root>/schema-inference/frontier.json`` -- beside the other
    schema-inference operator state, so the declaration follows the archive the
    generation run is bound to.
    """
    if explicit is not None:
        return explicit.expanduser()
    raw = os.environ.get(FRONTIER_ENV_VAR, "").strip()
    if raw:
        return Path(raw).expanduser()
    from polylogue.paths import archive_root

    return archive_root() / "schema-inference" / FRONTIER_FILENAME


def _exclusions_from_payload(payload: object) -> tuple[FrontierExclusion, ...]:
    if payload is None:
        return ()
    if not isinstance(payload, list):
        raise SchemaFrontierError("root exclusions must be a list")
    rows: list[FrontierExclusion] = []
    for item in payload:
        if not isinstance(item, dict):
            raise SchemaFrontierError("each exclusion must be an object")
        pattern = item.get("pattern")
        reason = item.get("reason")
        if not isinstance(pattern, str) or not isinstance(reason, str):
            raise SchemaFrontierError("each exclusion needs a pattern and a reason")
        owner = item.get("owner")
        rows.append(FrontierExclusion(pattern, reason, owner if isinstance(owner, str) else None))
    return tuple(rows)


def _root_from_payload(payload: object) -> FrontierRoot:
    if not isinstance(payload, dict):
        raise SchemaFrontierError("each declared root must be an object")
    path = payload.get("path")
    scope = payload.get("scope")
    if not isinstance(path, str) or not path:
        raise SchemaFrontierError("each declared root needs a path")
    if not isinstance(scope, str) or not scope:
        raise SchemaFrontierError(f"declared root {path} needs a scope")
    zero_material = payload.get("zero_material_reason")
    admit = payload.get("admit")
    if admit is not None and not (isinstance(admit, list) and all(isinstance(item, str) for item in admit)):
        raise SchemaFrontierError(f"declared root {path} has a malformed admit list")
    mutability = payload.get("mutability", "frozen")
    if not isinstance(mutability, str):
        raise SchemaFrontierError(f"declared root {path} has a malformed mutability")
    return FrontierRoot(
        path=Path(path).expanduser(),
        scope=scope,
        exclusions=_exclusions_from_payload(payload.get("exclusions")),
        admit=tuple(admit) if isinstance(admit, list) else (),
        mutability=mutability,
        zero_material_reason=zero_material if isinstance(zero_material, str) else None,
    )


def _member_from_payload(payload: object) -> FrontierMember:
    if not isinstance(payload, dict):
        raise SchemaFrontierError("each baseline member must be an object")
    relative = payload.get("relative")
    byte_count = payload.get("byte_count")
    sha256 = payload.get("sha256")
    if not isinstance(relative, str) or not isinstance(byte_count, int) or not isinstance(sha256, str):
        raise SchemaFrontierError("each baseline member needs relative, byte_count and sha256")
    return FrontierMember(relative, byte_count, sha256)


def frontier_from_payload(payload: object) -> SchemaFrontier:
    """Build a frontier from its declared JSON payload."""
    if not isinstance(payload, dict):
        raise SchemaFrontierError("the frontier document must be an object")
    declared_schema = payload.get("schema")
    if declared_schema != FRONTIER_SCHEMA:
        raise SchemaFrontierError(f"unsupported frontier schema: {declared_schema!r}")
    subjects_payload = payload.get("subjects")
    if not isinstance(subjects_payload, list) or not subjects_payload:
        raise SchemaFrontierError("the frontier declares no subjects")
    subjects: list[FrontierSubject] = []
    for item in subjects_payload:
        if not isinstance(item, dict):
            raise SchemaFrontierError("each subject must be an object")
        token = item.get("subject")
        roots_payload = item.get("roots")
        if not isinstance(token, str) or not token:
            raise SchemaFrontierError("each subject needs a token")
        if not isinstance(roots_payload, list) or not roots_payload:
            raise SchemaFrontierError(f"subject {token} declares no roots")
        excluded = inference_exclusion_reason(token)
        if excluded is not None:
            # A subject declared outside the inference denominator cannot carry
            # a declared root or a recorded baseline: its denominator is zero by
            # declaration, and a recorded member list would be exactly the
            # "eligible material we then refused" the exclusion denies exists.
            raise SchemaFrontierError(
                f"subject {token} is declared outside the schema-inference denominator and must not "
                f"declare source roots: {excluded}"
            )
        subjects.append(FrontierSubject(token, tuple(_root_from_payload(row) for row in roots_payload)))
    baselines_payload = payload.get("baselines")
    baselines: list[RootBaseline] = []
    if isinstance(baselines_payload, list):
        for item in baselines_payload:
            if not isinstance(item, dict):
                raise SchemaFrontierError("each baseline must be an object")
            subject_token = item.get("subject")
            root = item.get("root")
            members_payload = item.get("members")
            if not isinstance(subject_token, str) or not isinstance(root, str):
                raise SchemaFrontierError("each baseline needs a subject and a root")
            excluded_baseline = inference_exclusion_reason(subject_token)
            if excluded_baseline is not None:
                raise SchemaFrontierError(
                    f"subject {subject_token} is declared outside the schema-inference denominator and must not "
                    f"carry a recorded baseline: {excluded_baseline}"
                )
            if not isinstance(members_payload, list):
                raise SchemaFrontierError(f"baseline {subject_token}:{root} needs a member list")
            members = tuple(_member_from_payload(row) for row in members_payload)
            baselines.append(
                RootBaseline(
                    subject=subject_token,
                    root=root,
                    member_count=len(members),
                    byte_count=sum(member.byte_count for member in members),
                    members=members,
                )
            )
    recorded_at = payload.get("recorded_at")
    notes = payload.get("notes")
    frontier = SchemaFrontier(
        subjects=tuple(subjects),
        baselines=tuple(baselines),
        recorded_at=recorded_at if isinstance(recorded_at, str) else None,
        notes=tuple(note for note in notes if isinstance(note, str)) if isinstance(notes, list) else (),
    )
    stored_digest = payload.get("baseline_digest")
    if isinstance(stored_digest, str) and frontier.baseline_digest != stored_digest:
        raise SchemaFrontierError(
            "the recorded baseline digest does not match the recorded declaration and members; "
            "the frontier document was edited without re-recording"
        )
    return frontier


def load_frontier(path: Path | None = None) -> SchemaFrontier:
    """Load the declared frontier, refusing a missing or malformed document."""
    resolved = frontier_path(path)
    try:
        raw = resolved.read_text(encoding="utf-8")
    except OSError as exc:
        raise SchemaFrontierError(f"no declared schema source frontier at {resolved}") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SchemaFrontierError(f"the frontier at {resolved} is not valid JSON: {exc}") from exc
    return frontier_from_payload(payload)


def write_frontier(frontier: SchemaFrontier, path: Path | None = None) -> Path:
    """Write the frontier document, creating its directory."""
    resolved = frontier_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(frontier.to_payload(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return resolved


def frontier_source_inputs(frontier: SchemaFrontier, subject: str) -> tuple[SchemaSourceInput, ...]:
    """Return the declared generation inputs for one schema subject.

    A root that declares ``admit`` or ``exclusions`` is not handed to
    generation as a directory: the inference route has no notion of the
    declared denominator, so it would read every member of a shared drop root
    under this subject. Such a root is expanded into its recorded members, one
    explicit file input each, which binds generation to exactly the baseline
    that was recorded and checked. An unrestricted root is passed whole.
    """
    declared = frontier.subject(subject)
    if declared is None:
        raise SchemaFrontierError(f"the frontier declares no roots for subject {subject}")
    inputs: list[SchemaSourceInput] = []
    for root in declared.roots:
        if not root.admit and not root.exclusions:
            inputs.append(SchemaSourceInput(provider=subject, root=root.path))
            continue
        baseline = frontier.baseline_for(subject, str(root.path))
        if baseline is None:
            raise SchemaFrontierError(
                f"root {root.path} restricts its denominator but has no recorded baseline to expand"
            )
        inputs.extend(
            SchemaSourceInput(provider=subject, root=root.path / member.relative) for member in baseline.members
        )
    return tuple(inputs)


def _relative_member(root: FrontierRoot, path: Path) -> str:
    if root.path.is_file():
        return path.name
    try:
        return path.relative_to(root.path).as_posix()
    except ValueError:
        return path.name


def observe_root(subject: str, root: FrontierRoot, *, content_digest: bool) -> RootBaseline:
    """Inventory one declared root through the production admission route.

    Raises :class:`SchemaFrontierError` when the root cannot be resolved -- a
    moved or deleted root is a refusal, never an empty member list.
    """
    try:
        candidates = inventory_schema_sources((SchemaSourceInput(provider=subject, root=root.path),))
    except SourceInferenceError as exc:
        raise SchemaFrontierError(f"declared root is unresolvable: {root.path}") from exc
    members: list[FrontierMember] = []
    byte_count = 0
    for candidate in candidates:
        relative = _relative_member(root, candidate.path)
        if not root.admits(relative):
            continue
        try:
            size = candidate.path.stat().st_size
        except OSError as exc:
            raise SchemaFrontierError(f"declared member is unreadable: {candidate.path}") from exc
        digest = hash_file(candidate.path) if content_digest else ""
        members.append(FrontierMember(relative, size, digest))
        byte_count += size
    members.sort(key=lambda member: member.relative)
    return RootBaseline(
        subject=subject,
        root=str(root.path),
        member_count=len(members),
        byte_count=byte_count,
        members=tuple(members),
    )


def record_frontier(frontier: SchemaFrontier, *, subjects: Sequence[str] | None = None) -> SchemaFrontier:
    """Record the baseline for the declared subjects, hashing every member."""
    selected = set(subjects) if subjects is not None else None
    retained = [item for item in frontier.baselines if selected is not None and item.subject not in selected]
    recorded: list[RootBaseline] = list(retained)
    for subject in frontier.subjects:
        if selected is not None and subject.subject not in selected:
            continue
        for root in subject.roots:
            recorded.append(observe_root(subject.subject, root, content_digest=True))
    return replace(
        frontier,
        baselines=tuple(sorted(recorded, key=lambda item: (item.subject, item.root))),
        recorded_at=datetime.now(tz=timezone.utc).isoformat(),
    )


def check_frontier(
    frontier: SchemaFrontier,
    *,
    verify_content: bool = False,
    subjects: Iterable[str] | None = None,
) -> FrontierCheck:
    """Compare the live roots against the recorded baseline.

    Structural checking (the default) stats every admitted member and is cheap
    enough to gate on. ``verify_content`` additionally re-hashes every member's
    bytes and proves the recorded content digests.
    """
    selected = set(subjects) if subjects is not None else None
    findings: list[FrontierFinding] = []
    checked_roots = 0
    checked_members = 0
    if not frontier.baselines:
        findings.append(
            FrontierFinding(
                "baseline_not_recorded",
                "-",
                "-",
                "the frontier has never been recorded; there is no conserved baseline to conserve",
            )
        )
    for subject in frontier.subjects:
        if selected is not None and subject.subject not in selected:
            continue
        for root in subject.roots:
            baseline = frontier.baseline_for(subject.subject, str(root.path))
            if baseline is None:
                findings.append(
                    FrontierFinding(
                        "root_not_recorded",
                        subject.subject,
                        str(root.path),
                        "the declared root has no recorded baseline",
                    )
                )
                continue
            checked_roots += 1
            try:
                observed = observe_root(subject.subject, root, content_digest=verify_content)
            except SchemaFrontierError as exc:
                findings.append(FrontierFinding("root_unresolvable", subject.subject, str(root.path), str(exc)))
                continue
            checked_members += observed.member_count
            if observed.member_count == 0 and root.zero_material_reason is None:
                findings.append(
                    FrontierFinding(
                        "root_empty",
                        subject.subject,
                        str(root.path),
                        "the declared root admits no members and declares no zero-material reason",
                    )
                )
            recorded_by_relative = {member.relative: member for member in baseline.members}
            observed_by_relative = {member.relative: member for member in observed.members}
            for relative in sorted(set(recorded_by_relative) - set(observed_by_relative)):
                findings.append(
                    FrontierFinding(
                        "member_missing",
                        subject.subject,
                        str(root.path),
                        "a recorded member is no longer admitted by the declared root",
                        member=relative,
                    )
                )
            live = root.mutability == "append"
            for relative in sorted(set(observed_by_relative) - set(recorded_by_relative)):
                findings.append(
                    FrontierFinding(
                        "member_added",
                        subject.subject,
                        str(root.path),
                        "the declared root admits a member the baseline does not record",
                        member=relative,
                        severity="notice" if live else "error",
                    )
                )
            for relative in sorted(set(recorded_by_relative) & set(observed_by_relative)):
                recorded_member = recorded_by_relative[relative]
                observed_member = observed_by_relative[relative]
                if recorded_member.byte_count != observed_member.byte_count:
                    grew = observed_member.byte_count > recorded_member.byte_count
                    findings.append(
                        FrontierFinding(
                            "member_grew" if live and grew else "member_changed",
                            subject.subject,
                            str(root.path),
                            f"byte count moved from {recorded_member.byte_count} to {observed_member.byte_count}",
                            member=relative,
                            severity="notice" if live and grew else "error",
                        )
                    )
                elif verify_content and recorded_member.sha256 != observed_member.sha256:
                    findings.append(
                        FrontierFinding(
                            "member_content_changed",
                            subject.subject,
                            str(root.path),
                            "the member's content digest does not match the recorded baseline",
                            member=relative,
                        )
                    )
    return FrontierCheck(
        baseline_digest=frontier.baseline_digest,
        declaration_digest=frontier.declaration_digest,
        findings=tuple(findings),
        checked_roots=checked_roots,
        checked_members=checked_members,
        content_verified=verify_content,
    )


__all__ = [
    "FRONTIER_ENV_VAR",
    "FRONTIER_FILENAME",
    "FRONTIER_SCHEMA",
    "FrontierCheck",
    "FrontierExclusion",
    "FrontierFinding",
    "FrontierMember",
    "FrontierRoot",
    "FrontierSubject",
    "RootBaseline",
    "SchemaFrontier",
    "SchemaFrontierError",
    "check_frontier",
    "frontier_from_payload",
    "frontier_path",
    "frontier_source_inputs",
    "load_frontier",
    "observe_root",
    "record_frontier",
    "write_frontier",
]
