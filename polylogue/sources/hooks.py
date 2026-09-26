"""Append-only hook-event carriers for Claude Code, Codex, and Hermes.

Hook commands must return promptly and cannot rely on the archive daemon being
up. They therefore append exactly one newline-terminated line -- the validated
envelope, compact JSON -- to their own carrier under
``carriers/<provider>/<UTC day>/<pid>.ndjson``, with a single ``O_APPEND``
write and no fsync. The archive then acquires that carrier like any other
append-only source and materializes its events out of the retained bytes.

That ordering is the whole design. The retired shape was one atomically
renamed JSON file per event, drained one commit at a time: measured at 67 ms
and four fsyncs per event against a 690,297-event backlog, it projected to
roughly 13 hours of serial writer time for hooks alone -- more than the entire
session corpus (polylogue-xa33k). The cost was never the parsing; it was
paying blob publication, a ``synchronous=FULL`` reservation commit and two
directory fsyncs per *event*. A carrier pays them per *carrier revision*.

One file per producer process per day is what makes a bare ``O_APPEND`` write
sufficient. The earlier objection to an append-only journal was real -- a
concurrent append is only atomic below ``PIPE_BUF`` and hook payloads (tool
output previews) are not reliably under that bound -- but it assumed one
shared journal. Two harness processes never share a carrier, so there is no
concurrent appender to interleave with, and a short write can only ever be
completed by the process that started it.

The write side -- envelope validation and the carrier append -- lives in
:mod:`polylogue.sources.hook_producer`, which stays runnable without this
package so the installed hook command pays no package import. This module
imports it rather than keeping a second copy that could read a carrier
differently than the producer wrote it.

Hermes support (fs1.7) reuses this exact mechanism rather than inventing a
parallel capture surface: Hermes lifecycle hooks are best-effort in the same
way Claude Code/Codex hooks are. The one Hermes-specific addition is a payload
hygiene guard (``reject_duplicated_transcript``) enforcing that lifecycle
events carry ids/hashes/timings/outcomes, never a second copy of message text.
See ``polylogue.sources.parsers.hermes_lifecycle`` for the event-type taxonomy
and ``docs/design/hermes-archival-export-contract.md`` for the durability
semantics this capture exists to serve.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from polylogue.logging import emit, get_logger
from polylogue.sources.hook_producer import (
    CARRIERS_DIRNAME as _CARRIERS_DIRNAME,
)
from polylogue.sources.hook_producer import (
    HookSpoolRecordError,
    append_event,
)
from polylogue.sources.hook_producer import (
    validated_record as _validated_record,
)

# Hook producers run in fresh interpreters. Keep their imports independent of
# archive DDL; drain functions load archive dependencies locally.

if TYPE_CHECKING:
    from polylogue.storage.sqlite.archive_tiers.source_write import CarrierHookEvent

logger = get_logger(__name__)

_ORIGIN_TOKEN_BY_PROVIDER: dict[str, str] = {
    "claude-code": "claude-code-session",
    "codex": "codex-session",
    "hermes": "hermes-session",
}
_ACKNOWLEDGED_DIRNAME = "acknowledged"


class HookSpoolTopologyError(ValueError):
    """The declared hook spool topology is not safe to seal."""


@dataclass(frozen=True, slots=True)
class HookSpoolSourceSpec:
    """One stable logical identity for a hook spool root."""

    source_id: str
    role: Literal["primary-writable", "legacy-read-only"]
    root: Path


def hook_spool_sources(
    *,
    primary_root: Path | None = None,
    legacy_roots: tuple[Path, ...] | list[Path] | None = None,
) -> tuple[HookSpoolSourceSpec, ...]:
    """Return the canonical primary plus finite legacy hook-spool topology."""
    from polylogue.paths import data_home, hooks_sidecar_dir

    primary = (primary_root or hooks_sidecar_dir()).expanduser().resolve()
    if legacy_roots is not None:
        configured_legacy: tuple[Path, ...] = tuple(Path(root) for root in legacy_roots)
    else:
        # polylogue-e9y76: the global XDG spool is an *implicit* legacy root
        # only for the archive that owns it. ``hooks_sidecar_dir`` tracks
        # ``archive_root()``, so for any archive rooted elsewhere this default
        # silently admitted the operator's global hook spool with no opt-in and
        # mixed evidence across archives. An archive that wants it says so
        # through ``legacy_roots``.
        default_legacy = (data_home() / "hooks").expanduser().resolve()
        if primary == default_legacy:
            configured_legacy = (default_legacy,)
        else:
            configured_legacy = ()
            emit(
                "source.hook_spool.implicit_legacy_root_skipped",
                outcome="ok",
                path=str(default_legacy),
                root=str(primary),
                reason="primary_not_default_xdg_hook_spool",
            )
    sources = [HookSpoolSourceSpec("primary-hook-spool", "primary-writable", primary)]
    for index, root_value in enumerate(configured_legacy):
        root = Path(root_value).expanduser().resolve()
        if root == primary or root in primary.parents or primary in root.parents:
            if legacy_roots is not None:
                sources.append(HookSpoolSourceSpec(f"legacy-hook-spool-{index}", "legacy-read-only", root))
            continue
        sources.append(HookSpoolSourceSpec(f"legacy-hook-spool-{index}", "legacy-read-only", root))
    return validate_hook_spool_topology(sources)


def validate_hook_spool_topology(
    sources: tuple[HookSpoolSourceSpec, ...] | list[HookSpoolSourceSpec],
    *,
    sealed_primary_identity: tuple[int, int] | None = None,
    require_existing: bool = False,
) -> tuple[HookSpoolSourceSpec, ...]:
    """Validate the finite declared topology before acquisition or sealing."""

    from polylogue.maintenance.source_manifest_continuity import (
        SourceContinuityError,
        SourceDeclaration,
        SourceRole,
        canonical_source_declarations,
    )

    declared = tuple(sources)
    if not declared or sum(spec.role == "primary-writable" for spec in declared) != 1:
        raise HookSpoolTopologyError("topology requires exactly one primary-writable source")
    if any(spec.role not in {"primary-writable", "legacy-read-only"} for spec in declared):
        raise HookSpoolTopologyError("hook spool source role must be primary-writable or legacy-read-only")
    try:
        canonical = canonical_source_declarations(
            configured=(
                SourceDeclaration(spec.source_id, SourceRole.SPOOL, spec.root, mutable=True) for spec in declared
            )
        )
    except SourceContinuityError as exc:
        raise HookSpoolTopologyError(str(exc)) from exc
    ids = [spec.source_id for spec in declared]
    roots = [declaration.root.resolve() for declaration in canonical]
    if any(not source_id.strip() for source_id in ids):
        raise HookSpoolTopologyError("source_id must be nonempty")
    if len(set(ids)) != len(ids):
        raise HookSpoolTopologyError("duplicate source_id in hook spool topology")
    if len(set(roots)) != len(roots):
        raise HookSpoolTopologyError("duplicate root path in hook spool topology")
    for index, root in enumerate(roots):
        if require_existing and not root.exists():
            raise HookSpoolTopologyError(f"required hook spool root vanished: {root}")
        if any(root in other.parents or other in root.parents for other in roots[index + 1 :]):
            raise HookSpoolTopologyError("nested hook spool roots are not allowed")
    primary = next(spec for spec in declared if spec.role == "primary-writable")
    if primary.role != "primary-writable":  # defensive: keeps the type contract explicit
        raise HookSpoolTopologyError("legacy source cannot be writable")
    if sealed_primary_identity is not None:
        try:
            identity = (primary.root.stat().st_dev, primary.root.stat().st_ino)
        except OSError as exc:
            raise HookSpoolTopologyError(f"required hook spool root vanished: {primary.root}") from exc
        if identity != sealed_primary_identity:
            raise HookSpoolTopologyError("primary hook spool identity changed after seal")
    return tuple(
        HookSpoolSourceSpec(spec.source_id, spec.role, root) for spec, root in zip(declared, roots, strict=True)
    )


def hook_spool_root() -> Path:
    """Resolve the hook spool root shared by producer and daemon.

    Derives from ``hooks_sidecar_dir()``, which is itself scoped under the
    resolved archive root (polylogue-o7hx). There is no separate ad hoc env
    override at this layer: ``POLYLOGUE_ARCHIVE_ROOT`` is the one knob that
    already has to be set to isolate a scratch/test daemon, so a second,
    hook-specific override was a manual escape hatch operators had to
    remember on top of it -- and repeatedly didn't.
    """

    from polylogue.paths import hooks_sidecar_dir

    return hooks_sidecar_dir()


def hook_carrier_dir(root: Path | None = None) -> Path:
    """The carrier tree hook producers append to under one spool root."""

    return (root or hook_spool_root()) / _CARRIERS_DIRNAME


def hook_carrier_provider_dir(provider: str, root: Path | None = None) -> Path:
    """The one watched directory holding ``provider``'s carriers.

    Carriers are partitioned by harness because acquisition is
    provider-scoped: the watch source, the origin-spec artifact rule and the
    materialized ``origin`` all follow the directory, so a carrier never has
    to be opened to learn which provider wrote it.
    """

    return hook_carrier_dir(root) / provider


def append_hook_event(
    *,
    event_type: str,
    session_id: str,
    provider: str,
    timestamp: str,
    payload: dict[str, object],
    root: Path | None = None,
    event_id: str | None = None,
) -> Path:
    """Append one hook event to this process's carrier."""

    return Path(
        append_event(
            event_type=event_type,
            session_id=session_id,
            provider=provider,
            timestamp=timestamp,
            payload=payload,
            root=str(hook_spool_root() if root is None else root),
            event_id=event_id,
        )
    )


def validated_hook_record(value: dict[str, object]) -> dict[str, object]:
    """Normalize one envelope exactly as materialization does.

    The record the archive stores is not the carrier line's bytes:
    ``observed_at_ms`` is derived here and serialization is independent on
    both sides. Any comparison against stored hook material must go through
    this route rather than compare bytes.
    """

    return _validated_record(value)


@dataclass(frozen=True, slots=True)
class CarrierLine:
    """One decoded carrier line and the byte offset it starts at."""

    byte_offset: int
    line_bytes: int
    record: dict[str, object]


@dataclass(frozen=True, slots=True)
class CarrierRefusal:
    """One carrier line materialization will not admit, and why."""

    byte_offset: int
    reason: str


def read_hook_carrier(payload: bytes) -> tuple[tuple[CarrierLine, ...], tuple[CarrierRefusal, ...]]:
    """Decode one carrier's retained bytes into admissible lines and refusals.

    A refusal is counted and carried back, never dropped and never fatal: one
    malformed line in a carrier must not strand every well-formed event
    beside it, and it must not make the carrier eternally unmaterialized
    either. The offset is the coordinate the source tier keys the event by, so
    it is computed from the byte stream rather than from a line ordinal --
    lines are fixed in place once written, ordinals are not.

    A trailing partial line (no terminating newline) is deliberately *not*
    returned: the producer appends whole lines, so an unterminated tail is a
    write in flight, and the next acquisition of the same carrier will see it
    complete.
    """

    lines: list[CarrierLine] = []
    refusals: list[CarrierRefusal] = []
    offset = 0
    for raw in payload.splitlines(keepends=True):
        length = len(raw)
        start = offset
        offset += length
        if not raw.endswith(b"\n"):
            break
        body = raw.strip()
        if not body:
            continue
        try:
            value = json.loads(body)
            if not isinstance(value, dict):
                raise HookSpoolRecordError("carrier line must be a JSON object")
            record = _validated_record(value)
        except (UnicodeDecodeError, json.JSONDecodeError, HookSpoolRecordError) as exc:
            refusals.append(CarrierRefusal(start, f"{type(exc).__name__}: {exc}"))
            continue
        lines.append(CarrierLine(start, length, record))
    return tuple(lines), tuple(refusals)


def find_carrier_event(root: Path | None, event_id: str) -> tuple[Path, dict[str, object]] | None:
    """Locate one event by id anywhere in a spool root's carriers.

    Only ``carriers/`` counts as present: that is the one tree acquisition
    watches. An ``acknowledged/`` file is a record of what the one-shot legacy
    fold already folded, which nothing acquires; treating one as the
    destination's copy would report an event restored while leaving it
    unacquirable.
    """

    for carrier in sorted(hook_carrier_dir(root).rglob("*.ndjson")):
        if not carrier.is_file():
            continue
        try:
            lines, _refusals = read_hook_carrier(carrier.read_bytes())
        except OSError:
            continue
        for line in lines:
            if line.record.get("event_id") == event_id:
                return carrier, line.record
    return None


def hook_event_origin(provider_token: str) -> object:
    """Map a hook provider wire token onto its public source origin."""

    from polylogue.core.enums import Origin

    try:
        origin_token = _ORIGIN_TOKEN_BY_PROVIDER[provider_token]
    except KeyError as exc:
        # ``_validated_record`` already rejects any provider outside
        # ``SUPPORTED_PROVIDERS`` before a record reaches this point, so this
        # should be unreachable -- but silently defaulting an unrecognized
        # provider to "codex-session" would misclassify genuinely-unknown
        # providers as Codex if that upstream invariant ever drifts. Raise
        # instead of guessing.
        raise HookSpoolRecordError(f"no origin mapping for hook provider: {provider_token!r}") from exc
    return Origin.from_string(origin_token)


def carrier_hook_events(
    lines: Sequence[CarrierLine],
    *,
    source_path: str,
    base_offset: int = 0,
) -> tuple[CarrierHookEvent, ...]:
    """Build the source-tier rows one carrier's decoded lines materialize into.

    A hook event is evidence WITHIN a session, keyed to it by
    ``session_native_id`` -- never a session of its own. Persisting one as a
    ``raw_sessions`` row (as an earlier route did) minted an empty standalone
    session per hook and inflated the archive with tens of thousands of
    content-less session shells (polylogue-31r1). ``origin`` and
    ``session_native_id`` travel on every row so session excision reaches hook
    evidence by construction (polylogue-14ucm).

    ``base_offset`` is where these bytes begin in the whole carrier. An append
    revision retains only the delta, so its own offsets start at zero; adding
    the revision's start offset is what keeps every coordinate absolute in the
    file. Without it the first event of every append would claim the
    coordinate of the carrier's very first event.
    """

    from polylogue.storage.sqlite.archive_tiers.source_write import ArchiveHookEvent, CarrierHookEvent

    built: list[CarrierHookEvent] = []
    for line in lines:
        record = line.record
        observed_at_ms = record["observed_at_ms"]
        if not isinstance(observed_at_ms, int):
            raise HookSpoolRecordError("hook carrier line has an invalid observed timestamp")
        built.append(
            CarrierHookEvent(
                byte_offset=base_offset + line.byte_offset,
                line_bytes=line.line_bytes,
                event=ArchiveHookEvent(
                    hook_event_id=f"hook:{record['event_id']}",
                    origin=hook_event_origin(str(record["provider"])),  # type: ignore[arg-type]
                    source_path=source_path,
                    event_type=str(record["event_type"]),
                    payload=record,
                    observed_at_ms=observed_at_ms,
                    native_id=f"{record['session_id']}:{record['event_type']}:{record['event_id']}",
                    session_native_id=str(record["session_id"]),
                ),
            )
        )
    return tuple(built)


__all__ = [
    "CarrierLine",
    "CarrierRefusal",
    "HookSpoolRecordError",
    "HookSpoolSourceSpec",
    "HookSpoolTopologyError",
    "append_hook_event",
    "carrier_hook_events",
    "find_carrier_event",
    "hook_carrier_dir",
    "hook_carrier_provider_dir",
    "hook_event_origin",
    "hook_spool_root",
    "hook_spool_sources",
    "read_hook_carrier",
    "validate_hook_spool_topology",
    "validated_hook_record",
]
