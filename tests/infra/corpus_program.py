"""Typed, composable corpus programs for production-route property tests.

The program layer owns only synthetic evidence and scheduling.  It does not
know how to write an archive.  :class:`ProductionCorpusRuntime` delegates
those effects to the existing acquisition, parsing, convergence, hook, and
rebuild seams, which keeps this harness from becoming a second archive
engine.
"""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Protocol, cast

from polylogue.core.json import JSONDocument, JSONValue, is_json_document

if TYPE_CHECKING:
    from polylogue.maintenance.rebuild_index import RebuildIndexReceipt
    from polylogue.pipeline.services.parsing_models import ParseResult


class CorpusProgramError(ValueError):
    """Invalid corpus program or artifact reference."""


class CorpusRuntimeCrashedError(RuntimeError):
    """An operation attempted to use a crashed runtime before Restart."""


CorpusRuntimeCrashed = CorpusRuntimeCrashedError


def _validate_identifier(value: str, *, field_name: str) -> str:
    if not value or value.strip() != value or any(char.isspace() for char in value):
        raise CorpusProgramError(f"{field_name} must be a non-empty token")
    return value


def _validate_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise CorpusProgramError("source_path must be a non-empty relative path without '..'")
    return path.as_posix()


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _b64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _unb64(value: object, *, field_name: str) -> bytes:
    if not isinstance(value, str):
        raise CorpusProgramError(f"{field_name} must be base64 text")
    try:
        return base64.b64decode(value.encode("ascii"), validate=True)
    except (ValueError, UnicodeEncodeError) as exc:
        raise CorpusProgramError(f"{field_name} is not valid base64") from exc


@dataclass(frozen=True, slots=True)
class AttachmentArtifact:
    """Synthetic attachment bytes carried by a corpus artifact."""

    attachment_id: str
    name: str
    mime_type: str = "application/octet-stream"
    payload: bytes = b""

    def __post_init__(self) -> None:
        _validate_identifier(self.attachment_id, field_name="attachment_id")
        if not self.name:
            raise CorpusProgramError("attachment name must be non-empty")

    def to_dict(self) -> JSONDocument:
        return {
            "attachment_id": self.attachment_id,
            "name": self.name,
            "mime_type": self.mime_type,
            "payload_b64": _b64(self.payload),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> AttachmentArtifact:
        return cls(
            attachment_id=_required_string(payload, "attachment_id"),
            name=_required_string(payload, "name"),
            mime_type=_optional_string(payload, "mime_type") or "application/octet-stream",
            payload=_unb64(payload.get("payload_b64"), field_name="payload_b64"),
        )


@dataclass(frozen=True, slots=True)
class RawArtifact:
    """One deterministic source artifact before it reaches production ingest."""

    artifact_id: str
    payload: bytes
    source_name: str = "codex"
    source_path: str = "sources/artifact.jsonl"
    source_index: int = 0
    parent_artifact_id: str | None = None
    attachments: tuple[AttachmentArtifact, ...] = ()
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_identifier(self.artifact_id, field_name="artifact_id")
        _validate_relative_path(self.source_path)
        if self.source_index < 0:
            raise CorpusProgramError("source_index must be non-negative")
        if not is_json_document(dict(self.metadata)):
            raise CorpusProgramError("metadata must be JSON-compatible")

    def with_payload(self, payload: bytes) -> RawArtifact:
        return replace(self, payload=payload)

    def with_attachment(self, attachment: AttachmentArtifact) -> RawArtifact:
        if any(item.attachment_id == attachment.attachment_id for item in self.attachments):
            raise CorpusProgramError(f"attachment already exists: {attachment.attachment_id}")
        return replace(self, attachments=(*self.attachments, attachment))

    def to_dict(self) -> JSONDocument:
        return {
            "artifact_id": self.artifact_id,
            "payload_b64": _b64(self.payload),
            "source_name": self.source_name,
            "source_path": self.source_path,
            "source_index": self.source_index,
            "parent_artifact_id": self.parent_artifact_id,
            "attachments": [attachment.to_dict() for attachment in self.attachments],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> RawArtifact:
        attachments_value = payload.get("attachments", [])
        if not isinstance(attachments_value, list):
            raise CorpusProgramError("attachments must be a list")
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise CorpusProgramError("metadata must be an object")
        return cls(
            artifact_id=_required_string(payload, "artifact_id"),
            payload=_unb64(payload.get("payload_b64"), field_name="payload_b64"),
            source_name=_optional_string(payload, "source_name") or "codex",
            source_path=_optional_string(payload, "source_path") or "sources/artifact.jsonl",
            source_index=_required_int(payload, "source_index", default=0),
            parent_artifact_id=_optional_string(payload, "parent_artifact_id"),
            attachments=tuple(
                AttachmentArtifact.from_dict(cast(Mapping[str, object], item))
                for item in attachments_value
                if isinstance(item, Mapping)
            ),
            metadata=cast(Mapping[str, JSONValue], dict(metadata)),
        )


@dataclass(frozen=True, slots=True)
class HookArtifact:
    """A hook payload routed through the source-tier hook writer."""

    hook_event_id: str
    provider: str
    event_type: str
    session_native_id: str
    payload: bytes
    source_path: str = "hooks/events.jsonl"
    observed_at_ms: int = 1_735_689_600_000

    def __post_init__(self) -> None:
        for name, value in (
            ("hook_event_id", self.hook_event_id),
            ("provider", self.provider),
            ("event_type", self.event_type),
            ("session_native_id", self.session_native_id),
        ):
            _validate_identifier(value, field_name=name)
        _validate_relative_path(self.source_path)

    def to_dict(self) -> JSONDocument:
        return {
            "hook_event_id": self.hook_event_id,
            "provider": self.provider,
            "event_type": self.event_type,
            "session_native_id": self.session_native_id,
            "payload_b64": _b64(self.payload),
            "source_path": self.source_path,
            "observed_at_ms": self.observed_at_ms,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> HookArtifact:
        return cls(
            hook_event_id=_required_string(payload, "hook_event_id"),
            provider=_required_string(payload, "provider"),
            event_type=_required_string(payload, "event_type"),
            session_native_id=_required_string(payload, "session_native_id"),
            payload=_unb64(payload.get("payload_b64"), field_name="payload_b64"),
            source_path=_optional_string(payload, "source_path") or "hooks/events.jsonl",
            observed_at_ms=_required_int(payload, "observed_at_ms", default=1_735_689_600_000),
        )


@dataclass(frozen=True, slots=True)
class CorpusState:
    """Immutable state returned after each operation, making shrinking local."""

    artifacts: tuple[RawArtifact, ...] = ()
    hooks: tuple[HookArtifact, ...] = ()
    crashed: bool = False
    applied_operation_ids: tuple[str, ...] = ()

    def artifact(self, artifact_id: str) -> RawArtifact:
        for artifact in self.artifacts:
            if artifact.artifact_id == artifact_id:
                return artifact
        raise CorpusProgramError(f"unknown artifact: {artifact_id}")

    def add_artifact(self, artifact: RawArtifact) -> CorpusState:
        if any(item.artifact_id == artifact.artifact_id for item in self.artifacts):
            raise CorpusProgramError(f"artifact already exists: {artifact.artifact_id}")
        return replace(self, artifacts=(*self.artifacts, artifact))

    def replace_artifact(self, artifact: RawArtifact) -> CorpusState:
        if not any(item.artifact_id == artifact.artifact_id for item in self.artifacts):
            raise CorpusProgramError(f"unknown artifact: {artifact.artifact_id}")
        return replace(
            self,
            artifacts=tuple(artifact if item.artifact_id == artifact.artifact_id else item for item in self.artifacts),
        )

    def mark(
        self,
        operation_id: str,
        *,
        crashed: bool | None = None,
        hooks: tuple[HookArtifact, ...] | None = None,
    ) -> CorpusState:
        return replace(
            self,
            crashed=self.crashed if crashed is None else crashed,
            hooks=self.hooks if hooks is None else hooks,
            applied_operation_ids=(*self.applied_operation_ids, operation_id),
        )


class OperationKind(StrEnum):
    ACQUIRE = "Acquire"
    APPEND = "Append"
    REPLACE = "Replace"
    DUPLICATE = "Duplicate"
    FORK = "Fork"
    ATTACH = "Attach"
    EMIT_HOOK = "EmitHook"
    CRASH = "Crash"
    RESTART = "Restart"
    CONVERGE = "Converge"
    REBUILD = "Rebuild"
    PROMOTE = "Promote"


class CorpusRunner(Protocol):
    """Effects that a program may delegate to a production route."""

    def acquire(self, artifact: RawArtifact) -> object: ...

    def emit_hook(self, hook: HookArtifact) -> object: ...

    def crash(self) -> object: ...

    def restart(self) -> object: ...

    def converge(self) -> object: ...

    def rebuild(self) -> object: ...

    def promote(self) -> object: ...


@dataclass(frozen=True, slots=True)
class _Operation:
    operation_id: str
    kind: OperationKind = field(init=False)

    def __post_init__(self) -> None:
        _validate_identifier(self.operation_id, field_name="operation_id")

    def to_dict(self) -> JSONDocument:
        return {"operation_id": self.operation_id, "kind": self.kind.value}

    def apply(self, state: CorpusState, runner: CorpusRunner | None) -> CorpusState:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class Acquire(_Operation):
    artifact: RawArtifact
    kind: OperationKind = field(default=OperationKind.ACQUIRE, init=False)

    def apply(self, state: CorpusState, runner: CorpusRunner | None) -> CorpusState:
        next_state = (
            state.replace_artifact(self.artifact)
            if any(item.artifact_id == self.artifact.artifact_id for item in state.artifacts)
            else state.add_artifact(self.artifact)
        )
        if runner is not None:
            runner.acquire(self.artifact)
        return next_state.mark(self.operation_id)

    def to_dict(self) -> JSONDocument:
        return {**super().to_dict(), "artifact": self.artifact.to_dict()}


@dataclass(frozen=True, slots=True)
class Append(_Operation):
    artifact_id: str
    payload_delta: bytes
    kind: OperationKind = field(default=OperationKind.APPEND, init=False)

    def apply(self, state: CorpusState, runner: CorpusRunner | None) -> CorpusState:
        artifact = state.artifact(self.artifact_id).with_payload(
            state.artifact(self.artifact_id).payload + self.payload_delta
        )
        return state.replace_artifact(artifact).mark(self.operation_id)

    def to_dict(self) -> JSONDocument:
        return {**super().to_dict(), "artifact_id": self.artifact_id, "payload_delta_b64": _b64(self.payload_delta)}


@dataclass(frozen=True, slots=True)
class Replace(_Operation):
    artifact_id: str
    payload: bytes
    kind: OperationKind = field(default=OperationKind.REPLACE, init=False)

    def apply(self, state: CorpusState, runner: CorpusRunner | None) -> CorpusState:
        return state.replace_artifact(state.artifact(self.artifact_id).with_payload(self.payload)).mark(
            self.operation_id
        )

    def to_dict(self) -> JSONDocument:
        return {**super().to_dict(), "artifact_id": self.artifact_id, "payload_b64": _b64(self.payload)}


@dataclass(frozen=True, slots=True)
class Duplicate(_Operation):
    source_artifact_id: str
    new_artifact_id: str
    new_source_path: str | None = None
    kind: OperationKind = field(default=OperationKind.DUPLICATE, init=False)

    def apply(self, state: CorpusState, runner: CorpusRunner | None) -> CorpusState:
        source = state.artifact(self.source_artifact_id)
        copy = replace(
            source,
            artifact_id=self.new_artifact_id,
            source_path=self.new_source_path or f"duplicates/{self.new_artifact_id}.jsonl",
        )
        return state.add_artifact(copy).mark(self.operation_id)

    def to_dict(self) -> JSONDocument:
        return {
            **super().to_dict(),
            "source_artifact_id": self.source_artifact_id,
            "new_artifact_id": self.new_artifact_id,
            "new_source_path": self.new_source_path,
        }


def _fork_payload(payload: bytes, new_session_id: str, parent_session_id: str) -> bytes:
    """Update common session identity envelopes without parsing provider code."""
    lines = payload.splitlines(keepends=True)
    output: list[bytes] = []
    changed = False
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            output.append(line)
            continue
        if isinstance(value, dict):
            for container in (value, value.get("payload")):
                if not isinstance(container, dict):
                    continue
                for key in ("id", "sessionId", "session_id", "thread_id"):
                    if container.get(key) == parent_session_id:
                        container[key] = new_session_id
                        changed = True
                if container.get("type") == "session_meta":
                    container["parent_id"] = parent_session_id
            encoded = _canonical_json(value).encode("utf-8")
            output.append(encoded + (b"\n" if line.endswith(b"\n") else b""))
        else:
            output.append(line)
    return b"".join(output) if changed else payload


@dataclass(frozen=True, slots=True)
class Fork(_Operation):
    source_artifact_id: str
    new_artifact_id: str
    new_session_id: str
    kind: OperationKind = field(default=OperationKind.FORK, init=False)

    def apply(self, state: CorpusState, runner: CorpusRunner | None) -> CorpusState:
        source = state.artifact(self.source_artifact_id)
        parent_id = str(source.metadata.get("session_id", source.artifact_id))
        child = replace(
            source,
            artifact_id=self.new_artifact_id,
            source_path=f"forks/{self.new_artifact_id}.jsonl",
            parent_artifact_id=source.artifact_id,
            payload=_fork_payload(source.payload, self.new_session_id, parent_id),
            metadata={**source.metadata, "session_id": self.new_session_id, "parent_session_id": parent_id},
        )
        return state.add_artifact(child).mark(self.operation_id)

    def to_dict(self) -> JSONDocument:
        return {
            **super().to_dict(),
            "source_artifact_id": self.source_artifact_id,
            "new_artifact_id": self.new_artifact_id,
            "new_session_id": self.new_session_id,
        }


@dataclass(frozen=True, slots=True)
class Attach(_Operation):
    artifact_id: str
    attachment: AttachmentArtifact
    kind: OperationKind = field(default=OperationKind.ATTACH, init=False)

    def apply(self, state: CorpusState, runner: CorpusRunner | None) -> CorpusState:
        return state.replace_artifact(state.artifact(self.artifact_id).with_attachment(self.attachment)).mark(
            self.operation_id
        )

    def to_dict(self) -> JSONDocument:
        return {**super().to_dict(), "artifact_id": self.artifact_id, "attachment": self.attachment.to_dict()}


@dataclass(frozen=True, slots=True)
class EmitHook(_Operation):
    hook: HookArtifact
    kind: OperationKind = field(default=OperationKind.EMIT_HOOK, init=False)

    def apply(self, state: CorpusState, runner: CorpusRunner | None) -> CorpusState:
        if any(item.hook_event_id == self.hook.hook_event_id for item in state.hooks):
            raise CorpusProgramError(f"hook already exists: {self.hook.hook_event_id}")
        if runner is not None:
            runner.emit_hook(self.hook)
        return state.mark(self.operation_id, hooks=(*state.hooks, self.hook))

    def to_dict(self) -> JSONDocument:
        return {**super().to_dict(), "hook": self.hook.to_dict()}


@dataclass(frozen=True, slots=True)
class Crash(_Operation):
    kind: OperationKind = field(default=OperationKind.CRASH, init=False)

    def apply(self, state: CorpusState, runner: CorpusRunner | None) -> CorpusState:
        if runner is not None:
            runner.crash()
        return state.mark(self.operation_id, crashed=True)


@dataclass(frozen=True, slots=True)
class Restart(_Operation):
    kind: OperationKind = field(default=OperationKind.RESTART, init=False)

    def apply(self, state: CorpusState, runner: CorpusRunner | None) -> CorpusState:
        if runner is not None:
            runner.restart()
        return state.mark(self.operation_id, crashed=False)


@dataclass(frozen=True, slots=True)
class Converge(_Operation):
    kind: OperationKind = field(default=OperationKind.CONVERGE, init=False)

    def apply(self, state: CorpusState, runner: CorpusRunner | None) -> CorpusState:
        if runner is not None:
            runner.converge()
        return state.mark(self.operation_id)


@dataclass(frozen=True, slots=True)
class Rebuild(_Operation):
    kind: OperationKind = field(default=OperationKind.REBUILD, init=False)

    def apply(self, state: CorpusState, runner: CorpusRunner | None) -> CorpusState:
        if runner is not None:
            runner.rebuild()
        return state.mark(self.operation_id)


@dataclass(frozen=True, slots=True)
class Promote(_Operation):
    kind: OperationKind = field(default=OperationKind.PROMOTE, init=False)

    def apply(self, state: CorpusState, runner: CorpusRunner | None) -> CorpusState:
        if runner is not None:
            runner.promote()
        return state.mark(self.operation_id)


CorpusOperation = (
    Acquire | Append | Replace | Duplicate | Fork | Attach | EmitHook | Crash | Restart | Converge | Rebuild | Promote
)


@dataclass(frozen=True, slots=True)
class CorpusRun:
    state: CorpusState
    schedule: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CorpusProgram:
    """A serializable operation graph with an explicit execution schedule."""

    operations: tuple[CorpusOperation, ...]
    schedule: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        operation_ids = tuple(operation.operation_id for operation in self.operations)
        if len(set(operation_ids)) != len(operation_ids):
            raise CorpusProgramError("operation ids must be unique")
        schedule = tuple(operation_ids) if self.schedule is None else tuple(self.schedule)
        if schedule != tuple(dict.fromkeys(schedule)) or set(schedule) != set(operation_ids):
            raise CorpusProgramError("schedule must be a complete permutation of operation ids")
        object.__setattr__(self, "schedule", schedule)

    def ordered_operations(self) -> tuple[CorpusOperation, ...]:
        by_id = {operation.operation_id: operation for operation in self.operations}
        assert self.schedule is not None
        return tuple(by_id[operation_id] for operation_id in self.schedule)

    def run(self, runner: CorpusRunner | None = None) -> CorpusRun:
        state = CorpusState()
        for operation in self.ordered_operations():
            state = operation.apply(state, runner)
        assert self.schedule is not None
        return CorpusRun(state=state, schedule=self.schedule)

    def to_dict(self) -> JSONDocument:
        return {
            "operations": [operation.to_dict() for operation in self.operations],
            "schedule": list(self.schedule or ()),
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, serialized: str | bytes) -> CorpusProgram:
        value = json.loads(serialized)
        if not isinstance(value, Mapping):
            raise CorpusProgramError("corpus program JSON must be an object")
        return cls.from_dict(cast(Mapping[str, object], value))

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> CorpusProgram:
        raw_operations = payload.get("operations")
        raw_schedule = payload.get("schedule")
        if not isinstance(raw_operations, list) or not isinstance(raw_schedule, list):
            raise CorpusProgramError("program requires operations and schedule lists")
        operations = tuple(_operation_from_dict(cast(Mapping[str, object], item)) for item in raw_operations)
        if not all(isinstance(item, str) for item in raw_schedule):
            raise CorpusProgramError("schedule entries must be strings")
        return cls(operations=operations, schedule=tuple(cast(list[str], raw_schedule)))


def _operation_from_dict(payload: Mapping[str, object]) -> CorpusOperation:
    kind = OperationKind(_required_string(payload, "kind"))
    operation_id = _required_string(payload, "operation_id")
    if kind is OperationKind.ACQUIRE:
        artifact = payload.get("artifact")
        if not isinstance(artifact, Mapping):
            raise CorpusProgramError("Acquire requires artifact")
        return Acquire(operation_id, RawArtifact.from_dict(cast(Mapping[str, object], artifact)))
    if kind is OperationKind.APPEND:
        return Append(
            operation_id,
            _required_string(payload, "artifact_id"),
            _unb64(payload.get("payload_delta_b64"), field_name="payload_delta_b64"),
        )
    if kind is OperationKind.REPLACE:
        return Replace(
            operation_id,
            _required_string(payload, "artifact_id"),
            _unb64(payload.get("payload_b64"), field_name="payload_b64"),
        )
    if kind is OperationKind.DUPLICATE:
        return Duplicate(
            operation_id,
            _required_string(payload, "source_artifact_id"),
            _required_string(payload, "new_artifact_id"),
            _optional_string(payload, "new_source_path"),
        )
    if kind is OperationKind.FORK:
        return Fork(
            operation_id,
            _required_string(payload, "source_artifact_id"),
            _required_string(payload, "new_artifact_id"),
            _required_string(payload, "new_session_id"),
        )
    if kind is OperationKind.ATTACH:
        attachment = payload.get("attachment")
        if not isinstance(attachment, Mapping):
            raise CorpusProgramError("Attach requires attachment")
        return Attach(
            operation_id,
            _required_string(payload, "artifact_id"),
            AttachmentArtifact.from_dict(cast(Mapping[str, object], attachment)),
        )
    if kind is OperationKind.EMIT_HOOK:
        hook = payload.get("hook")
        if not isinstance(hook, Mapping):
            raise CorpusProgramError("EmitHook requires hook")
        return EmitHook(operation_id, HookArtifact.from_dict(cast(Mapping[str, object], hook)))
    constructors: dict[OperationKind, type[_Operation]] = {
        OperationKind.CRASH: Crash,
        OperationKind.RESTART: Restart,
        OperationKind.CONVERGE: Converge,
        OperationKind.REBUILD: Rebuild,
        OperationKind.PROMOTE: Promote,
    }
    return cast(CorpusOperation, constructors[kind](operation_id))


def _required_string(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise CorpusProgramError(f"{key} must be a string")
    return value


def _optional_string(payload: Mapping[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is not None and not isinstance(value, str):
        raise CorpusProgramError(f"{key} must be a string or null")
    return value


def _required_int(payload: Mapping[str, object], key: str, *, default: int | None = None) -> int:
    value = payload.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise CorpusProgramError(f"{key} must be an integer")
    return value


def operation_strategy(*, artifact_ids: Sequence[str] = ("a", "b")) -> Any:
    """Return a bounded Hypothesis strategy whose values shrink structurally."""
    from hypothesis import strategies as st

    ids = tuple(artifact_ids)
    artifact = st.builds(
        RawArtifact,
        artifact_id=st.sampled_from(ids),
        payload=st.binary(max_size=96),
        source_path=st.sampled_from(tuple(f"sources/{artifact_id}.jsonl" for artifact_id in ids)),
    )
    return st.one_of(
        st.builds(Acquire, operation_id=st.text("op", min_size=2, max_size=8), artifact=artifact),
        st.builds(
            Append,
            operation_id=st.text("op", min_size=2, max_size=8),
            artifact_id=st.sampled_from(ids),
            payload_delta=st.binary(max_size=32),
        ),
        st.builds(
            Replace,
            operation_id=st.text("op", min_size=2, max_size=8),
            artifact_id=st.sampled_from(ids),
            payload=st.binary(max_size=96),
        ),
        st.builds(
            Duplicate,
            operation_id=st.text("op", min_size=2, max_size=8),
            source_artifact_id=st.sampled_from(ids),
            new_artifact_id=st.sampled_from(ids),
        ),
        st.builds(
            Fork,
            operation_id=st.text("op", min_size=2, max_size=8),
            source_artifact_id=st.sampled_from(ids),
            new_artifact_id=st.sampled_from(ids),
            new_session_id=st.text("s", min_size=2, max_size=8),
        ),
        st.builds(
            Attach,
            operation_id=st.text("op", min_size=2, max_size=8),
            artifact_id=st.sampled_from(ids),
            attachment=st.builds(
                AttachmentArtifact,
                attachment_id=st.text("a", min_size=2, max_size=8),
                name=st.just("fixture.bin"),
                payload=st.binary(max_size=32),
            ),
        ),
        st.builds(
            EmitHook,
            operation_id=st.text("op", min_size=2, max_size=8),
            hook=st.builds(
                HookArtifact,
                hook_event_id=st.text("hook", min_size=4, max_size=10),
                provider=st.just("codex"),
                event_type=st.just("session.created"),
                session_native_id=st.text("session", min_size=4, max_size=12),
                payload=st.binary(max_size=64),
            ),
        ),
        st.builds(Crash, operation_id=st.text("op", min_size=2, max_size=8)),
        st.builds(Restart, operation_id=st.text("op", min_size=2, max_size=8)),
        st.builds(Converge, operation_id=st.text("op", min_size=2, max_size=8)),
        st.builds(Rebuild, operation_id=st.text("op", min_size=2, max_size=8)),
        st.builds(Promote, operation_id=st.text("op", min_size=2, max_size=8)),
    )


def corpus_program_strategy(*, max_operations: int = 8) -> Any:
    """Generate programs with unique operation ids and a shrinkable schedule."""
    from hypothesis import strategies as st

    return st.lists(operation_strategy(), min_size=1, max_size=max_operations).map(
        lambda operations: CorpusProgram(
            operations=tuple(
                replace(operation, operation_id=f"op-{index}") for index, operation in enumerate(operations)
            ),
        )
    )


class ProductionCorpusRuntime:
    """Adapter from corpus operations to the live archive production seams."""

    def __init__(self, archive_root: Path) -> None:
        self.archive_root = archive_root.resolve()
        self.source_root = self.archive_root / "corpus-program-sources"
        self._raw_ids: dict[str, tuple[str, ...]] = {}
        self._source_paths: dict[str, Path] = {}
        self._rebuild_receipt: RebuildIndexReceipt | None = None
        self._crashed = False
        self.last_results: list[object] = []

    def _ensure_running(self) -> None:
        if self._crashed:
            raise CorpusRuntimeCrashedError("runtime is crashed; apply Restart before the next effect")

    def acquire(self, artifact: RawArtifact) -> object:
        self._ensure_running()
        from polylogue.config import Source
        from polylogue.pipeline.services.acquisition import AcquisitionService
        from polylogue.storage.sqlite import SQLiteBackend

        path = self.source_root / _validate_relative_path(artifact.source_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(artifact.payload)

        async def run() -> object:
            backend = SQLiteBackend(db_path=self.archive_root / "index.db")
            try:
                result = await AcquisitionService(backend).acquire_sources(
                    [Source(name=artifact.source_name, path=path)]
                )
                self._raw_ids[artifact.artifact_id] = tuple(result.raw_ids)
                self._source_paths[artifact.artifact_id] = path
                return result
            finally:
                await backend.close()

        result = asyncio.run(run())
        self.last_results.append(result)
        return result

    def emit_hook(self, hook: HookArtifact) -> object:
        self._ensure_running()
        from polylogue.core.enums import Provider
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from polylogue.storage.sqlite.archive_tiers.source_write import ArchiveHookEvent

        provider = Provider.from_string(hook.provider)
        payload = hook.payload
        with ArchiveStore.open_existing(self.archive_root, read_only=False) as archive:
            result = archive.write_hook_event(
                provider=provider,
                payload=payload,
                source_path=hook.source_path,
                acquired_at_ms=hook.observed_at_ms,
                hook_event=ArchiveHookEvent(
                    hook_event_id=hook.hook_event_id,
                    origin=provider.value,
                    source_path=hook.source_path,
                    event_type=hook.event_type,
                    payload={"session_id": hook.session_native_id},
                    observed_at_ms=hook.observed_at_ms,
                    native_id=hook.hook_event_id,
                    session_native_id=hook.session_native_id,
                ),
            )
        self.last_results.append(result)
        return result

    def crash(self) -> object:
        self._crashed = True
        return None

    def restart(self) -> object:
        self._crashed = False
        return None

    def converge(self) -> object:
        self._ensure_running()
        from polylogue.config import Config
        from polylogue.daemon.convergence import DaemonConverger
        from polylogue.daemon.convergence_stages import make_default_convergence_stages
        from polylogue.pipeline.services.parsing import ParsingService
        from polylogue.storage.repository import SessionRepository
        from polylogue.storage.sqlite import SQLiteBackend

        raw_ids = tuple(dict.fromkeys(raw_id for ids in self._raw_ids.values() for raw_id in ids))
        paths = tuple(dict.fromkeys(self._source_paths.values()))

        async def parse() -> ParseResult:
            backend = SQLiteBackend(db_path=self.archive_root / "index.db")
            try:
                config = Config(
                    archive_root=self.archive_root,
                    render_root=self.archive_root / "render",
                    sources=[],
                    db_path=self.archive_root / "index.db",
                )
                service = ParsingService(
                    repository=SessionRepository(backend=backend),
                    archive_root=self.archive_root,
                    config=config,
                    ingest_workers=1,
                )
                return await service.parse_from_raw(raw_ids=list(raw_ids), force_write=True)
            finally:
                await backend.close()

        parse_result = asyncio.run(parse())
        states = DaemonConverger(make_default_convergence_stages(self.archive_root / "index.db")).converge_all(paths)
        result = {"parse": parse_result, "convergence": states}
        self.last_results.append(result)
        return result

    def rebuild(self) -> object:
        self._ensure_running()
        from polylogue.maintenance.rebuild_index import RebuildIndexRequest, rebuild_index_from_source

        async def run() -> RebuildIndexReceipt:
            return await rebuild_index_from_source(RebuildIndexRequest(archive_root=self.archive_root, promote=False))

        result = asyncio.run(run())
        self._rebuild_receipt = result
        self.last_results.append(result)
        return result

    def promote(self) -> object:
        self._ensure_running()
        if self._rebuild_receipt is None:
            raise CorpusProgramError("Promote requires a preceding Rebuild")
        from polylogue.storage.index_generation import IndexGenerationStore

        generation_id = self._rebuild_receipt.generation.get("generation_id")
        if not isinstance(generation_id, str):
            raise CorpusProgramError("Rebuild did not return an inactive generation")
        store = IndexGenerationStore.for_archive_root(self.archive_root)
        result = store.promote(store.load(generation_id))
        self.last_results.append(result)
        return result


__all__ = [
    "Acquire",
    "Append",
    "Attach",
    "AttachmentArtifact",
    "CorpusOperation",
    "CorpusProgram",
    "CorpusProgramError",
    "CorpusRun",
    "CorpusRuntimeCrashed",
    "CorpusRuntimeCrashedError",
    "CorpusState",
    "Crash",
    "Converge",
    "Duplicate",
    "EmitHook",
    "Fork",
    "HookArtifact",
    "OperationKind",
    "ProductionCorpusRuntime",
    "Promote",
    "RawArtifact",
    "Rebuild",
    "Replace",
    "Restart",
    "corpus_program_strategy",
    "operation_strategy",
]
