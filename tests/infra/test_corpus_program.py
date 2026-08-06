"""Focused contracts for the composable corpus-program harness."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from hypothesis import given

from tests.infra.corpus_program import (
    Acquire,
    Append,
    Attach,
    AttachmentArtifact,
    Converge,
    CorpusProgram,
    CorpusProgramError,
    Crash,
    Duplicate,
    EmitHook,
    Fork,
    HookArtifact,
    ProductionCorpusRuntime,
    Promote,
    RawArtifact,
    Rebuild,
    Replace,
    Restart,
    corpus_program_schedule_strategy,
    corpus_program_strategy,
)


class RecordingRunner:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.acquired: list[RawArtifact] = []

    def acquire(self, artifact: RawArtifact) -> None:
        self.calls.append(f"acquire:{artifact.artifact_id}")
        self.acquired.append(artifact)

    def emit_hook(self, hook: HookArtifact) -> None:
        self.calls.append(f"hook:{hook.hook_event_id}")

    def crash(self) -> None:
        self.calls.append("crash")

    def restart(self) -> None:
        self.calls.append("restart")

    def converge(self) -> None:
        self.calls.append("converge")

    def rebuild(self) -> None:
        self.calls.append("rebuild")

    def promote(self) -> None:
        self.calls.append("promote")


def _artifact(artifact_id: str, payload: bytes = b"payload") -> RawArtifact:
    return RawArtifact(
        artifact_id=artifact_id,
        payload=payload,
        source_path=f"sources/{artifact_id}.jsonl",
        metadata={"session_id": artifact_id},
    )


def _hook() -> HookArtifact:
    return HookArtifact(
        hook_event_id="hook-1",
        provider="claude-code",
        event_type="SessionStart",
        session_native_id="session-1",
        payload=b'{"session_id":"session-1"}',
    )


def test_program_serialization_is_canonical_and_round_trips_all_operation_shapes() -> None:
    attachment = AttachmentArtifact("att-1", "fixture.txt", "text/plain", b"hello")
    operations = (
        Acquire("acquire", _artifact("a")),
        Append("append", "a", b"tail"),
        Replace("replace", "a", b"replacement"),
        Duplicate("duplicate", "a", "b"),
        Fork("fork", "a", "c", "session-c"),
        Attach("attach", "a", attachment),
        EmitHook("hook", _hook()),
        Crash("crash"),
        Restart("restart"),
        Converge("converge"),
        Rebuild("rebuild"),
        Promote("promote"),
    )
    program = CorpusProgram(operations, schedule=tuple(operation.operation_id for operation in operations))

    serialized = program.to_json()
    assert serialized == program.to_json()
    assert " \n" not in serialized
    assert CorpusProgram.from_json(serialized) == program


def test_composition_applies_transformations_in_declared_schedule() -> None:
    original = _artifact("a", b"one")
    updated = _artifact("a", b"onetwo")
    program = CorpusProgram(
        operations=(
            Acquire("first", original),
            Append("append", "a", b"two"),
            Acquire("second", updated),
            Attach("attach", "a", AttachmentArtifact("att", "x.txt", payload=b"x")),
            Duplicate("duplicate", "a", "b"),
            Fork("fork", "a", "c", "session-c"),
        )
    )

    run = program.run()
    assert [artifact.artifact_id for artifact in run.state.artifacts] == ["a", "b", "c"]
    assert run.state.artifact("a").payload == b"onetwo"
    assert run.state.artifact("a").attachments[0].attachment_id == "att"
    assert run.state.artifact("c").parent_artifact_id == "a"


def test_mutations_reacquire_the_current_transformed_artifact() -> None:
    attachment = AttachmentArtifact("att", "fixture.txt", "text/plain", b"bytes")
    program = CorpusProgram(
        operations=(
            Acquire("acquire", _artifact("a", b"one")),
            Append("append", "a", b"two"),
            Replace("replace", "a", b"replacement"),
            Duplicate("duplicate", "a", "b"),
            Fork("fork", "a", "c", "session-c"),
            Attach("attach", "a", attachment),
        )
    )
    runner = RecordingRunner()

    program.run(runner)

    assert [artifact.artifact_id for artifact in runner.acquired] == ["a", "a", "a", "b", "c", "a"]
    assert runner.acquired[1].payload == b"onetwo"
    assert runner.acquired[2].payload == b"replacement"
    assert runner.acquired[3].payload == b"replacement"
    assert runner.acquired[4].parent_artifact_id == "a"
    assert runner.acquired[5].attachments == (attachment,)


def test_adversarial_schedule_is_observable_and_cannot_be_ignored() -> None:
    program = CorpusProgram(
        operations=(Acquire("a-op", _artifact("a")), Acquire("b-op", _artifact("b"))),
        schedule=("b-op", "a-op"),
    )
    runner = RecordingRunner()

    program.run(runner)

    assert runner.calls == ["acquire:b", "acquire:a"]


@given(corpus_program_strategy(max_operations=5))
def test_generated_programs_have_shrinkable_canonical_round_trips(program: CorpusProgram) -> None:
    assert CorpusProgram.from_json(program.to_json()) == program


@given(corpus_program_strategy(max_operations=8))
def test_generated_programs_execute_from_evolving_acquired_state(program: CorpusProgram) -> None:
    run = program.run()

    assert run.state.applied_operation_ids == run.schedule
    assert set(run.schedule) == {operation.operation_id for operation in program.operations}


@given(corpus_program_schedule_strategy(("op-a", "op-b", "op-c")))
def test_schedule_strategy_returns_operation_id_permutations(schedule: tuple[str, ...]) -> None:
    assert set(schedule) == {"op-a", "op-b", "op-c"}


def test_production_route_composes_acquire_append_and_converge(
    workspace_env: dict[str, Path],
) -> None:
    fixture_path = Path(__file__).parents[1] / "data" / "codex_event_stream" / "text_only_stream.jsonl"
    initial = fixture_path.read_bytes()
    delta = b'{"type":"response_item","payload":{"type":"message","id":"msg-appended","role":"user","timestamp":"2025-01-15T10:01:00Z","content":[{"type":"input_text","text":"Append this turn."}]}}\n'
    program = CorpusProgram(
        operations=(
            Acquire("acquire-v1", _artifact("session", initial)),
            Append("append", "session", delta),
            Converge("converge"),
        )
    )
    runtime = ProductionCorpusRuntime(workspace_env["archive_root"])

    run = program.run(runtime)

    assert run.state.artifact("session").payload.endswith(delta)
    assert runtime.last_results
    with runtime.archive_root.joinpath("index.db").open("rb"):
        pass
    import sqlite3

    with sqlite3.connect(runtime.archive_root / "index.db") as conn:
        session_count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        message_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    assert session_count == 1
    assert message_count >= 3


def test_production_route_carries_attachment_identity_metadata_and_bytes(
    workspace_env: dict[str, Path],
) -> None:
    fixture_path = Path(__file__).parents[1] / "data" / "codex_event_stream" / "text_only_stream.jsonl"
    attachment = AttachmentArtifact("att-corpus", "fixture.txt", "text/plain", b"attachment bytes")
    program = CorpusProgram(
        operations=(
            Acquire("acquire", _artifact("session", fixture_path.read_bytes())),
            Attach("attach", "session", attachment),
            Converge("converge"),
        )
    )
    runtime = ProductionCorpusRuntime(workspace_env["archive_root"])

    program.run(runtime)

    with sqlite3.connect(runtime.archive_root / "index.db") as conn:
        row = conn.execute(
            "SELECT a.display_name, a.media_type, a.byte_count, a.acquisition_status, a.blob_hash, n.native_id "
            "FROM attachments AS a "
            "JOIN attachment_refs AS r ON r.attachment_id = a.attachment_id "
            "JOIN attachment_native_ids AS n ON n.ref_id = r.ref_id AND n.id_kind = 'attachment' "
            "WHERE n.native_id = ?",
            (attachment.attachment_id,),
        ).fetchone()
    assert row is not None
    assert row[:4] == (attachment.name, attachment.mime_type, len(attachment.payload), "acquired")
    assert row[5] == attachment.attachment_id
    blob_hash = row[4]
    assert isinstance(blob_hash, bytes)
    assert len(blob_hash) == 32
    assert runtime._raw_ids["session"]


def test_production_route_persists_canonical_hook_envelope(workspace_env: dict[str, Path]) -> None:
    fixture_path = Path(__file__).parents[1] / "data" / "codex_event_stream" / "text_only_stream.jsonl"
    hook = _hook()
    runtime = ProductionCorpusRuntime(workspace_env["archive_root"])
    runtime.acquire(_artifact("session", fixture_path.read_bytes()))

    runtime.emit_hook(hook)

    with sqlite3.connect(runtime.archive_root / "source.db") as conn:
        origin, payload_json = conn.execute(
            "SELECT origin, payload_json FROM raw_hook_events WHERE hook_event_id = ?",
            (hook.hook_event_id,),
        ).fetchone()
    payload = json.loads(payload_json)
    assert origin == "claude-code-session"
    assert payload == {
        "event_id": hook.hook_event_id,
        "event_type": hook.event_type,
        "observed_at_ms": hook.observed_at_ms,
        "payload": {"session_id": hook.session_native_id},
        "provider": "claude-code",
        "session_id": hook.session_native_id,
        "timestamp": "2025-01-01T00:00:00Z",
    }


def test_promote_reenters_owned_rebuild_boundary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runtime = ProductionCorpusRuntime(tmp_path / "archive")
    runtime._rebuild_receipt = SimpleNamespace(transaction={"operation_id": "rebuild-op"})  # type: ignore[assignment]
    calls: list[object] = []

    async def resume(request: object) -> object:
        calls.append(request)
        return object()

    monkeypatch.setattr("polylogue.maintenance.rebuild_index.rebuild_index_from_source", resume)

    result = runtime.promote()

    assert result is runtime._rebuild_receipt
    request = cast(Any, calls[0])
    assert request.operation_id == "rebuild-op"
    assert request.promote is True


def test_emit_hook_refuses_non_object_payload_with_named_reason(tmp_path: Path) -> None:
    runtime = ProductionCorpusRuntime(tmp_path / "archive")
    with pytest.raises(CorpusProgramError, match="EmitHook refused: payload is not a JSON object"):
        runtime.emit_hook(
            HookArtifact(
                hook_event_id="bad-hook",
                provider="codex",
                event_type="SessionStart",
                session_native_id="session-1",
                payload=b"[]",
            )
        )
