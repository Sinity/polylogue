"""Focused contracts for the composable corpus-program harness."""

from __future__ import annotations

from pathlib import Path

from hypothesis import given

from tests.infra.corpus_program import (
    Acquire,
    Append,
    Attach,
    AttachmentArtifact,
    Converge,
    CorpusProgram,
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
    corpus_program_strategy,
)


class RecordingRunner:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def acquire(self, artifact: RawArtifact) -> None:
        self.calls.append(f"acquire:{artifact.artifact_id}")

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
            Acquire("acquire-v2", _artifact("session", initial + delta)),
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
