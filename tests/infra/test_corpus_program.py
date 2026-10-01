"""Focused contracts for the composable corpus-program harness."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, find, given, settings

from polylogue.pipeline.services.parsing_models import ParseResult
from polylogue.storage.blob_store import BlobStore
from tests.infra.corpus_program import (
    Acquire,
    Append,
    Attach,
    AttachmentArtifact,
    Converge,
    CorpusAcquisitionRejectedError,
    CorpusConvergenceRejectedError,
    CorpusProgram,
    CorpusProgramError,
    Crash,
    Duplicate,
    EmitHook,
    Fork,
    HookArtifact,
    ProductionCorpusRuntime,
    RawArtifact,
    Replace,
    Restart,
    _codex_transcript,
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
    replacement = initial.replace(b"The capital of France is Paris.", b"Replacement reached production.")
    assert replacement != initial
    program = CorpusProgram(
        operations=(
            Append("append", "session", delta),
            Converge("converge"),
            Replace("replace", "session", replacement),
            Acquire("acquire-v1", _artifact("session", initial)),
        ),
        schedule=("acquire-v1", "append", "replace", "converge"),
    )
    runtime = ProductionCorpusRuntime(workspace_env["archive_root"])

    run = program.run(runtime)

    assert run.state.artifact("session").payload == replacement
    assert runtime.last_results
    with runtime.archive_root.joinpath("index.db").open("rb"):
        pass
    import sqlite3

    with sqlite3.connect(runtime.archive_root / "index.db") as conn:
        session_count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        message_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        block_text = "\n".join(str(row[0]) for row in conn.execute("SELECT text FROM blocks ORDER BY block_id"))
    assert session_count == 1
    assert message_count >= 2
    assert "Replacement reached production" in block_text
    assert "Append this turn" not in block_text


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
    with BlobStore(runtime.archive_root / "blob").open(blob_hash.hex()) as retained:
        assert retained.read() == attachment.payload
    assert runtime._raw_ids["session"]
    # Replacing the capture turns with one synthetic turn must turn this red.
    with sqlite3.connect(runtime.archive_root / "index.db") as conn:
        turns = conn.execute(
            "SELECT m.native_id, m.role, b.text FROM messages m JOIN blocks b ON b.message_id = m.message_id "
            "WHERE b.block_type = 'text' ORDER BY m.position, b.position"
        ).fetchall()
    assert turns == [
        ("msg-user-1", "user", "What is the capital of France?"),
        ("msg-asst-1", "assistant", "The capital of France is Paris."),
    ]


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


@settings(max_examples=4, suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
@given(corpus_program_strategy(max_operations=5))
def test_generated_programs_acquire_and_parse_on_production_route(
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    program: CorpusProgram,
) -> None:
    """Arbitrary binary transcript draws fail at the actual acquisition/parser."""
    import uuid

    root = workspace_env["archive_root"].parent / f"generated-{uuid.uuid4().hex}"
    # Start every draw from the same empty fixture, not a previous draw's rows.
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(root)
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(root))
    runtime = ProductionCorpusRuntime(root)
    run = program.run(runtime)
    runtime.restart()
    result = runtime.converge()
    assert result["parse"].parse_failures == 0
    assert all(state.converged and state.error_count == 0 for state in result["convergence"].values())
    with sqlite3.connect(root / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] > 0
    assert run.state.applied_operation_ids == program.schedule


def test_rejected_acquisition_does_not_advance_reference_state(
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ignoring the production error AcquireResult permits the rejected append."""
    from polylogue.pipeline.services import acquisition
    from polylogue.pipeline.services.acquisition_streams import iter_raw_record_stream

    runtime = ProductionCorpusRuntime(workspace_env["archive_root"])
    initial = _artifact("session", _codex_transcript("session", "first", "authored"))
    from tests.infra.corpus_program import CorpusState

    state = Acquire("acquire", initial).apply(CorpusState(), runtime)
    original_stream = iter_raw_record_stream

    async def unreadable_stream(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        async for record in original_stream(*args, **kwargs):
            raise OSError("synthetic source read failure")
            yield record

    monkeypatch.setattr(acquisition, "iter_raw_record_stream", unreadable_stream)
    with pytest.raises(CorpusAcquisitionRejectedError) as refused:
        Replace("bad-replace", "session", _codex_transcript("session", "second", "new authored turn")).apply(
            state, runtime
        )
    assert refused.value.artifact_id == "session"
    assert refused.value.result.errors > 0 or not refused.value.result.raw_ids
    assert state.artifact("session").payload == initial.payload
    assert state.applied_operation_ids == ("acquire",)
    assert all(not isinstance(result, dict) or "convergence" not in result for result in runtime.last_results)


def test_retained_parser_invalid_input_reports_convergence_failure(workspace_env: dict[str, Path]) -> None:
    runtime = ProductionCorpusRuntime(workspace_env["archive_root"])
    result = runtime.acquire(_artifact("invalid", b"\xff\x00"))
    assert result.acquired > 0
    with pytest.raises(CorpusConvergenceRejectedError):
        runtime.converge()
    parsed = runtime.last_results[-1]
    assert isinstance(parsed, ParseResult)
    assert parsed.parse_failures > 0


@pytest.mark.parametrize("pending", [False, True], ids=["failed", "pending"])
def test_unfinished_daemon_stages_do_not_advance_converge_operation(
    workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pending: bool
) -> None:
    """Ignoring the real daemon FileState lets failed/pending work look applied."""
    from polylogue.daemon import convergence_stages
    from polylogue.daemon.convergence import ConvergenceStage, StageState
    from tests.infra.corpus_program import CorpusState

    runtime = ProductionCorpusRuntime(workspace_env["archive_root"])
    state = Acquire("acquire", _artifact("session", _codex_transcript("session", "first", "authored"))).apply(
        CorpusState(), runtime
    )
    original_stages = convergence_stages.make_default_convergence_stages

    def stages_with_unfinished_work(*args: Any, **kwargs: Any) -> tuple[ConvergenceStage, ...]:
        return (
            *original_stages(*args, **kwargs),
            ConvergenceStage(
                name="synthetic-unfinished",
                description="Synthetic unfinished derivation",
                check=lambda path: True,
                execute=lambda path: False,
                false_means_pending=pending,
            ),
        )

    monkeypatch.setattr(convergence_stages, "make_default_convergence_stages", stages_with_unfinished_work)
    with pytest.raises(CorpusConvergenceRejectedError) as refused:
        Converge("converge").apply(state, runtime)
    result = refused.value.result
    assert isinstance(result, dict)
    assert result["parse"].parse_failures == 0
    assert len(result["convergence"]) == 1
    actual = next(iter(result["convergence"].values()))
    assert actual.stages["synthetic-unfinished"] == (StageState.PENDING if pending else StageState.FAILED)
    assert not actual.converged
    assert state.applied_operation_ids == ("acquire",)


def test_unchanged_reacquisition_preserves_proven_raw_evidence(workspace_env: dict[str, Path]) -> None:
    runtime = ProductionCorpusRuntime(workspace_env["archive_root"])
    artifact = _artifact("session", _codex_transcript("session", "first", "authored"))
    first = runtime.acquire(artifact)
    raw_ids = runtime._raw_ids["session"]
    repeated = runtime.acquire(artifact)
    assert first.acquired > 0
    assert repeated.skipped > 0
    assert runtime._raw_ids["session"] == raw_ids
    assert runtime.converge()["parse"].parse_failures == 0


@pytest.mark.parametrize("mutation_type", [Append, Replace], ids=["append", "replace"])
def test_generated_mutation_transcripts_reach_production(
    workspace_env: dict[str, Path],
    mutation_type: type[Append] | type[Replace],
) -> None:
    """Arbitrary bytes at the actual selected generator branch make this red."""
    program = find(
        corpus_program_strategy(max_operations=2),
        lambda candidate: any(isinstance(operation, mutation_type) for operation in candidate.operations),
        settings=settings(max_examples=100, database=None, deadline=None, derandomize=True),
    )
    runtime = ProductionCorpusRuntime(workspace_env["archive_root"])
    program.run(runtime)
    result = runtime.converge()
    assert result["parse"].parse_failures == 0
    with sqlite3.connect(runtime.archive_root / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] > 0
