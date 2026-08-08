"""Focused contracts for the composable corpus-program harness."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from hypothesis import given

from polylogue.sources.revision_backfill import backfill_historical_revision_evidence
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.index_generation import IndexGenerationStore
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
from tests.infra.rebuild_receipt import write_valid_rebuild_receipt


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


def _freeze_runtime_source(
    runtime: ProductionCorpusRuntime,
    receipt_path: Path,
    *,
    expected_raws: int = 1,
) -> None:
    backfill = backfill_historical_revision_evidence(runtime.archive_root)
    assert backfill.scanned == expected_raws
    assert backfill.classified_full == expected_raws
    assert backfill.replayed_logical_sources + backfill.adoption_deferred == 1
    runtime.bind_schema_inference_receipt(write_valid_rebuild_receipt(runtime.archive_root, receipt_path))


def test_production_route_promotes_the_owned_candidate_with_terminal_transaction(
    workspace_env: dict[str, Path],
) -> None:
    fixture_path = Path(__file__).parents[1] / "data" / "codex_event_stream" / "text_only_stream.jsonl"
    runtime = ProductionCorpusRuntime(workspace_env["archive_root"])
    runtime.acquire(_artifact("session", fixture_path.read_bytes()))
    _freeze_runtime_source(
        runtime,
        workspace_env["archive_root"].parent / "corpus-program-schema-inference-receipt.json",
    )
    store = IndexGenerationStore.for_archive_root(runtime.archive_root)
    active_before = store.active_pointer.resolve(strict=True)

    built = runtime.rebuild()

    assert built.status == "replayed"
    assert built.generation["state"] == "inactive"
    assert built.transaction is not None
    assert built.transaction["status"] == "ready"
    operation_id = built.transaction["operation_id"]
    assert store.active_pointer.resolve(strict=True) == active_before

    promoted = runtime.promote()

    assert promoted.status == "replayed"
    assert promoted.generation["state"] == "active"
    assert promoted.transaction is not None
    assert promoted.transaction["operation_id"] == operation_id
    assert promoted.transaction["status"] == "promoted"
    assert store.active_pointer.resolve(strict=True) == Path(str(promoted.generation["index_path"])).resolve()


def test_production_route_refuses_promotion_after_source_drift(
    workspace_env: dict[str, Path],
) -> None:
    fixture_path = Path(__file__).parents[1] / "data" / "codex_event_stream" / "text_only_stream.jsonl"
    runtime = ProductionCorpusRuntime(workspace_env["archive_root"])
    artifact = _artifact("session", fixture_path.read_bytes())
    runtime.acquire(artifact)
    _freeze_runtime_source(
        runtime,
        workspace_env["archive_root"].parent / "corpus-program-source-drift-receipt.json",
    )
    store = IndexGenerationStore.for_archive_root(runtime.archive_root)
    active_before = store.active_pointer.resolve(strict=True)

    built = runtime.rebuild()
    assert built.transaction is not None
    operation_id = str(built.transaction["operation_id"])
    runtime.acquire(artifact.with_payload(artifact.payload + b"\n"))
    _freeze_runtime_source(
        runtime,
        workspace_env["archive_root"].parent / "corpus-program-post-drift-receipt.json",
        expected_raws=2,
    )

    with pytest.raises(RuntimeError, match="source evidence changed"):
        runtime.promote()

    transaction = store.load_transaction(operation_id)
    assert transaction.status == "stale"
    assert store.active_pointer.resolve(strict=True) == active_before


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
