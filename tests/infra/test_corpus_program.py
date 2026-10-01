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
    _codex_turn,
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


def test_native_attachment_preserves_semantics_lineage_and_retained_replay(
    workspace_env: dict[str, Path], tmp_path: Path
) -> None:
    from polylogue.core.enums import Provider
    from polylogue.sources.dispatch import parse_payload
    from polylogue.sources.revision_backfill import backfill_historical_revision_evidence
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
    from tests.infra.corpus_program import CorpusState

    payload = (Path(__file__).parents[1] / "fixtures" / "corpus-program-codex-native.jsonl").read_bytes()
    runtime = ProductionCorpusRuntime(workspace_env["archive_root"])
    initial = CorpusProgram(
        operations=(
            Acquire("acquire", _artifact("tool-call-session-1", payload)),
            Fork("fork", "tool-call-session-1", "child", "child-native"),
            Append("append", "child", _codex_turn("child-tail", "A divergent authored turn")),
            Converge("converge"),
        )
    ).run(runtime)
    state: CorpusState = initial.state
    child = state.artifact("child")
    native = parse_payload(Provider.CODEX, [json.loads(line) for line in child.payload.splitlines()], "child")[0]
    assert native.parent_session_provider_id == "tool-call-session-1"
    assert any(message.model_name == "gpt-5-codex" and message.model_effort == "high" for message in native.messages)
    assert any(message.input_tokens for message in native.messages)
    assert any(message.blocks for message in native.messages)

    def semantics(root: Path) -> tuple[list[tuple[Any, ...]], list[tuple[Any, ...]]]:
        with sqlite3.connect(root / "index.db") as conn:
            messages = conn.execute(
                "SELECT native_id, role, message_type, material_origin, input_tokens, output_tokens, "
                "model_name, model_effort, stop_reason, variant_index, sender_name, recipient, "
                "delivery_status, end_turn, user_context_text FROM messages ORDER BY message_id"
            ).fetchall()
            blocks = conn.execute("SELECT * FROM blocks ORDER BY message_id, position").fetchall()
        return messages, blocks

    before = semantics(runtime.archive_root)
    state = Attach("attach", "child", AttachmentArtifact("native-att", "fixture.txt", "text/plain", b"bytes")).apply(
        state, runtime
    )
    state = Converge("attached-converge").apply(state, runtime)
    assert semantics(runtime.archive_root) == before
    with ArchiveStore.open_existing(runtime.archive_root, read_only=True) as archive:
        with archive.open_raw_revision_material(runtime._raw_ids["child"][0]) as (provider, handle, _, _):
            retained = handle.read()
    capture = json.loads(retained)
    replayed = parse_payload(provider, capture, "retained")[0]
    assert replayed.messages == native.messages
    assert replayed.parent_session_provider_id == native.parent_session_provider_id
    assert replayed.attachments[0].provider_attachment_id == "native-att"
    assert replayed.attachments[0].inline_bytes == b"bytes"

    replay_root = tmp_path / "retained-native-replay"
    initialize_active_archive_root(replay_root)
    with ArchiveStore.open_existing(replay_root, read_only=False) as archive:
        parent_raw = archive.write_raw_payload(
            provider=Provider.CODEX, payload=payload, source_path="parent.jsonl", acquired_at_ms=1
        )
        child_raw = archive.write_raw_payload(
            provider=provider, payload=retained, source_path="child.json", acquired_at_ms=2
        )
    result = backfill_historical_revision_evidence(replay_root, selected_raw_ids=[parent_raw, child_raw])
    assert result.quarantined == result.adoption_deferred == 0
    assert result.replayed_logical_sources == 2
    assert semantics(replay_root) == before
    with sqlite3.connect(replay_root / "index.db") as conn:
        links = conn.execute("SELECT src_session_id, resolved_dst_session_id FROM session_links").fetchall()
        assert links and all(parent is not None for _, parent in links)
        assert conn.execute("SELECT native_id FROM attachment_native_ids WHERE id_kind = 'attachment'").fetchall() == [
            ("native-att",)
        ]
    runtime.restart()
    assert runtime.converge()["parse"].parse_failures == 0
    assert semantics(runtime.archive_root) == before


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


def test_content_identical_duplicate_reuses_actual_raw_evidence(workspace_env: dict[str, Path]) -> None:
    runtime = ProductionCorpusRuntime(workspace_env["archive_root"])
    program = CorpusProgram(
        operations=(
            Acquire("acquire", _artifact("session", _codex_transcript("session", "first", "authored"))),
            Duplicate("duplicate", "session", "copy"),
            Converge("converge"),
        )
    )
    run = program.run(runtime)
    assert run.state.applied_operation_ids == ("acquire", "duplicate", "converge")
    assert runtime._raw_ids["copy"] == runtime._raw_ids["session"]
    assert runtime.converge()["parse"].parse_failures == 0
    with sqlite3.connect(runtime.archive_root / "index.db") as conn:
        assert conn.execute("SELECT native_id FROM messages").fetchall() == [("first",)]


def test_replacement_can_reacquire_an_earlier_retained_revision(workspace_env: dict[str, Path]) -> None:
    runtime = ProductionCorpusRuntime(workspace_env["archive_root"])
    initial = _codex_transcript("session", "first", "authored")
    program = CorpusProgram(
        operations=(
            Acquire("acquire", _artifact("session", initial)),
            Replace("replace", "session", _codex_transcript("session", "second", "changed")),
            Replace("restore", "session", initial),
            Converge("converge"),
        )
    )
    run = program.run(runtime)
    assert run.state.applied_operation_ids == ("acquire", "replace", "restore", "converge")
    with sqlite3.connect(runtime.archive_root / "index.db") as conn:
        assert conn.execute("SELECT native_id FROM messages").fetchall() == [("first",)]


def test_legacy_codex_fork_retains_the_actual_parent_carrier(workspace_env: dict[str, Path]) -> None:
    payload = (Path(__file__).parents[1] / "fixtures" / "corpus-program-codex-legacy.jsonl").read_bytes()
    runtime = ProductionCorpusRuntime(workspace_env["archive_root"])
    CorpusProgram(
        operations=(
            Acquire("acquire", _artifact("legacy-parent", payload)),
            Fork("fork", "legacy-parent", "child", "legacy-child"),
            Converge("converge"),
        )
    ).run(runtime)
    with sqlite3.connect(runtime.archive_root / "index.db") as conn:
        assert conn.execute(
            "SELECT src.native_id, dst.native_id FROM session_links l "
            "JOIN sessions src ON src.session_id = l.src_session_id "
            "JOIN sessions dst ON dst.session_id = l.resolved_dst_session_id"
        ).fetchall() == [("legacy-child", "legacy-parent")]


@pytest.mark.parametrize("payload", [b"\xff", b"{}\n{bad}"])
def test_malformed_attach_has_typed_refusal(workspace_env: dict[str, Path], payload: bytes) -> None:
    from tests.infra.corpus_program import CorpusState

    runtime = ProductionCorpusRuntime(workspace_env["archive_root"])
    state = Acquire("acquire", _artifact("invalid", payload)).apply(CorpusState(), runtime)
    with pytest.raises(CorpusProgramError):
        Attach("attach", "invalid", AttachmentArtifact("att", "fixture.txt", payload=b"bytes")).apply(state, runtime)
    assert state.applied_operation_ids == ("acquire",)
    assert not state.artifact("invalid").attachments


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
        native_ids = [row[0] for row in conn.execute("SELECT native_id FROM messages ORDER BY position")]
    # An empty binary append can leave the valid initial turn untouched. The
    # actual new authored turn must arrive, even in the smallest generated case.
    assert native_ids == (["message-0", "message-1"] if mutation_type is Append else ["message-1"])
