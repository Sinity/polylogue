"""Production raw-adapter laws independent of the legacy backlog census."""

from __future__ import annotations

import errno
import json
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from unittest.mock import Mock

import pytest

from polylogue.core.enums import Provider
from polylogue.daemon.derivation import Budget, DerivationRegistry, DerivationReport, converge
from polylogue.operations.raw_observation_derivation import (
    converge_raw_observations,
    raw_observation_frame,
    raw_observation_pending_roots,
)
from polylogue.storage.derived.raw import RawFrame, RawObservationDerivation, RawObservationReplacement
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.archive_templates import bootstrap_archive_root


def _admit(root: Path, names: tuple[str, ...], *, path: str = "bundle.json") -> str:
    payload = [
        {
            "id": name,
            "title": name,
            "create_time": 1,
            "current_node": "m",
            "mapping": {
                "m": {
                    "id": "m",
                    "parent": None,
                    "children": [],
                    "message": {
                        "id": "m",
                        "author": {"role": "user"},
                        "create_time": 1,
                        "content": {"content_type": "text", "parts": [name]},
                    },
                },
            },
        }
        for name in names
    ]
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        return archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=json.dumps(payload).encode(),
            source_path=path,
            acquired_at_ms=1,
        )


def _run(root: Path) -> DerivationReport:
    return converge(DerivationRegistry((RawObservationDerivation(root),)), raw_observation_frame(root))


def _snapshot(root: Path) -> tuple[tuple[tuple[object, ...], ...], ...]:
    with sqlite3.connect(root / "source.db") as conn:
        return tuple(
            tuple(conn.execute(f"SELECT * FROM {table} ORDER BY raw_id"))
            for table in (
                "raw_sessions",
                "raw_session_memberships",
                "raw_membership_census",
                "raw_authority_parser_census",
            )
        )


def test_split_member_loss_is_recovered_by_kernel_without_legacy_scanner(tmp_path: Path) -> None:
    """Anti-vacuity: a raw-id/session-exists probe misses the lost split member."""
    bootstrap_archive_root(tmp_path)
    raw_id = _admit(tmp_path, ("split-a", "split-b"))

    first = _run(tmp_path)
    assert first.done == 1 and first.failed == 0
    with sqlite3.connect(tmp_path / "index.db") as conn:
        conn.execute("DELETE FROM sessions WHERE native_id = 'split-b'")
        conn.commit()
    adapter = RawObservationDerivation(tmp_path)
    frame = raw_observation_frame(tmp_path)
    assert adapter.inspect(frame, (raw_id,))[raw_id] == "missing"
    restored = _run(tmp_path)
    assert restored.done == 1 and restored.failed == 0
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT native_id FROM sessions ORDER BY native_id").fetchall() == [
            ("split-a",),
            ("split-b",),
        ]
    before = _snapshot(tmp_path)
    unchanged = _run(tmp_path)
    assert unchanged.work.computed == unchanged.work.published == 0
    assert _snapshot(tmp_path) == before


def test_one_pass_replays_a_shared_raw_component_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A sibling made current by the first replay must not replay the component again.

    Anti-vacuity: process the initially stale status of every raw ID without
    rechecking before preparation, and the second raw below invokes the real
    replay route a second time despite the first publication having already
    made its whole authoritative component current.
    """
    from polylogue.sources import revision_backfill

    bootstrap_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_ids = tuple(
            archive.write_raw_payload(
                provider=Provider.CHATGPT,
                payload=b"[]",
                source_path="shared-component.json",
                source_index=index,
                acquired_at_ms=1,
            )
            for index in range(2)
        )
        component, _logical_keys = archive.expand_raw_membership_selection([raw_ids[0]])
    assert set(component) == set(raw_ids)

    replay = Mock(wraps=revision_backfill.backfill_historical_revision_evidence)
    monkeypatch.setattr(revision_backfill, "backfill_historical_revision_evidence", replay)

    report = converge(
        DerivationRegistry((RawObservationDerivation(tmp_path),)),
        raw_observation_frame(tmp_path),
        budget=Budget(page=2, discovery=2, inspection=4, compute=2, publication=2),
    )

    assert report.done == 2 and report.failed == report.pending == 0
    assert replay.call_count == 1
    adapter = RawObservationDerivation(tmp_path)
    assert adapter.inspect(raw_observation_frame(tmp_path), raw_ids) == dict.fromkeys(raw_ids, "valid")


def test_restart_without_ops_hints_recovers_index_loss_and_new_admission(tmp_path: Path) -> None:
    """Anti-vacuity: retained pending caches hide reset or later admissions."""
    bootstrap_archive_root(tmp_path)
    _admit(tmp_path, ("first",))
    assert _run(tmp_path).done == 1
    with sqlite3.connect(tmp_path / "ops.db") as conn:
        conn.execute("DELETE FROM convergence_debt")
        conn.commit()
    with sqlite3.connect(tmp_path / "index.db") as conn:
        conn.execute("DELETE FROM sessions")
        conn.commit()
    _admit(tmp_path, ("second",), path="second.json")
    restarted = _run(tmp_path)
    assert restarted.done == 2 and restarted.failed == 0
    assert _run(tmp_path).made_no_publication_attempts


def test_missing_prepared_raw_retries_without_quarantining_source(tmp_path: Path) -> None:
    """A lost worker carrier must not become a parser verdict about retained bytes."""
    from polylogue.sources.revision_backfill import (
        RetainedPreparationRetryableError,
        backfill_historical_revision_evidence,
    )

    bootstrap_archive_root(tmp_path)
    raw_id = _admit(tmp_path, ("prepared-retry",))
    with pytest.raises(RetainedPreparationRetryableError, match="missing"):
        backfill_historical_revision_evidence(tmp_path, selected_raw_ids=[raw_id], prepared_inputs={})
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT parse_error FROM raw_sessions WHERE raw_id = ?", (raw_id,)).fetchone() == (None,)
        assert conn.execute(
            "SELECT COUNT(*) FROM raw_authority_parser_census WHERE raw_id = ?", (raw_id,)
        ).fetchone() == (0,)


def test_retained_jsonl_replay_consumes_worker_carrier_without_inline_parse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Losing the strict carrier path would reparse this raw inside replay."""
    from polylogue.sources import revision_backfill

    bootstrap_archive_root(tmp_path)
    payload = (
        b'{"type":"session_meta","payload":{"id":"prepared-session"}}\n'
        b'{"type":"response_item","payload":{"type":"message","role":"user",'
        b'"content":[{"type":"input_text","text":"prepared text"}]}}\n'
    )
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX, payload=payload, source_path="prepared-session.jsonl", acquired_at_ms=1
        )
    adapter = RawObservationDerivation(tmp_path)
    frame = raw_observation_frame(tmp_path)
    replacement = adapter.compute(frame, raw_id)
    assert replacement.prepared_inputs is not None
    monkeypatch.setattr(
        revision_backfill,
        "parse_retained_raw_sessions",
        lambda *_args: pytest.fail("retained replay parsed inline"),
    )
    assert adapter.publish(frame, replacement)
    assert replacement.scratch_directory is not None and not replacement.scratch_directory.exists()
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT native_id FROM sessions").fetchone() == ("prepared-session",)


def test_retained_worker_exit_keeps_raw_retryable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A dead process reports preparation failure without a source parser refusal."""
    from concurrent.futures.process import BrokenProcessPool

    from polylogue.sources.revision_backfill import RetainedPreparationRetryableError
    from polylogue.storage.derived import raw as raw_module

    class DeadPool:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def __enter__(self) -> DeadPool:
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def submit(self, *_args: object) -> DeadPool:
            return self

        def result(self, **_kwargs: object) -> str:
            raise BrokenProcessPool("worker died")

    bootstrap_archive_root(tmp_path)
    payload = (
        b'{"type":"session_meta","payload":{"id":"worker-exit"}}\n'
        b'{"type":"response_item","payload":{"type":"message","role":"user",'
        b'"content":[{"type":"input_text","text":"retry"}]}}\n'
    )
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX, payload=payload, source_path="worker-exit.jsonl", acquired_at_ms=1
        )
    monkeypatch.setattr(raw_module, "ProcessPoolExecutor", DeadPool)
    with pytest.raises(RetainedPreparationRetryableError, match="worker exited"):
        RawObservationDerivation(tmp_path).compute(raw_observation_frame(tmp_path), raw_id)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT parse_error FROM raw_sessions WHERE raw_id = ?", (raw_id,)).fetchone() == (None,)


def test_retained_blob_io_failure_retries_without_quarantine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A transient blob read error must not become a durable parser refusal."""
    from polylogue.sources.revision_backfill import RetainedPreparationRetryableError
    from polylogue.storage.blob_publication import ArchiveBlobPublisher
    from polylogue.storage.derived import raw as raw_module

    class InlinePool:
        def __init__(self, **_kwargs: object) -> None:
            self._task: Callable[..., tuple[str | None, str | None]] | None = None
            self._args: tuple[object, ...] = ()

        def __enter__(self) -> InlinePool:
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def submit(self, task: Callable[..., tuple[str | None, str | None]], *args: object) -> InlinePool:
            self._task, self._args = task, args
            return self

        def result(self, **_kwargs: object) -> tuple[str | None, str | None]:
            assert self._task is not None
            return self._task(*self._args)

    bootstrap_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b'{"type":"session_meta","payload":{"id":"io-retry"}}\n',
            source_path="io-retry.jsonl",
            acquired_at_ms=1,
        )

    def fail_blob_open(*_args: object) -> None:
        raise OSError(errno.EMFILE, "too many open files")

    monkeypatch.setattr(raw_module, "ProcessPoolExecutor", InlinePool)
    monkeypatch.setattr(ArchiveBlobPublisher, "open", fail_blob_open)
    with pytest.raises(RetainedPreparationRetryableError, match="read failed"):
        RawObservationDerivation(tmp_path).compute(raw_observation_frame(tmp_path), raw_id)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT parse_error FROM raw_sessions WHERE raw_id = ?", (raw_id,)).fetchone() == (None,)
        assert conn.execute("SELECT COUNT(*) FROM raw_membership_census WHERE raw_id = ?", (raw_id,)).fetchone() == (0,)


def test_retained_parser_error_keeps_semantic_quarantine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A parser verdict from a live worker follows the canonical source census."""
    from polylogue.storage.derived import raw as raw_module

    class ParserRefusalPool:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def __enter__(self) -> ParserRefusalPool:
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def submit(self, *_args: object) -> ParserRefusalPool:
            return self

        def result(self, **_kwargs: object) -> tuple[None, str]:
            return None, "synthetic parser refusal"

    bootstrap_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX, payload=b"{bad json}\n", source_path="bad-session.jsonl", acquired_at_ms=1
        )
    adapter = RawObservationDerivation(tmp_path)
    frame = raw_observation_frame(tmp_path)
    monkeypatch.setattr(raw_module, "ProcessPoolExecutor", ParserRefusalPool)
    replacement = adapter.compute(frame, raw_id)
    assert replacement.prepared_inputs is not None
    assert adapter.publish(frame, replacement)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT status FROM raw_membership_census WHERE raw_id = ?", (raw_id,)).fetchone() == (
            "failed",
        )


@pytest.mark.parametrize("mutation", ["descriptor", "blob", "generation"])
def test_publish_rejects_changed_source_or_generation(tmp_path: Path, mutation: str) -> None:
    """Anti-vacuity: bypassing publish revalidation materializes stale inputs."""
    bootstrap_archive_root(tmp_path)
    raw_id = _admit(tmp_path, ("bound",))
    adapter = RawObservationDerivation(tmp_path)
    frame = raw_observation_frame(tmp_path)
    prepared = adapter.compute(frame, raw_id)
    if mutation == "descriptor":
        with sqlite3.connect(tmp_path / "source.db") as conn:
            conn.execute("UPDATE raw_sessions SET source_path = 'changed.json' WHERE raw_id = ?", (raw_id,))
            conn.commit()
    elif mutation == "blob":
        with ArchiveStore.open_existing(tmp_path, read_only=True) as archive:
            _provider, blob_hash, _path, _kind, _size = archive.raw_revision_descriptor(raw_id)
            blob_path = archive.blob_path_for_hash(blob_hash)
            assert blob_path is not None
            blob_path.write_bytes(b"changed")
    else:
        from dataclasses import replace

        frame = replace(frame, source_revision=str(tmp_path / "another-index.db"))
    assert adapter.publish(frame, prepared) is False
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone() == (0,)


def test_poison_observation_does_not_suppress_healthy_sibling(tmp_path: Path) -> None:
    bootstrap_archive_root(tmp_path)
    _admit(tmp_path, ("healthy",), path="healthy.json")
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=b"not json",
            source_path="poison.json",
            acquired_at_ms=1,
        )
    report = _run(tmp_path)
    assert report.done == 1 and report.failed == 1
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT native_id FROM sessions").fetchall() == [("healthy",)]


def test_zero_output_requires_parser_evidence(tmp_path: Path) -> None:
    """Anti-vacuity: a bare empty index cannot certify a zero-output raw."""
    bootstrap_archive_root(tmp_path)
    raw_id = _admit(tmp_path, ())
    assert _run(tmp_path).done == 1
    adapter = RawObservationDerivation(tmp_path)
    frame = raw_observation_frame(tmp_path)
    assert adapter.inspect(frame, (raw_id,))[raw_id] == "valid"
    assert _run(tmp_path).work.published == 0
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute("DELETE FROM raw_membership_census WHERE raw_id = ?", (raw_id,))
        conn.execute("DELETE FROM raw_artifacts WHERE raw_id = ?", (raw_id,))
        conn.commit()
    assert adapter.inspect(frame, (raw_id,))[raw_id] == "stale"


def test_inspection_rejects_stale_parser_recipe_and_excess_identity(tmp_path: Path) -> None:
    bootstrap_archive_root(tmp_path)
    raw_id = _admit(tmp_path, ("expected",))
    assert _run(tmp_path).done == 1
    adapter = RawObservationDerivation(tmp_path)
    frame = raw_observation_frame(tmp_path)
    with sqlite3.connect(tmp_path / "index.db") as conn:
        conn.execute("UPDATE sessions SET parser_fingerprint = 'stale'")
        conn.commit()
    assert adapter.inspect(frame, (raw_id,))[raw_id] == "stale"
    with sqlite3.connect(tmp_path / "index.db") as conn:
        conn.execute("UPDATE sessions SET native_id = 'excess'")
        conn.commit()
    assert adapter.inspect(frame, (raw_id,))[raw_id] == "excess"


def test_existing_head_does_not_certify_missing_observation_application(tmp_path: Path) -> None:
    """Anti-vacuity: a valid sibling head cannot replace this raw's decision."""
    bootstrap_archive_root(tmp_path)
    raw_id = _admit(tmp_path, ("application",))
    assert _run(tmp_path).done == 1
    with sqlite3.connect(tmp_path / "index.db") as conn:
        conn.execute("DELETE FROM raw_revision_applications WHERE raw_id = ?", (raw_id,))
        conn.commit()
    adapter = RawObservationDerivation(tmp_path)
    assert adapter.inspect(raw_observation_frame(tmp_path), (raw_id,))[raw_id] == "missing"
    assert _run(tmp_path).done == 1


def test_discovery_budget_bounds_raw_enumeration(tmp_path: Path) -> None:
    """A discovery bound stops enumeration mid-domain and defers, never drops, the rest.

    Anti-vacuity: let the adapter enumerate past its page/discovery bound, mark
    the cursor swept while keys remain unread, or fail to reach the deferred
    raws on later passes, and this goes red.
    """
    bootstrap_archive_root(tmp_path)
    for index in range(5):
        _admit(tmp_path, (f"session-{index}",), path=f"{index}.json")

    registry = DerivationRegistry((RawObservationDerivation(tmp_path),))
    budget = Budget(page=2, discovery=2, inspection=2, compute=2)

    report = converge(registry, raw_observation_frame(tmp_path), budget=budget)

    # The bound is a bound: one pass may not read the whole five-raw domain.
    assert report.work.discovered == 2
    assert report.work.published <= 2
    assert not report.cursor.position("raw_observation").swept

    # What the bound withheld is deferred, not dropped: bounded passes reach
    # every raw, and the domain then settles with nothing left to enumerate.
    cursor = report.cursor
    for _ in range(16):
        report = converge(registry, raw_observation_frame(tmp_path), budget=budget, cursor=cursor)
        cursor = report.cursor

    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 5

    assert _run(tmp_path).made_no_publication_attempts


@pytest.mark.parametrize("limit", [1, 2])
def test_bounded_source_pass_publishes_every_selected_observation(tmp_path: Path, limit: int) -> None:
    """Anti-vacuity: spending the inspection budget on discovery prevents all writes."""
    bootstrap_archive_root(tmp_path)
    source = tmp_path / "source"
    for index in range(limit):
        _admit(tmp_path, (f"selected-{index}",), path=str(source / f"{index}.json"))
    _admit(tmp_path, ("outside",), path=str(tmp_path / "outside.json"))

    report = converge_raw_observations(tmp_path, source_roots=(source,), limit=limit, max_payload_bytes=1_000_000)

    assert report.failed == report.pending == 0
    assert report.done == report.work.computed == report.work.published == limit
    assert report.work.discovered == limit
    assert report.work.inspected == 2 * limit
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT native_id FROM sessions ORDER BY native_id").fetchall() == [
            (f"selected-{index}",) for index in range(limit)
        ]
    before = _snapshot(tmp_path)
    unchanged = converge_raw_observations(tmp_path, source_roots=(source,), limit=limit, max_payload_bytes=1_000_000)
    assert unchanged.made_no_publication_attempts
    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize(
    "field", ["source_revision", "accepted_source_revision", "decision_id", "accepted_content_hash"]
)
def test_inspection_requires_exact_application_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    """Red twin: bypassing receipt validation falsely certifies a forged application."""
    from polylogue.storage.derived import raw as raw_adapter

    bootstrap_archive_root(tmp_path)
    raw_id = _admit(tmp_path, ("receipt",))
    assert _run(tmp_path).done == 1
    frame = raw_observation_frame(tmp_path)
    adapter = RawObservationDerivation(tmp_path)
    assert adapter.inspect(frame, (raw_id,))[raw_id] == "valid"
    with sqlite3.connect(tmp_path / "index.db") as conn:
        value = bytes(32) if field == "accepted_content_hash" else "forged"
        conn.execute(f"UPDATE raw_revision_applications SET {field} = ? WHERE raw_id = ?", (value, raw_id))
    assert adapter.inspect(frame, (raw_id,))[raw_id] == "stale"
    monkeypatch.setattr(raw_adapter, "validate_raw_replay_application_receipt", lambda *_args: (True, ()))
    assert adapter.inspect(frame, (raw_id,))[raw_id] == "valid"


@pytest.mark.parametrize(
    "error",
    [
        "OperationalError: database is locked",
        "MembershipReplayConflictError: retained comparison",
        "RuntimeError: raw revision CAS rejected an older accepted frontier",
        "RuntimeError: membership replay cannot replace an unconvertible byte head",
        "decode: No such file or directory retained-blob",
    ],
)
def test_historical_replay_refusal_is_retried_by_canonical_inspection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: str
) -> None:
    """Red twin: removing the legacy spelling bridge strands retained source work."""
    from polylogue.storage.derived import raw as raw_adapter

    bootstrap_archive_root(tmp_path)
    raw_id = _admit(tmp_path, ("retry",))
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute("UPDATE raw_sessions SET parse_error = ? WHERE raw_id = ?", (error, raw_id))
    adapter = RawObservationDerivation(tmp_path)
    frame = raw_observation_frame(tmp_path)
    assert adapter.inspect(frame, (raw_id,))[raw_id] == "missing"
    monkeypatch.setattr(raw_adapter, "raw_replay_error_is_retryable", lambda *_args: False)
    assert adapter.inspect(frame, (raw_id,))[raw_id] == "valid"


@pytest.mark.timeout(600)
def test_all_valid_prefix_has_a_total_discovery_bound_and_continuation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Red twin: exhausting all valid pages exceeds the invocation's read bound."""
    from tests.infra.sqlite_work_counter import sqlite_work_counter

    bootstrap_archive_root(tmp_path)
    source = tmp_path / "sources"
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_ids = [
            archive.write_raw_payload(
                provider=Provider.CHATGPT,
                payload=b"[]",
                source_path=str(source / "prefix.json"),
                source_index=index,
                acquired_at_ms=1,
            )
            for index in range(4096)
        ]
    assert len(set(raw_ids)) == 4096
    # A real source component lets one canonical publication census every
    # acquired observation. The observer counts successful publication members,
    # not setup rows or a synthetic inspection verdict.
    published: list[str] = []
    publish = RawObservationDerivation.publish

    def counted_publish(
        self: RawObservationDerivation, frame: RawFrame, replacement: RawObservationReplacement
    ) -> bool:
        committed = publish(self, frame, replacement)
        if committed:
            published.extend(replacement.raw_ids)
        return committed

    with monkeypatch.context() as setup:
        setup.setattr(RawObservationDerivation, "publish", counted_publish)
        publication = converge(
            DerivationRegistry((RawObservationDerivation(tmp_path),)),
            raw_observation_frame(tmp_path, raw_ids=(raw_ids[0],)),
            budget=Budget(page=1, discovery=1, inspection=2, compute=1, publication=1),
        )
    assert publication.done == 1 and publication.failed == publication.pending == 0
    assert len(published) == len(set(published)) == 4096
    assert set(published) == set(raw_ids)
    adapter = RawObservationDerivation(tmp_path)
    frame = raw_observation_frame(tmp_path, source_roots=(source,))
    with sqlite_work_counter(step_interval=1) as indexed:
        keys, continuation = adapter.required_page(frame, cursor=None, limit=128)
    assert len(keys) == 128 and continuation is not None
    assert indexed.metric("vm_steps", "source") < 10_000, indexed.summary()
    # Red twin: LIMIT after source filtering/sorting still consumes the full
    # matching scope. Returned-row counts alone cannot detect that work.
    with sqlite_work_counter(step_interval=1) as sorted_scope:
        with sqlite3.connect(tmp_path / "source.db") as conn:
            rows = conn.execute(
                "SELECT raw_id FROM raw_sessions INDEXED BY idx_raw_sessions_source_path "
                "WHERE source_path >= ? AND source_path < ? ORDER BY raw_id LIMIT 128",
                (str(source) + "/", str(source) + "0"),
            ).fetchall()
    assert len(rows) == 128
    assert sorted_scope.metric("vm_steps", "source") > 10_000
    before = _snapshot(tmp_path)
    seen: list[str] = []
    inspect = RawObservationDerivation.inspect

    def counted(self: RawObservationDerivation, frame: RawFrame, keys: Sequence[str]) -> Mapping[str, str]:
        seen.extend(keys)
        return inspect(self, frame, keys)

    monkeypatch.setattr(RawObservationDerivation, "inspect", counted)
    continuations: dict[tuple[Path, ...], tuple[str, str | None]] = {}
    for page in range(32):
        with sqlite_work_counter(step_interval=1) as work:
            assert raw_observation_pending_roots(
                tmp_path,
                (source,),
                continuations=continuations,
                limit=128,
            ) == {source}
        assert len(seen) == (page + 1) * 128
        assert work.metric("vm_steps", "source") < 500_000, work.summary()
    assert raw_observation_pending_roots(tmp_path, (source,), continuations=continuations, limit=128) == set()
    assert len(seen) == len(set(seen)) == 4096
    assert _snapshot(tmp_path) == before

    def exhaustive(root: Path) -> set[Path]:
        adapter = RawObservationDerivation(root)
        frame = raw_observation_frame(root, source_roots=(source,))
        cursor = None
        while True:
            keys, cursor = adapter.required_page(frame, cursor=cursor, limit=128)
            adapter.inspect(frame, keys)
            if cursor is None:
                return set()

    seen.clear()
    with sqlite_work_counter(step_interval=1) as unbounded:
        assert exhaustive(tmp_path) == set()
    assert len(seen) > 128
    assert unbounded.metric("vm_steps", "source") > 500_000

    # A fresh traversal must not treat its first all-valid page as ready when
    # an output obligation follows the long valid prefix.
    late = _admit(tmp_path, ("late-obligation",), path=str(source / "zz-late.json"))
    restarted_continuations: dict[tuple[Path, ...], tuple[str, str | None]] = {}
    for _ in range(33):
        assert raw_observation_pending_roots(
            tmp_path,
            (source,),
            continuations=restarted_continuations,
            limit=128,
        ) == {source}
    assert seen[-1] == late
    report = converge_raw_observations(
        tmp_path, source_roots=(source / "zz-late.json",), limit=2, max_payload_bytes=64 * 1024 * 1024
    )
    assert report.done == 1 and report.failed == report.pending == 0
    assert (
        raw_observation_pending_roots(
            tmp_path,
            (source,),
            continuations=restarted_continuations,
            limit=128,
        )
        == set()
    )
