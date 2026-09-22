"""The daemon's cold build as an owned inactive generation (polylogue-b7dkb).

The active-generation cold-build shape (``test_live_cold_build_route.py``)
cannot take index deferral, ``journal_mode=MEMORY`` or ``locking_mode=
EXCLUSIVE``, because live readers hold the file it writes. These tests pin the
shape that can: the same dispatcher-fed ``LiveBatchProcessor.ingest_files``
pass, writing its index rows into a generation created by
``IndexGenerationStore`` and invisible to readers until ``promote()``.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from polylogue.maintenance.candidate_capacity import (
    InsufficientCapacityError,
    read_capacity_receipts,
)
from polylogue.maintenance.source_manifest_continuity import (
    WANTED_SOURCE_RECEIPT_DIRNAME,
    WANTED_SOURCE_RECEIPT_FILENAME,
    SourceDeclaration,
    SourceRole,
    WantedSourceReceiptError,
    write_wanted_source_receipt,
)
from polylogue.sources.live import WatchSource
from polylogue.sources.live.batch import LiveBatchProcessor
from polylogue.sources.live.cold_build import (
    ColdBuildGeneration,
    active_index_generation_is_empty,
    clear_cold_build_generation,
    register_cold_build_generation,
)
from polylogue.sources.live.cursor import CursorStore
from polylogue.storage.archive_identity import GENERATIONS_DIRNAME, MAINTENANCE_STATE_DIRNAME
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore


def _codex_session(native_id: str, text: str) -> bytes:
    return (
        f'{{"type":"session_meta","payload":{{"id":"{native_id}",'
        '"timestamp":"2026-06-02T00:00:00Z"}}\n'
        '{"type":"response_item","payload":{"type":"message","id":"message-0",'
        f'"role":"user","content":[{{"type":"input_text","text":"{text}"}}]}}}}\n'
    ).encode()


def _processor(archive_root: Path, root: Path) -> LiveBatchProcessor:
    index_db = archive_root / "index.db"
    return LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=archive_root, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=CursorStore(index_db),
        parser_fingerprint="test-parser",
    )


def _active_session_count(archive_root: Path) -> int:
    """Count sessions the way a reader resolving the active pointer sees them."""
    from polylogue.storage.archive_identity import resolve_active_index_path

    conn = sqlite3.connect(f"file:{resolve_active_index_path(archive_root)}?mode=ro", uri=True)
    try:
        return int(conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0])
    finally:
        conn.close()


def _candidate_index_names(generation: ColdBuildGeneration) -> set[str]:
    conn = sqlite3.connect(f"file:{generation.generation.index_path}?mode=ro", uri=True)
    try:
        return {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'")}
    finally:
        conn.close()


@pytest.fixture
def cold_build(tmp_path: Path) -> Iterator[ColdBuildGeneration]:
    assert active_index_generation_is_empty(tmp_path) is True
    generation = ColdBuildGeneration.begin(tmp_path, reason="test")
    register_cold_build_generation(generation)
    try:
        yield generation
    finally:
        clear_cold_build_generation()


def _ingest(archive_root: Path, root: Path, name: str, native_id: str) -> None:
    (root / name).write_bytes(_codex_session(native_id, native_id))
    metrics = asyncio.run(_processor(archive_root, root).ingest_files([root / name], emit_event=False))
    assert metrics.succeeded_file_count == 1, metrics


def test_the_live_pass_writes_into_the_owned_generation_not_the_active_one(
    tmp_path: Path, cold_build: ColdBuildGeneration
) -> None:
    """Readers keep the previous active generation for the whole build.

    Anti-vacuity: making ``_open_archive_for_live_write`` ignore the
    registered generation (returning ``open_active_cold_build``) writes the
    rows into the active index and makes the during-build assertion red;
    deleting the ``promote()`` pointer swap makes the after-promotion one red.
    """
    root = tmp_path / "sessions"
    root.mkdir()
    _ingest(tmp_path, root, "one.jsonl", "owned-one")

    # During the build the candidate holds the row and no reader can see it.
    assert cold_build.session_count() == 1
    assert _active_session_count(tmp_path) == 0
    with ArchiveStore.open_existing(tmp_path, read_only=True) as reader:
        assert reader._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0

    promoted = cold_build.promote()
    assert promoted.state == "active"
    assert _active_session_count(tmp_path) == 1
    with ArchiveStore.open_existing(tmp_path, read_only=True) as reader:
        assert reader._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1


def test_the_build_spans_passes_and_keeps_the_deferred_indexes_dropped(
    tmp_path: Path, cold_build: ColdBuildGeneration
) -> None:
    """A cold build is many dispatcher pages against one generation.

    The second page re-opens a generation that now has rows; the deferral must
    survive that rather than being refused for non-emptiness or silently
    recreated by the runtime-index ensure.

    Anti-vacuity: restoring the ``SELECT 1 FROM sessions`` refusal in the
    deferral branch makes the second ingest raise; letting
    ``ensure_runtime_indexes_sync`` run on a deferring open makes the
    ``idx_messages_role`` assertion red.
    """
    root = tmp_path / "sessions"
    root.mkdir()
    _ingest(tmp_path, root, "one.jsonl", "owned-first")
    _ingest(tmp_path, root, "two.jsonl", "owned-second")

    assert cold_build.session_count() == 2
    assert "idx_messages_role" not in _candidate_index_names(cold_build)

    cold_build.promote()
    assert _active_session_count(tmp_path) == 2


def test_the_readiness_pass_restores_the_reader_shape_before_promotion(
    tmp_path: Path, cold_build: ColdBuildGeneration
) -> None:
    """One CREATE INDEX pass and one FTS build, then the generation is ordinary.

    A read-only open projects ``sqlite_master`` including indexes, so a
    promoted generation still missing its deferred indexes refuses with a
    schema mismatch -- which is exactly why the deferral needs the owned
    generation in the first place.

    Anti-vacuity: deleting ``restore_deferred_secondary_indexes_sync`` from
    ``run_generation_readiness_pass`` makes the read-only open raise; deleting
    the FTS rebuild makes the search assertion red.
    """
    root = tmp_path / "sessions"
    root.mkdir()
    _ingest(tmp_path, root, "one.jsonl", "owned-ready")

    cold_build.promote()

    assert "idx_messages_role" in _candidate_index_names(cold_build)
    with ArchiveStore.open_existing(tmp_path, read_only=True) as reader:
        hits = reader._conn.execute("SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH 'owned'").fetchone()
        assert hits[0] >= 1


def test_a_never_promoted_generation_is_discarded_and_leaves_readers_alone(
    tmp_path: Path, cold_build: ColdBuildGeneration
) -> None:
    """Crash semantics: the build simply never becomes visible.

    Anti-vacuity: making ``discard`` promote instead, or having the ingest
    pass write through to the active generation, makes the active count
    assertion red.
    """
    root = tmp_path / "sessions"
    root.mkdir()
    _ingest(tmp_path, root, "one.jsonl", "owned-doomed")
    generation_root = cold_build.generation_root

    assert cold_build.discard() is True
    assert not generation_root.exists()
    assert _active_session_count(tmp_path) == 0


def test_acquisition_stays_on_the_real_durable_tiers(tmp_path: Path, cold_build: ColdBuildGeneration) -> None:
    """The candidate directory carries read-through symlinks, not a second archive.

    The cold build acquires and materializes in one pass, so the raw row and
    its blob must land in the archive's own ``source.db`` -- otherwise a
    discarded generation would take the durable evidence with it.

    Anti-vacuity: dropping ``durable_writer`` (so the store keeps the inactive
    candidate's refusing blob publisher) makes the ingest fail outright; a
    generation created before the durable tiers exist would grow its own
    ``source.db`` and make the symlink assertion red.
    """
    root = tmp_path / "sessions"
    root.mkdir()
    _ingest(tmp_path, root, "one.jsonl", "owned-durable")

    assert (cold_build.generation_root / "source.db").is_symlink()
    conn = sqlite3.connect(f"file:{tmp_path / 'source.db'}?mode=ro", uri=True)
    try:
        assert int(conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0]) >= 1
    finally:
        conn.close()


def _free_space(monkeypatch: pytest.MonkeyPatch, available_bytes: int) -> None:
    """Pin what the filesystem reports as usable for the generations root."""
    monkeypatch.setattr(
        "polylogue.maintenance.candidate_capacity._available_bytes",
        lambda _path: available_bytes,
    )


def test_the_cold_build_refuses_before_it_allocates_a_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The production build route is the one that has to refuse on free space.

    The guard used to sit only on the manual rebuild lifecycle, which has no
    production caller, so before this lift a real daemon cold build -- the
    route that writes a second whole index beside the one serving reads --
    allocated with no headroom check at all.

    Anti-vacuity: deleting the ``require_candidate_capacity`` call from
    ``ColdBuildGeneration.begin`` makes this red -- ``begin`` returns a live
    generation instead of raising, and a ``gen-*`` directory appears.
    """
    assert active_index_generation_is_empty(tmp_path) is True
    _free_space(monkeypatch, 0)

    with pytest.raises(InsufficientCapacityError) as refusal:
        ColdBuildGeneration.begin(tmp_path, reason="test")

    assert refusal.value.projection.shortfall_bytes > 0
    assert list((tmp_path / GENERATIONS_DIRNAME).glob("gen-*")) == []
    # A refused build records nothing: a prediction receipt for a build that
    # never ran would calibrate the next projection from fiction.
    assert read_capacity_receipts(tmp_path) == ()


def test_a_first_daemon_start_is_not_refused_by_the_preflight(tmp_path: Path) -> None:
    """The preflight is unconditional, so it must clear a fresh root on its own.

    A freshly bootstrapped archive needs 276.8 MiB (the 256 MiB reserve floor
    plus the 16 MiB receipt floor dominate an evidence-proportional term of
    under 4 MiB), which is why this guard does not need an operator-intent
    gate to avoid blocking an ordinary first start.

    Anti-vacuity: raising ``RESERVE_FLOOR_BYTES`` above real free space, or
    making the projection scale from the filesystem rather than the archive,
    makes this red.
    """
    generation = ColdBuildGeneration.begin(tmp_path, reason="test")
    try:
        receipts = read_capacity_receipts(tmp_path)
        assert [receipt.operation_id for receipt in receipts] == [generation.operation_id]
        receipt = receipts[0]
        assert receipt.required_free_bytes < 512 * 1024 * 1024
        assert receipt.available_bytes_at_prediction >= receipt.required_free_bytes
        assert receipt.actual_peak_index_bytes == 0
    finally:
        generation.discard()


def test_a_promoted_cold_build_calibrates_the_next_projection(tmp_path: Path, cold_build: ColdBuildGeneration) -> None:
    """Prediction without observation leaves ``calibrated_index_ratio`` at its default.

    ``record_capacity_observation`` used to have exactly one caller, on the
    manual rebuild lifecycle that had no production entry point, so every
    recorded receipt kept ``actual_peak_index_bytes == 0`` and every
    projection forever used the unmeasured 4.0 constant.

    Anti-vacuity: deleting the ``observe_candidate_capacity`` call from
    ``ColdBuildGeneration.promote`` leaves the peak at zero and the
    calibration source at ``default``, making both assertions red.
    """
    from polylogue.maintenance.candidate_capacity import calibrated_index_ratio

    root = tmp_path / "sessions"
    root.mkdir()
    _ingest(tmp_path, root, "one.jsonl", "owned-calibrated")
    assert calibrated_index_ratio(tmp_path) == (4.0, "default")

    cold_build.promote()

    receipt = read_capacity_receipts(tmp_path)[0]
    assert receipt.operation_id == cold_build.operation_id
    assert receipt.actual_peak_index_bytes > 0
    assert receipt.observations == 1
    ratio, source = calibrated_index_ratio(tmp_path)
    assert source == "recorded"
    assert ratio > 0


def test_a_failed_capacity_observation_does_not_block_promotion(
    tmp_path: Path, cold_build: ColdBuildGeneration, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Calibration is evidence about the next build, never a gate on this one.

    Anti-vacuity: letting the observation error escape
    ``observe_candidate_capacity`` makes ``promote`` raise ``OSError`` here.
    """
    root = tmp_path / "sessions"
    root.mkdir()
    _ingest(tmp_path, root, "one.jsonl", "owned-unmeasured")

    def explode(*_args: object, **_kwargs: object) -> None:
        raise OSError("receipt directory is gone")

    monkeypatch.setattr("polylogue.maintenance.candidate_capacity.record_capacity_observation", explode)

    assert cold_build.promote().state == "active"
    assert _active_session_count(tmp_path) == 1


def _declare_sources(monkeypatch: pytest.MonkeyPatch, *roots: Path) -> tuple[SourceDeclaration, ...]:
    """Pin what ``config.source_declarations`` returns for one test.

    The declarations are the shape ``config.source_declarations`` builds from
    ``source_paths.explicit`` -- the only thing the campaign policy admits into
    the denominator.
    """
    declarations = tuple(
        SourceDeclaration(f"configured-{position}", SourceRole.DIRECTORY, root, True)
        for position, root in enumerate(roots)
    )
    monkeypatch.setattr("polylogue.config.configured_source_declarations", lambda _runtime: declarations)
    return declarations


def _declared_source_root(tmp_path: Path) -> Path:
    root = tmp_path / "declared"
    root.mkdir()
    (root / "one.jsonl").write_bytes(_codex_session("declared-one", "declared"))
    return root


def _fresh_archive_root(tmp_path: Path) -> Path:
    archive = tmp_path / "archive"
    archive.mkdir()
    return archive


def _receipt_path(archive_root: Path) -> Path:
    return archive_root / MAINTENANCE_STATE_DIRNAME / WANTED_SOURCE_RECEIPT_DIRNAME / WANTED_SOURCE_RECEIPT_FILENAME


def test_cold_build_refuses_an_unfrozen_denominator(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Declared source roots and no receipt is a refusal, not a fresh walk.

    ``require_rebuild_preflight`` existed with exactly one caller -- the
    operator CLI -- so the build driver that the frozen denominator is *for*
    started with no authorization at all (polylogue-co2iz AC3).

    Anti-vacuity: deleting the ``_require_frozen_wanted_sources`` call from
    ``ColdBuildGeneration.begin`` makes this red -- ``begin`` returns a live
    generation instead of raising, and a ``gen-*`` directory appears.
    """
    archive = _fresh_archive_root(tmp_path)
    _declare_sources(monkeypatch, _declared_source_root(tmp_path))

    with pytest.raises(WantedSourceReceiptError) as refusal:
        ColdBuildGeneration.begin(archive, reason="empty active index generation")

    detail = str(refusal.value)
    assert "wanted-source receipt is missing" in detail
    assert "1 declared source root(s)" in detail
    # The refusal has to be actionable: one command produces a receipt.
    assert "polylogue ops maintenance wanted-sources --freeze" in detail
    assert list((archive / GENERATIONS_DIRNAME).glob("gen-*")) == []
    # Refused before the capacity walk, so nothing was predicted either.
    assert read_capacity_receipts(archive) == ()


def test_cold_build_refuses_a_tampered_receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A published receipt is re-verified by the build, not trusted by presence.

    Anti-vacuity: accepting the receipt on existence (or catching the
    ``WantedSourceReceiptError`` and continuing) makes this red -- the edited
    denominator authorizes a build whose conservation proof would then be
    measured against a number nobody enumerated.
    """
    archive = _fresh_archive_root(tmp_path)
    declarations = _declare_sources(monkeypatch, _declared_source_root(tmp_path))
    write_wanted_source_receipt(archive, declarations)
    receipt_path = _receipt_path(archive)
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload["item_count"] = int(payload["item_count"]) + 41
    receipt_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(WantedSourceReceiptError) as refusal:
        ColdBuildGeneration.begin(archive, reason="explicit cold build")

    assert "denominators mismatch" in str(refusal.value)
    assert list((archive / GENERATIONS_DIRNAME).glob("gen-*")) == []


def test_a_frozen_receipt_authorizes_the_cold_build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The authorized case still builds: the guard refuses defects, not work.

    Anti-vacuity: a guard that refused whenever declarations exist -- or one
    that re-enumerated the roots instead of reading the frozen receipt, which
    would see the file added after the freeze -- makes this red.
    """
    source_root = _declared_source_root(tmp_path)
    archive = _fresh_archive_root(tmp_path)
    declarations = _declare_sources(monkeypatch, source_root)
    receipt = write_wanted_source_receipt(archive, declarations)
    assert receipt.item_count == 1
    (source_root / "two.jsonl").write_bytes(_codex_session("declared-two", "later"))

    generation = ColdBuildGeneration.begin(archive, reason="explicit cold build")
    try:
        assert generation.generation.state == "inactive"
    finally:
        generation.discard()


def test_an_undeclared_denominator_still_builds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No declared root means no receipt can exist, so requiring one is a deadlock.

    ``build_wanted_source_receipt`` refuses an empty selection ("no source is
    declared: the rebuild denominator would be empty"), so a strictly
    unconditional receipt requirement would make every live-capture-only
    archive permanently unbuildable -- the freeze route could never produce
    the receipt the build demands.

    Anti-vacuity: requiring a receipt regardless of the declarations makes
    this red, and ``--freeze`` cannot make it green again.
    """
    archive = _fresh_archive_root(tmp_path)
    _declare_sources(monkeypatch)
    with pytest.raises(WantedSourceReceiptError):
        write_wanted_source_receipt(archive, ())

    generation = ColdBuildGeneration.begin(archive, reason="empty active index generation")
    try:
        assert generation.generation.state == "inactive"
    finally:
        generation.discard()


def test_undeclared_denominator_build_is_reported_degraded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A build with no frozen denominator is a named gap, not a clean run.

    With no declared source root ``build_wanted_source_receipt`` itself
    refuses, so requiring a receipt here would make every live-capture-only
    archive permanently unbuildable -- the skip is deliberate. What was not
    deliberate is that it emitted ``outcome="ok"``, making an
    unauthorized-denominator whole-archive rebuild indistinguishable from an
    authorized one in the event stream.

    Anti-vacuity: restoring ``outcome="ok"`` on the
    ``wanted_sources_undeclared`` emit makes this red. The positive control
    below fails instead if the event is turned into a refusal, which would
    make live-capture-only archives unbuildable.
    """
    from polylogue.sources.live import cold_build as cold_build_module

    archive = _fresh_archive_root(tmp_path)
    emitted: list[tuple[str, dict[str, Any]]] = []

    def _record(kind: str, **fields: Any) -> None:
        emitted.append((kind, fields))

    monkeypatch.setattr(cold_build_module, "emit", _record)
    cold_build_module._require_frozen_wanted_sources(
        archive, reason="empty active index generation", operation_id="op-test"
    )

    undeclared = [fields for kind, fields in emitted if kind.endswith("wanted_sources_undeclared")]
    assert undeclared, f"no wanted_sources_undeclared event: {emitted}"
    assert undeclared[0]["outcome"] == "degraded"
    # Positive control: the build is NOT refused. A live-capture-only archive
    # must stay buildable, so this path returns rather than raising.
    assert undeclared[0]["sources"] == 0
