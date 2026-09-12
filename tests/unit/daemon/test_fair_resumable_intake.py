"""Fair, resumable intake: no class starves, no item wedges its siblings.

The regression this replaces gave one spool class absolute priority: while
any browser-capture file existed, raw materialization was skipped entirely.
Four undrainable files stopped unrelated work for as long as they sat
there. These tests are written so that restoring class-global preemption,
disabling weighted fairness, or losing the isolation of a poison item makes
one of them red.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from polylogue.core.enums import Provider
from polylogue.daemon.derivation import DerivationFrame
from polylogue.daemon.intake import (
    AdmissionOutcome,
    AdmissionResult,
    FairIntakeDispatcher,
    IntakeClassSpec,
    IntakeItem,
)
from polylogue.daemon.observation import ObservationBoard, ObservationState
from polylogue.daemon.service_halt import HaltReason, HaltRegistry, UnitKind, unit_id
from polylogue.operations.intake_adapters import (
    RawMaterializationDiscovery,
    RawMaterializationIntakeAdapter,
    _bounded_source_paths,
    discover_pending_raw_ids,
)
from polylogue.sources.live.watcher import WatchSource
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.archive_templates import bootstrap_archive_root


class FakeAdapter:
    """A spool whose queue authority is its own pending list."""

    def __init__(
        self,
        class_name: str,
        pending: Sequence[str],
        *,
        outcome_for: Callable[[IntakeItem], AdmissionResult] | None = None,
    ) -> None:
        self.class_name = class_name
        self.pending = list(pending)
        self.acknowledged: list[str] = []
        self.admitted: list[str] = []
        self.discover_calls: list[int] = []
        self._outcome_for = outcome_for or (lambda _item: AdmissionResult(AdmissionOutcome.ADMITTED))

    async def discover(self, *, limit: int) -> Sequence[IntakeItem]:
        self.discover_calls.append(limit)
        return [IntakeItem(item_id=name, class_name=self.class_name) for name in self.pending[:limit]]

    async def admit(self, item: IntakeItem) -> AdmissionResult:
        result = self._outcome_for(item)
        if result.outcome is AdmissionOutcome.ADMITTED:
            self.admitted.append(item.item_id)
        return result

    async def acknowledge(self, item: IntakeItem) -> None:
        self.acknowledged.append(item.item_id)
        # Atomic and idempotent: acknowledging twice releases one entry.
        if item.item_id in self.pending:
            self.pending.remove(item.item_id)


def test_bounded_source_paths_prunes_ignored_subtrees_and_keeps_nested_sources(tmp_path: Path) -> None:
    """An ignored directory cannot consume a page or hide an accepted child."""
    ignored = tmp_path / "ignored"
    ignored.mkdir()
    (ignored / "not-an-intake.json").write_text("ignored")
    nested = tmp_path / "accepted" / "deeper"
    nested.mkdir(parents=True)
    accepted = nested / "intake.json"
    accepted.write_text("accepted")
    source = WatchSource(
        name="test",
        root=tmp_path,
        suffixes=(".json",),
        ignored_dir_names=frozenset({"ignored"}),
    )

    first_page = _bounded_source_paths(source, (source,), limit=1, after=None)
    assert first_page == [accepted]
    assert _bounded_source_paths(source, (source,), limit=8, after=str(accepted)) == []


def test_raw_discovery_uses_canonical_adapter_and_returns_payload_costs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A census selector cannot hide a pending raw from fair intake."""
    bootstrap_archive_root(tmp_path)
    payloads = {
        "a.json": b"raw-a",
        "b.json": b"raw-b-longer",
        "c.json": b"raw-c",
    }
    raw_ids: dict[str, str] = {}
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        for path, payload in payloads.items():
            raw_ids[path] = archive.write_raw_payload(
                provider=Provider.CHATGPT,
                payload=payload,
                source_path=path,
                acquired_at_ms=1,
            )

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("legacy raw census selector was called")

    monkeypatch.setattr("polylogue.storage.raw_convergence.raw_materialization_pending_census_raw_ids", forbidden)
    result = discover_pending_raw_ids(tmp_path, limit=2, max_payload_bytes=1024)

    expected = tuple(
        (raw_id, len(payloads[path])) for path, raw_id in sorted(raw_ids.items(), key=lambda item: item[1])[:2]
    )
    assert result == expected


def test_raw_discovery_bounds_valid_prefix_and_resumes_after_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutation: scan until a pending raw is found, and the first call exceeds one page."""
    bootstrap_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        valid = archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=b"v",
            source_path="valid.json",
            acquired_at_ms=1,
        )
        pending = archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=b"pending",
            source_path="pending.json",
            acquired_at_ms=1,
        )
    calls: list[tuple[str | None, int]] = []

    class FakeRawObservationDerivation:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def required_page(
            self, _frame: object, *, cursor: str | None, limit: int
        ) -> tuple[tuple[str, ...], str | None]:
            calls.append((cursor, limit))
            return ((valid,), valid) if cursor is None else ((pending,), None)

        def inspect(self, _frame: object, keys: Sequence[str]) -> dict[str, str]:
            return {key: "valid" if key == valid else "missing" for key in keys}

    monkeypatch.setattr("polylogue.storage.derived.raw.RawObservationDerivation", FakeRawObservationDerivation)
    discovery = RawMaterializationDiscovery(tmp_path, max_payload_bytes=1024)

    assert discovery.discover_pending_raw_ids(1) == ()
    assert calls == [(None, 1)]
    assert discovery.discover_pending_raw_ids(1) == ((pending, len(b"pending")),)
    assert calls == [(None, 1), (valid, 1)]


@pytest.mark.asyncio
async def test_raw_discovery_moves_past_an_isolated_poison_in_the_fair_dispatcher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutation: reset discovery each pass, and the isolated head starves the healthy raw."""
    bootstrap_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        valid = archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=b"v",
            source_path="valid.json",
            acquired_at_ms=1,
        )
        poison = archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=b"p",
            source_path="poison.json",
            acquired_at_ms=1,
        )
        healthy = archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=b"h",
            source_path="healthy.json",
            acquired_at_ms=1,
        )

    class FakeRawObservationDerivation:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def required_page(
            self, _frame: object, *, cursor: str | None, limit: int
        ) -> tuple[tuple[str, ...], str | None]:
            assert limit <= 32
            if cursor is None:
                return (valid,), valid
            if cursor == valid:
                return (poison,), poison
            assert cursor == poison
            return (healthy,), None

        def inspect(self, _frame: object, keys: Sequence[str]) -> dict[str, str]:
            return {key: "valid" if key == valid else "missing" for key in keys}

    monkeypatch.setattr("polylogue.storage.derived.raw.RawObservationDerivation", FakeRawObservationDerivation)
    admitted: list[str] = []

    async def admit(raw_id: str) -> AdmissionResult:
        if raw_id == poison:
            return AdmissionResult(AdmissionOutcome.RETRYABLE, reason="poison")
        admitted.append(raw_id)
        return AdmissionResult(AdmissionOutcome.ADMITTED)

    discovery = RawMaterializationDiscovery(tmp_path, max_payload_bytes=1024)
    dispatcher = FairIntakeDispatcher(
        [
            IntakeClassSpec(
                name="raw_materialization",
                adapter=RawMaterializationIntakeAdapter(discovery.discover_pending_raw_ids, admit),
                max_attempts=1,
            )
        ]
    )

    await dispatcher.run_once(budget=1)
    await dispatcher.run_once(budget=1)
    await dispatcher.run_once(budget=1)

    assert dispatcher.isolated_items("raw_materialization") == frozenset({poison})
    assert admitted == [healthy]


def test_raw_discovery_resets_only_for_a_new_generation_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutation: reset the cursor every pass, or carry it into a replacement generation."""
    bootstrap_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        first = archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=b"first",
            source_path="first.json",
            acquired_at_ms=1,
        )
        second = archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=b"second",
            source_path="second.json",
            acquired_at_ms=1,
        )
    frames = iter(
        (
            DerivationFrame(str(tmp_path), "index-v1", recipe_versions={"raw_observation": "recipe-v1"}),
            DerivationFrame(str(tmp_path), "index-v1", recipe_versions={"raw_observation": "recipe-v1"}),
            DerivationFrame(str(tmp_path), "index-v2", recipe_versions={"raw_observation": "recipe-v1"}),
        )
    )
    cursors: list[str | None] = []

    class FakeRawObservationDerivation:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def required_page(
            self, _frame: object, *, cursor: str | None, limit: int
        ) -> tuple[tuple[str, ...], str | None]:
            cursors.append(cursor)
            return ((first,), first) if cursor is None else ((second,), None)

        def inspect(self, _frame: object, keys: Sequence[str]) -> dict[str, str]:
            return dict.fromkeys(keys, "missing")

    monkeypatch.setattr(
        "polylogue.operations.raw_observation_derivation.raw_observation_frame",
        lambda _archive_root: next(frames),
    )
    monkeypatch.setattr("polylogue.storage.derived.raw.RawObservationDerivation", FakeRawObservationDerivation)
    discovery = RawMaterializationDiscovery(tmp_path, max_payload_bytes=1024)

    assert discovery.discover_pending_raw_ids(1)[0][0] == first
    assert discovery.discover_pending_raw_ids(1)[0][0] == second
    assert discovery.discover_pending_raw_ids(1)[0][0] == first
    assert cursors == [None, first, None]


@pytest.mark.asyncio
async def test_intake_coalesces_hints_received_during_discovery() -> None:
    """Clearing a wake after discovery loses the event and waits sixty seconds."""
    from polylogue.daemon.intake_adapters import DaemonIntakeService

    wakeup = asyncio.Event()
    rediscovered = asyncio.Event()

    class HintingAdapter(FakeAdapter):
        async def discover(self, *, limit: int) -> Sequence[IntakeItem]:
            self.discover_calls.append(limit)
            if len(self.discover_calls) == 1:
                for _ in range(20):
                    wakeup.set()
            else:
                rediscovered.set()
            return ()

    adapter = HintingAdapter("local", [])
    service = DaemonIntakeService(
        FairIntakeDispatcher([IntakeClassSpec(name="local", adapter=adapter)]),
        wakeup=wakeup,
        idle_delay_s=60,
    )
    task = asyncio.create_task(service.run())
    try:
        await asyncio.wait_for(rediscovered.wait(), timeout=1)
        await asyncio.sleep(0)
        assert len(adapter.discover_calls) == 2
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_a_huge_class_cannot_starve_its_siblings() -> None:
    """Mutation: give ``huge`` absolute priority and ``small`` admits nothing."""
    huge = FakeAdapter("huge", [f"h{index}" for index in range(10_000)])
    small = FakeAdapter("small", ["s0", "s1"])
    dispatcher = FairIntakeDispatcher(
        [
            IntakeClassSpec(name="huge", adapter=huge, page_size=8),
            IntakeClassSpec(name="small", adapter=small, page_size=8),
        ]
    )

    result = await dispatcher.run_once(budget=16)

    assert result.require_report("small").admitted == 2
    assert result.require_report("huge").admitted > 0
    assert small.pending == []


@pytest.mark.asyncio
async def test_no_pass_enumerates_a_whole_spool() -> None:
    """Discovery is paged; a per-tick full scan cannot converge at spool scale."""
    huge = FakeAdapter("huge", [f"h{index}" for index in range(830_789)])
    dispatcher = FairIntakeDispatcher([IntakeClassSpec(name="huge", adapter=huge, page_size=32)])

    await dispatcher.run_once(budget=64)

    assert huge.discover_calls == [32]


@pytest.mark.asyncio
async def test_weight_decides_the_share_of_one_pass() -> None:
    """Mutation: ignore ``weight`` and both classes admit the same count."""
    heavy = FakeAdapter("heavy", [f"a{index}" for index in range(100)])
    light = FakeAdapter("light", [f"b{index}" for index in range(100)])
    dispatcher = FairIntakeDispatcher(
        [
            IntakeClassSpec(name="heavy", adapter=heavy, weight=3, page_size=64),
            IntakeClassSpec(name="light", adapter=light, weight=1, page_size=64),
        ]
    )

    result = await dispatcher.run_once(budget=40)

    assert result.require_report("heavy").admitted > result.require_report("light").admitted


@pytest.mark.asyncio
async def test_a_poison_item_is_isolated_and_its_siblings_continue() -> None:
    """Mutation: drop the attempt bound and the head item blocks the class."""

    def outcome(item: IntakeItem) -> AdmissionResult:
        if item.item_id == "poison":
            return AdmissionResult(AdmissionOutcome.RETRYABLE, reason="cannot parse")
        return AdmissionResult(AdmissionOutcome.ADMITTED)

    adapter = FakeAdapter("hooks", ["poison", "good0", "good1"], outcome_for=outcome)
    dispatcher = FairIntakeDispatcher([IntakeClassSpec(name="hooks", adapter=adapter, max_attempts=2, page_size=8)])

    await dispatcher.run_once(budget=8)
    await dispatcher.run_once(budget=8)

    assert dispatcher.isolated_items("hooks") == frozenset({"poison"})
    assert adapter.admitted == ["good0", "good1", "good0", "good1"] or set(adapter.admitted) == {"good0", "good1"}
    assert adapter.pending == ["poison"]


@pytest.mark.asyncio
async def test_an_adapter_that_raises_is_one_item_retried_not_a_dead_class() -> None:
    def outcome(item: IntakeItem) -> AdmissionResult:
        if item.item_id == "boom":
            raise RuntimeError("adapter exploded")
        return AdmissionResult(AdmissionOutcome.ADMITTED)

    adapter = FakeAdapter("hooks", ["boom", "fine"], outcome_for=outcome)
    dispatcher = FairIntakeDispatcher([IntakeClassSpec(name="hooks", adapter=adapter, max_attempts=1, page_size=8)])

    result = await dispatcher.run_once(budget=8)

    assert result.require_report("hooks").admitted == 1
    assert dispatcher.isolated_items("hooks") == frozenset({"boom"})


@pytest.mark.asyncio
async def test_a_terminal_item_is_set_aside_without_further_attempts() -> None:
    def outcome(item: IntakeItem) -> AdmissionResult:
        if item.item_id == "unparseable":
            return AdmissionResult(AdmissionOutcome.TERMINAL, reason="unknown envelope version")
        return AdmissionResult(AdmissionOutcome.ADMITTED)

    adapter = FakeAdapter("browser", ["unparseable", "ok"], outcome_for=outcome)
    dispatcher = FairIntakeDispatcher([IntakeClassSpec(name="browser", adapter=adapter, page_size=8)])

    await dispatcher.run_once(budget=8)

    assert dispatcher.isolated_items("browser") == frozenset({"unparseable"})
    assert adapter.acknowledged == ["ok"]


@pytest.mark.asyncio
async def test_duplicate_delivery_is_acknowledged_without_double_admission() -> None:
    """Crash after admission but before acknowledgement replays the item."""
    adapter = FakeAdapter(
        "hooks",
        ["already"],
        outcome_for=lambda _item: AdmissionResult(AdmissionOutcome.DUPLICATE, reason="content hash present"),
    )
    dispatcher = FairIntakeDispatcher([IntakeClassSpec(name="hooks", adapter=adapter, page_size=8)])

    result = await dispatcher.run_once(budget=8)

    assert result.require_report("hooks").duplicates == 1
    assert result.require_report("hooks").admitted == 0
    assert adapter.acknowledged == ["already"]
    assert adapter.admitted == []


@pytest.mark.asyncio
async def test_losing_every_scheduling_hint_preserves_correctness() -> None:
    """Deficits, attempts and the ready set are hints; the spool is authority."""
    adapter = FakeAdapter("hooks", [f"h{index}" for index in range(6)])
    first = FairIntakeDispatcher([IntakeClassSpec(name="hooks", adapter=adapter, page_size=2)])
    await first.run_once(budget=2)

    # A restart constructs a new dispatcher with no hints at all.
    second = FairIntakeDispatcher([IntakeClassSpec(name="hooks", adapter=adapter, page_size=2)])
    while adapter.pending:
        await second.run_once(budget=2)

    assert sorted(adapter.acknowledged) == [f"h{index}" for index in range(6)]
    assert len(adapter.acknowledged) == len(set(adapter.acknowledged))


@pytest.mark.asyncio
async def test_a_class_in_terminal_refusal_is_excluded_at_selection(tmp_path: Path) -> None:
    """The claude-code wedge: refusing late is the defect.

    Mutation: consult the halt inside ``admit`` instead of
    ``schedulable_classes`` and this reddens -- discovery runs again.
    """
    halts = HaltRegistry(tmp_path)
    dead = FakeAdapter(
        "claude-code",
        ["c0", "c1"],
        outcome_for=lambda _item: AdmissionResult(
            AdmissionOutcome.CLASS_TERMINAL, reason="refusing further ingest until restart"
        ),
    )
    live = FakeAdapter("codex", ["x0", "x1"])
    dispatcher = FairIntakeDispatcher(
        [
            IntakeClassSpec(name="claude-code", adapter=dead, page_size=8),
            IntakeClassSpec(name="codex", adapter=live, page_size=8),
        ],
        halts=halts,
        frame="daemon:1",
    )

    await dispatcher.run_once(budget=8)
    discover_calls_at_halt = len(dead.discover_calls)

    for _ in range(5):
        await dispatcher.run_once(budget=8)

    assert len(dead.discover_calls) == discover_calls_at_halt, "a halted class was planned again"
    assert live.acknowledged == ["x0", "x1"]
    assert halts.is_halted(unit_id(UnitKind.INTAKE_CLASS, "claude-code"))


@pytest.mark.asyncio
async def test_a_halted_class_is_named_in_its_status_observation(tmp_path: Path) -> None:
    halts = HaltRegistry(tmp_path)
    halts.halt(
        unit_id(UnitKind.INTAKE_CLASS, "claude-code"),
        reason=HaltReason.TERMINAL_REFUSAL,
        message="refusing further ingest until restart",
        frame="daemon:1",
    )
    board = ObservationBoard()
    adapter = FakeAdapter("claude-code", ["c0"])
    dispatcher = FairIntakeDispatcher(
        [IntakeClassSpec(name="claude-code", adapter=adapter, page_size=8)],
        halts=halts,
        board=board,
    )

    result = await dispatcher.run_once(budget=8)

    assert result.skipped_halted == ("claude-code",)
    observation = board.get_or_unavailable("intake.claude-code")
    assert observation.state is ObservationState.FAILED
    assert "refusing further ingest until restart" in (observation.reason or "")
    assert observation.value is None


@pytest.mark.asyncio
async def test_a_halted_class_survives_a_restart(tmp_path: Path) -> None:
    halts = HaltRegistry(tmp_path)
    halts.halt(
        unit_id(UnitKind.INTAKE_CLASS, "claude-code"),
        reason=HaltReason.TERMINAL_REFUSAL,
        message="refusing further ingest until restart",
        frame="daemon:1",
    )

    adapter = FakeAdapter("claude-code", ["c0"])
    restarted = FairIntakeDispatcher(
        [IntakeClassSpec(name="claude-code", adapter=adapter, page_size=8)],
        halts=HaltRegistry(tmp_path),
    )

    await restarted.run_once(budget=8)

    assert adapter.discover_calls == []
    assert adapter.acknowledged == []


@pytest.mark.asyncio
async def test_a_discovery_failure_reports_rather_than_raising() -> None:
    class BrokenAdapter(FakeAdapter):
        async def discover(self, *, limit: int) -> Sequence[IntakeItem]:
            raise OSError("spool directory vanished")

    live = FakeAdapter("codex", ["x0"])
    dispatcher = FairIntakeDispatcher(
        [
            IntakeClassSpec(name="broken", adapter=BrokenAdapter("broken", []), page_size=8),
            IntakeClassSpec(name="codex", adapter=live, page_size=8),
        ]
    )

    result = await dispatcher.run_once(budget=8)

    assert "spool directory vanished" in (result.require_report("broken").reason or "")
    assert result.require_report("codex").admitted == 1


def test_duplicate_class_names_are_refused() -> None:
    adapter = FakeAdapter("hooks", [])
    with pytest.raises(ValueError, match="duplicate intake class name"):
        FairIntakeDispatcher(
            [
                IntakeClassSpec(name="hooks", adapter=adapter),
                IntakeClassSpec(name="hooks", adapter=adapter),
            ]
        )
