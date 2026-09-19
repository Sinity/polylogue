"""A writable connection is obtainable only from a held lease.

polylogue-8qm4k: the daemon claims to be the sole SQLite writer, but that was a
convention. The gate is advisory, holds were unbounded, and dispatch to the
coordinator fell open on a duck-typing miss, so a writer off the gate exhausted
the busy timeout while a holder ran for hours (a measured
``maintenance.drive_catchup`` hold of 18,623 s) and the live catch-up chunk died
with ``database is locked``.

These laws make the boundary structural: where enforcement is armed, opening a
write-mode connection without the lease raises instead of contending.
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from contextlib import closing
from pathlib import Path

import pytest

from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root, initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.connection_profile import (
    open_connection,
    open_daemon_connection,
    open_isolated_write_connection,
)
from polylogue.storage.sqlite.write_lease import (
    UnleasedWriteError,
    WriteHoldExceededError,
    adopt_write_lease,
    arm_write_lease_enforcement,
    current_write_lease,
    delegate_write_lease,
    require_write_lease,
    write_lease,
    write_lease_enforced,
)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "tier.db"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE t (x INTEGER)")
        conn.commit()
    return path


def test_an_unleased_write_open_is_refused_where_enforcement_is_armed(db_path: Path) -> None:
    """The anti-vacuity case named on the bead: a writer with no lease is a bug.

    Anti-vacuity: delete the ``require_write_lease`` call from
    ``open_connection`` and this goes green while an unserialized writer opens a
    write-mode connection exactly as before.
    """
    with arm_write_lease_enforcement():
        with pytest.raises(UnleasedWriteError):
            open_connection(db_path, validate_schema=False)
        with pytest.raises(UnleasedWriteError):
            open_daemon_connection(db_path, validate_schema=False)
        with pytest.raises(UnleasedWriteError):
            open_isolated_write_connection(db_path, purpose="probe")


def test_a_leased_write_open_succeeds(db_path: Path) -> None:
    """The lease authorizes; it does not merely record."""
    with arm_write_lease_enforcement(), write_lease("test.writer"):
        with closing(open_connection(db_path, validate_schema=False)) as conn:
            conn.execute("INSERT INTO t VALUES (1)")
            conn.commit()
    with closing(sqlite3.connect(db_path)) as conn:
        assert conn.execute("SELECT count(*) FROM t").fetchone()[0] == 1


def test_archive_insight_writer_refuses_a_different_archive_root(tmp_path: Path) -> None:
    """A generation path cannot turn one archive's lease into another's writer.

    Anti-vacuity: omitting the explicit ``archive_root`` from the convergence
    writer lets this open succeed because the factory has no root to compare.
    """
    from polylogue.daemon.convergence_stages import _open_archive_insight_write_connection

    owner_root = tmp_path / "owner"
    target_root = tmp_path / "target"
    owner_root.mkdir()
    target_root.mkdir()
    target_db = target_root / "index.db"
    initialize_archive_database(target_db, ArchiveTier.INDEX)

    with (
        write_lease("test.owner", archive_root=owner_root),
        pytest.raises(UnleasedWriteError, match="outside the archive"),
    ):
        _open_archive_insight_write_connection(target_db, archive_root=target_root)


def test_checkpoint_writer_refuses_a_different_archive_root(tmp_path: Path) -> None:
    """The periodic checkpoint route carries the root it was admitted for.

    Anti-vacuity: dropping ``archive_root=archive_root`` from
    ``checkpoint_archive_wals`` reopens the target database under this lease.
    """
    from polylogue.storage.sqlite.wal_checkpoint import checkpoint_archive_wals

    owner_root = tmp_path / "owner"
    target_root = tmp_path / "target"
    owner_root.mkdir()
    target_root.mkdir()
    initialize_archive_database(target_root / "index.db", ArchiveTier.INDEX)

    with (
        write_lease("test.owner", archive_root=owner_root),
        pytest.raises(UnleasedWriteError, match="outside the archive"),
    ):
        checkpoint_archive_wals(target_root, reason="test", warn_bytes=0)


def test_index_generation_bootstrap_requires_the_archive_bound_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A generation's direct writable index open cannot bypass admission.

    Anti-vacuity: removing the lease assertion from ``IndexGenerationStore``
    lets this production bootstrap create an archive-tier ``index.db`` while
    the daemon's connection guard is armed but no writer owns the archive.
    """
    from polylogue.storage.index_generation import IndexGenerationStore

    root = tmp_path / "archive"
    initialize_active_archive_root(root)
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(root))
    store = IndexGenerationStore.for_archive_root(root)

    with arm_write_lease_enforcement(), pytest.raises(UnleasedWriteError):
        store.create(source_snapshot="snapshot-unleased")

    with arm_write_lease_enforcement(), write_lease("test.generation", archive_root=root):
        generation = store.create(source_snapshot="snapshot-leased")
    assert Path(generation.index_path).is_file()

    with arm_write_lease_enforcement(), pytest.raises(UnleasedWriteError):
        store.seal_candidate_membership(generation, source_snapshot="snapshot-unleased")
    with arm_write_lease_enforcement(), pytest.raises(UnleasedWriteError):
        store.commit_candidate_membership(generation, ["raw-id"])
    with arm_write_lease_enforcement(), pytest.raises(UnleasedWriteError):
        store.promote(generation)


def test_cold_generation_open_binds_to_the_declared_archive_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The candidate path is not a substitute for its archive authority."""
    from polylogue.storage.index_generation import IndexGenerationStore
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    root = tmp_path / "archive"
    initialize_active_archive_root(root)
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(root))
    store = IndexGenerationStore.for_archive_root(root)
    with write_lease("test.generation", archive_root=root):
        generation = store.create(source_snapshot="snapshot-leased")
        candidate = Path(generation.index_path).parent
        with ArchiveStore.open_cold_build_generation(
            candidate,
            generation_id=generation.generation_id,
            owner_id=generation.owner_id,
        ) as archive:
            assert archive.archive_root == Path(generation.index_path).parent

    wrong_root = tmp_path / "other"
    wrong_root.mkdir()
    with (
        arm_write_lease_enforcement(),
        write_lease("test.wrong-generation", archive_root=wrong_root),
        pytest.raises(UnleasedWriteError, match="outside the archive"),
    ):
        ArchiveStore.open_cold_build_generation(
            candidate,
            generation_id=generation.generation_id,
            owner_id=generation.owner_id,
        )


def test_cold_generation_discard_requires_the_archive_bound_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Empty-build teardown cannot remove a candidate outside writer admission.

    Anti-vacuity: before ``discard_if_inactive`` asserted its archive-root
    lease, this production ``ColdBuildGeneration.discard`` route succeeded
    while enforcement was armed and removed the candidate's writable
    ``index.db`` concurrently with any admitted writer.
    """
    from polylogue.sources.live.cold_build import ColdBuildGeneration

    root = tmp_path / "archive"
    initialize_active_archive_root(root)
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(root))
    with write_lease("test.generation", archive_root=root):
        generation = ColdBuildGeneration.begin(root, reason="test")

    with arm_write_lease_enforcement(), pytest.raises(UnleasedWriteError):
        generation.discard()

    assert generation.generation_root.is_dir()
    with arm_write_lease_enforcement(), write_lease("test.generation", archive_root=root):
        assert generation.discard() is True
    assert generation.settled
    assert not generation.generation_root.exists()


def test_embedding_failure_resolution_refuses_an_unleased_writer(tmp_path: Path) -> None:
    """The CLI failure-resolution path cannot bypass archive-bound admission.

    Anti-vacuity: restoring the generic ``sqlite_connection`` open makes this
    mutation run despite the daemon's armed process-wide writer boundary.
    """
    from polylogue.storage.embeddings.materialization import resolve_embedding_failure_with_lifecycle

    embeddings_db = tmp_path / "embeddings.db"
    initialize_archive_database(embeddings_db, ArchiveTier.EMBEDDINGS)

    with arm_write_lease_enforcement(), pytest.raises(UnleasedWriteError):
        resolve_embedding_failure_with_lifecycle(
            embeddings_db,
            failure_id="failure-does-not-matter-before-admission",
            action="acknowledge",
        )


def test_enforcement_is_off_by_default_so_one_shot_writers_are_unaffected(db_path: Path) -> None:
    """A CLI or API process is its own single writer and has no gate to be outside of."""
    assert write_lease_enforced() is False
    with closing(open_connection(db_path, validate_schema=False)) as conn:
        conn.execute("INSERT INTO t VALUES (2)")
        conn.commit()


def test_the_lease_does_not_leak_past_its_block(db_path: Path) -> None:
    with arm_write_lease_enforcement():
        with write_lease("test.writer"):
            assert current_write_lease() is not None
        assert current_write_lease() is None
        with pytest.raises(UnleasedWriteError):
            open_connection(db_path, validate_schema=False)


def test_enforcement_does_not_leak_past_its_block(db_path: Path) -> None:
    with arm_write_lease_enforcement():
        assert write_lease_enforced() is True
    assert write_lease_enforced() is False


def test_a_nested_acquisition_returns_the_outer_lease(db_path: Path) -> None:
    """A publish inside a batch must not reset the outer hold's budget.

    Anti-vacuity: make the inner acquisition a second lease with its own clock
    and a long outer hold stops being reported, which is the exact blindness the
    budget exists to remove.
    """
    with write_lease("outer", max_hold_seconds=10.0) as outer:
        with write_lease("inner", max_hold_seconds=0.001) as inner:
            assert inner is outer
            assert inner.actor == "outer"


def test_a_hold_past_its_declared_budget_is_a_typed_failure() -> None:
    """An over-long hold fails rather than being absorbed as a longer wait.

    The budget cannot preempt a writer already inside a SQLite transaction, so
    it fires at release: its job is to make an 18,623 s hold impossible to miss.
    """
    with pytest.raises(WriteHoldExceededError, match="budget"):
        with write_lease("test.slow", max_hold_seconds=0.0):
            pass


def test_a_hold_within_its_budget_is_silent() -> None:
    with write_lease("test.fast", max_hold_seconds=60.0) as lease:
        assert lease.over_budget is False


def test_require_write_lease_returns_the_lease_for_an_authorized_caller() -> None:
    with write_lease("test.writer") as lease:
        assert require_write_lease("probe") is lease


def test_a_child_task_cannot_inherit_its_parents_write_lease(db_path: Path) -> None:
    """A copied context marker is not authority to start another writer.

    Anti-vacuity: omitting the task-owner check in ``require_write_lease`` lets
    ``create_task`` inherit the context variable and open an overlapping
    writer while the owning task still holds the coordinator lease.
    """

    async def scenario() -> None:
        with arm_write_lease_enforcement(), write_lease("test.owner"):

            async def child_writer() -> None:
                with closing(open_connection(db_path, validate_schema=False)):
                    pass

            child = asyncio.create_task(child_writer())
            with pytest.raises(UnleasedWriteError, match="inherited by a child task"):
                await child

            # The owning task retains its own authority after refusing the
            # inherited child context.
            with closing(open_connection(db_path, validate_schema=False)):
                pass

    asyncio.run(scenario())


def test_require_write_lease_is_permissive_when_unarmed() -> None:
    assert require_write_lease("probe") is None


def test_a_readonly_open_never_needs_a_lease(db_path: Path) -> None:
    """The boundary is about writers; refusing readers would be overreach."""
    with arm_write_lease_enforcement():
        with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as conn:
            assert conn.execute("SELECT count(*) FROM t").fetchone()[0] >= 0


def test_every_write_mode_factory_in_storage_routes_through_the_lease() -> None:
    """The census the bead asks for: no second door into a writable tier.

    A new write-mode factory that forgets ``require_write_lease`` reintroduces
    exactly the defect this closes -- an in-process writer outside the gate,
    exhausting the busy timeout of whoever holds it. Enumerating the factories
    is what makes that a test failure rather than a review miss.

    Anti-vacuity: drop the ``require_write_lease`` call from any listed factory
    and this fails naming it.
    """
    import ast
    from pathlib import Path as _Path

    #: Every function in ``polylogue/storage`` that opens a write-mode SQLite
    #: connection to an archive tier. Read-only opens are deliberately absent.
    write_mode_factories = {
        "polylogue/storage/sqlite/connection_profile.py": {
            "open_connection",
            "open_daemon_connection",
            "open_isolated_write_connection",
        },
        "polylogue/storage/sqlite/connection.py": {"_get_cached_connection"},
        "polylogue/storage/sqlite/audit_leaf.py": {
            "open_verified_audit_connection",
            "open_verified_sqlite_write_connection",
        },
    }

    repo_root = _Path(__file__).resolve().parents[3]
    unguarded: list[str] = []
    for relative, functions in write_mode_factories.items():
        module = ast.parse((repo_root / relative).read_text(encoding="utf-8"))
        found = {
            node.name: node for node in ast.walk(module) if isinstance(node, ast.FunctionDef) and node.name in functions
        }
        assert set(found) == functions, f"{relative}: {functions - set(found)} no longer exist; update the census"
        for name, node in found.items():
            calls = {
                call.func.id
                for call in ast.walk(node)
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
            }
            if "require_write_lease" not in calls:
                unguarded.append(f"{relative}:{name}")
    assert unguarded == [], f"write-mode factories that do not take the lease: {unguarded}"


def test_the_cached_write_connection_is_refused_without_a_lease(tmp_path: Path) -> None:
    """The async runtime's thread-local writer is on the lease like any other."""
    from polylogue.storage.sqlite.connection import open_connection as cached_write_connection

    with arm_write_lease_enforcement():
        with pytest.raises(UnleasedWriteError):
            with cached_write_connection(tmp_path / "cached.db"):
                pass


def test_an_over_budget_hold_does_not_displace_its_own_failure() -> None:
    """A failing over-budget hold reports its own error, not the budget breach.

    Determinism: ``max_hold_seconds=0.0`` puts the hold over budget on every
    run without a sleep, since ``held_seconds`` is strictly positive by the
    time release is reached. No timing window is raced.

    Anti-vacuity: raising ``WriteHoldExceededError`` from the release
    ``finally`` -- the previous behavior -- turns this red. That masking is not
    cosmetic: ``_publication_commit_known`` in ``daemon/convergence.py``
    recovers a partial-write fact by ``isinstance`` on the raised exception, so
    a displaced error makes a committed index replacement with an unlowered
    marker report as an ordinary failure carrying no committed fact.
    """

    class PartialWriteError(RuntimeError):
        def __init__(self) -> None:
            super().__init__("index committed, marker lowering failed")
            self.index_family_committed = True

    with pytest.raises(PartialWriteError) as caught:
        with write_lease("test.failing_slow_writer", max_hold_seconds=0.0):
            raise PartialWriteError

    # The typed fact an ``except`` clause needs survives the over-budget release.
    assert caught.value.index_family_committed is True


def test_a_failing_over_budget_hold_still_releases_the_lease() -> None:
    """The contextvar is restored on the failing path, not only the clean one.

    Deterministic for the same reason as above; a leaked lease would let the
    next unrelated caller in this context open a write connection unleased.
    """

    class BoomError(RuntimeError):
        pass

    with pytest.raises(BoomError):
        with write_lease("test.failing_slow_writer", max_hold_seconds=0.0):
            raise BoomError

    assert current_write_lease() is None


def test_an_unbound_thread_that_inherits_the_lease_is_still_refused() -> None:
    """A spawned thread must not write on a lease it merely inherited.

    This interpreter is a free-threading (no-GIL) CPython build, and on it a
    new ``threading.Thread`` starts from a *copy* of the creating thread's
    context rather than an empty one. So a thread spawned while the daemon
    holds the write lease observes that lease in ``_ACTIVE`` -- the
    ContextVar does not isolate it. What actually keeps the single-writer
    boundary is the bound-thread check: the inherited lease names the owner's
    thread id, and an unbound thread is refused.

    Determinism: no sleeps. The worker is joined before the assertion, so the
    result is observed after the thread has certainly finished, and the lease
    is held for the whole join. Nothing races.

    Anti-vacuity: deleting the ``bound_thread_ids`` check in
    ``require_write_lease`` turns this green-to-red -- the inherited lease
    would authorize an arbitrary thread to open a durable write connection
    while the daemon believes it is the sole writer. It is red today only by
    that check, not by contextvar isolation.
    """
    observed: dict[str, object] = {}

    def worker() -> None:
        observed["inherited_lease"] = current_write_lease()
        try:
            require_write_lease("worker durable write")
        except UnleasedWriteError as exc:
            observed["outcome"] = f"refused: {exc}"
        else:
            observed["outcome"] = "allowed"

    with arm_write_lease_enforcement(), write_lease("daemon.writer") as lease:
        thread = threading.Thread(target=worker, name="inheriting-worker")
        thread.start()
        thread.join()

    # The inheritance itself is real: the guard, not isolation, is the defense.
    assert observed["inherited_lease"] is lease
    assert str(observed["outcome"]).startswith("refused: ")
    assert "unauthorized thread" in str(observed["outcome"])
    # The worker must not have smuggled itself into the owner's bound set.
    assert lease.bound_thread_ids == {lease.owner_thread_id}


def test_delegation_authorizes_a_foreign_thread_and_loop_but_nothing_else() -> None:
    """Ownership travels as a value, so a hand-off survives thread + loop changes.

    Anti-vacuity: drop the ``adopt_write_lease`` block from ``worker`` and the
    adopted probe raises ``UnleasedWriteError`` -- the ambient lease does not
    reach a worker thread running its own event loop, which is exactly the
    daemon HTTP write gate's shape (polylogue-h5l6i).
    """
    seen: list[str] = []

    with arm_write_lease_enforcement():
        with write_lease("owner"):
            delegation = delegate_write_lease()

            def worker() -> None:
                async def body() -> None:
                    with adopt_write_lease(delegation):
                        lease = require_write_lease("user.db write")
                        seen.append("adopted" if lease is not None else "unleased")

                asyncio.run(body())

            thread = threading.Thread(target=worker)
            thread.start()
            thread.join()

    assert seen == ["adopted"]


def test_a_thread_without_the_delegation_is_still_refused_inside_the_hold() -> None:
    """The load-bearing negative: admission is not ambient authorization.

    Anti-vacuity: authorize the rogue thread ambiently -- call
    ``bind_write_lease_thread()`` inside ``rogue`` -- and the assertion below
    fails, because the lease's ContextVar *is* inherited by threads on this
    build. Only the explicit hand-off keeps it out.
    """
    refused: list[BaseException | None] = []

    with arm_write_lease_enforcement():
        with write_lease("owner"):
            delegate_write_lease()

            def rogue() -> None:
                try:
                    require_write_lease("user.db write")
                except UnleasedWriteError as exc:
                    refused.append(exc)
                else:
                    refused.append(None)

            thread = threading.Thread(target=rogue)
            thread.start()
            thread.join()

    assert len(refused) == 1
    assert isinstance(refused[0], UnleasedWriteError)


def test_delegation_admits_one_writer_at_a_time() -> None:
    """One admission cannot fan out into concurrent writers.

    Anti-vacuity: remove the ``_adopted_by`` guard in ``adopt_write_lease``
    and the second adoption succeeds while the first still holds it.
    """
    entered = threading.Event()
    release = threading.Event()

    with arm_write_lease_enforcement():
        with write_lease("owner"):
            delegation = delegate_write_lease()

            def first() -> None:
                with adopt_write_lease(delegation):
                    entered.set()
                    release.wait(timeout=5.0)

            thread = threading.Thread(target=first)
            thread.start()
            assert entered.wait(timeout=5.0)
            try:
                with pytest.raises(UnleasedWriteError, match="already executing"):
                    with adopt_write_lease(delegation):
                        pass
            finally:
                release.set()
                thread.join()


def test_delegation_is_revoked_when_its_lease_is_released() -> None:
    """A stashed grant authorizes nothing once the admission is over.

    Anti-vacuity: delete the revoke loop in ``write_lease``'s finally block and
    this adoption succeeds outside any admission.
    """
    with arm_write_lease_enforcement():
        with write_lease("owner"):
            delegation = delegate_write_lease()
        assert not delegation.live
        with pytest.raises(UnleasedWriteError, match="revoked"):
            with adopt_write_lease(delegation):
                pass


def test_delegation_cannot_be_minted_without_holding_the_lease() -> None:
    """Delegation is a hand-off, never an escalation.

    Anti-vacuity: mint a ``WriteLeaseDelegation`` directly instead of routing
    through ``require_write_lease`` and an unleased caller gains authority.
    """
    with arm_write_lease_enforcement():
        with pytest.raises(UnleasedWriteError):
            delegate_write_lease()


def test_delegation_is_revoked_when_its_lease_fails() -> None:
    """A failing hold releases the lease, so it must revoke the same grants.

    The delegation contract is that a stashed delegation authorizes nothing
    once its lease is gone. A hold that raises releases the lease exactly as a
    successful one does.

    Anti-vacuity: delete the revoke loop from the ``except BaseException``
    branch in ``write_lease`` and this adoption succeeds outside any
    admission, with ``delegation.live`` still True.
    """
    with arm_write_lease_enforcement():
        delegation = None
        with pytest.raises(RuntimeError, match="hold failed"):
            with write_lease("owner"):
                delegation = delegate_write_lease()
                raise RuntimeError("hold failed")
        assert delegation is not None
        assert not delegation.live
        with pytest.raises(UnleasedWriteError, match="revoked"):
            with adopt_write_lease(delegation):
                pass


def test_an_inheriting_thread_cannot_bind_itself_into_the_live_lease() -> None:
    """polylogue-1oa7o residual 1: binding must be delegated, not self-served.

    ``bind_write_lease_thread()`` used to read the ambient lease and add the
    calling thread's id to it. On this free-threading build *every* thread
    spawned during a hold inherits that lease, so any of them could join the
    daemon's live authority by calling it. Binding now requires a single-use
    grant the owner minted before the thread existed.

    Anti-vacuity: restore the no-argument self-binding form and the worker
    below joins ``bound_thread_ids``, so both assertions go red. The
    inheritance itself is real (asserted first), so this is not vacuous.
    """
    from polylogue.storage.sqlite.write_lease import bind_write_lease_thread

    observed: dict[str, object] = {}

    def worker() -> None:
        observed["inherited_lease"] = current_write_lease()
        try:
            bind_write_lease_thread(None)  # type: ignore[arg-type]
        except (UnleasedWriteError, AttributeError) as exc:
            observed["outcome"] = f"refused: {type(exc).__name__}"
        else:
            observed["outcome"] = "bound"

    with arm_write_lease_enforcement(), write_lease("daemon.writer") as lease:
        thread = threading.Thread(target=worker, name="self-binding-worker")
        thread.start()
        thread.join()

        assert observed["inherited_lease"] is lease
        assert str(observed["outcome"]).startswith("refused: ")
        assert lease.authorized_threads() == frozenset({lease.owner_thread_id})


def test_a_granted_thread_may_bind_and_write() -> None:
    """The owner-minted grant is the admitted path the coordinator uses."""
    from polylogue.storage.sqlite.write_lease import bind_write_lease_thread, grant_write_lease_thread

    observed: dict[str, object] = {}

    with arm_write_lease_enforcement(), write_lease("daemon.writer") as lease:
        grant = grant_write_lease_thread()

        def worker() -> None:
            bind_write_lease_thread(grant)
            observed["lease"] = require_write_lease("granted worker write")

        thread = threading.Thread(target=worker, name="granted-worker")
        thread.start()
        thread.join()

        assert observed["lease"] is lease
        assert threading.get_ident() in lease.authorized_threads()


def test_a_grant_authorizes_exactly_one_thread() -> None:
    from polylogue.storage.sqlite.write_lease import bind_write_lease_thread, grant_write_lease_thread

    outcomes: list[str] = []

    with arm_write_lease_enforcement(), write_lease("daemon.writer"):
        grant = grant_write_lease_thread()

        def worker() -> None:
            try:
                bind_write_lease_thread(grant)
            except UnleasedWriteError:
                outcomes.append("refused")
            else:
                outcomes.append("bound")

        first = threading.Thread(target=worker)
        first.start()
        first.join()
        second = threading.Thread(target=worker)
        second.start()
        second.join()

    assert outcomes == ["bound", "refused"]


def test_a_nested_lease_in_an_inheriting_thread_is_refused() -> None:
    """polylogue-1oa7o residual 2: the re-entrant branch had no thread check.

    An inheriting thread asking for ``write_lease(...)`` took the re-entrant
    path and was handed the parent's lease, creating neither its own authority
    nor a refusal.

    Anti-vacuity: drop the ``require_write_lease`` call from the re-entrant
    branch of ``write_lease`` and the worker below reports "granted".
    """
    observed: dict[str, object] = {}

    def worker() -> None:
        try:
            with write_lease("nested.worker"):
                observed["outcome"] = "granted"
        except UnleasedWriteError as exc:
            observed["outcome"] = f"refused: {exc}"

    with arm_write_lease_enforcement(), write_lease("daemon.writer") as lease:
        assert lease is not None
        thread = threading.Thread(target=worker, name="nested-inheriting-worker")
        thread.start()
        thread.join()

    assert str(observed["outcome"]).startswith("refused: ")
    assert "unauthorized thread" in str(observed["outcome"])
