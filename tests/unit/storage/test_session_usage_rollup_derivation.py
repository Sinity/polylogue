"""Canonical usage reconciliation is a stage, not a side effect of publication.

The measured defect (polylogue-bp12n.1 AC2): the session-profile publisher ran
``_refresh_provider_usage_rollup`` and committed it *before* checking whether
the partition it had already prepared was still applicable. The prepared bundle
had read the pre-refresh rollup, so the check that followed necessarily failed
for every session whose rollup had drifted: the first computation was doomed by
construction, a second pass did the real work, and ``False`` came back from a
call that had already committed a usage change.

These laws run against a real archive written by the production writer.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from polylogue.storage.derived.session.derivation import (
    SESSION_PROFILE_DOMAIN,
    SESSION_PROFILE_RECIPE_VERSION,
    SessionProfileDerivation,
    inspect_session_profiles,
    publish_prepared_session_profile,
    publish_session_profile,
)
from polylogue.storage.derived.session.input_binding import session_input_bindings
from polylogue.storage.derived.session.summary import (
    SESSION_SUMMARY_DOMAIN,
    SESSION_SUMMARY_RECIPE_VERSION,
    SessionSummaryDerivation,
)
from polylogue.storage.derived.session.usage_rollup import (
    SESSION_USAGE_ROLLUP_DOMAIN,
    SessionUsageRollupDerivation,
    inspect_session_usage_rollups,
    publish_session_usage_rollup,
    session_usage_rollup_recipe_version,
)
from polylogue.storage.runtime import SESSION_INSIGHT_MATERIALIZER_VERSION
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.connection_profile import open_connection
from polylogue.storage.sqlite.write_lease import write_lease
from tests.infra.storage_records import SessionBuilder

if TYPE_CHECKING:
    from polylogue.daemon.derivation import DerivationFrame, DerivationRegistry

_MATERIALIZER_VERSION = SESSION_INSIGHT_MATERIALIZER_VERSION


@pytest.fixture
def archive(tmp_path: Path) -> Iterator[tuple[Path, str]]:
    """One archive whose messages actually carry token counts.

    Token lanes are what the rollup aggregates; a session whose messages leave
    them NULL cannot exercise a reconciliation at all.
    """
    root = tmp_path / "archive"
    root.mkdir()
    index_db = root / "index.db"
    initialize_active_archive_root(root)
    builder = SessionBuilder(index_db, "usage-rollup")
    builder.add_message(
        role="user",
        text="how much did this cost",
        model_name="claude-sonnet-4-5",
        input_tokens=1200,
        output_tokens=0,
    )
    builder.add_message(
        role="assistant",
        text="the rollup knows",
        model_name="claude-sonnet-4-5",
        input_tokens=0,
        output_tokens=340,
    )
    builder.save()
    yield index_db, builder.native_session_id()


def _write_connection(index_db: Path) -> sqlite3.Connection:
    conn = open_connection(index_db)
    conn.row_factory = sqlite3.Row
    return conn


def _read_connection(index_db: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{index_db}?mode=ro", uri=True)


def _usage_rows(index_db: Path, session_id: str) -> tuple[tuple[object, ...], ...]:
    with closing(_read_connection(index_db)) as conn:
        return tuple(
            tuple(row)
            for row in conn.execute(
                """
                SELECT model_name, input_tokens, output_tokens, cache_read_tokens,
                       cache_write_tokens, message_count, catalog_cost_usd
                FROM session_model_usage
                WHERE session_id = ?
                ORDER BY model_name
                """,
                (session_id,),
            )
        )


def _rollup_status(index_db: Path, session_id: str) -> str:
    with closing(_read_connection(index_db)) as conn:
        return inspect_session_usage_rollups(
            conn,
            (session_id,),
            recipe_version=session_usage_rollup_recipe_version(),
        )[session_id]


def _profile_status(index_db: Path, session_id: str) -> str:
    with closing(_read_connection(index_db)) as conn:
        return inspect_session_profiles(conn, (session_id,), materializer_version=_MATERIALIZER_VERSION)[session_id]


def _bump_message_input_tokens(index_db: Path, session_id: str, amount: int) -> None:
    """Move a real profile input without touching an identity or a count."""
    with write_lease("test.mutate"), closing(_write_connection(index_db)) as conn:
        changed = conn.execute(
            "UPDATE messages SET input_tokens = COALESCE(input_tokens, 0) + ? WHERE session_id = ? AND position = 0",
            (amount, session_id),
        ).rowcount
        conn.commit()
    assert changed == 1, "the mutation must reach exactly one message"


def _registry(index_db: Path, session_id: str) -> DerivationRegistry:
    from polylogue.daemon.derivation import DerivationRegistry

    scope = [session_id]
    return DerivationRegistry(
        [
            SessionSummaryDerivation(
                lambda: _read_connection(index_db),
                lambda: _write_connection(index_db),
                session_scope=lambda _frame: scope,
            ),
            SessionUsageRollupDerivation(
                lambda: _read_connection(index_db),
                lambda: _write_connection(index_db),
                session_scope=lambda _frame: scope,
            ),
            SessionProfileDerivation(
                lambda: _read_connection(index_db),
                lambda: _write_connection(index_db),
                materializer_version=_MATERIALIZER_VERSION,
                session_scope=lambda _frame: scope,
            ),
        ]
    )


def _frame(index_db: Path) -> DerivationFrame:
    from polylogue.daemon.derivation import DerivationFrame

    return DerivationFrame(
        archive_root=str(index_db.parent),
        source_revision="r1",
        recipe_versions={
            SESSION_SUMMARY_DOMAIN: SESSION_SUMMARY_RECIPE_VERSION,
            SESSION_USAGE_ROLLUP_DOMAIN: session_usage_rollup_recipe_version(),
            SESSION_PROFILE_DOMAIN: SESSION_PROFILE_RECIPE_VERSION,
        },
    )


def _materialize(index_db: Path, session_id: str) -> bool:
    with write_lease("test.publish"), closing(_write_connection(index_db)) as conn:
        binding = session_input_bindings(conn, (session_id,))[session_id]
        return publish_session_profile(conn, session_id, input_binding=binding)


def test_a_drifted_rollup_no_longer_dooms_the_first_profile_computation(
    archive: tuple[Path, str],
) -> None:
    """One pass reconciles usage and publishes the profile built from it.

    This is the whole point of the separation. A message token change moves
    both the rollup's inputs and the profile's, and the profile names the
    rollup's key as a prerequisite, so the kernel reconciles first and the
    profile publishes on its first attempt.

    Anti-vacuity (measured): restore ``_refresh_provider_usage_rollup`` and its
    commit inside ``publish_prepared_session_profile``, drop
    ``SESSION_USAGE_ROLLUP_DOMAIN`` from the profile's ``prerequisites`` and
    ``prerequisite_keys``, and drop the adapter from the registry -- the
    profile comes back PENDING/BINDING_MOVED from the pass below, because its
    prepared bundle read the pre-refresh rollup that publication then moved.
    Delete the refresh from ``publish_session_usage_rollup`` instead and the
    reconciled-total assertion goes red.
    """
    from polylogue.daemon.derivation import Outcome, converge

    index_db, session_id = archive
    assert _materialize(index_db, session_id) is True
    assert _profile_status(index_db, session_id) == "valid"

    _bump_message_input_tokens(index_db, session_id, 5000)
    assert _profile_status(index_db, session_id) == "stale"

    report = converge(_registry(index_db, session_id), _frame(index_db))

    # The load-bearing assertion: the profile is DONE in the *first* pass
    # after its input moved. Under the previous publisher it was PENDING here
    # and only reached DONE on a second pass.
    profile = [outcome for outcome in report.outcomes if outcome.key.domain == SESSION_PROFILE_DOMAIN]
    assert [outcome.outcome for outcome in profile] == [Outcome.DONE], report.outcomes
    assert not report.by_outcome(Outcome.FAILED), report.outcomes
    assert [outcome.key.domain for outcome in report.by_outcome(Outcome.DONE)] == [
        SESSION_USAGE_ROLLUP_DOMAIN,
        SESSION_PROFILE_DOMAIN,
    ]
    assert _profile_status(index_db, session_id) == "valid"
    # The reconciliation actually ran: the rollup now carries the new total.
    assert [row[1] for row in _usage_rows(index_db, session_id)] == [6200]
    assert converge(_registry(index_db, session_id), _frame(index_db)).wrote_nothing


def test_a_refused_reconciliation_commits_nothing(archive: tuple[Path, str]) -> None:
    """``False`` from the rollup publisher means no effect, not "usage moved anyway".

    Anti-vacuity: move ``_refresh_provider_usage_rollup`` above the binding
    comparison in ``publish_session_usage_rollup``, or commit before it, and
    the stored rollup changes while the call reports refusal -- exactly the
    untruth the profile publisher used to tell.
    """
    index_db, session_id = archive
    assert _materialize(index_db, session_id) is True
    _bump_message_input_tokens(index_db, session_id, 5000)
    before = _usage_rows(index_db, session_id)

    with write_lease("test.rollup"), closing(_write_connection(index_db)) as conn:
        refused = publish_session_usage_rollup(
            conn,
            session_id,
            input_binding="a-binding-this-archive-never-had",
            recipe_version=session_usage_rollup_recipe_version(),
        )

    assert refused is False
    assert _usage_rows(index_db, session_id) == before


def test_the_profile_publisher_writes_no_canonical_usage(archive: tuple[Path, str]) -> None:
    """Profile publication owns the four-table family and nothing else.

    Anti-vacuity: restore the ``_refresh_provider_usage_rollup`` +
    ``conn.commit()`` pair inside ``publish_prepared_session_profile`` and the
    stored rollup moves inside a call that is supposed to publish a profile.
    """
    from polylogue.storage.derived.session.rebuild import prepare_session_insight_partition

    index_db, session_id = archive
    assert _materialize(index_db, session_id) is True
    _bump_message_input_tokens(index_db, session_id, 5000)
    before = _usage_rows(index_db, session_id)

    with closing(_write_connection(index_db)) as conn:
        prepared = prepare_session_insight_partition(conn, session_id)

    with write_lease("test.publish"), closing(_write_connection(index_db)) as conn:
        assert publish_prepared_session_profile(conn, prepared) is True

    assert _usage_rows(index_db, session_id) == before
    assert _profile_status(index_db, session_id) == "valid"


def test_a_bulk_rebuild_stamps_the_binding_it_reconciled(archive: tuple[Path, str]) -> None:
    """A build leaves no rollup work behind for the first recurring pass.

    ``rebuild_session_insights_sync`` refreshes every chunk's rollup already.
    Without the stamp, the derivation would report every one of those sessions
    MISSING and reconcile the whole archive a second time for no change.

    Anti-vacuity: delete the ``_stamp_refreshed_usage_bindings`` call from the
    rebuild chunk loop and this reports ``missing``.
    """
    index_db, session_id = archive
    assert _rollup_status(index_db, session_id) == "missing"

    assert _materialize(index_db, session_id) is True

    assert _rollup_status(index_db, session_id) == "valid"


def test_a_pricing_catalog_change_invalidates_the_stored_rollup(
    archive: tuple[Path, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The catalog is data the reprice step reads, so it belongs in the recipe.

    Anti-vacuity: drop ``_pricing_catalog_digest`` from
    ``session_usage_rollup_recipe_version`` and a catalog update leaves every
    ``catalog_cost_usd`` stale while inspection reports valid.
    """
    from polylogue.archive.semantic import pricing
    from polylogue.storage.derived.session import usage_rollup

    index_db, session_id = archive
    assert _materialize(index_db, session_id) is True
    assert _rollup_status(index_db, session_id) == "valid"

    sample = next(iter(pricing.PRICING.values()))
    monkeypatch.setitem(pricing.PRICING, "a-model-the-catalog-just-learned", sample)
    usage_rollup._pricing_catalog_digest.cache_clear()
    try:
        assert _rollup_status(index_db, session_id) == "stale"
    finally:
        monkeypatch.undo()
        usage_rollup._pricing_catalog_digest.cache_clear()

    assert _rollup_status(index_db, session_id) == "valid"


def test_a_binding_row_whose_session_is_gone_retires(archive: tuple[Path, str]) -> None:
    """The binding row is the one part of this partition that can be stranded.

    ``session_model_usage`` cascades with ``sessions``; a maintenance write
    with foreign keys disabled can leave the binding behind, and a binding
    with no session would certify a rollup that no longer exists.

    Anti-vacuity: return ``(), None`` from ``excess_page`` and the stranded
    row survives every pass.
    """
    index_db, session_id = archive
    assert _materialize(index_db, session_id) is True
    assert _rollup_status(index_db, session_id) == "valid"

    with write_lease("test.delete"), closing(_write_connection(index_db)) as conn:
        conn.execute("PRAGMA foreign_keys=OFF")
        conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        conn.commit()

    adapter = SessionUsageRollupDerivation(
        lambda: _read_connection(index_db),
        lambda: _write_connection(index_db),
        session_scope=lambda _frame: [session_id],
    )
    assert adapter.excess_page(None, cursor=None, limit=10) == ((session_id,), None)

    replacement = adapter.compute(None, session_id)
    assert replacement.empty is True
    assert adapter.publish(None, replacement) is True

    assert adapter.excess_page(None, cursor=None, limit=10) == ((), None)
