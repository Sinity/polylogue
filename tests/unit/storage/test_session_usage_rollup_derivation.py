"""Canonical usage reconciliation is a stage, not a side effect of publication.

The measured defect (polylogue-bp12n.1 AC2): the session-profile publisher ran
``reconcile_session_usage_rollup`` and committed it *before* checking whether
the partition it had already prepared was still applicable. The prepared bundle
had read the pre-refresh rollup, so the check that followed necessarily failed
for every session whose rollup had drifted: the first computation was doomed by
construction, a second pass did the real work, and ``False`` came back from a
call that had already committed a usage change.

The surviving half, fixed here: the *unprepared* publisher
``publish_session_profile`` reached the same reconciliation through the bulk
rebuild it calls, so every profile publication also rewrote canonical usage and
stamped the usage domain's binding. It no longer does; the reconciliation is
owned by ``usage_rollup`` and the publisher refuses a session that domain has
not settled.

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
    """Converge the usage prerequisite, then publish the profile it feeds.

    Two calls because they are two domains. ``publish_session_profile`` refuses
    a session whose rollup ``SESSION_USAGE_ROLLUP_DOMAIN`` has not settled; it
    does not reconcile one behind the caller's back.
    """
    with write_lease("test.publish"), closing(_write_connection(index_db)) as conn:
        binding = session_input_bindings(conn, (session_id,))[session_id]
        assert (
            publish_session_usage_rollup(
                conn,
                session_id,
                input_binding=binding,
                recipe_version=session_usage_rollup_recipe_version(),
            )
            is True
        )
        return publish_session_profile(conn, session_id, input_binding=binding)


def test_a_drifted_rollup_no_longer_dooms_the_first_profile_computation(
    archive: tuple[Path, str],
) -> None:
    """One pass reconciles usage and publishes the profile built from it.

    This is the whole point of the separation. A message token change moves
    both the rollup's inputs and the profile's, and the profile names the
    rollup's key as a prerequisite, so the kernel reconciles first and the
    profile publishes on its first attempt.

    Anti-vacuity (measured): restore ``reconcile_session_usage_rollup`` and its
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
    assert converge(_registry(index_db, session_id), _frame(index_db)).made_no_publication_attempts


def test_a_refused_reconciliation_commits_nothing(archive: tuple[Path, str]) -> None:
    """``False`` from the rollup publisher means no effect, not "usage moved anyway".

    Anti-vacuity: move ``reconcile_session_usage_rollup`` above the binding
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
    """Profile publication must not repair stale canonical usage or certify it.

    Anti-vacuity: restore the ``reconcile_session_usage_rollup`` +
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
        assert publish_prepared_session_profile(conn, prepared) is False

    assert _usage_rows(index_db, session_id) == before
    assert _profile_status(index_db, session_id) == "stale"


def test_a_bulk_rebuild_stamps_the_binding_it_reconciled(archive: tuple[Path, str]) -> None:
    """A build leaves no rollup work behind for the first recurring pass.

    The bulk index rebuild owns both jobs and runs the reconciliation before
    the profiles that read it. Without the stamp, the derivation would report
    every one of those sessions MISSING and reconcile the whole archive a
    second time for no change.

    This drives ``rebuild_session_insights_sync`` directly: it is the route
    that reconciles usage, and reaching it through a profile publisher was the
    conflation polylogue-bp12n.1 AC2 removed.

    Anti-vacuity: delete the ``reconcile_session_usage_rollups`` call from the
    rebuild chunk loop and this reports ``missing``.
    """
    from polylogue.storage.derived.session.rebuild import rebuild_session_insights_sync

    index_db, session_id = archive
    assert _rollup_status(index_db, session_id) == "missing"

    with write_lease("test.rebuild"), closing(_write_connection(index_db)) as conn:
        rebuild_session_insights_sync(conn, session_ids=[session_id])

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


def _skew_stored_rollup(index_db: Path, session_id: str, amount: int) -> None:
    """Move the rollup's stored *rows* without moving its binding.

    The binding digests the rollup's inputs -- messages and provider usage
    events -- not the totals it produced, so a direct row edit leaves the
    derivation reporting VALID. That skew is what makes a reconciliation
    observable: on an archive whose rollup already agrees with its inputs, a
    refresh rewrites the same numbers and no assertion can tell a publisher
    that reconciles from one that does not.
    """
    with write_lease("test.skew"), closing(_write_connection(index_db)) as conn:
        changed = conn.execute(
            "UPDATE session_model_usage SET input_tokens = input_tokens + ? WHERE session_id = ?",
            (amount, session_id),
        ).rowcount
        conn.commit()
    assert changed >= 1, "the skew must reach a stored rollup row"


def test_profile_publish_writes_no_usage(archive: tuple[Path, str]) -> None:
    """Publishing a profile touches no canonical usage row (AC2/AC7).

    ``publish_session_profile`` rebuilds through
    ``rebuild_session_insights_sync``, which is also the bulk index rebuild and
    therefore owns the canonical-usage stage. Reaching that stage from a
    profile publication is the surviving half of the defect PR #5312 fixed
    only for the prepared publisher: a usage change committed by a call whose
    subject is a profile, which the call's own refusal cannot take back.

    The stored rollup is skewed first *without* moving its binding, so the
    derivation still reports it settled and this publication has no business
    revisiting it. Whether the rollup is right is the other domain's job.

    Anti-vacuity (executed): pass ``reconcile_usage_rollup=True`` from
    ``publish_session_profile`` and the skew is silently reconciled away
    inside a profile publication -- the assertion below reports the ingest
    totals instead of the skewed ones.
    """
    index_db, session_id = archive
    assert _materialize(index_db, session_id) is True
    assert _rollup_status(index_db, session_id) == "valid"

    _skew_stored_rollup(index_db, session_id, 9000)
    skewed = _usage_rows(index_db, session_id)
    assert _rollup_status(index_db, session_id) == "valid", "a row edit does not move the binding"

    with write_lease("test.publish"), closing(_write_connection(index_db)) as conn:
        binding = session_input_bindings(conn, (session_id,))[session_id]
        # True, not a blanket refusal: the profile is published, and it is
        # published without writing usage.
        assert publish_session_profile(conn, session_id, input_binding=binding) is True

    assert _usage_rows(index_db, session_id) == skewed
    assert _profile_status(index_db, session_id) == "valid"


def test_profile_needs_a_settled_rollup(archive: tuple[Path, str]) -> None:
    """An unsettled rollup is a refusal with no effects, not a silent refresh.

    The profile's stored binding covers the rollup's inputs, not its rows, so
    a profile published against a rollup this derivation has never reconciled
    would certify cost values nothing checked. The publisher refuses instead
    of reconciling, which is what makes the prerequisite real: the key stays
    pending for the pass that converges the owner first.

    The second half pins the opposite direction. A publisher that refused
    everything would satisfy the first half and fail here.

    Anti-vacuity (executed): delete the ``inspect_session_usage_rollups``
    guard from ``publish_session_profile`` and the first publication succeeds
    against a MISSING rollup.
    """
    index_db, session_id = archive
    assert _rollup_status(index_db, session_id) == "missing"
    before = _usage_rows(index_db, session_id)

    with write_lease("test.publish"), closing(_write_connection(index_db)) as conn:
        binding = session_input_bindings(conn, (session_id,))[session_id]
        assert publish_session_profile(conn, session_id, input_binding=binding) is False

    # A refusal for this cause commits nothing at all: no profile row, no
    # usage row, and no binding stamped by a domain that did not run.
    assert _profile_status(index_db, session_id) == "missing"
    assert _rollup_status(index_db, session_id) == "missing"
    assert _usage_rows(index_db, session_id) == before

    with write_lease("test.rollup"), closing(_write_connection(index_db)) as conn:
        binding = session_input_bindings(conn, (session_id,))[session_id]
        assert (
            publish_session_usage_rollup(
                conn,
                session_id,
                input_binding=binding,
                recipe_version=session_usage_rollup_recipe_version(),
            )
            is True
        )
        assert publish_session_profile(conn, session_id, input_binding=binding) is True

    assert _profile_status(index_db, session_id) == "valid"


@pytest.mark.parametrize("upstream_state", ["missing", "changed-input", "old-recipe"])
def test_prepared_profile_requires_a_current_usage_certificate(archive: tuple[Path, str], upstream_state: str) -> None:
    """An unchanged stale rollup is not a valid dependency snapshot.

    The changed-input case is the deterministic schedule: settle the prerequisite,
    commit another input write, then prepare and publish. The writer must refuse
    without reconciling usage or replacing any retained profile rows. Re-running
    the prerequisite followed by fresh preparation must succeed.
    """
    from polylogue.storage.derived.session.rebuild import prepare_session_insight_partition

    index_db, session_id = archive
    if upstream_state != "missing":
        assert _materialize(index_db, session_id)
        if upstream_state == "changed-input":
            _bump_message_input_tokens(index_db, session_id, 5000)
        else:
            with write_lease("test.old_recipe"), closing(_write_connection(index_db)) as conn:
                conn.execute(
                    "UPDATE session_usage_rollup_bindings SET recipe_version = ? WHERE session_id = ?",
                    ("previous-software-recipe", session_id),
                )
                conn.commit()
    assert _rollup_status(index_db, session_id) != "valid"
    before_usage = _usage_rows(index_db, session_id)
    with closing(_read_connection(index_db)) as conn:
        before_profile = tuple(conn.execute("SELECT * FROM session_profiles WHERE session_id = ?", (session_id,)))
        before_latency = tuple(
            conn.execute("SELECT * FROM session_latency_profiles WHERE session_id = ?", (session_id,))
        )
        conn.row_factory = sqlite3.Row
        conn.execute("BEGIN")
        prepared = prepare_session_insight_partition(conn, session_id)
        pending = conn.execute(
            "SELECT revision FROM session_profile_demand WHERE session_id = ?", (session_id,)
        ).fetchone()
        expected_demand_revision = 0 if pending is None else int(pending[0])
    with write_lease("test.prepared_refusal"), closing(_write_connection(index_db)) as conn:
        assert (
            publish_prepared_session_profile(conn, prepared, expected_demand_revision=expected_demand_revision) is False
        )
    assert _usage_rows(index_db, session_id) == before_usage
    with closing(_read_connection(index_db)) as conn:
        assert (
            tuple(conn.execute("SELECT * FROM session_profiles WHERE session_id = ?", (session_id,))) == before_profile
        )
        assert (
            tuple(conn.execute("SELECT * FROM session_latency_profiles WHERE session_id = ?", (session_id,)))
            == before_latency
        )

    with write_lease("test.settle_usage"), closing(_write_connection(index_db)) as conn:
        binding = session_input_bindings(conn, (session_id,))[session_id]
        assert publish_session_usage_rollup(
            conn,
            session_id,
            input_binding=binding,
            recipe_version=session_usage_rollup_recipe_version(),
        )
    with closing(_read_connection(index_db)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("BEGIN")
        fresh = prepare_session_insight_partition(conn, session_id)
        pending = conn.execute(
            "SELECT revision FROM session_profile_demand WHERE session_id = ?", (session_id,)
        ).fetchone()
        expected_demand_revision = 0 if pending is None else int(pending[0])
    with write_lease("test.prepared_success"), closing(_write_connection(index_db)) as conn:
        assert publish_prepared_session_profile(conn, fresh, expected_demand_revision=expected_demand_revision) is True
    assert _rollup_status(index_db, session_id) == "valid"
    assert _profile_status(index_db, session_id) == "valid"
