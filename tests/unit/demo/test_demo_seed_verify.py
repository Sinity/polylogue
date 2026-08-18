from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from polylogue.archive.query.unit_results import query_unit_envelope, query_unit_request
from polylogue.config import load_polylogue_config
from polylogue.demo import (
    DemoSeedTargetUnsafeError,
    apply_demo_post_ingest_augmentation,
    seed_demo_archive,
    verify_demo_archive,
)
from polylogue.demo.seed import (
    DEMO_OWNERSHIP_MANIFEST_FILENAME,
    DEMO_SOURCE_DIRNAME,
    demo_source_specs,
    materialize_demo_source,
)
from polylogue.pipeline.services.archive_ingest import parse_sources_archive
from polylogue.scenarios import (
    DEMO_CHATGPT_SESSION_ID,
    DEMO_CLAUDE_AI_TEMPORARY_SESSION_ID,
    DEMO_CLAUDE_CODE_LINEAGE_SIDECHAIN_SESSION_ID,
    DEMO_CLAUDE_CODE_SESSION_ID,
    DEMO_EMBEDDING_PROSE_SESSION_ID,
    DEMO_SESSION_IDS,
)
from polylogue.storage.embeddings.identity import EmbeddingRecipe
from polylogue.storage.embeddings.materialization import select_pending_archive_session_window
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.run_projection_relations import context_snapshot_relation_sql, run_relation_sql
from polylogue.storage.sqlite.sqlite_vec_extension import try_load_sqlite_vec


@pytest.mark.asyncio
async def test_seed_demo_archive_creates_ready_queryable_archive(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"

    seed = await seed_demo_archive(archive_root, force=True, with_overlays=True)
    verify = verify_demo_archive(archive_root, require_overlays=True)

    assert seed.archive_root == archive_root
    assert seed.session_count == len(DEMO_SESSION_IDS)
    assert seed.message_count >= 35
    assert seed.session_ids == tuple(sorted(DEMO_SESSION_IDS))
    assert seed.overlays_seeded is True
    assert seed.assertion_count >= 4
    assert seed.construct_coverage
    assert all(row.ok for row in seed.construct_coverage)

    assert verify.ok is True
    assert verify.session_count == len(DEMO_SESSION_IDS)
    assert verify.message_count >= 35
    assert DEMO_CLAUDE_CODE_SESSION_ID in verify.query_hits
    assert verify.overlays_present is True
    assert verify.absolute_path_leaks == ()
    assert verify.construct_coverage
    assert all(row.ok for row in verify.construct_coverage)
    assert verify.problems == ()

    with ArchiveStore.open_existing(archive_root) as archive:
        delegation_rows = query_unit_envelope(
            archive,
            query_unit_request(
                expression="delegations where parent:demo-lineage-parent AND mapping_state:resolved",
                limit=10,
            ),
        )
        [delegation_item] = delegation_rows.items
        delegation = delegation_item.model_dump(mode="json")
        assert delegation["parent_session_id"] == "codex-session:demo-lineage-parent"
        assert delegation["child_session_id"] == "codex-session:demo-lineage-subagent"
        assert delegation["mapping_state"] == "resolved"
        assert delegation["evidence_basis"] == "action"
        assert delegation["instruction_preview"] == "Inspect the demo lineage child and report caveats."
        expected_instruction = "Inspect the demo lineage child and report caveats."
        assert delegation["instruction_sha256"] == hashlib.sha256(expected_instruction.encode()).hexdigest()
        assert delegation["instruction_truncated"] is False
        instruction_block_id = delegation["instruction_tool_use_block_id"]
        assert isinstance(instruction_block_id, str)
        card = archive.get_delegation_card(instruction_tool_use_block_id=instruction_block_id)
        assert card is not None
        assert card.instruction == expected_instruction
        assert card.dispatch_result == "Subagent completed. Session: codex-session:demo-lineage-subagent"
        assert card.child_excerpt == (
            "Subagent report: lineage fixture has a parent, a fork, and a resolved child link."
        )

    with sqlite3.connect(archive_root / "index.db") as conn:
        conn.execute("ATTACH DATABASE ? AS source", (str(archive_root / "source.db"),))
        links = conn.execute("SELECT link_type, inheritance FROM session_links ORDER BY src_session_id").fetchall()
        temporary_sessions = conn.execute("SELECT session_id FROM sessions WHERE session_kind = 'temporary'").fetchall()
        capture_gap_events = conn.execute(
            "SELECT session_id, summary FROM session_events WHERE event_type = 'capture_gap'"
        ).fetchall()
        chatgpt_raw_rows = conn.execute(
            """
            SELECT COUNT(*)
            FROM source.raw_sessions
            WHERE origin = 'chatgpt-export'
              AND native_id = ?
            """,
            (DEMO_CHATGPT_SESSION_ID.split(":", maxsplit=1)[1],),
        ).fetchone()[0]
        chatgpt_session_rows = conn.execute(
            """
            SELECT COUNT(*)
            FROM sessions AS s
            JOIN source.raw_sessions AS r
              ON r.raw_id = s.raw_id
            WHERE s.session_id = ?
            """,
            (DEMO_CHATGPT_SESSION_ID,),
        ).fetchone()[0]
        compaction_events = conn.execute(
            "SELECT session_id, summary FROM session_events WHERE event_type = 'compaction'"
        ).fetchall()
        sidechain_sessions = conn.execute("SELECT session_id FROM sessions WHERE branch_type = 'sidechain'").fetchall()
        subagent_snapshots = conn.execute(
            f"{context_snapshot_relation_sql()} SELECT COUNT(*) FROM context_snapshots WHERE boundary = 'subagent_start'"
        ).fetchone()[0]
        subagent_runs = conn.execute(
            f"{run_relation_sql()} SELECT COUNT(*) FROM runs WHERE role = 'subagent'"
        ).fetchone()[0]
        unfinished_terminal_states = conn.execute(
            "SELECT terminal_state, COUNT(*) FROM session_profiles GROUP BY terminal_state"
        ).fetchall()
    with sqlite3.connect(archive_root / "embeddings.db") as conn:
        embedding_rows = conn.execute(
            "SELECT COUNT(*) FROM message_embeddings_meta WHERE model = 'demo-synthetic-embedding'"
        ).fetchone()[0]
        embedding_status = conn.execute(
            "SELECT session_id, message_count_embedded, needs_reindex, error_message FROM embedding_status"
        ).fetchall()

    assert temporary_sessions == [(DEMO_CLAUDE_AI_TEMPORARY_SESSION_ID,)]
    assert len(capture_gap_events) == 1
    assert "DOM browser-capture fallback" in capture_gap_events[0][1]
    assert chatgpt_raw_rows == 3
    assert chatgpt_session_rows == 1
    assert ("branch", "prefix-sharing") in links
    # A genuine main-session agent-acompact-* replays its dedicated parent's
    # content verbatim (>= 90% membership, polylogue-4ts.3), so the archive
    # writer's content-membership gate confirms real prefix-sharing rather
    # than an unverified "continuation" label with no actual shared content.
    assert ("continuation", "prefix-sharing") in links
    assert ("subagent", "spawned-fresh") in links
    assert len(compaction_events) >= 1
    assert sidechain_sessions == [(DEMO_CLAUDE_CODE_LINEAGE_SIDECHAIN_SESSION_ID,)]
    assert subagent_snapshots >= 1
    assert subagent_runs >= 1
    terminal_state_counts = dict(unfinished_terminal_states)
    assert terminal_state_counts.get("question_left", 0) + terminal_state_counts.get("tool_left", 0) >= 1
    assert terminal_state_counts.get("error_left", 0) >= 1
    assert embedding_rows >= 1
    assert embedding_status == [(DEMO_EMBEDDING_PROSE_SESSION_ID, embedding_rows, 0, None)]


@pytest.mark.asyncio
async def test_seed_demo_embedding_session_is_classified_fresh_not_pending(tmp_path: Path) -> None:
    """The demo-seeded synthetic embedding must never look selectable for a paid re-embed.

    ``_seed_demo_embeddings`` writes vectors and a clean ``embedding_status``
    row directly (no provider call, no ``embedding_derivation_state`` row was
    ever created before this fix). Because the v3 archive uses the exact-key
    freshness predicate (``_archive_embedding_freshness_predicate``) for every
    production selector, a missing derivation-state row made
    ``d.session_id IS NOT NULL`` always false, so the demo session was
    classified pending forever regardless of the compatibility
    ``embedding_status.needs_reindex`` flag. An embedding-enabled daemon
    pointed at a demo archive would then attempt to re-embed content that was
    deliberately seeded to be cost-free (polylogue PR #3067 review). Guards
    that the demo session is excluded from the same real selector every
    production call site shares.
    """

    archive_root = tmp_path / "archive"
    await seed_demo_archive(archive_root, force=True)

    cfg = load_polylogue_config()
    recipe = EmbeddingRecipe.current(model=cfg.embedding_model, dimensions=cfg.embedding_dimension)

    with sqlite3.connect(archive_root / "index.db") as conn:
        conn.execute("ATTACH DATABASE ? AS embeddings", (str(archive_root / "embeddings.db"),))
        loaded, error = try_load_sqlite_vec(conn)
        if not loaded:
            pytest.skip(str(error) if error else "sqlite-vec extension is unavailable")
        pending = select_pending_archive_session_window(
            conn,
            status_table="embeddings.embedding_status",
            recipe=recipe,
        )

    assert DEMO_EMBEDDING_PROSE_SESSION_ID not in {item.session_id for item in pending}


@pytest.mark.asyncio
async def test_seed_materializes_session_profiles_for_postmortem(tmp_path: Path) -> None:
    """The no-daemon seed must materialize the session-profile insight read model.

    Without it ``analyze --postmortem`` (and the session-digest surfaces) render
    an empty bundle on the demo archive because the postmortem aggregator fetches
    profiles that ``parse_sources_archive`` never wrote. Guards the #2196 fix.
    """

    archive_root = tmp_path / "archive"
    await seed_demo_archive(archive_root, force=True, with_overlays=True)

    with sqlite3.connect(archive_root / "index.db") as conn:
        profile_count = conn.execute("SELECT count(*) FROM session_profiles").fetchone()[0]

    assert profile_count == len(DEMO_SESSION_IDS)


@pytest.mark.asyncio
async def test_demo_verify_reports_missing_overlays(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"

    await seed_demo_archive(archive_root, force=True, with_overlays=False)
    verify = verify_demo_archive(archive_root, require_overlays=True)

    assert verify.ok is False
    assert "expected demo overlays" in "\n".join(verify.problems)
    failed_constructs = [row.to_payload() for row in verify.construct_coverage if not row.ok]
    assert not failed_constructs, failed_constructs


@pytest.mark.asyncio
async def test_demo_verify_can_skip_daemon_source_path_leak_posture(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"

    await seed_demo_archive(archive_root, force=True, with_overlays=True)
    with sqlite3.connect(archive_root / "source.db") as conn:
        conn.execute("UPDATE raw_sessions SET source_path = ?", (str(archive_root / "inbox" / "demo.jsonl"),))

    strict = verify_demo_archive(archive_root, require_overlays=True)
    daemon_wait = verify_demo_archive(
        archive_root,
        require_overlays=True,
        check_source_path_leaks=False,
    )

    assert strict.ok is False
    assert "raw source paths contain absolute paths" in "\n".join(strict.problems)
    assert daemon_wait.ok is True
    assert daemon_wait.absolute_path_leaks == ()


@pytest.mark.asyncio
async def test_seed_injects_demo_cost_for_postmortem(tmp_path: Path) -> None:
    """The demo claude-code session must carry usage so the postmortem blade
    renders real cost + token lanes (not $0) on the demo archive. Guards #2196
    slice 2 and the SessionProfile cost/token round-trip."""

    from polylogue.scenarios import DEMO_CLAUDE_CODE_SESSION_ID

    archive_root = tmp_path / "archive"
    await seed_demo_archive(archive_root, force=True, with_overlays=True)

    with sqlite3.connect(archive_root / "index.db") as conn:
        row = conn.execute(
            "SELECT total_cost_usd, total_input_tokens, total_output_tokens FROM session_profiles WHERE session_id = ?",
            (DEMO_CLAUDE_CODE_SESSION_ID,),
        ).fetchone()

    assert row is not None
    total_cost_usd, total_input_tokens, total_output_tokens = row
    assert total_cost_usd > 0
    assert total_input_tokens > 0
    assert total_output_tokens > 0


@pytest.mark.asyncio
async def test_apply_demo_post_ingest_augmentation_matches_direct_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Any ingest path must converge to the same demo-only enrichments once
    ``apply_demo_post_ingest_augmentation`` runs standalone.

    Simulates the shape of a daemon-driven ingest: materialize the fixture
    world and ingest it via ``parse_sources_archive`` directly, skipping
    ``seed_demo_archive``'s inline augmentation calls, then apply the shared
    post-ingest augmentation function standalone -- exactly what
    ``polylogue import --demo --wait`` does after daemon convergence. The
    resulting usage/repo/embedding facts must match what the direct seeder
    produces inline (polylogue-z1c6)."""

    archive_root = tmp_path / "archive"
    source_root = materialize_demo_source(archive_root, force=True)
    monkeypatch.chdir(source_root)
    result = await parse_sources_archive(archive_root, demo_source_specs(source_root))
    assert result.counts["sessions"] > 0

    # Base ingest alone (no augmentation yet) never materializes session
    # profiles at all -- that table is one of the derived insights
    # ``apply_demo_post_ingest_augmentation`` materializes.
    with sqlite3.connect(archive_root / "index.db") as conn:
        pre_row = conn.execute(
            "SELECT total_cost_usd FROM session_profiles WHERE session_id = ?",
            (DEMO_CLAUDE_CODE_SESSION_ID,),
        ).fetchone()
    assert pre_row is None or not pre_row[0]

    apply_demo_post_ingest_augmentation(archive_root)
    # Idempotent: a repeated call (e.g. a second ``--wait``) must not error or
    # change the outcome.
    apply_demo_post_ingest_augmentation(archive_root)

    with sqlite3.connect(archive_root / "index.db") as conn:
        cost_row = conn.execute(
            "SELECT total_cost_usd, total_input_tokens, total_output_tokens, repo_names_json "
            "FROM session_profiles WHERE session_id = ?",
            (DEMO_CLAUDE_CODE_SESSION_ID,),
        ).fetchone()
    assert cost_row is not None
    total_cost_usd, total_input_tokens, total_output_tokens, repo_names_json = cost_row
    assert total_cost_usd > 0
    assert total_input_tokens > 0
    assert total_output_tokens > 0
    assert "polylogue" in repo_names_json

    with sqlite3.connect(archive_root / "embeddings.db") as conn:
        embedding_rows = conn.execute(
            "SELECT COUNT(*) FROM message_embeddings_meta WHERE model = 'demo-synthetic-embedding'"
        ).fetchone()[0]
    assert embedding_rows >= 1


@pytest.mark.asyncio
async def test_seed_gives_demo_session_canonical_repo(tmp_path: Path) -> None:
    """The demo claude-code session must carry a canonical repo so the
    postmortem `repos_touched` metric renders a project, not an empty list."""

    import json as _json

    from polylogue.scenarios import DEMO_CLAUDE_CODE_SESSION_ID

    archive_root = tmp_path / "archive"
    await seed_demo_archive(archive_root, force=True, with_overlays=True)

    with sqlite3.connect(archive_root / "index.db") as conn:
        row = conn.execute(
            "SELECT repo_names_json FROM session_profiles WHERE session_id = ?",
            (DEMO_CLAUDE_CODE_SESSION_ID,),
        ).fetchone()

    assert row is not None
    assert "polylogue" in _json.loads(row[0])


@pytest.mark.asyncio
async def test_seed_demo_archive_forces_sequential_parse_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The demo seeder must never route its fixed, dozen-file synthetic corpus
    through the real subprocess parse pool.

    Guards polylogue-b054.1.1.1: ``parse_sources_archive``'s ambient worker
    count defaults to up to ``cpus-1`` real ``ProcessPoolExecutor`` (spawn)
    subprocesses. Under pytest-xdist every worker process independently
    spawns its own sub-pool, and a per-file worker task failure under host
    load (BrokenProcessPool, transient spawn/OOM) is caught by
    ``except Exception: failed += 1; continue`` in the pool driver and
    silently drops that file's sessions with only a log line -- no raised
    error. That is the reproduced root cause of the nondeterministic
    declared-construct-below-minimum failure observed under xdist. Spies on
    the real ``parse_sources_archive`` call the demo seeder makes (through a
    delegating wrapper, so the actual archive tiers are still populated by
    the genuine ingest route) and asserts it always pins ``parse_workers=1``.
    """

    import polylogue.demo.seed as seed_module

    captured: dict[str, object] = {}

    async def spy(archive_root: Path, sources: object, **kwargs: object):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return await parse_sources_archive(archive_root, sources, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(seed_module, "parse_sources_archive", spy)

    archive_root = tmp_path / "archive"
    result = await seed_demo_archive(archive_root, force=True)

    assert captured.get("parse_workers") == 1
    assert result.session_count == len(DEMO_SESSION_IDS)


@pytest.mark.asyncio
async def test_seed_demo_archive_self_heals_a_stale_schema_on_a_demo_owned_root(tmp_path: Path) -> None:
    """Re-seeding a demo archive whose index.db predates the current schema self-heals.

    Reproduces polylogue-3ycw: ``demo seed`` documented as the safe,
    private-data-free onboarding path failed outright with
    ``RuntimeError: index.db schema version N is not the current index tier
    version M; move it aside and rebuild the archive root`` and no
    self-heal, even though a demo archive's tiers hold only synthetic
    fixture content this exact command regenerates every run.

    ANTI-VACUITY: the production caller exercised is
    ``polylogue.demo.seed.seed_demo_archive`` (imported directly, no
    test-local reimplementation). Deleting the
    ``_self_heal_stale_demo_archive_tiers`` call from ``seed_demo_archive``
    (or deleting that helper's move-aside-and-reinitialize body) makes this
    exact test fail with the original unhandled ``RuntimeError`` instead of
    seeding successfully a second time.
    """

    archive_root = tmp_path / "archive"
    await seed_demo_archive(archive_root, force=True)

    # Roll index.db's schema version back, as if this demo archive had been
    # seeded by an older Polylogue release whose index tier version was lower
    # than the current code's.
    with sqlite3.connect(archive_root / "index.db") as conn:
        conn.execute("PRAGMA user_version = 1")
        conn.commit()

    seed = await seed_demo_archive(archive_root, force=False)

    assert seed.healed_tiers == ("index.db",)
    assert seed.session_count == len(DEMO_SESSION_IDS)
    assert seed.construct_coverage
    assert all(row.ok for row in seed.construct_coverage)

    verify = verify_demo_archive(archive_root)
    assert verify.ok is True

    stale_backups = list(archive_root.glob("index.db.stale-*"))
    assert len(stale_backups) == 1


@pytest.mark.asyncio
async def test_seed_demo_archive_refuses_the_default_root_collision(tmp_path: Path) -> None:
    """Demo seed refuses to write synthetic content into a root holding real sessions.

    Guards polylogue-o3a1t: ``demo seed``'s default archive-root resolution
    shares ``archive_root()``/``polylogue.toml`` with the live daemon, so an
    operator or agent who forgets ``--root`` can have it resolve straight to
    their live production archive. ``demo seed`` adds rows rather than
    refusing to run against existing content, so without this guard the
    collision would silently seed synthetic fixture sessions into a real
    archive.

    ANTI-VACUITY: the production entry point exercised is
    ``polylogue.demo.seed.seed_demo_archive`` with ``explicit_root=False``
    (what the CLI passes when neither ``--root`` nor
    ``POLYLOGUE_ARCHIVE_ROOT`` was given). Deleting the
    ``_guard_demo_seed_target`` call from ``seed_demo_archive`` makes this
    test fail by seeding successfully instead of raising.
    """

    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
    from tests.infra.storage_records import SessionBuilder

    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)
    SessionBuilder(archive_root / "index.db", "real-session").provider("claude-code").save()

    with pytest.raises(DemoSeedTargetUnsafeError, match="real ingested session"):
        await seed_demo_archive(archive_root, force=False, explicit_root=False)

    # Nothing was touched: no demo fixture source, no ownership manifest.
    assert not (archive_root / DEMO_SOURCE_DIRNAME).exists()
    assert not (archive_root / DEMO_OWNERSHIP_MANIFEST_FILENAME).exists()


@pytest.mark.asyncio
async def test_seed_demo_archive_explicit_root_bypasses_the_collision_guard(tmp_path: Path) -> None:
    """An operator-supplied --root/POLYLOGUE_ARCHIVE_ROOT is never second-guessed.

    Guards the "no friction for the deliberate/CI/cloud path" half of
    polylogue-o3a1t: passing ``explicit_root=True`` (what the CLI passes for
    an explicit ``--root`` flag or a ``POLYLOGUE_ARCHIVE_ROOT`` override)
    must seed successfully even against a root that already holds real
    content, because the caller took explicit responsibility for that root.
    """

    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
    from tests.infra.storage_records import SessionBuilder

    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)
    SessionBuilder(archive_root / "index.db", "real-session").provider("claude-code").save()

    seed = await seed_demo_archive(archive_root, force=False, explicit_root=True)

    assert seed.session_count == len(DEMO_SESSION_IDS) + 1


@pytest.mark.asyncio
async def test_seed_demo_archive_never_self_heals_a_root_that_held_real_content(tmp_path: Path) -> None:
    """A root holding real content when first seeded never becomes self-heal eligible.

    Guards polylogue-wyvio: the prior fix (PR #3515) gated self-heal on
    ``DEMO_SOURCE_DIRNAME``'s mere presence, which only proves "a demo seed
    touched this root at some point" -- not that the root's tiers are
    exclusively synthetic. A root that already held real content (here,
    reached via an explicit ``--force`` override past the polylogue-o3a1t
    guard, mirroring an operator who deliberately or accidentally pointed
    demo seed at a real archive) must never become eligible for self-heal,
    even after being reseeded, so a later schema mismatch cannot authorize
    moving aside real (potentially durable/irreplaceable) tier data.

    ANTI-VACUITY: exercises the real ``seed_demo_archive`` entry point twice
    (matching the exact sequence an operator would hit) and asserts on the
    real ``_self_heal_stale_demo_archive_tiers`` outcome (``healed_tiers``)
    and on-disk side effects (no ``.stale-`` backup file written). Reverting
    ``_archive_root_is_demo_owned`` to trust ``DEMO_SOURCE_DIRNAME`` alone
    makes this test fail: the second call would then self-heal instead of
    raising.
    """

    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
    from tests.infra.storage_records import SessionBuilder

    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)
    SessionBuilder(archive_root / "index.db", "real-session").provider("claude-code").save()

    # Force past the o3a1t guard, exactly like an operator who consciously
    # (or via the pre-fix collision bug) seeded demo content into this root.
    first = await seed_demo_archive(archive_root, force=True)
    assert first.healed_tiers == ()
    assert (archive_root / DEMO_SOURCE_DIRNAME).exists()  # legacy marker persists

    with sqlite3.connect(archive_root / "index.db") as conn:
        conn.execute("PRAGMA user_version = 1")
        conn.commit()

    with pytest.raises(RuntimeError, match="move it aside and rebuild the archive root"):
        await seed_demo_archive(archive_root, force=True)

    # Nothing was moved aside: the mixed-content root's tiers are untouched,
    # despite the legacy demo-source-directory marker being present.
    assert not list(archive_root.glob("index.db.stale-*"))
    manifest = json.loads((archive_root / DEMO_OWNERSHIP_MANIFEST_FILENAME).read_text())
    assert manifest["demo_only"] is False


@pytest.mark.asyncio
async def test_seed_demo_archive_grants_self_heal_to_a_genuinely_fresh_root(tmp_path: Path) -> None:
    """A brand-new root is recorded demo-only on first seed and stays self-heal eligible.

    Complements the mixed-content test above: an empty/nonexistent root has
    no real content to protect, so the ownership manifest records
    ``demo_only: true`` and self-heal keeps working across repeated
    reseeds, exactly like before the polylogue-wyvio fix.
    """

    archive_root = tmp_path / "archive"

    first = await seed_demo_archive(archive_root, force=True)
    assert first.healed_tiers == ()
    manifest = json.loads((archive_root / DEMO_OWNERSHIP_MANIFEST_FILENAME).read_text())
    assert manifest["demo_only"] is True

    with sqlite3.connect(archive_root / "index.db") as conn:
        conn.execute("PRAGMA user_version = 1")
        conn.commit()

    second = await seed_demo_archive(archive_root, force=False)
    assert second.healed_tiers == ("index.db",)


@pytest.mark.asyncio
async def test_seed_demo_archive_revokes_self_heal_once_a_demo_only_root_gains_real_content(
    tmp_path: Path,
) -> None:
    """A demo-only root that later gains real content loses self-heal eligibility.

    Guards polylogue-dl6af gap 1: the ownership manifest's ``demo_only: true``
    bit was previously treated as PERMANENT proof of exclusive demo
    ownership. If an operator seeds the zero-friction demo against a fresh
    default root and then starts using that same root normally -- real
    sessions accumulate through ordinary ingest -- the manifest still says
    ``demo_only: true`` forever. A later ``demo seed`` hitting schema drift
    must not trust that historical bit alone and self-heal by moving aside
    the now-real ``source.db``/``user.db``.

    ANTI-VACUITY: exercises the real ``seed_demo_archive`` entry point twice,
    with a real (non-demo) session inserted directly into the index tier in
    between -- mirroring an operator's normal-use ingest, not a test-only
    reimplementation. Reverting ``_archive_root_is_demo_owned`` to trust the
    historical ``demo_only`` bit without revalidating against the recorded
    ``demo_session_ids`` baseline makes this test fail: the second call
    would self-heal (``healed_tiers == ("index.db",)``) instead of raising
    the original unhandled schema-mismatch ``RuntimeError``, and the
    ``index.db.stale-*`` backup would appear on disk.
    """

    from tests.infra.storage_records import SessionBuilder

    archive_root = tmp_path / "archive"

    first = await seed_demo_archive(archive_root, force=True)
    assert first.healed_tiers == ()
    manifest = json.loads((archive_root / DEMO_OWNERSHIP_MANIFEST_FILENAME).read_text())
    assert manifest["demo_only"] is True
    assert manifest["demo_session_ids"]

    # Normal use: a real session lands in the same root through ordinary
    # ingest, independent of anything the demo seeder wrote.
    SessionBuilder(archive_root / "index.db", "real-session-after-seed").provider("claude-code").save()

    with sqlite3.connect(archive_root / "index.db") as conn:
        conn.execute("PRAGMA user_version = 1")
        conn.commit()

    with pytest.raises(RuntimeError, match="move it aside and rebuild the archive root"):
        await seed_demo_archive(archive_root, force=True)

    assert not list(archive_root.glob("index.db.stale-*"))


@pytest.mark.asyncio
async def test_record_demo_ownership_treats_missing_index_as_unsafe_not_empty(tmp_path: Path) -> None:
    """A missing/unreadable index.db must never authorize moving aside real durable content.

    Guards polylogue-dl6af gap 2: ``index.db`` is explicitly the rebuildable
    tier (this repo's "Schema regimes" doc) and can legitimately be absent
    on a real archive -- e.g. right after ``polylogue ops reset --index``,
    before the daemon has rebuilt it. Classifying that absence as "0
    sessions" (proof of emptiness) let the first-touch ownership manifest
    record ``demo_only: true`` for a root whose durable tiers (``source.db``,
    ``user.db``) already held real content, which would then authorize
    self-heal to move that real content aside on a later schema mismatch.

    ANTI-VACUITY: the production entry point exercised is
    ``polylogue.demo.seed.seed_demo_archive`` with ``explicit_root=True``
    (bypassing the separate default-root collision guard so this test
    isolates the ownership-manifest predicate specifically). Reverting
    ``_archive_root_has_real_content`` to trust ``_archive_tier_session_count``
    (index-only) alone makes this test fail: the manifest would record
    ``demo_only: true`` instead of ``false``.
    """

    from polylogue.storage.sqlite.archive_tiers.bootstrap import (
        initialize_active_archive_root,
        initialize_archive_database,
    )
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    archive_root = tmp_path / "archive"
    archive_root.mkdir(parents=True)

    # Populate only the durable tiers with real content; deliberately leave no
    # index.db, mirroring a root whose rebuildable tier was reset or never
    # rebuilt. Bootstrap the whole root first and then drop the rebuildable
    # tier: a source.db created on its own has no provenance marker, so durable
    # admission rejects the root before the missing-index case is reached.
    initialize_active_archive_root(archive_root)
    (archive_root / "index.db").unlink(missing_ok=True)
    initialize_archive_database(archive_root / "source.db", ArchiveTier.SOURCE)
    with sqlite3.connect(archive_root / "source.db") as conn:
        conn.execute(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, native_id, source_path, source_index,
                blob_hash, blob_size, acquired_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "real-raw-0",
                "claude-code-session",
                "real-session-0",
                "/real/source/path",
                0,
                b"\x00" * 32,
                1,
                0,
            ),
        )
        conn.commit()
    assert not (archive_root / "index.db").exists()

    seed = await seed_demo_archive(archive_root, force=True, explicit_root=True)

    manifest = json.loads((archive_root / DEMO_OWNERSHIP_MANIFEST_FILENAME).read_text())
    assert manifest["demo_only"] is False
    assert seed.healed_tiers == ()
