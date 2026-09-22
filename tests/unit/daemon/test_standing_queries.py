from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, cast

from polylogue.archive.query.evaluator import QueryEvaluation, QueryEvaluationRequest
from polylogue.core.enums import AssertionKind, AssertionStatus
from polylogue.daemon.convergence import DaemonConverger
from polylogue.daemon.convergence_stages import make_standing_query_stage
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.user_write import (
    FindingAssertion,
    mark_assertion_status,
    upsert_findings_as_assertions,
)
from polylogue.storage.sqlite.query_objects import (
    EvaluationReceipt,
    get_result_set_members,
    get_watched_query_baseline,
    put_query,
    put_query_name,
)


class _Evaluator:
    def __init__(
        self,
        *,
        members: tuple[str, ...],
        cache_only: bool = False,
        exactness: Literal["exact", "capped", "sampled", "estimate"] = "exact",
    ) -> None:
        self.members = members
        self.cache_only = cache_only
        self.exactness = exactness
        self.corpus_epoch = "index:g1"
        self.requests: list[QueryEvaluationRequest] = []

    def evaluate(self, request: QueryEvaluationRequest) -> QueryEvaluation:
        self.requests.append(request)
        return QueryEvaluation(
            grain="session",
            member_refs=self.members,
            corpus_epoch=self.corpus_epoch,
            exactness=self.exactness,
            cache_only=self.cache_only,
            receipt=EvaluationReceipt(
                receipt_id=f"receipt-{len(self.requests)}-{request.purpose}",
                source_generation="source:g1",
                user_generation="user:g1",
                index_generation="index:g1",
                runtime_build_ref="build:test",
            ),
        )

    def resolve_cohort(self, _operand: object) -> QueryEvaluation:
        raise AssertionError("standing queries do not resolve cohorts directly")


def _seed_watch(tmp_path: Path, *, watch: bool = True) -> tuple[Path, str]:
    index_db = tmp_path / "index.db"
    user_db = tmp_path / "user.db"
    initialize_archive_database(user_db, ArchiveTier.USER)
    with sqlite3.connect(user_db) as conn:
        query = put_query(
            conn,
            {"field": "origin", "value": "codex-session"},
            grain="session",
            lane="dialogue",
            rank_policy="mixed",
            created_at_ms=1,
        )
        if watch:
            put_query_name(conn, name="codex", query_hash=query.query_hash, watch=True, updated_at_ms=2)
        conn.commit()
    return index_db, query.query_hash


def test_watched_session_delta_is_candidate_once_and_has_self_firewall(tmp_path: Path) -> None:
    index_db, query_hash = _seed_watch(tmp_path)
    evaluator = _Evaluator(members=("session:one",))
    stage = make_standing_query_stage(index_db, evaluator=evaluator)

    assert stage.check_sessions is not None and stage.execute_sessions is not None
    assert stage.check_sessions(("session:changed",)) == {"session:changed"}
    assert stage.execute_sessions(("session:changed",)) is True  # baseline only
    evaluator.members = ("session:one", "session:two")
    assert stage.execute_sessions(("session:changed",)) is True
    assert stage.execute_sessions(("session:changed",)) is True

    with sqlite3.connect(tmp_path / "user.db") as conn:
        rows = conn.execute("SELECT status, target_ref, value_json FROM assertions WHERE kind = 'finding'").fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "candidate"
        assert rows[0][1] == f"query:{query_hash}"
        assert "query-delta" in str(rows[0][2])
        assert conn.execute("SELECT COUNT(*) FROM query_evaluation_receipts").fetchone()[0] == 3
    request = evaluator.requests[0]
    assert request.excluded_scope_refs == (f"query:{query_hash}",)
    assert request.excluded_origin_prefixes == ("notice.",)


def test_cache_only_evaluation_after_index_reset_never_compares_baseline(tmp_path: Path) -> None:
    index_db, _ = _seed_watch(tmp_path)
    evaluator = _Evaluator(members=("session:one",))
    stage = make_standing_query_stage(index_db, evaluator=evaluator)
    assert stage.execute_sessions is not None
    assert stage.execute_sessions(("session:changed",)) is True
    evaluator.members = ("session:one", "session:two")
    evaluator.cache_only = True
    assert stage.execute_sessions(("session:changed",)) is True

    with sqlite3.connect(tmp_path / "user.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM assertions WHERE kind = 'finding'").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM result_sets WHERE persistence_class = 'watch'").fetchone()[0] == 1


def test_watched_aliases_materialize_one_query_once_per_convergence(tmp_path: Path) -> None:
    index_db, query_hash = _seed_watch(tmp_path)
    with sqlite3.connect(tmp_path / "user.db") as conn:
        put_query_name(conn, name="codex-alias", query_hash=query_hash, watch=True, updated_at_ms=3)
        conn.commit()

    evaluator = _Evaluator(members=("session:one",))
    stage = make_standing_query_stage(index_db, evaluator=evaluator)
    assert stage.execute_sessions is not None
    assert stage.execute_sessions(("session:changed",)) is True

    assert len(evaluator.requests) == 1
    with sqlite3.connect(tmp_path / "user.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM result_sets WHERE persistence_class = 'watch'").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM assertions WHERE kind = 'finding'").fetchone()[0] == 0


def test_watch_baseline_tracks_return_to_a_prior_membership(tmp_path: Path) -> None:
    index_db, query_hash = _seed_watch(tmp_path)
    evaluator = _Evaluator(members=("session:a",))
    stage = make_standing_query_stage(index_db, evaluator=evaluator)
    assert stage.execute_sessions is not None
    assert stage.execute_sessions(("session:changed",)) is True  # baseline A
    evaluator.members = ("session:b",)
    assert stage.execute_sessions(("session:changed",)) is True  # A -> B
    evaluator.members = ("session:a",)
    assert stage.execute_sessions(("session:changed",)) is True  # B -> A

    with sqlite3.connect(tmp_path / "user.db") as conn:
        rows = conn.execute(
            "SELECT value_json FROM assertions WHERE kind = 'finding' ORDER BY created_at_ms, assertion_id"
        ).fetchall()
        assert len(rows) == 2
        assert "baseline_ref" in str(rows[1][0])
        assert "result-set:watch-" in str(rows[1][0])
        baseline = get_watched_query_baseline(conn, query_hash)
        assert baseline is not None
        assert get_result_set_members(conn, baseline.result_set_id) == ("session:a",)

    evaluator.members = ("session:b",)
    assert stage.execute_sessions(("session:changed",)) is True  # A -> B again
    with sqlite3.connect(tmp_path / "user.db") as conn:
        # Stable finding identity deduplicates the repeated B delta, but the
        # durable pointer must still advance to B for the next transition.
        assert conn.execute("SELECT COUNT(*) FROM assertions WHERE kind = 'finding'").fetchone()[0] == 2
        baseline = get_watched_query_baseline(conn, query_hash)
        assert baseline is not None
        assert get_result_set_members(conn, baseline.result_set_id) == ("session:b",)


def test_watch_snapshot_metadata_creates_a_new_manifest_without_membership_drift(tmp_path: Path) -> None:
    index_db, query_hash = _seed_watch(tmp_path)
    evaluator = _Evaluator(members=("session:a", "session:b"))
    stage = make_standing_query_stage(index_db, evaluator=evaluator)
    assert stage.execute_sessions is not None
    assert stage.execute_sessions(("session:changed",)) is True

    evaluator.members = ("session:b", "session:a")  # same membership, new rank order
    assert stage.execute_sessions(("session:changed",)) is True
    evaluator.corpus_epoch = "index:g2"
    assert stage.execute_sessions(("session:changed",)) is True

    with sqlite3.connect(tmp_path / "user.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM result_sets WHERE persistence_class = 'watch'").fetchone()[0] == 3
        assert conn.execute("SELECT COUNT(*) FROM assertions WHERE kind = 'finding'").fetchone()[0] == 0
        baseline = get_watched_query_baseline(conn, query_hash)
        assert baseline is not None
        assert baseline.corpus_epoch == "index:g2"
        assert get_result_set_members(conn, baseline.result_set_id) == ("session:b", "session:a")


def test_promoted_expected_count_divergence_targets_original_finding_without_watch(tmp_path: Path) -> None:
    index_db, query_hash = _seed_watch(tmp_path, watch=False)
    with sqlite3.connect(tmp_path / "user.db") as conn:
        original = upsert_findings_as_assertions(
            conn,
            [
                FindingAssertion(
                    claim_key="expected-count",
                    target_ref=f"query:{query_hash}",
                    body_text="Expected one member.",
                    finding_kind="measure",
                    statistic={"op": "count", "value": 1, "unit": "members"},
                    n=1,
                    query_ref=f"query:{query_hash}",
                    result_set_ref="result-set:original",
                    detector_ref="agent:test-detector",
                    expected={"measure": "member_count", "op": "=", "value": 1},
                )
            ],
            now_ms=1,
        )[0]
        mark_assertion_status(conn, original.assertion_id, AssertionStatus.ACCEPTED, now_ms=2)
        conn.commit()
    stage = make_standing_query_stage(index_db, evaluator=_Evaluator(members=("session:one", "session:two")))
    converger = DaemonConverger(stages=(stage,))
    states, _ = converger.converge_sessions(("session:changed",))
    assert states["session:changed"].stages["standing-queries"].value == "done"

    with sqlite3.connect(tmp_path / "user.db") as conn:
        rows = conn.execute(
            "SELECT target_ref, value_json, status FROM assertions WHERE kind = ? ORDER BY created_at_ms, assertion_id",
            (AssertionKind.FINDING.value,),
        ).fetchall()
        assert len(rows) == 2
        assert rows[1][0] == f"assertion:{original.assertion_id}"
        assert '"finding_kind":"query-drift"' in str(rows[1][1])
        assert rows[1][2] == "candidate"


# ---------------------------------------------------------------------------
# Planner-derived candidate-set narrowing (polylogue-bv1w.1)
# ---------------------------------------------------------------------------

#: An origin-scoped leaf in the executable predicate grammar the production
#: evaluator inverts -- the shape a real origin-scoped watch persists.
_ORIGIN_LEAF = {"kind": "field", "field": "origin", "op": "=", "values": ["codex-session"]}
_UNSCOPABLE_LEAF = {"kind": "field", "field": "tag", "op": "=", "values": ["release"]}


def _seed_origin_archive(root: Path, layout: Mapping[str, int]) -> dict[str, tuple[str, ...]]:
    """Seed real sessions across origins through the production writer."""
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
    from tests.infra.storage_records import SessionBuilder

    root.mkdir(parents=True, exist_ok=True)
    initialize_active_archive_root(root)
    seeded: dict[str, list[str]] = {}
    for provider, count in layout.items():
        for index in range(count):
            builder = SessionBuilder(root / "index.db", f"{provider}-{index}")
            builder.provider(provider)
            builder.add_message(role="user", text=f"standing scope {provider} {index}")
            builder.save()
            seeded.setdefault(provider, []).append(builder.native_session_id())
    return {provider: tuple(ids) for provider, ids in seeded.items()}


def _put_watch(root: Path, ast: Mapping[str, object], *, name: str = "watched") -> str:
    from polylogue.storage.sqlite.query_objects import put_query, put_query_name

    with sqlite3.connect(root / "user.db") as conn:
        query = put_query(
            conn,
            dict(ast),  # type: ignore[arg-type]
            grain="session",
            lane="dialogue",
            rank_policy="mixed",
            created_at_ms=1,
        )
        put_query_name(conn, name=name, query_hash=query.query_hash, watch=True, updated_at_ms=2)
        conn.commit()
    return query.query_hash


def _tick(stage: object, session_ids: tuple[str, ...]) -> set[str]:
    """One convergence tick, shaped exactly as the driver runs one."""
    check = stage.check_sessions  # type: ignore[attr-defined]
    execute = stage.execute_sessions  # type: ignore[attr-defined]
    assert check is not None and execute is not None
    candidates = set(check(session_ids)).intersection(session_ids)
    if candidates:
        assert execute(tuple(sorted(candidates))) is True
    return candidates


def _durable_watch_state(root: Path) -> dict[str, object]:
    """Every durable membership fact the stage is allowed to change.

    Result-set identifiers are resolved to their member lists because a
    ``result_set_id`` hashes ``corpus_epoch``, which moves with the archive's
    ``max(updated_at_ms)`` watermark -- an identifier that differs between two
    separately seeded archives even when they record identical membership.
    What a caller can observe, and what this compares, is which sessions each
    watched definition holds and which deltas were reported.
    """
    import json

    from polylogue.storage.sqlite.query_objects import get_result_set_members

    with sqlite3.connect(root / "user.db") as conn:

        def resolve(value: object) -> object:
            if isinstance(value, str) and value.startswith("result-set:"):
                return get_result_set_members(conn, value.removeprefix("result-set:"))
            if isinstance(value, dict):
                return {key: resolve(item) for key, item in sorted(value.items())}
            if isinstance(value, list):
                return [resolve(item) for item in value]
            return value

        baselines = conn.execute(
            "SELECT query_hash, result_set_id FROM watched_query_baselines ORDER BY query_hash"
        ).fetchall()
        membership = {
            str(query_hash): get_result_set_members(conn, str(result_set_id)) for query_hash, result_set_id in baselines
        }
        findings = [
            (str(target_ref), resolve(json.loads(str(value_json))))
            for target_ref, value_json in conn.execute(
                "SELECT target_ref, value_json FROM assertions WHERE kind = 'finding' ORDER BY target_ref"
            )
        ]
    return {"membership": membership, "findings": findings}


class _UnscopedEvaluator:
    """The production evaluator with its scope claim withheld.

    This is the exhaustive-global control: identical membership, no bound, so
    ``check_sessions`` must keep every changed session.
    """

    def __init__(self, index_db: Path) -> None:
        from polylogue.archive.query.production_evaluator import ArchiveCanonicalPlanEvaluator

        self._inner = ArchiveCanonicalPlanEvaluator(index_db)

    def evaluate(self, request: QueryEvaluationRequest) -> QueryEvaluation:
        return self._inner.evaluate(request)

    def resolve_cohort(self, operand: object) -> QueryEvaluation:
        return self._inner.resolve_cohort(operand)  # type: ignore[arg-type]


def test_planner_scope_is_never_narrower_than_the_evaluators_own_membership(tmp_path: Path) -> None:
    """The bound is a claim about the planner's own results; check it against them.

    A narrowed candidate set is a correctness change, so the bound must be a
    provable over-approximation of what the same planner actually matches.
    This runs the real evaluator over a real multi-origin archive and requires
    every returned member's origin to lie inside the declared bound.

    Anti-vacuity: the `bounded` counter below requires at least one shape that
    both produced a bound and returned members, so a `declared_origin_scope`
    that always answered ``None`` -- or an evaluator that matched nothing --
    cannot satisfy this.
    """
    from polylogue.archive.query.evaluator import declared_origin_scope, session_origin
    from polylogue.archive.query.production_evaluator import ArchiveCanonicalPlanEvaluator
    from polylogue.storage.sqlite.query_objects import put_query

    root = tmp_path / "archive"
    _seed_origin_archive(root, {"codex": 2, "claude_code": 3, "chatgpt": 1})
    evaluator = ArchiveCanonicalPlanEvaluator(root / "index.db")

    claude_leaf = {"kind": "field", "field": "origin", "op": "=", "values": ["claude-code-session"]}
    shapes: tuple[dict[str, object], ...] = (
        dict(_ORIGIN_LEAF),
        {"kind": "or", "children": [dict(_ORIGIN_LEAF), dict(claude_leaf)]},
        {"kind": "and", "children": [dict(_ORIGIN_LEAF), dict(_UNSCOPABLE_LEAF)]},
        {"kind": "and", "children": [dict(_ORIGIN_LEAF), dict(claude_leaf)]},
        {"kind": "or", "children": [dict(_ORIGIN_LEAF), dict(_UNSCOPABLE_LEAF)]},
        {"kind": "not", "child": dict(_ORIGIN_LEAF)},
        dict(_UNSCOPABLE_LEAF),
    )

    bounded_with_members = 0
    with sqlite3.connect(root / "user.db") as conn:
        for index, ast in enumerate(shapes):
            query = put_query(
                conn,
                ast,  # type: ignore[arg-type]
                grain="session",
                lane="dialogue",
                rank_policy="mixed",
                created_at_ms=index + 1,
            )
            conn.commit()
            scope = declared_origin_scope(query)
            assert scope == evaluator.session_origin_scope(query)
            members = evaluator.evaluate(QueryEvaluationRequest(query=query, purpose="standing-watch")).member_refs
            if scope is None:
                continue
            observed = {session_origin(ref.removeprefix("session:")) for ref in members}
            assert observed <= scope, f"shape {index} matched an origin its declared bound excludes"
            if members:
                bounded_with_members += 1
    assert bounded_with_members >= 2


def test_changed_sessions_outside_every_watch_scope_are_not_candidates(tmp_path: Path) -> None:
    """B1-1, and the case a wrongly-narrowed scope would drop.

    Anti-vacuity, executed in one call: ``retained`` is a session of the
    watched origin whose change is exactly what the old unconditional
    ``return set(session_ids)`` caught, and ``dropped`` is a session of another
    origin the old scan needlessly evaluated. A scope that excluded the watched
    origin -- or one that returned an empty bound -- fails on ``retained``.
    """
    from polylogue.archive.query.production_evaluator import ArchiveCanonicalPlanEvaluator

    root = tmp_path / "archive"
    seeded = _seed_origin_archive(root, {"codex": 1, "claude_code": 1})
    _put_watch(root, _ORIGIN_LEAF)
    index_db = root / "index.db"
    stage = make_standing_query_stage(index_db, evaluator=ArchiveCanonicalPlanEvaluator(index_db))

    retained = seeded["codex"][0]
    dropped = seeded["claude_code"][0]

    # The first tick establishes the baseline and is never narrowed.
    assert _tick(stage, (retained, dropped)) == {retained, dropped}

    assert stage.check_sessions is not None
    assert stage.check_sessions((dropped,)) == set()
    assert stage.check_sessions((retained,)) == {retained}
    assert stage.check_sessions((retained, dropped)) == {retained}


def test_a_changed_in_scope_session_still_reports_its_membership_delta(tmp_path: Path) -> None:
    """Narrowing must not cost a firing the broad scan would have produced."""
    from polylogue.archive.query.production_evaluator import ArchiveCanonicalPlanEvaluator
    from tests.infra.storage_records import SessionBuilder

    root = tmp_path / "archive"
    seeded = _seed_origin_archive(root, {"codex": 1, "claude_code": 1})
    query_hash = _put_watch(root, _ORIGIN_LEAF)
    index_db = root / "index.db"
    stage = make_standing_query_stage(index_db, evaluator=ArchiveCanonicalPlanEvaluator(index_db))

    _tick(stage, (seeded["codex"][0],))  # baseline: one codex member
    state = _durable_watch_state(root)
    assert state["findings"] == []

    added = SessionBuilder(index_db, "codex-added")
    added.provider("codex")
    added.add_message(role="user", text="a new codex session")
    added.save()

    assert _tick(stage, (added.native_session_id(),)) == {added.native_session_id()}
    after = _durable_watch_state(root)
    assert len(cast("list[object]", after["findings"])) == 1
    assert cast("dict[str, tuple[str, ...]]", after["membership"])[query_hash] == (
        f"session:{added.native_session_id()}",
        f"session:{seeded['codex'][0]}",
    )


def test_an_unscopable_predicate_retains_the_global_baseline(tmp_path: Path) -> None:
    """B1-2: a predicate the planner cannot bound is never narrowed.

    Anti-vacuity: make the unsupported-predicate branch narrow instead of
    falling back (return an empty bound rather than ``None`` from
    ``declared_origin_scope``) and this reports an empty candidate set.
    """
    from polylogue.archive.query.evaluator import declared_origin_scope
    from polylogue.archive.query.production_evaluator import ArchiveCanonicalPlanEvaluator
    from polylogue.storage.sqlite.query_objects import get_query

    root = tmp_path / "archive"
    seeded = _seed_origin_archive(root, {"codex": 1, "claude_code": 1})
    query_hash = _put_watch(root, _UNSCOPABLE_LEAF)
    index_db = root / "index.db"
    stage = make_standing_query_stage(index_db, evaluator=ArchiveCanonicalPlanEvaluator(index_db))

    with sqlite3.connect(root / "user.db") as conn:
        query = get_query(conn, query_hash)
    assert query is not None
    assert declared_origin_scope(query) is None

    changed = (seeded["codex"][0], seeded["claude_code"][0])
    assert _tick(stage, changed) == set(changed)
    assert stage.check_sessions is not None
    assert stage.check_sessions(changed) == set(changed)


def test_one_unscopable_watch_beside_a_scoped_one_retains_the_global_baseline(tmp_path: Path) -> None:
    """The bound is the union of *proved* bounds; one gap abandons narrowing."""
    from polylogue.archive.query.production_evaluator import ArchiveCanonicalPlanEvaluator

    root = tmp_path / "archive"
    seeded = _seed_origin_archive(root, {"codex": 1, "claude_code": 1})
    _put_watch(root, _ORIGIN_LEAF, name="scoped")
    _put_watch(root, _UNSCOPABLE_LEAF, name="unscoped")
    index_db = root / "index.db"
    stage = make_standing_query_stage(index_db, evaluator=ArchiveCanonicalPlanEvaluator(index_db))

    changed = (seeded["codex"][0], seeded["claude_code"][0])
    assert _tick(stage, changed) == set(changed)
    assert stage.check_sessions is not None
    assert stage.check_sessions(changed) == set(changed)


def test_a_watch_without_a_durable_baseline_retains_the_global_baseline(tmp_path: Path) -> None:
    """B1-4: the first observation is silent, so it must not be deferred.

    If narrowing skipped the tick that establishes a baseline, the next tick
    would become the first observation and would record the changed membership
    silently instead of reporting it as a delta.
    """
    from polylogue.archive.query.production_evaluator import ArchiveCanonicalPlanEvaluator

    root = tmp_path / "archive"
    seeded = _seed_origin_archive(root, {"codex": 1, "claude_code": 1})
    index_db = root / "index.db"
    _put_watch(root, _ORIGIN_LEAF)
    stage = make_standing_query_stage(index_db, evaluator=ArchiveCanonicalPlanEvaluator(index_db))

    assert stage.check_sessions is not None
    out_of_scope = (seeded["claude_code"][0],)
    assert stage.check_sessions(out_of_scope) == set(out_of_scope)
    assert _tick(stage, out_of_scope) == set(out_of_scope)
    assert stage.check_sessions(out_of_scope) == set()


def test_a_definition_version_change_recomputes_the_scope(tmp_path: Path) -> None:
    """B1-4: nothing is persisted, so a new definition is scoped on its own terms.

    Repointing the watched name at a different definition both expands and
    moves the bound. The newly watched origin becomes a candidate on the very
    next tick without any stored fingerprint to invalidate.
    """
    from polylogue.archive.query.production_evaluator import ArchiveCanonicalPlanEvaluator

    root = tmp_path / "archive"
    seeded = _seed_origin_archive(root, {"codex": 1, "claude_code": 1})
    index_db = root / "index.db"
    _put_watch(root, _ORIGIN_LEAF)
    stage = make_standing_query_stage(index_db, evaluator=ArchiveCanonicalPlanEvaluator(index_db))

    codex, claude = seeded["codex"][0], seeded["claude_code"][0]
    _tick(stage, (codex, claude))
    assert stage.check_sessions is not None
    assert stage.check_sessions((claude,)) == set()

    _put_watch(
        root,
        {"kind": "field", "field": "origin", "op": "=", "values": ["claude-code-session"]},
    )
    # The new definition has no baseline of its own, so this tick is global.
    assert stage.check_sessions((claude,)) == {claude}
    _tick(stage, (claude,))
    assert stage.check_sessions((claude,)) == {claude}
    assert stage.check_sessions((codex,)) == set()


def test_an_accepted_expected_count_finding_retains_the_global_baseline(tmp_path: Path) -> None:
    """Expected-count drift is measured against a stored expectation, not a delta.

    Such a finding can already be drifting before anything in scope changes,
    so its report must not wait for an in-scope session.
    """
    from polylogue.archive.query.production_evaluator import ArchiveCanonicalPlanEvaluator

    root = tmp_path / "archive"
    seeded = _seed_origin_archive(root, {"codex": 1, "claude_code": 1})
    index_db = root / "index.db"
    query_hash = _put_watch(root, _ORIGIN_LEAF)
    stage = make_standing_query_stage(index_db, evaluator=ArchiveCanonicalPlanEvaluator(index_db))
    _tick(stage, (seeded["codex"][0],))
    assert stage.check_sessions is not None
    assert stage.check_sessions((seeded["claude_code"][0],)) == set()

    with sqlite3.connect(root / "user.db") as conn:
        promoted = upsert_findings_as_assertions(
            conn,
            [
                FindingAssertion(
                    claim_key="expected-count",
                    target_ref=f"query:{query_hash}",
                    body_text="Expected one member.",
                    finding_kind="measure",
                    statistic={"op": "count", "value": 1, "unit": "members"},
                    n=1,
                    query_ref=f"query:{query_hash}",
                    result_set_ref="result-set:original",
                    detector_ref="agent:test-detector",
                    expected={"measure": "member_count", "op": "=", "value": 1},
                )
            ],
            now_ms=1,
        )[0]
        mark_assertion_status(conn, promoted.assertion_id, AssertionStatus.ACCEPTED, now_ms=2)
        conn.commit()

    assert stage.check_sessions((seeded["claude_code"][0],)) == {seeded["claude_code"][0]}


def test_scoped_and_exhaustive_ticks_record_the_same_membership_and_findings(tmp_path: Path) -> None:
    """B1-3: narrowing changes which ticks run, never what they record.

    Two identically seeded archives run the same change sequence -- one with
    the planner's bound available, one with it withheld so every tick is
    exhaustive. The durable watched-query membership and the findings must
    match exactly.
    """
    from polylogue.archive.query.production_evaluator import ArchiveCanonicalPlanEvaluator
    from tests.infra.storage_records import SessionBuilder

    def run(root: Path, *, scoped: bool) -> dict[str, object]:
        seeded = _seed_origin_archive(root, {"codex": 1, "claude_code": 2})
        index_db = root / "index.db"
        _put_watch(root, _ORIGIN_LEAF)
        evaluator = ArchiveCanonicalPlanEvaluator(index_db) if scoped else _UnscopedEvaluator(index_db)
        stage = make_standing_query_stage(index_db, evaluator=evaluator)

        _tick(stage, (seeded["codex"][0],))
        for step in range(3):
            # Out-of-scope churn: cannot change a codex-scoped membership.
            _tick(stage, seeded["claude_code"])
            if step == 1:
                added = SessionBuilder(index_db, "codex-late")
                added.provider("codex")
                added.add_message(role="user", text="late codex arrival")
                added.save()
                _tick(stage, (added.native_session_id(),))
        return _durable_watch_state(root)

    scoped_state = run(tmp_path / "scoped", scoped=True)
    global_state = run(tmp_path / "global", scoped=False)
    assert scoped_state == global_state
    assert cast("list[object]", scoped_state["findings"]), "the sequence must produce a real delta to compare"


def test_an_evaluator_without_a_scope_claim_keeps_every_changed_session(tmp_path: Path) -> None:
    """A planner that publishes no bound can never shrink the candidate set."""
    index_db, _ = _seed_watch(tmp_path)
    stage = make_standing_query_stage(index_db, evaluator=_Evaluator(members=("session:one",)))
    assert stage.check_sessions is not None
    assert not hasattr(_Evaluator, "session_origin_scope")
    assert stage.check_sessions(("codex-session:a", "claude-code-session:b")) == {
        "codex-session:a",
        "claude-code-session:b",
    }


def test_a_capped_evaluation_claims_no_membership_delta(tmp_path: Path) -> None:
    """A non-exact evaluation never emits ``query-delta`` (polylogue-uwm6y).

    ``member_refs`` from a capped, sampled or estimated evaluation is a bounded
    view of the watched relation, not the relation, so its merkle root answers
    a different question than the baseline's.  Comparing them would let a
    shifting cap window alone emit an unconditional "membership changed"
    assertion.

    Emitting nothing would be the mirror defect -- an unmeasured negative
    reported as a measured one -- so the degraded receipt is always written and
    a baselined watch gets exactly one ``query-delta-unmeasured`` candidate
    naming the condition.

    Anti-vacuity: drop the ``exactness != "exact"`` branch and this asserts
    ``query-delta`` on a membership the evaluator never enumerated, and the
    baseline advances to a capped root.
    """
    index_db, query_hash = _seed_watch(tmp_path)
    evaluator = _Evaluator(members=("session:one",))
    stage = make_standing_query_stage(index_db, evaluator=evaluator)
    assert stage.execute_sessions is not None

    # An exact tick first, so there is a real drift question to answer.
    assert stage.execute_sessions(("session:changed",)) is True
    with sqlite3.connect(tmp_path / "user.db") as conn:
        before = conn.execute("SELECT result_set_id, updated_at_ms FROM watched_query_baselines").fetchone()
    assert before is not None

    evaluator.members = ("session:one", "session:two")
    evaluator.exactness = "capped"
    assert stage.execute_sessions(("session:changed",)) is True

    with sqlite3.connect(tmp_path / "user.db") as conn:
        rows = conn.execute("SELECT status, value_json FROM assertions WHERE kind = 'finding'").fetchall()
        after = conn.execute("SELECT result_set_id, updated_at_ms FROM watched_query_baselines").fetchone()
    kinds = [str(row[1]) for row in rows]
    assert not any('"query-delta"' in kind for kind in kinds), "a capped view is not a membership"
    assert any("query-delta-unmeasured" in kind for kind in kinds), "the unanswered question is still recorded"
    assert after == before, "a non-exact tick never advances the baseline"


def _install_legacy_capped_baseline(user_db: Path, query_hash: str, members: tuple[str, ...]) -> str:
    """Write the baseline shape a pre-guard ``user.db`` can already contain.

    Before the non-exact guard landed, every evaluation was persisted as the
    watch baseline, so a durable ``watched_query_baselines`` row can point at a
    result set whose ``exactness`` is ``capped``/``sampled``/``estimate``.
    """
    from polylogue.storage.sqlite.query_objects import put_result_set, put_watched_query_baseline

    result_set_id = "watch-legacy-capped"
    with sqlite3.connect(user_db) as conn:
        put_result_set(
            conn,
            result_set_id=result_set_id,
            query_hash=query_hash,
            grain="session",
            corpus_epoch="index:g1",
            member_refs=members,
            exactness="capped",
            persistence_class="watch",
            created_at_ms=3,
        )
        put_watched_query_baseline(
            conn,
            query_hash=query_hash,
            result_set_id=result_set_id,
            updated_at_ms=3,
        )
        conn.commit()
    return result_set_id


def test_legacy_capped_baseline_replaced_silently(tmp_path: Path) -> None:
    """A non-exact stored baseline is unmeasured, not a prior observation.

    Its merkle root is over a bounded view, so comparing an exact root against
    it emits a definitive ``query-delta`` for a membership change nobody
    observed -- the same false claim the guard prevents for new baselines.

    Anti-vacuity: drop ``comparable_baseline`` and this asserts a
    ``query-delta`` candidate against the legacy capped root. The fixture's
    legacy members deliberately differ from the exact tick's, because an equal
    root would be silent under both implementations.
    """
    index_db, query_hash = _seed_watch(tmp_path)
    legacy_id = _install_legacy_capped_baseline(tmp_path / "user.db", query_hash, ("session:one",))
    evaluator = _Evaluator(members=("session:one", "session:two"))
    stage = make_standing_query_stage(index_db, evaluator=evaluator)
    assert stage.execute_sessions is not None
    assert stage.execute_sessions(("session:changed",)) is True

    with sqlite3.connect(tmp_path / "user.db") as conn:
        kinds = [str(row[0]) for row in conn.execute("SELECT value_json FROM assertions WHERE kind = 'finding'")]
        baseline = get_watched_query_baseline(conn, query_hash)
    assert kinds == [], "an exact root compared against a capped one claims nothing"
    assert baseline is not None
    assert baseline.result_set_id != legacy_id, "the first exact evaluation replaces the legacy baseline"
    assert baseline.exactness == "exact"


def test_exact_delta_after_legacy_baseline_replaced(tmp_path: Path) -> None:
    """Opposite direction: the replacement baseline still reports real drift."""
    index_db, query_hash = _seed_watch(tmp_path)
    _install_legacy_capped_baseline(tmp_path / "user.db", query_hash, ("session:one",))
    evaluator = _Evaluator(members=("session:one", "session:two"))
    stage = make_standing_query_stage(index_db, evaluator=evaluator)
    assert stage.execute_sessions is not None
    assert stage.execute_sessions(("session:changed",)) is True
    evaluator.members = ("session:one", "session:two", "session:three")
    assert stage.execute_sessions(("session:changed",)) is True

    with sqlite3.connect(tmp_path / "user.db") as conn:
        kinds = [str(row[0]) for row in conn.execute("SELECT value_json FROM assertions WHERE kind = 'finding'")]
    assert len(kinds) == 1
    assert '"query-delta"' in kinds[0]


def test_unmeasured_tick_ignores_legacy_baseline(tmp_path: Path) -> None:
    """A capped tick against a capped baseline has no drift question to name."""
    index_db, query_hash = _seed_watch(tmp_path)
    _install_legacy_capped_baseline(tmp_path / "user.db", query_hash, ("session:one",))
    evaluator = _Evaluator(members=("session:one", "session:two"), exactness="capped")
    stage = make_standing_query_stage(index_db, evaluator=evaluator)
    assert stage.execute_sessions is not None
    assert stage.execute_sessions(("session:changed",)) is True

    with sqlite3.connect(tmp_path / "user.db") as conn:
        kinds = [str(row[0]) for row in conn.execute("SELECT value_json FROM assertions WHERE kind = 'finding'")]
        assert conn.execute("SELECT COUNT(*) FROM query_evaluation_receipts").fetchone()[0] == 1
    assert kinds == []
