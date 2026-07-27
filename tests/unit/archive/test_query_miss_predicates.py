"""Real-fixture contracts for bounded predicate-drop miss diagnosis (polylogue-jnj.12).

Complements the mock-based ``test_query_miss_diagnostics.py`` with real
archive fixtures so the clause-drop attribution, since/until relaxation, and
FTS-vs-structured disagreement probes are proven against an actual queryable
index, not just wiring.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.api import Polylogue
from polylogue.archive.query.miss_diagnostics import diagnose_query_miss
from polylogue.archive.query.miss_predicates import (
    probe_date_relaxation_reasons,
    probe_fts_structured_disagreement,
    probe_predicate_zeroing,
)
from polylogue.archive.query.predicate import QueryFieldPredicate
from polylogue.archive.query.spec import SessionQuerySpec
from tests.infra.storage_records import SessionBuilder, db_setup


def _codes(reasons: object) -> list[str]:
    return [reason.code for reason in reasons]  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_probe_predicate_zeroing_names_both_culprits(workspace_env: dict[str, Path]) -> None:
    """Two disjoint filters: each one's removal alone should surface the other's matches."""
    db_path = db_setup(workspace_env)
    await (
        SessionBuilder(db_path, "conv-claude")
        .provider("claude-code")
        .git_repository_url("https://example.com/repo-a")
        .add_message("m1", text="hello")
        .build()
    )
    await (
        SessionBuilder(db_path, "conv-chatgpt")
        .provider("chatgpt")
        .git_repository_url("https://example.com/repo-b")
        .add_message("m1", text="hi")
        .build()
    )

    async with Polylogue(archive_root=db_path.parent, db_path=db_path) as facade:
        # No session is both claude-code AND repo-b -- zero hits, but each
        # filter alone matches the *other* session.
        spec = SessionQuerySpec(origins=("claude-code-session",), repo_names=("repo-b",))
        assert await spec.count(facade.config) == 0

        result = await probe_predicate_zeroing(spec, facade.config)

        assert _codes(result.reasons) == ["predicate_zeroed_set", "predicate_zeroed_set"]
        assert set(result.culprit_fields) == {"origins", "repo_names"}
        by_field = {reason.detail: reason for reason in result.reasons}
        assert by_field["field=origins"].count == 1  # repo-b alone matches conv-chatgpt
        assert by_field["field=repo_names"].count == 1  # claude-code alone matches conv-claude


@pytest.mark.asyncio
async def test_probe_predicate_zeroing_skips_single_active_predicate(workspace_env: dict[str, Path]) -> None:
    """A single active predicate is not attributed -- there's nothing to compare it against."""
    db_path = db_setup(workspace_env)
    await SessionBuilder(db_path, "conv-1").provider("claude-code").add_message("m1", text="hi").build()

    async with Polylogue(archive_root=db_path.parent, db_path=db_path) as facade:
        spec = SessionQuerySpec(origins=("chatgpt-export",))
        result = await probe_predicate_zeroing(spec, facade.config)

        assert result.reasons == ()
        assert result.culprit_fields == ()


@pytest.mark.asyncio
async def test_probe_predicate_zeroing_scopes_out_boolean_predicate_tree(workspace_env: dict[str, Path]) -> None:
    """A DSL-compiled boolean predicate tree is out of scope -- named, not guessed."""
    db_path = db_setup(workspace_env)
    async with Polylogue(archive_root=db_path.parent, db_path=db_path) as facade:
        spec = SessionQuerySpec(
            origins=("chatgpt-export",),
            repo_names=("repo-b",),
            boolean_predicate=QueryFieldPredicate(field="repo", values=("repo-b",)),
        )

        result = await probe_predicate_zeroing(spec, facade.config)

        assert _codes(result.reasons) == ["predicate_attribution_skipped_boolean_tree"]
        assert result.culprit_fields == ()


@pytest.mark.asyncio
async def test_probe_date_relaxation_suggests_nearest_boundary(workspace_env: dict[str, Path]) -> None:
    """since is the confirmed culprit: relaxation names the actual nearest match date."""
    db_path = db_setup(workspace_env)
    await (
        SessionBuilder(db_path, "conv-old")
        .provider("claude-code")
        .created_at("2020-06-15T00:00:00+00:00")
        .updated_at("2020-06-15T00:00:00+00:00")
        .add_message("m1", text="hi")
        .build()
    )

    async with Polylogue(archive_root=db_path.parent, db_path=db_path) as facade:
        spec = SessionQuerySpec(origins=("claude-code-session",), since="2030-01-01")
        assert await spec.count(facade.config) == 0

        probe_result = await probe_predicate_zeroing(spec, facade.config)
        assert "since" in probe_result.culprit_fields

        relaxation = await probe_date_relaxation_reasons(
            spec, facade.config, culprit_fields=probe_result.culprit_fields
        )

        assert len(relaxation) == 1
        assert relaxation[0].code == "predicate_relaxation_suggested"
        assert "2020-06-15" in (relaxation[0].detail or "")


@pytest.mark.asyncio
async def test_probe_date_relaxation_reports_unavailable_without_boundary(workspace_env: dict[str, Path]) -> None:
    """No culprit fields to relax -- nothing to probe, returns no reasons."""
    db_path = db_setup(workspace_env)
    async with Polylogue(archive_root=db_path.parent, db_path=db_path) as facade:
        spec = SessionQuerySpec(origins=("chatgpt-export",), since="2030-01-01")

        relaxation = await probe_date_relaxation_reasons(spec, facade.config, culprit_fields=())

        assert relaxation == ()


@pytest.mark.asyncio
async def test_probe_fts_structured_disagreement_detects_disjoint_sets(workspace_env: dict[str, Path]) -> None:
    """FTS text and structured filters each independently match, but never the same session."""
    db_path = db_setup(workspace_env)
    await (
        SessionBuilder(db_path, "conv-claude")
        .provider("claude-code")
        .add_message("m1", text="UniqueNeedlePhraseZQX")
        .build()
    )
    await (
        SessionBuilder(db_path, "conv-chatgpt")
        .provider("chatgpt")
        .add_message("m1", text="totally unrelated content")
        .build()
    )

    async with Polylogue(archive_root=db_path.parent, db_path=db_path) as facade:
        spec = SessionQuerySpec(query_terms=("UniqueNeedlePhraseZQX",), origins=("chatgpt-export",))
        assert await spec.count(facade.config) == 0

        reason = await probe_fts_structured_disagreement(spec, facade.config)

        assert reason is not None
        assert reason.code == "fts_structured_disagreement"
        assert "fts_only_count=1" in (reason.detail or "")
        assert "structured_only_count=1" in (reason.detail or "")


@pytest.mark.asyncio
async def test_probe_fts_structured_disagreement_none_without_structured_filters(
    workspace_env: dict[str, Path],
) -> None:
    """A pure FTS miss (no structured filters) isn't a 'disagreement' -- nothing to disagree with."""
    db_path = db_setup(workspace_env)
    async with Polylogue(archive_root=db_path.parent, db_path=db_path) as facade:
        spec = SessionQuerySpec(query_terms=("nonexistent",))

        reason = await probe_fts_structured_disagreement(spec, facade.config)

        assert reason is None


@pytest.mark.asyncio
async def test_diagnose_query_miss_default_is_bounded_full_adds_breakdown(
    workspace_env: dict[str, Path],
) -> None:
    """full=False still attributes the culprit; full=True adds relaxation + FTS-disagreement."""
    db_path = db_setup(workspace_env)
    await (
        SessionBuilder(db_path, "conv-old")
        .provider("claude-code")
        .created_at("2020-06-15T00:00:00+00:00")
        .updated_at("2020-06-15T00:00:00+00:00")
        .add_message("m1", text="UniqueNeedlePhraseZQX")
        .build()
    )
    await SessionBuilder(db_path, "conv-chatgpt").provider("chatgpt").add_message("m1", text="other").build()

    async with Polylogue(archive_root=db_path.parent, db_path=db_path) as facade:
        spec = SessionQuerySpec(
            query_terms=("UniqueNeedlePhraseZQX",),
            origins=("claude-code-session",),
            since="2030-01-01",
        )
        assert await spec.count(facade.config) == 0

        bounded = await diagnose_query_miss(facade, spec, config=facade.config, full=False)
        full = await diagnose_query_miss(facade, spec, config=facade.config, full=True)

        bounded_codes = _codes(bounded.reasons)
        full_codes = _codes(full.reasons)
        assert "predicate_zeroed_set" in bounded_codes
        assert "predicate_relaxation_suggested" not in bounded_codes
        assert "predicate_relaxation_suggested" in full_codes
        # full is a strict superset of bounded for this scenario.
        assert set(bounded_codes) <= set(full_codes)
