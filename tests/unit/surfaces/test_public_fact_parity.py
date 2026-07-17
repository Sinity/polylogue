"""One public session-profile fact across stable Polylogue read surfaces.

The survivor uses one authored :class:`ArchiveScenario` fact set and the real
materializer, then compares semantic meaning rather than serialized bytes.
MCP tool and browser-DOM obligations remain with their active rewrites; this
module exercises the current daemon HTTP substrate, not an obsolete renderer.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from http import HTTPStatus
from pathlib import Path

import pytest

from polylogue.insights.archive_models import ARCHIVE_INSIGHT_CONTRACT_VERSION
from tests.infra.archive_scenarios import (
    ArchiveScenario,
    ScenarioAttachment,
    ScenarioContentBlock,
    ScenarioMessage,
    archive_for_scenario_db,
    seed_workspace_scenarios,
)
from tests.infra.json_contracts import json_object
from tests.infra.semantic_facts import (
    SessionProfileFacts,
    assert_same_session_profile_facts,
)
from tests.infra.surfaces import CLISurface, DaemonHTTPSurface, FacadeSurface, RepositorySurface

_SELECTED_ORIGIN = "claude-code-session"
_DECOY_ORIGIN = "chatgpt-export"
_MISSING_ORIGIN = "codex-session"
_SELECTED_UPDATED_AT = "2026-07-01T09:05:00+00:00"


def _timestamp_millis(value: str) -> int:
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)


def _profile_fact_scenarios() -> tuple[ArchiveScenario, ArchiveScenario, ArchiveScenario]:
    selected = ArchiveScenario(
        name="public-fact-selected",
        provider="claude-code",
        title="Selected profile fact",
        created_at="2026-07-01T09:00:00+00:00",
        updated_at=_SELECTED_UPDATED_AT,
        messages=(
            ScenarioMessage(
                message_id="selected-u1",
                role="user",
                text="alpha beta gamma",
                timestamp="2026-07-01T09:00:00+00:00",
                attachments=(
                    ScenarioAttachment(
                        attachment_id="selected-attachment",
                        mime_type="text/plain",
                        size_bytes=11,
                        path="/workspace/fact.txt",
                    ),
                ),
            ),
            ScenarioMessage(
                message_id="selected-a1",
                role="assistant",
                text="delta epsilon zeta",
                timestamp="2026-07-01T09:02:00+00:00",
                blocks=(
                    ScenarioContentBlock.text_block("delta epsilon zeta"),
                    ScenarioContentBlock.tool_use(
                        tool_name="Read",
                        tool_input={"path": "/workspace/fact.txt"},
                        tool_id="tool-selected-1",
                    ),
                ),
            ),
            ScenarioMessage(
                message_id="selected-a2",
                role="assistant",
                text="eta theta iota",
                timestamp="2026-07-01T09:04:00+00:00",
                blocks=(
                    ScenarioContentBlock.text_block("eta theta iota"),
                    ScenarioContentBlock.thinking("verify the public fact"),
                ),
            ),
        ),
    )
    decoy = ArchiveScenario(
        name="public-fact-decoy",
        provider="chatgpt",
        title="Origin filter decoy",
        created_at="2026-07-01T10:00:00+00:00",
        updated_at="2026-07-01T10:01:00+00:00",
        messages=(
            ScenarioMessage(
                message_id="decoy-u1",
                role="user",
                text="decoy profile",
                timestamp="2026-07-01T10:00:00+00:00",
            ),
        ),
    )
    missing = ArchiveScenario(
        name="public-fact-unmaterialized",
        provider="codex",
        title="Existing but unmaterialized",
        created_at="2026-07-01T11:00:00+00:00",
        updated_at="2026-07-01T11:01:00+00:00",
        messages=(
            ScenarioMessage(
                message_id="missing-u1",
                role="user",
                text="not in the materialization",
                timestamp="2026-07-01T11:00:00+00:00",
            ),
        ),
    )
    return selected, decoy, missing


@pytest.mark.asyncio()
async def test_session_profile_fact_survives_repository_facade_cli_and_daemon_http(
    workspace_env: Mapping[str, Path],
) -> None:
    """Compare one materialized fact and its two public absence states.

    Production dependencies exercised:

    * ``SessionRepository.get_session_profile_record`` plus
      ``SessionProfileInsight.from_record``;
    * ``Polylogue.get_session_profile_insight``;
    * the descriptor-routed ``analyze insights profiles`` Click command;
    * ``GET /api/insights/sessions/{id}?include=profile``.

    Anti-vacuity mutations: dropping the CLI origin filter admits the decoy;
    dropping the archive HWM-source projection or daemon provenance projection
    disagrees with the repository fact; coercing an existing unmaterialized
    session to 404 disagrees with the explicit ``q-missing`` state.
    """
    selected, decoy, missing = _profile_fact_scenarios()
    db_path, _ = seed_workspace_scenarios(workspace_env, (selected, decoy))

    materializer = archive_for_scenario_db(db_path)
    try:
        rebuild = await materializer.rebuild_insights()
        assert rebuild.profiles == 2
    finally:
        await materializer.close()

    # This session exists in the archive but was deliberately planted after
    # the one rebuild. It is the independent q-missing fact, not a fabricated
    # daemon response or a deleted profile row.
    missing.seed(db_path)

    repository = RepositorySurface(db_path)
    facade = FacadeSurface(archive_root=workspace_env["archive_root"], db_path=db_path)
    cli = CLISurface(db_path=db_path)
    daemon = DaemonHTTPSurface(db_path=db_path)
    try:
        repository_insight = await repository.session_profile_insight(selected.native_session_id)
        facade_insight = await facade.session_profile_insight(selected.native_session_id)
        selected_cli = await cli.session_profile_payloads(origin=_SELECTED_ORIGIN)
        daemon_status, daemon_payload = await daemon.session_profile_response(selected.native_session_id)

        assert repository_insight is not None
        assert facade_insight is not None
        assert len(selected_cli) == 1
        assert selected_cli[0]["session_id"] == selected.native_session_id
        assert daemon_status == HTTPStatus.OK

        repository_facts = SessionProfileFacts.from_insight(repository_insight, context="repository")
        facade_facts = SessionProfileFacts.from_insight(facade_insight, context="facade")
        cli_facts = SessionProfileFacts.from_insight_payload(selected_cli[0], context="CLI")
        daemon_facts = SessionProfileFacts.from_daemon_payload(daemon_payload, context="daemon HTTP")
        assert_same_session_profile_facts(
            repository_facts,
            facade_facts,
            cli_facts,
            daemon_facts,
        )

        expected_hwm_ms = _timestamp_millis(_SELECTED_UPDATED_AT)
        assert repository_facts.session_id == selected.native_session_id
        assert repository_facts.logical_session_id == selected.native_session_id
        assert repository_facts.origin == _SELECTED_ORIGIN
        assert repository_facts.title == selected.title
        assert repository_facts.message_count == 3
        assert repository_facts.substantive_count == 1
        assert repository_facts.attachment_count == 1
        assert repository_facts.tool_use_count == 1
        assert repository_facts.thinking_count == 1
        assert repository_facts.word_count == 9
        assert repository_facts.provenance.materializer_version > 0
        assert repository_facts.provenance.materialized_at_ms > 0
        assert repository_facts.provenance.source_updated_at_ms == expected_hwm_ms
        assert repository_facts.provenance.source_sort_key == pytest.approx(expected_hwm_ms / 1000.0)
        assert repository_facts.provenance.input_high_water_mark_ms == expected_hwm_ms
        assert repository_facts.provenance.input_high_water_mark_source == "provider_ts"
        assert repository_facts.provenance.time_confidence == "recorded"

        for insight in (repository_insight, facade_insight):
            assert insight.contract_version == ARCHIVE_INSIGHT_CONTRACT_VERSION
            assert insight.insight_kind == "session_profile"
            assert insight.semantic_tier == "evidence"
        assert selected_cli[0]["contract_version"] == ARCHIVE_INSIGHT_CONTRACT_VERSION
        assert selected_cli[0]["insight_kind"] == "session_profile"
        assert selected_cli[0]["semantic_tier"] == "evidence"

        # The selected origin has exactly one profile even though the same
        # materialization contains a second origin. Removing root-filter
        # forwarding from the CLI command makes this assertion fail.
        decoy_cli = await cli.session_profile_payloads(origin=_DECOY_ORIGIN)
        assert len(decoy_cli) == 1
        assert decoy_cli[0]["session_id"] == decoy.native_session_id

        assert await repository.session_profile_insight(missing.native_session_id) is None
        assert await facade.session_profile_insight(missing.native_session_id) is None
        assert await cli.session_profile_payloads(origin=_MISSING_ORIGIN) == ()

        missing_status, missing_payload = await daemon.session_profile_response(missing.native_session_id)
        assert missing_status == HTTPStatus.OK
        assert missing_payload["session_id"] == missing.native_session_id
        assert missing_payload["origin"] == _MISSING_ORIGIN
        missing_kinds = json_object(missing_payload["kinds"], context="missing daemon kinds")
        missing_panel = json_object(missing_kinds["profile"], context="missing daemon profile panel")
        assert missing_panel["readiness_tag"] == "q-missing"
        assert missing_panel["materialized"] is False
        assert missing_panel["profile"] is None
        assert missing_panel["provenance"] is None

        unknown_ref = "totally-unknown-profile-ref"
        assert await repository.session_profile_insight(unknown_ref) is None
        assert await facade.session_profile_insight(unknown_ref) is None
        unknown_status, unknown_payload = await daemon.session_profile_response(unknown_ref)
        assert unknown_status == HTTPStatus.NOT_FOUND
        assert unknown_payload["ok"] is False
        assert unknown_payload["error"] == "not_found"
        assert unknown_payload["detail"] is None
        assert unknown_payload["field"] is None
    finally:
        await daemon.close()
        await cli.close()
        await facade.close()
        await repository.close()
