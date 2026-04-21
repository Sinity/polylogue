"""Tests for shared archive scenario and oracle helpers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

from polylogue.storage.backends.connection import open_connection
from tests.infra.archive_scenarios import (
    ArchiveScenario,
    ScenarioAttachment,
    ScenarioMessage,
    repository_for_scenario_db,
    seed_workspace_scenarios,
)
from tests.infra.oracles import assert_conversation_surfaces_agree
from tests.infra.semantic_facts import ArchiveFacts, assert_same_archive_facts


def test_archive_scenario_seeds_record_and_hydration_facts(workspace_env: Mapping[str, Path]) -> None:
    scenario = ArchiveScenario(
        name="scenario-harness",
        provider="claude-code",
        title="Harness conversation",
        messages=(
            ScenarioMessage(role="user", text="Run the checks", message_id="m-user"),
            ScenarioMessage(
                role="assistant",
                text="Using pytest",
                message_id="m-tool",
                content_blocks=(
                    {
                        "type": "tool_use",
                        "tool_name": "shell",
                        "tool_input": {"command": "pytest -q"},
                    },
                ),
                attachments=(ScenarioAttachment(attachment_id="att-report", mime_type="text/plain"),),
            ),
        ),
    )
    db_path, seeds = seed_workspace_scenarios(workspace_env, [scenario])

    assert [seed.conversation_id for seed in seeds] == ["scenario-harness"]
    with open_connection(db_path) as conn:
        assert_conversation_surfaces_agree(
            scenario.facts_from_connection(conn),
            scenario.hydrated_facts_from_connection(conn),
        )


@pytest.mark.asyncio()
async def test_archive_scenario_repository_projection_agrees(workspace_env: Mapping[str, Path]) -> None:
    scenario = ArchiveScenario(
        name="repository-harness",
        provider="chatgpt",
        title="Repository view",
        messages=(
            ScenarioMessage(role="user", text="Question", message_id="m1"),
            ScenarioMessage(role="assistant", text="Answer", message_id="m2"),
        ),
    )
    db_path, _ = seed_workspace_scenarios(workspace_env, [scenario])
    repository = repository_for_scenario_db(db_path)
    try:
        with open_connection(db_path) as conn:
            assert_conversation_surfaces_agree(
                scenario.facts_from_connection(conn),
                await scenario.facts_from_repository(repository),
            )
    finally:
        await repository.close()


@pytest.mark.asyncio()
async def test_archive_facts_compare_db_and_repository(workspace_env: Mapping[str, Path]) -> None:
    scenarios = [
        ArchiveScenario(name="archive-a", provider="chatgpt"),
        ArchiveScenario(name="archive-b", provider="codex"),
    ]
    db_path, _ = seed_workspace_scenarios(workspace_env, scenarios)
    repository = repository_for_scenario_db(db_path)
    try:
        with open_connection(db_path) as conn:
            db_facts = ArchiveFacts.from_db_connection(conn)
        repository_facts = ArchiveFacts.from_conversations(await repository.list(limit=None))
    finally:
        await repository.close()

    assert_same_archive_facts(db_facts, repository_facts)
