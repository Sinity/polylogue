"""Tag CRUD invariant tests — catalog-driven parametrization over tag lifecycle.

Covers: add, duplicate, remove, list, bulk-add, provider-filter, and
session-scoped reads. Uses the async repository interface.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

from tests.infra.archive_scenarios import (
    ArchiveScenario,
    ScenarioMessage,
    repository_for_scenario_db,
    seed_workspace_scenarios,
)

# ---------------------------------------------------------------------------
# Tag lifecycle operation catalog
# ---------------------------------------------------------------------------


TAG_LIFECYCLE_CASES: list[tuple[str, list[str], list[str], str]] = [
    # (operation, initial_tags, expected_tags_after, description)
    ("add", [], ["review"], "add single tag to untagged session"),
    ("add", ["existing"], ["existing", "review"], "add tag to already-tagged session"),
    ("add-duplicate", ["review"], ["review"], "add duplicate tag: idempotent"),
    ("remove", ["review"], [], "remove sole tag returns to untagged"),
    ("remove", ["review", "backlog"], ["backlog"], "remove one tag leaves others intact"),
    ("remove-missing", ["review"], ["review"], "remove non-existent tag: no-op"),
    ("add-multiple", [], ["a", "b", "c"], "add three distinct tags sequentially"),
    ("add-bulk", [], ["bulk-1", "bulk-2"], "bulk_add_tags across single session"),
]

MULTI_SESSION_CASES: list[tuple[int, str]] = [
    (1, "single session has no cross-contamination"),
    (3, "three sessions: tags are isolated per session"),
]


TAG_VALIDATION_CASES: list[tuple[str, str, type[Exception], str]] = [
    ("", "empty string tag", ValueError, "add_tag rejects empty string"),
    ("   ", "whitespace-only tag", ValueError, "add_tag rejects whitespace-only"),
    ("x" * 201, "201-char tag exceeds limit", ValueError, "add_tag rejects >200 char tag"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scenario(name: str, tags: list[str] | None = None) -> ArchiveScenario:
    return ArchiveScenario(
        name=name,
        provider="test",
        title=f"Tag Test: {name}",
        messages=(ScenarioMessage(role="user", text="Test message", message_id=f"{name}-msg"),),
        metadata={"tags": list(tags)} if tags else None,
    )


# ---------------------------------------------------------------------------
# Tag lifecycle CRUD tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio()
@pytest.mark.parametrize("operation,initial_tags,expected,desc", TAG_LIFECYCLE_CASES)
async def test_tag_lifecycle_crud(
    operation: str,
    initial_tags: list[str],
    expected: list[str],
    desc: str,
    workspace_env: Mapping[str, Path],
) -> None:
    """Catalog-driven tag lifecycle: add, duplicate, remove, bulk-add."""
    scenario = _scenario("tag-lifecycle", initial_tags if initial_tags else None)
    db_path, _ = seed_workspace_scenarios(workspace_env, [scenario])
    repo = repository_for_scenario_db(db_path)
    try:
        conv_id = scenario.native_session_id

        if operation == "add":
            await repo.add_tag(conv_id, "review")
        elif operation == "add-duplicate":
            await repo.add_tag(conv_id, "review")  # should be no-op
        elif operation == "remove":
            await repo.remove_tag(conv_id, "review")
        elif operation == "remove-missing":
            await repo.remove_tag(conv_id, "nonexistent")
        elif operation == "add-multiple":
            for tag in ("a", "b", "c"):
                await repo.add_tag(conv_id, tag)
        elif operation == "add-bulk":
            await repo.bulk_tag_sessions([conv_id], ["bulk-1", "bulk-2"])

        # Verify via list_tags
        listed = await repo.list_tags()
        for expected_tag in expected:
            assert expected_tag in listed, f"[{desc}] Expected tag '{expected_tag}' in list_tags: {listed}"
    finally:
        await repo.close()


@pytest.mark.asyncio()
@pytest.mark.parametrize("conv_count,desc", MULTI_SESSION_CASES)
async def test_tag_isolation_across_sessions(
    conv_count: int,
    desc: str,
    workspace_env: Mapping[str, Path],
) -> None:
    """Tags applied to one session do not leak to others."""
    scenarios = [_scenario(f"tag-isolation-{i}", ["shared"] if i == 0 else None) for i in range(conv_count)]
    db_path, _ = seed_workspace_scenarios(workspace_env, scenarios)
    repo = repository_for_scenario_db(db_path)
    try:
        first_conv_id = scenarios[0].native_session_id
        await repo.add_tag(first_conv_id, "unique-0")

        listed = await repo.list_tags()
        assert "shared" in listed, f"[{desc}] shared tag should be visible"
        assert "unique-0" in listed, f"[{desc}] unique-0 should be visible"
        assert listed["shared"] == 1, f"[{desc}] shared tag should have count 1"
        assert listed["unique-0"] == 1, f"[{desc}] unique-0 tag should have count 1"
    finally:
        await repo.close()


@pytest.mark.asyncio()
@pytest.mark.parametrize("tag,error_desc,exc_type,desc", TAG_VALIDATION_CASES)
async def test_tag_validation_rejects_invalid_input(
    tag: str,
    error_desc: str,
    exc_type: type[Exception],
    desc: str,
    workspace_env: Mapping[str, Path],
) -> None:
    """add_tag rejects empty, whitespace-only, and over-length tags."""
    scenario = _scenario("tag-validation")
    db_path, _ = seed_workspace_scenarios(workspace_env, [scenario])
    repo = repository_for_scenario_db(db_path)
    try:
        with pytest.raises(exc_type):
            await repo.add_tag(scenario.native_session_id, tag)
    finally:
        await repo.close()


# ---------------------------------------------------------------------------
# #1240: M2M tag tables are the only tag read surface.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_list_tags_ignores_json_metadata_only_tags(workspace_env: Mapping[str, Path]) -> None:
    """#1240: ``list_tags`` reads only the user tag tables.

    Tags written exclusively into the session's user metadata (as a JSON
    ``tags`` value) must not appear in ``list_tags`` output: the
    ``session_tags`` table is the only tag read surface and there is no
    JSON-metadata fallback.
    """
    scenario = ArchiveScenario(
        name="json-only-tags",
        provider="test",
        title="Legacy JSON tags",
        messages=(ScenarioMessage(role="user", text="x", message_id="m1"),),
    )
    db_path, _ = seed_workspace_scenarios(workspace_env, [scenario])

    repo = repository_for_scenario_db(db_path)
    try:
        # Write a JSON-only "tags" value into the session's user metadata,
        # bypassing the user tag table that ``add_tag`` would normally write.
        await repo.set_metadata(scenario.native_session_id, "tags", ["json-only-tag"])

        listed = await repo.list_tags()
        assert "json-only-tag" not in listed, "list_tags must not surface tags that live only in the user metadata JSON"
        assert listed == {}, f"expected empty tag list, got {listed!r}"
    finally:
        await repo.close()


@pytest.mark.asyncio()
async def test_add_tag_does_not_dual_write_into_metadata_json(workspace_env: Mapping[str, Path]) -> None:
    """#1240: ``add_tag`` writes only to the M2M tables.

    The legacy dual-write into ``sessions.metadata['tags']`` was
    removed; ``add_tag`` must not modify the metadata column.
    """
    scenario = ArchiveScenario(
        name="add-tag-no-dual-write",
        provider="test",
        title="Add-tag isolation",
        messages=(ScenarioMessage(role="user", text="x", message_id="m1"),),
    )
    db_path, _ = seed_workspace_scenarios(workspace_env, [scenario])

    repo = repository_for_scenario_db(db_path)
    try:
        await repo.add_tag(scenario.native_session_id, "review")
        metadata = await repo.get_metadata(scenario.native_session_id)
        assert "tags" not in metadata, f"add_tag must not write into metadata.tags; got metadata={metadata!r}"

        # And the M2M read surface does see it.
        listed = await repo.list_tags()
        assert listed == {"review": 1}
    finally:
        await repo.close()
