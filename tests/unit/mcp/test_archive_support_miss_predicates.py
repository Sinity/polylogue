"""MCP zero-hit search payload merges the shared predicate-attribution probes (polylogue-jnj.12).

``archive_search_payload``/``_search_term_diagnostics`` are otherwise
synchronous (direct ``ArchiveStore`` calls); this proves the bridge into the
async :mod:`polylogue.archive.query.miss_predicates` substrate actually
surfaces clause-drop reasons on a real archive, not just that the wiring
compiles.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.api import Polylogue
from polylogue.archive.query.spec import SessionQuerySpec
from polylogue.mcp.archive_support import archive_search_payload
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.storage_records import SessionBuilder, db_setup


@pytest.mark.asyncio
async def test_archive_search_payload_merges_predicate_probe_reasons(workspace_env: dict[str, Path]) -> None:
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

    archive_root = db_path.parent
    async with Polylogue(archive_root=archive_root, db_path=db_path) as facade:
        spec = SessionQuerySpec(origins=("claude-code-session",), repo_names=("repo-b",))
        archive = ArchiveStore.open_existing(archive_root)
        try:
            envelope = archive_search_payload(
                archive,
                spec,
                query="",
                limit=10,
                offset=0,
                retrieval_lane="dialogue",
                sort=None,
                config=facade.config,
                archive_root=archive_root,
            )
        finally:
            archive.close()

    assert envelope.total == 0
    assert envelope.diagnostics is not None
    codes = [reason.code for reason in envelope.diagnostics.reasons]
    assert codes.count("predicate_zeroed_set") == 2


@pytest.mark.asyncio
async def test_archive_search_payload_without_config_omits_predicate_probe(workspace_env: dict[str, Path]) -> None:
    """No config -> no probes, not a crash (mirrors every other config-gated check)."""
    db_path = db_setup(workspace_env)
    await SessionBuilder(db_path, "conv-1").provider("claude-code").add_message("m1", text="hi").build()

    archive_root = db_path.parent
    spec = SessionQuerySpec(origins=("chatgpt-export",), repo_names=("repo-b",))
    archive = ArchiveStore.open_existing(archive_root)
    try:
        envelope = archive_search_payload(
            archive,
            spec,
            query="",
            limit=10,
            offset=0,
            retrieval_lane="dialogue",
            sort=None,
            config=None,
            archive_root=archive_root,
        )
    finally:
        archive.close()

    assert envelope.total == 0
    assert envelope.diagnostics is None
