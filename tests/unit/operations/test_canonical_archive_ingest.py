from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from polylogue.operations.canonical_archive_ingest import _ingest_selected_paths


@pytest.mark.asyncio
async def test_one_shot_ingest_reoffers_unattempted_paths_until_64_files_settle() -> None:
    paths = [Path(f"/synthetic/codex-{index:02d}.jsonl") for index in range(64)]
    offered_by_pass: list[list[Path]] = []

    async def ingest_pass(offered: list[Path]) -> SimpleNamespace:
        offered_by_pass.append(offered)
        succeeded = offered[:14]
        excluded = offered[14:]
        return SimpleNamespace(
            failed_file_count=0,
            deferred_file_count=0,
            succeeded_file_count=len(succeeded),
            succeeded_paths=tuple(succeeded),
            excluded_file_count=len(excluded),
            excluded_paths={str(path): "unattempted_time_budget" for path in excluded},
        )

    receipts = await _ingest_selected_paths(paths, ingest_pass)

    assert [len(batch) for batch in offered_by_pass] == [64, 50, 36, 22, 8]
    assert len(receipts) == 5
    assert offered_by_pass == [paths, paths[14:], paths[28:], paths[42:], paths[56:]]
    assert {path for batch in offered_by_pass for path in batch} == set(paths)


@pytest.mark.asyncio
async def test_one_shot_ingest_refuses_non_retryable_exclusions() -> None:
    path = Path("/synthetic/unsupported.jsonl")

    async def ingest_pass(offered: list[Path]) -> SimpleNamespace:
        return SimpleNamespace(
            failed_file_count=0,
            deferred_file_count=0,
            succeeded_file_count=0,
            succeeded_paths=(),
            excluded_file_count=1,
            excluded_paths={str(offered[0]): "unsupported_source"},
        )

    with pytest.raises(RuntimeError, match="non-retryable reason"):
        await _ingest_selected_paths([path], ingest_pass)
