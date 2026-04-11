from __future__ import annotations

from polylogue.pipeline.services.process_pool import process_pool_context


def test_process_pool_context_avoids_fork() -> None:
    assert process_pool_context().get_start_method() != "fork"
