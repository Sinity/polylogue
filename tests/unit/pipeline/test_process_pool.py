from __future__ import annotations

from typing import cast

from polylogue.pipeline.services.process_pool import process_pool_context, process_pool_executor


def _worker_wrapper_class_name() -> str:
    import structlog

    wrapper_class = structlog.get_config()["wrapper_class"]
    return cast(str, getattr(wrapper_class, "__name__", str(wrapper_class)))


def test_process_pool_context_avoids_fork() -> None:
    assert process_pool_context().get_start_method() != "fork"


def test_process_pool_workers_initialize_info_logging() -> None:
    with process_pool_executor(max_workers=1) as executor:
        wrapper_name = executor.submit(_worker_wrapper_class_name).result(timeout=10)

    assert wrapper_name == "BoundLoggerFilteringAtInfo"
