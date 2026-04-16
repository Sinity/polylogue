from __future__ import annotations

from polylogue.pipeline.services.process_pool import process_pool_context, process_pool_executor


def _worker_wrapper_class_name() -> str:
    import structlog

    return structlog.get_config()["wrapper_class"].__name__


def test_process_pool_context_avoids_fork() -> None:
    assert process_pool_context().get_start_method() != "fork"


def test_process_pool_workers_initialize_info_logging() -> None:
    with process_pool_executor(max_workers=1) as executor:
        wrapper_name = executor.submit(_worker_wrapper_class_name).result(timeout=10)

    assert wrapper_name == "BoundLoggerFilteringAtInfo"
