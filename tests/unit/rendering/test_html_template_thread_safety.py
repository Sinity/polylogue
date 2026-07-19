"""Thread-safety regression tests for get_cached_template() (polylogue-xikl.2).

get_cached_template()'s module-level Jinja2 Environment singleton used to be
built via an unguarded check-then-set. The daemon's real archive_query_executor
already dispatches concurrent request handlers (including HTML rendering) onto
real OS threads today under the standard GIL build, so this race is live now,
not merely a free-threading concern.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator

import pytest

import polylogue.rendering.renderers.html_template as html_template_module
from polylogue.rendering.renderers.html_template import get_cached_template


@pytest.fixture(autouse=True)
def _reset_cached_template_env() -> Iterator[None]:
    original = html_template_module._CACHED_TEMPLATE_ENV
    html_template_module._CACHED_TEMPLATE_ENV = None
    yield
    html_template_module._CACHED_TEMPLATE_ENV = original


def test_get_cached_template_singleton_is_race_safe_under_concurrent_first_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent first access must build exactly one Environment instance.

    A short delay is injected into ``_build_template_environment`` to force
    the interleaving window open regardless of scheduling luck: before the
    fix this reliably produced multiple distinct Environment instances
    (duplicate construction); with the lock guarding the check-then-build
    section, exactly one thread ever builds the Environment.
    """
    original_build = html_template_module._build_template_environment
    built_envs: list[object] = []
    built_envs_lock = threading.Lock()

    def delayed_build() -> object:
        time.sleep(0.02)
        env = original_build()
        with built_envs_lock:
            built_envs.append(env)
        return env

    monkeypatch.setattr(html_template_module, "_build_template_environment", delayed_build)

    templates: list[object] = []
    templates_lock = threading.Lock()

    def worker() -> None:
        template = get_cached_template()
        with templates_lock:
            templates.append(template)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert len(templates) == 8
    assert len(built_envs) == 1, f"concurrent first access built {len(built_envs)} Environments, expected exactly 1"
