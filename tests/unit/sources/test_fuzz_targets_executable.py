"""Behavior checks that every committed Atheris fuzz target stays executable.

This test does **not** invoke libFuzzer or run the targets. It only confirms
that each module under ``tests/fuzz/`` exposes the documented target
functions and a ``main()`` entrypoint so that:

* renaming or deleting a target is caught immediately by the unit suite
  rather than being silently dropped from the campaign surface
* every documented fuzz module remains a runnable script
  (``python tests/fuzz/fuzz_<name>.py``)
* the ``main()`` entrypoint actually invokes Atheris with the module's declared
  target, so a green test proves executable wiring rather than source spelling

See ``tests/fuzz/README.md`` for invocation and seed-corpus policy.
"""

from __future__ import annotations

import importlib
import inspect
from types import SimpleNamespace

import pytest

_FUZZ_MODULES: dict[str, tuple[str, ...]] = {
    "tests.fuzz.fuzz_fts5_escape": ("fuzz_fts5_escape",),
    "tests.fuzz.fuzz_json_parsers": (
        "fuzz_chatgpt_parser",
        "fuzz_codex_parser",
        "fuzz_claude_code_parser",
        "fuzz_claude_ai_parser",
        "fuzz_drive_parser",
        "fuzz_antigravity_parser",
        "fuzz_browser_capture_parser",
        "fuzz_local_agent_parser",
        "fuzz_all_parsers",
    ),
    "tests.fuzz.fuzz_path_sanitizer": (
        "fuzz_path_sanitizer",
        "fuzz_name_sanitizer",
    ),
    "tests.fuzz.fuzz_timestamp": (
        "fuzz_parse_timestamp",
        "fuzz_normalize_timestamp",
        "fuzz_format_timestamp",
        "fuzz_all_timestamps",
    ),
}

_FUZZ_ENTRY_TARGETS = {
    "tests.fuzz.fuzz_fts5_escape": "fuzz_fts5_escape",
    "tests.fuzz.fuzz_json_parsers": "fuzz_all_parsers",
    "tests.fuzz.fuzz_path_sanitizer": "fuzz_path_sanitizer",
    "tests.fuzz.fuzz_timestamp": "fuzz_all_timestamps",
}


@pytest.mark.parametrize("module_name,targets", list(_FUZZ_MODULES.items()))
def test_fuzz_module_exposes_targets(module_name: str, targets: tuple[str, ...]) -> None:
    module = importlib.import_module(module_name)

    for target in targets:
        fn = getattr(module, target, None)
        assert callable(fn), f"{module_name}.{target} is not callable"
        params = inspect.signature(fn).parameters
        assert len(params) == 1, (
            f"{module_name}.{target} should take exactly one bytes argument; got signature {inspect.signature(fn)}"
        )


@pytest.mark.parametrize("module_name", list(_FUZZ_MODULES.keys()))
def test_fuzz_module_runs_libfuzzer_entrypoint(module_name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.import_module(module_name)
    assert callable(getattr(module, "main", None)), (
        f"{module_name} is missing the libFuzzer main() entrypoint documented in tests/fuzz/README.md"
    )

    calls: list[tuple[str, tuple[list[str], object] | None]] = []

    def setup(args: list[str], target: object) -> None:
        calls.append(("setup", (args, target)))

    def fuzz() -> None:
        calls.append(("fuzz", None))

    monkeypatch.setattr(module, "HAS_ATHERIS", True)
    monkeypatch.setattr(
        module,
        "atheris",
        SimpleNamespace(Setup=setup, Fuzz=fuzz),
        raising=False,
    )
    monkeypatch.setenv("FUZZ_ITERATIONS", "7")

    module.main()

    assert [name for name, _payload in calls] == ["setup", "fuzz"]
    setup_payload = calls[0][1]
    assert setup_payload is not None
    setup_args, setup_target = setup_payload
    assert "-runs=7" in setup_args
    assert setup_target is getattr(module, _FUZZ_ENTRY_TARGETS[module_name])
