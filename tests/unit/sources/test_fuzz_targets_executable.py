"""Structural check that every committed Atheris fuzz target stays importable.

This test does **not** invoke libFuzzer or run the targets. It only confirms
that each module under ``tests/fuzz/`` exposes the documented target
functions and a ``main()`` entrypoint so that:

* renaming or deleting a target is caught immediately by the unit suite
  rather than being silently dropped from the campaign surface
* every documented fuzz module remains a runnable script
  (``python tests/fuzz/fuzz_<name>.py``)
* the ``main()`` entrypoint references ``atheris.Setup`` and ``atheris.Fuzz``
  so the libFuzzer wiring cannot drift unnoticed

See ``tests/fuzz/README.md`` for invocation and seed-corpus policy.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import pytest

_FUZZ_MODULES: dict[str, tuple[str, ...]] = {
    "tests.fuzz.fuzz_fts5_escape": ("fuzz_fts5_escape",),
    "tests.fuzz.fuzz_json_parsers": (
        "fuzz_chatgpt_parser",
        "fuzz_codex_parser",
        "fuzz_claude_code_parser",
        "fuzz_claude_ai_parser",
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
def test_fuzz_module_has_libfuzzer_entrypoint(module_name: str) -> None:
    module = importlib.import_module(module_name)
    assert callable(getattr(module, "main", None)), (
        f"{module_name} is missing the libFuzzer main() entrypoint documented in tests/fuzz/README.md"
    )

    source = Path(module.__file__ or "").read_text(encoding="utf-8") if module.__file__ else ""
    assert "atheris.Setup" in source, (
        f"{module_name} does not call atheris.Setup; libFuzzer wiring drifted from README contract"
    )
    assert "atheris.Fuzz" in source, (
        f"{module_name} does not call atheris.Fuzz; libFuzzer wiring drifted from README contract"
    )


def test_fuzz_readme_lists_every_module() -> None:
    readme = Path("tests/fuzz/README.md").read_text(encoding="utf-8")
    for module_name, targets in _FUZZ_MODULES.items():
        short = module_name.rsplit(".", 1)[-1]
        assert short in readme, f"tests/fuzz/README.md missing entry for {short}"
        for target in targets:
            assert target in readme, f"tests/fuzz/README.md missing target {target}"
