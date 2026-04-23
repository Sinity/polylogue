from __future__ import annotations

import runpy
from unittest.mock import patch


def test_module_entrypoints_delegate_to_click_main() -> None:
    with patch("polylogue.cli.click_app.main") as click_main:
        runpy.run_module("polylogue.__main__", run_name="__main__")
        runpy.run_module("polylogue.cli.__main__", run_name="__main__")

    assert click_main.call_count == 2
