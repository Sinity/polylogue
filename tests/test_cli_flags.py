from __future__ import annotations

from argparse import Namespace

import pytest

from polylogue.cli import CommandEnv, build_parser, run_sync_cli, _resolve_html_settings


class DummyConsole:
    def print(self, *args, **kwargs):
        pass


class DummyUI:
    plain = True
    console = DummyConsole()


def test_html_flag_default_and_overrides_render():
    parser = build_parser()
    args = parser.parse_args(["render", "input.json"])
    assert args.html_mode == "auto"

    args_on = parser.parse_args(["render", "input.json", "--html"])
    assert args_on.html_mode == "on"

    args_off = parser.parse_args(["render", "input.json", "--html", "off"])
    assert args_off.html_mode == "off"


def test_html_flag_sync_variants():
    parser = build_parser()
    args_auto = parser.parse_args(["sync", "codex"])
    assert args_auto.html_mode == "auto"

    args_explicit = parser.parse_args(["sync", "codex", "--html", "on"])
    assert args_explicit.html_mode == "on"


def test_html_flag_import_variants():
    parser = build_parser()
    args = parser.parse_args(["import", "chatgpt", "export.zip", "--html", "off"])
    assert args.html_mode == "off"


def test_resolve_html_settings_modes():
    auto, explicit_auto = _resolve_html_settings(Namespace(html_mode="auto"))
    assert auto in (True, False)  # falls back to global setting
    assert explicit_auto is False

    enabled, explicit_on = _resolve_html_settings(Namespace(html_mode="on"))
    assert enabled is True
    assert explicit_on is True

    disabled, explicit_off = _resolve_html_settings(Namespace(html_mode="off"))
    assert disabled is False
    assert explicit_off is True


def test_run_sync_cli_invalid_provider_raises():
    with pytest.raises(SystemExit):
        run_sync_cli(Namespace(provider="unknown"), CommandEnv(ui=DummyUI()))


def test_run_sync_cli_dispatch(monkeypatch):
    calls = []

    def fake_drive(args, env):  # noqa: ARG001
        calls.append("drive")

    monkeypatch.setattr("polylogue.cli.sync._run_sync_drive", fake_drive)
    run_sync_cli(Namespace(provider="drive"), CommandEnv(ui=DummyUI()))
    assert calls == ["drive"]
