from __future__ import annotations

import argparse
import json

from polylogue.cli.app import run_env_cli, run_help_cli, run_completions_cli, run_complete_cli
from polylogue.cli import CommandEnv


class DummyConsole:
    def __init__(self):
        self.lines: list[str] = []

    def print(self, *args, **kwargs):  # noqa: ANN001, ARG002
        text = " ".join(str(arg) for arg in args)
        self.lines.append(text)


class DummyUI:
    plain = True

    def __init__(self):
        self.console = DummyConsole()


def test_help_topic_outputs_details(capsys):
    env = CommandEnv(ui=DummyUI())
    run_help_cli(argparse.Namespace(topic="sync"), env)
    out = capsys.readouterr().out
    assert "Synchronize provider archives" in out


def test_help_unknown_command_reports_error():
    ui = DummyUI()
    env = CommandEnv(ui=ui)
    run_help_cli(argparse.Namespace(topic="nope"), env)
    assert any("Unknown command" in line for line in ui.console.lines)


def test_help_lists_command_descriptions():
    ui = DummyUI()
    env = CommandEnv(ui=ui)
    run_help_cli(argparse.Namespace(), env)
    joined = "\n".join(ui.console.lines)
    assert "sync" in joined
    assert "Synchronize provider archives" in joined


def test_env_json(capsys):
    env = CommandEnv(ui=DummyUI())
    run_env_cli(argparse.Namespace(json=True), env)
    parsed = json.loads(capsys.readouterr().out)
    assert "outputDirs" in parsed
    assert "statePath" in parsed


def test_completions_emits_script(capsys):
    env = CommandEnv(ui=DummyUI())
    run_completions_cli(argparse.Namespace(shell="bash"), env)
    script = capsys.readouterr().out
    assert "polylogue" in script
    assert "complete -F" in script


def test_fish_completions_include_descriptions(capsys):
    env = CommandEnv(ui=DummyUI())
    run_completions_cli(argparse.Namespace(shell="fish"), env)
    script = capsys.readouterr().out
    assert '-d "Synchronize provider archives"' in script


def test_complete_top_level(capsys):
    env = CommandEnv(ui=DummyUI())
    run_complete_cli(argparse.Namespace(shell="zsh", cword=1, words=["polylogue", ""]))
    lines = capsys.readouterr().out.strip().splitlines()
    assert any(line.startswith("render") for line in lines)


def test_complete_sync_provider(capsys):
    env = CommandEnv(ui=DummyUI())
    run_complete_cli(argparse.Namespace(shell="zsh", cword=2, words=["polylogue", "sync", ""]))
    output = capsys.readouterr().out
    assert "drive" in output
