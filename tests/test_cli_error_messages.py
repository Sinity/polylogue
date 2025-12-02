from argparse import Namespace

import pytest

from polylogue.cli.app import _dispatch_inspect, _dispatch_config
from polylogue.commands import CommandEnv
from polylogue.ui import create_ui


class DummyEnv(CommandEnv):
    def __init__(self):
        super().__init__(ui=create_ui(plain=True))


def test_inspect_requires_subcommand(capsys):
    env = DummyEnv()
    with pytest.raises(SystemExit):
        _dispatch_inspect(Namespace(inspect_cmd=None), env)
    captured = capsys.readouterr()
    assert "inspect requires a sub-command" in captured.out or captured.err


def test_config_requires_subcommand(capsys):
    env = DummyEnv()
    with pytest.raises(SystemExit):
        _dispatch_config(Namespace(config_cmd=None), env)
    captured = capsys.readouterr()
    assert "config requires a sub-command" in captured.out or captured.err
