from __future__ import annotations

from types import SimpleNamespace

from polylogue import util
from polylogue.cli.sync import _apply_resume_from
from polylogue.commands import CommandEnv


class DummyConsole:
    def print(self, *_args, **_kwargs):
        return None


class DummyUI:
    plain = True

    def __init__(self):
        self.console = DummyConsole()


def test_resume_from_drive_selects_failed_chat_ids(state_env):
    util.add_run(
        {
            "cmd": "sync drive",
            "provider": "drive",
            "failedChats": [{"id": "abc", "name": "A", "error": "nope"}, {"id": "def", "name": "B", "error": "nope"}],
        }
    )
    run_id = util.load_runs(limit=1)[0]["id"]
    env = CommandEnv(ui=DummyUI())
    args = SimpleNamespace(provider="drive", chat_ids=None, sessions=None, all=False, prune=True, resume_from=None)
    _apply_resume_from(args, env, run_id=run_id)
    assert args.chat_ids == ["abc", "def"]
    assert args.all is True
    assert args.prune is False
    assert args.resume_from == run_id


def test_resume_from_local_selects_failed_paths(state_env):
    util.add_run(
        {
            "cmd": "sync codex",
            "provider": "codex",
            "failedPaths": [{"path": "/tmp/demo.jsonl", "error": "bad json"}],
        }
    )
    run_id = util.load_runs(limit=1)[0]["id"]
    env = CommandEnv(ui=DummyUI())
    args = SimpleNamespace(provider="codex", chat_ids=None, sessions=None, all=False, prune=True, resume_from=None)
    _apply_resume_from(args, env, run_id=run_id)
    assert args.sessions == ["/tmp/demo.jsonl"]
    assert args.all is True
    assert args.prune is False
    assert args.resume_from == run_id

