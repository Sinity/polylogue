import json
import os
import sys
import types
from pathlib import Path

import pytest

from polylogue import util
from polylogue import commands as cmd_module
from polylogue import db as db_module
from polylogue import index_sqlite as index_sqlite_module

class _SimpleEncoding:
    def __init__(self, name: str):
        self.name = name

    def encode(self, text: str):
        if not text:
            return []
        # Split on whitespace as a deterministic approximation.
        return text.split()


def _build_tiktoken_stub() -> types.ModuleType:
    module = types.ModuleType("tiktoken")

    def _get_encoding(name: str) -> _SimpleEncoding:
        return _SimpleEncoding(name)

    def _encoding_for_model(model: str) -> _SimpleEncoding:
        return _SimpleEncoding(model)

    module.get_encoding = _get_encoding  # type: ignore[attr-defined]
    module.encoding_for_model = _encoding_for_model  # type: ignore[attr-defined]
    core_module = types.ModuleType("tiktoken.core")
    core_module.Encoding = _SimpleEncoding  # type: ignore[attr-defined]
    module.core = core_module  # type: ignore[attr-defined]
    sys.modules.setdefault("tiktoken.core", core_module)
    return module


sys.modules.setdefault("tiktoken", _build_tiktoken_stub())

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture
def state_env(tmp_path, monkeypatch):
    state_home = tmp_path / "state" / "polylogue"
    state_home.mkdir(parents=True, exist_ok=True)
    state_path = state_home / "state.json"
    runs_path = state_home / "runs.json"

    monkeypatch.setenv("XDG_STATE_HOME", str(state_home.parent))
    monkeypatch.setattr(util, "STATE_HOME", state_home, raising=False)
    monkeypatch.setattr(util, "STATE_PATH", state_path, raising=False)
    monkeypatch.setattr(util, "RUNS_PATH", runs_path, raising=False)

    monkeypatch.setattr(cmd_module, "STATE_PATH", state_path, raising=False)
    monkeypatch.setattr(cmd_module, "RUNS_PATH", runs_path, raising=False)
    monkeypatch.setattr(db_module, "STATE_HOME", state_home, raising=False)
    db_module.DB_PATH = state_home / "polylogue.db"
    monkeypatch.setattr(index_sqlite_module, "STATE_HOME", state_home, raising=False)

    # Seed empty state file for convenience
    state_path.write_text(json.dumps({}, indent=2), encoding="utf-8")
    return state_home
