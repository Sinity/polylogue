import os
import stat
import sys
import tempfile
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from polylogue import commands as cmd_module
from polylogue import db as db_module
from polylogue import paths as paths_module
from polylogue import util


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
    sys.modules["tiktoken.core"] = core_module
    return module


_tiktoken_stub = _build_tiktoken_stub()
# Force deterministic tokenization in tests even when `tiktoken` is installed.
sys.modules["tiktoken"] = _tiktoken_stub


def _configure_state(monkeypatch, root: Path) -> Path:
    state_root = root / "state"
    polylogue_state = state_root / "polylogue"
    polylogue_state.mkdir(parents=True, exist_ok=True)
    db_path = polylogue_state / "polylogue.db"

    monkeypatch.setenv("XDG_STATE_HOME", str(state_root))
    monkeypatch.setattr(paths_module, "STATE_HOME", polylogue_state, raising=False)
    for module in (util, cmd_module, db_module):
        monkeypatch.setattr(module, "STATE_HOME", polylogue_state, raising=False)

    cache_root = root / "cache"
    cache_home = cache_root / "polylogue"
    cache_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_root))
    monkeypatch.setattr(paths_module, "CACHE_HOME", cache_home, raising=False)

    monkeypatch.setattr(db_module, "DB_PATH", db_path, raising=False)
    return polylogue_state


@pytest.fixture
def state_env(tmp_path, monkeypatch):
    return _configure_state(monkeypatch, tmp_path)


def _write_stub(name: str, content: str, directory: Path) -> None:
    path = directory / name
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def _install_tool_shims() -> None:
    tool_dir = Path(tempfile.mkdtemp(prefix="polylogue-tools-"))
    scripts = {
        "sk": """#!/usr/bin/env python3
import sys

def main():
    data = sys.stdin.read()
    sys.stdout.write(data)

if __name__ == "__main__":
    main()
""",
        "bat": """#!/usr/bin/env python3
import sys
from pathlib import Path

def main():
    args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]
    for arg in args:
        p = Path(arg)
        if p.exists():
            try:
                sys.stdout.write(p.read_text(encoding="utf-8"))
            except Exception:
                pass

if __name__ == "__main__":
    main()
""",
        "glow": """#!/usr/bin/env python3
import sys

if __name__ == "__main__":
    sys.stdout.write("")
""",
    }
    for name, script in scripts.items():
        _write_stub(name, script, tool_dir)
    os.environ["PATH"] = f"{tool_dir}{os.pathsep}{os.environ.get('PATH', '')}"


_install_tool_shims()


@pytest.fixture(autouse=True)
def _disable_browser_launch(monkeypatch):
    monkeypatch.delenv("POLYLOGUE_BROWSER", raising=False)
    monkeypatch.delenv("BROWSER", raising=False)
    monkeypatch.setattr("polylogue.cli.editor.webbrowser.open", lambda *_a, **_k: False)
