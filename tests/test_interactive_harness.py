from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import pexpect


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _make_interactive_stubs(tmp_path: Path) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)

    _write_executable(
        bin_dir / "gum",
        """#!/usr/bin/env python3
import os
import sys

def _log(argv):
    log_path = os.environ.get("POLYLOGUE_TEST_CMD_LOG")
    if not log_path:
        return
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write("gum " + " ".join(argv[1:]) + "\\n")

def main():
    _log(sys.argv)
    if len(sys.argv) < 2:
        sys.exit(1)
    sub = sys.argv[1]
    if sub == "format":
        sys.stdout.write(sys.stdin.read())
        return
    if sub == "confirm":
        # Heuristic: treat --default as "yes", otherwise "no".
        sys.exit(0 if "--default" in sys.argv else 1)
    if sub == "input":
        out = ""
        if "--value" in sys.argv:
            idx = sys.argv.index("--value")
            if idx + 1 < len(sys.argv):
                out = sys.argv[idx + 1]
        sys.stdout.write(out + "\\n")
        return
    sys.exit(0)

if __name__ == "__main__":
    main()
""",
    )

    _write_executable(
        bin_dir / "sk",
        """#!/usr/bin/env python3
import os
import sys

def _log(argv):
    log_path = os.environ.get("POLYLOGUE_TEST_CMD_LOG")
    if not log_path:
        return
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write("sk " + " ".join(argv[1:]) + "\\n")

def main():
    _log(sys.argv)
    if os.environ.get("POLYLOGUE_TEST_SK_CANCEL") in {"1", "true", "yes", "on"}:
        sys.exit(130)
    forced = os.environ.get("POLYLOGUE_TEST_SK_CHOICE")
    if forced is not None:
        sys.stdout.write(forced + "\\n")
        return
    options = [line for line in sys.stdin.read().splitlines() if line.strip()]
    sys.stdout.write((options[0] if options else "") + "\\n")

if __name__ == "__main__":
    main()
""",
    )

    passthrough = """#!/usr/bin/env python3
import os
import sys

log_path = os.environ.get("POLYLOGUE_TEST_CMD_LOG")
if log_path:
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(os.path.basename(sys.argv[0]) + " " + " ".join(sys.argv[1:]) + "\\n")
sys.stdout.write(sys.stdin.read())
"""
    for name in ("bat", "glow", "delta"):
        _write_executable(bin_dir / name, passthrough)

    return bin_dir


def _spawn_polylogue(args: list[str], *, cwd: Path, env: dict[str, str]):
    cmd = [sys.executable, str(cwd / "polylogue.py"), *args]
    child = pexpect.spawn(cmd[0], cmd[1:], cwd=str(cwd), env=env, encoding="utf-8", timeout=30)
    return child


def test_interactive_config_init_runs_under_pty(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    stub_bin = _make_interactive_stubs(tmp_path)
    log_path = tmp_path / "cmd.log"

    env = os.environ.copy()
    env["PATH"] = f"{stub_bin}{os.pathsep}{env.get('PATH', '')}"
    env["XDG_CONFIG_HOME"] = str(tmp_path / "config")
    env["XDG_DATA_HOME"] = str(tmp_path / "data")
    env["XDG_CACHE_HOME"] = str(tmp_path / "cache")
    env["XDG_STATE_HOME"] = str(tmp_path / "state")
    env["POLYLOGUE_TEST_CMD_LOG"] = str(log_path)

    child = _spawn_polylogue(["config", "init", "--force"], cwd=repo_root, env=env)
    child.expect("Next steps", timeout=30)
    child.expect(pexpect.EOF)
    child.close()
    assert child.exitstatus == 0

    config_home = Path(env["XDG_CONFIG_HOME"]) / "polylogue"
    assert (config_home / "settings.json").exists()
    assert (config_home / "config.json").exists()

    log = log_path.read_text(encoding="utf-8")
    assert "gum input" in log
    assert "gum confirm" in log
    assert "sk --prompt" in log


def test_interactive_config_init_cancelled_when_config_exists(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    stub_bin = _make_interactive_stubs(tmp_path)

    env = os.environ.copy()
    env["PATH"] = f"{stub_bin}{os.pathsep}{env.get('PATH', '')}"
    env["XDG_CONFIG_HOME"] = str(tmp_path / "config")
    env["XDG_DATA_HOME"] = str(tmp_path / "data")
    env["XDG_CACHE_HOME"] = str(tmp_path / "cache")
    env["XDG_STATE_HOME"] = str(tmp_path / "state")

    # Seed config via a forced init run.
    first = _spawn_polylogue(["config", "init", "--force"], cwd=repo_root, env=env)
    first.expect("Next steps", timeout=30)
    first.expect(pexpect.EOF)
    first.close()
    assert first.exitstatus == 0

    # Second run should prompt and then cancel (gum confirm default is "no").
    second = _spawn_polylogue(["config", "init"], cwd=repo_root, env=env)
    second.expect("Init cancelled", timeout=30)
    second.expect(pexpect.EOF)
    second.close()
    assert second.exitstatus == 0


def test_interactive_sync_codex_selects_first_session(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    stub_bin = _make_interactive_stubs(tmp_path)

    # Prepare a codex base dir with one session.
    codex_base = tmp_path / "codex"
    codex_base.mkdir(parents=True, exist_ok=True)
    fixture = repo_root / "tests" / "fixtures" / "golden" / "codex" / "codex-golden.jsonl"
    session_path = codex_base / "session.jsonl"
    session_path.write_text(fixture.read_text(encoding="utf-8"), encoding="utf-8")

    env = os.environ.copy()
    env["PATH"] = f"{stub_bin}{os.pathsep}{env.get('PATH', '')}"
    env["XDG_CONFIG_HOME"] = str(tmp_path / "config")
    env["XDG_DATA_HOME"] = str(tmp_path / "data")
    env["XDG_CACHE_HOME"] = str(tmp_path / "cache")
    env["XDG_STATE_HOME"] = str(tmp_path / "state")

    child = _spawn_polylogue(
        ["--interactive", "sync", "codex", "--base-dir", str(codex_base)],
        cwd=repo_root,
        env=env,
    )
    child.expect(pexpect.EOF)
    child.close()
    assert child.exitstatus == 0

    output_dir = Path(env["XDG_DATA_HOME"]) / "polylogue" / "archive" / "codex"
    assert any((output_dir).rglob("conversation.md"))


def test_interactive_sync_codex_cancelled_when_sk_aborts(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    stub_bin = _make_interactive_stubs(tmp_path)

    codex_base = tmp_path / "codex"
    codex_base.mkdir(parents=True, exist_ok=True)
    fixture = repo_root / "tests" / "fixtures" / "golden" / "codex" / "codex-golden.jsonl"
    session_path = codex_base / "session.jsonl"
    session_path.write_text(fixture.read_text(encoding="utf-8"), encoding="utf-8")

    env = os.environ.copy()
    env["PATH"] = f"{stub_bin}{os.pathsep}{env.get('PATH', '')}"
    env["XDG_CONFIG_HOME"] = str(tmp_path / "config")
    env["XDG_DATA_HOME"] = str(tmp_path / "data")
    env["XDG_CACHE_HOME"] = str(tmp_path / "cache")
    env["XDG_STATE_HOME"] = str(tmp_path / "state")
    env["POLYLOGUE_TEST_SK_CANCEL"] = "1"

    child = _spawn_polylogue(
        ["--interactive", "sync", "codex", "--base-dir", str(codex_base)],
        cwd=repo_root,
        env=env,
    )
    child.expect(pexpect.EOF)
    child.close()
    assert child.exitstatus == 0


def test_interactive_sync_chatgpt_selects_first_export(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    stub_bin = _make_interactive_stubs(tmp_path)

    export_root = tmp_path / "exports"
    export_dir = export_root / "chatgpt-export"
    export_dir.mkdir(parents=True, exist_ok=True)
    fixture = repo_root / "tests" / "fixtures" / "golden" / "chatgpt" / "conversations.json"
    (export_dir / "conversations.json").write_text(fixture.read_text(encoding="utf-8"), encoding="utf-8")

    env = os.environ.copy()
    env["PATH"] = f"{stub_bin}{os.pathsep}{env.get('PATH', '')}"
    env["XDG_CONFIG_HOME"] = str(tmp_path / "config")
    env["XDG_DATA_HOME"] = str(tmp_path / "data")
    env["XDG_CACHE_HOME"] = str(tmp_path / "cache")
    env["XDG_STATE_HOME"] = str(tmp_path / "state")
    (Path(env["XDG_DATA_HOME"]) / "polylogue" / "inbox").mkdir(parents=True, exist_ok=True)

    child = _spawn_polylogue(
        ["--interactive", "sync", "chatgpt", "--base-dir", str(export_root)],
        cwd=repo_root,
        env=env,
    )
    child.expect(pexpect.EOF)
    child.close()
    assert child.exitstatus == 0, f"exit={child.exitstatus} output={child.before!r}"

    output_dir = Path(env["XDG_DATA_HOME"]) / "polylogue" / "archive" / "chatgpt"
    assert any(output_dir.rglob("conversation.md"))


def test_interactive_sync_chatgpt_cancelled_when_sk_aborts(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    stub_bin = _make_interactive_stubs(tmp_path)

    export_root = tmp_path / "exports"
    export_dir = export_root / "chatgpt-export"
    export_dir.mkdir(parents=True, exist_ok=True)
    fixture = repo_root / "tests" / "fixtures" / "golden" / "chatgpt" / "conversations.json"
    (export_dir / "conversations.json").write_text(fixture.read_text(encoding="utf-8"), encoding="utf-8")

    env = os.environ.copy()
    env["PATH"] = f"{stub_bin}{os.pathsep}{env.get('PATH', '')}"
    env["XDG_CONFIG_HOME"] = str(tmp_path / "config")
    env["XDG_DATA_HOME"] = str(tmp_path / "data")
    env["XDG_CACHE_HOME"] = str(tmp_path / "cache")
    env["XDG_STATE_HOME"] = str(tmp_path / "state")
    (Path(env["XDG_DATA_HOME"]) / "polylogue" / "inbox").mkdir(parents=True, exist_ok=True)
    env["POLYLOGUE_TEST_SK_CANCEL"] = "1"

    child = _spawn_polylogue(
        ["--interactive", "sync", "chatgpt", "--base-dir", str(export_root)],
        cwd=repo_root,
        env=env,
    )
    child.expect(pexpect.EOF)
    child.close()
    assert child.exitstatus == 0, f"exit={child.exitstatus} output={child.before!r}"


def test_interactive_sync_claude_selects_first_export(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    stub_bin = _make_interactive_stubs(tmp_path)

    export_root = tmp_path / "exports"
    export_dir = export_root / "claude-export"
    export_dir.mkdir(parents=True, exist_ok=True)
    fixture = repo_root / "tests" / "fixtures" / "golden" / "claude" / "conversations.json"
    (export_dir / "conversations.json").write_text(fixture.read_text(encoding="utf-8"), encoding="utf-8")

    env = os.environ.copy()
    env["PATH"] = f"{stub_bin}{os.pathsep}{env.get('PATH', '')}"
    env["XDG_CONFIG_HOME"] = str(tmp_path / "config")
    env["XDG_DATA_HOME"] = str(tmp_path / "data")
    env["XDG_CACHE_HOME"] = str(tmp_path / "cache")
    env["XDG_STATE_HOME"] = str(tmp_path / "state")
    (Path(env["XDG_DATA_HOME"]) / "polylogue" / "inbox").mkdir(parents=True, exist_ok=True)

    child = _spawn_polylogue(
        ["--interactive", "sync", "claude", "--base-dir", str(export_root)],
        cwd=repo_root,
        env=env,
    )
    child.expect(pexpect.EOF)
    child.close()
    assert child.exitstatus == 0, f"exit={child.exitstatus} output={child.before!r}"

    output_dir = Path(env["XDG_DATA_HOME"]) / "polylogue" / "archive" / "claude"
    assert any(output_dir.rglob("conversation.md"))


def test_interactive_sync_claude_cancelled_when_sk_aborts(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    stub_bin = _make_interactive_stubs(tmp_path)

    export_root = tmp_path / "exports"
    export_dir = export_root / "claude-export"
    export_dir.mkdir(parents=True, exist_ok=True)
    fixture = repo_root / "tests" / "fixtures" / "golden" / "claude" / "conversations.json"
    (export_dir / "conversations.json").write_text(fixture.read_text(encoding="utf-8"), encoding="utf-8")

    env = os.environ.copy()
    env["PATH"] = f"{stub_bin}{os.pathsep}{env.get('PATH', '')}"
    env["XDG_CONFIG_HOME"] = str(tmp_path / "config")
    env["XDG_DATA_HOME"] = str(tmp_path / "data")
    env["XDG_CACHE_HOME"] = str(tmp_path / "cache")
    env["XDG_STATE_HOME"] = str(tmp_path / "state")
    (Path(env["XDG_DATA_HOME"]) / "polylogue" / "inbox").mkdir(parents=True, exist_ok=True)
    env["POLYLOGUE_TEST_SK_CANCEL"] = "1"

    child = _spawn_polylogue(
        ["--interactive", "sync", "claude", "--base-dir", str(export_root)],
        cwd=repo_root,
        env=env,
    )
    child.expect(pexpect.EOF)
    child.close()
    assert child.exitstatus == 0, f"exit={child.exitstatus} output={child.before!r}"
