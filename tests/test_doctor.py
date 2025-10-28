import json
from pathlib import Path

from polylogue.doctor import run_doctor
from polylogue import doctor as doctor_module
from polylogue import db as db_module
from polylogue.config import Defaults, OutputDirs


def _fake_defaults(tmp_path):
    base = tmp_path / "outputs"
    render = base / "render"
    drive = base / "drive"
    codex = base / "codex"
    claude_code = base / "claude-code"
    chatgpt = base / "chatgpt"
    claude = base / "claude"
    for directory in (render, drive, codex, claude_code, chatgpt, claude):
        directory.mkdir(parents=True, exist_ok=True)
    return Defaults(
        collapse_threshold=30,
        html_previews=False,
        html_theme="light",
        output_dirs=OutputDirs(
            render=render,
            sync_drive=drive,
            sync_codex=codex,
            sync_claude_code=claude_code,
            import_chatgpt=chatgpt,
            import_claude=claude,
        ),
    )


def test_doctor_detects_invalid_codex(tmp_path):
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    bad_session = codex_dir / "bad.jsonl"
    bad_session.write_text("{not json}\n", encoding="utf-8")

    report = run_doctor(codex_dir=codex_dir, claude_code_dir=tmp_path / "claude", limit=None)
    assert any(issue.provider == "codex" for issue in report.issues)
    assert report.checked.get("codex") == 1


def test_doctor_prunes_state_and_db(tmp_path, monkeypatch):
    defaults = _fake_defaults(tmp_path)
    monkeypatch.setattr(doctor_module, "CONFIG", type("Cfg", (), {"defaults": defaults}))

    state_home = tmp_path / "state"
    state_home.mkdir()
    state_path = state_home / "state.json"
    missing_conv_dir = defaults.render / "orphan"
    attachments_dir = missing_conv_dir / "attachments"
    attachments_dir.mkdir(parents=True, exist_ok=True)
    (attachments_dir / "sample.txt").write_text("orphan", encoding="utf-8")
    state_data = {
        "conversations": {
            "render": {
                "conv-orphan": {
                    "outputPath": str(missing_conv_dir / "conversation.md"),
                    "attachmentsDir": str(attachments_dir),
                }
            }
        }
    }
    state_path.write_text(json.dumps(state_data), encoding="utf-8")

    monkeypatch.setattr(doctor_module, "STATE_PATH", state_path)
    monkeypatch.setattr(doctor_module, "STATE_HOME", state_home)
    monkeypatch.setattr(db_module, "STATE_HOME", state_home)
    db_module.DB_PATH = state_home / "polylogue.db"

    # Seed database with a stale conversation entry
    with db_module.open_connection() as conn:
        conn.execute(
            "INSERT INTO conversations (provider, conversation_id, slug) VALUES (?, ?, ?)",
            ("render", "conv-orphan", "orphan"),
        )
        conn.commit()

    codex_dir = tmp_path / "codex"
    claude_dir = tmp_path / "claude"
    codex_dir.mkdir()
    claude_dir.mkdir()

    report = run_doctor(codex_dir=codex_dir, claude_code_dir=claude_dir, limit=0)

    assert any(issue.provider == "state" and "Removed" in issue.message for issue in report.issues)
    assert any(issue.provider == "database" for issue in report.issues)
    updated_state = json.loads(state_path.read_text(encoding="utf-8"))
    assert not updated_state.get("conversations", {}).get("render")
    with db_module.open_connection() as conn:
        rows = conn.execute(
            "SELECT 1 FROM conversations WHERE provider='render' AND conversation_id='conv-orphan'"
        ).fetchall()
    assert not rows
    assert not attachments_dir.exists()


def test_doctor_skips_external_attachments(tmp_path, monkeypatch):
    defaults = _fake_defaults(tmp_path)
    monkeypatch.setattr(doctor_module, "CONFIG", type("Cfg", (), {"defaults": defaults}))

    state_home = tmp_path / "state"
    state_home.mkdir()
    state_path = state_home / "state.json"
    external_dir = tmp_path / "external"
    external_dir.mkdir()
    missing_conv_dir = defaults.render / "ghost"
    state_data = {
        "conversations": {
            "render": {
                "conv-external": {
                    "outputPath": str(missing_conv_dir / "conversation.md"),
                    "attachmentsDir": str(external_dir),
                }
            }
        }
    }
    state_path.write_text(json.dumps(state_data), encoding="utf-8")

    monkeypatch.setattr(doctor_module, "STATE_PATH", state_path)
    monkeypatch.setattr(doctor_module, "STATE_HOME", state_home)
    monkeypatch.setattr(db_module, "STATE_HOME", state_home)
    db_module.DB_PATH = state_home / "polylogue.db"

    codex_dir = tmp_path / "codex"
    claude_dir = tmp_path / "claude"
    codex_dir.mkdir()
    claude_dir.mkdir()

    report = run_doctor(codex_dir=codex_dir, claude_code_dir=claude_dir, limit=0)

    assert external_dir.exists()
    assert any(
        issue.provider == "state" and "Skipped removing attachment path outside managed directories" in issue.message
        for issue in report.issues
    )
