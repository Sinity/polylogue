from pathlib import Path

from polylogue.doctor import run_doctor
from polylogue import doctor as doctor_module
from polylogue import db as db_module
from polylogue.config import Defaults, OutputDirs
from polylogue.services.conversation_registrar import ConversationRegistrar
from polylogue.services.conversation_service import ConversationService
from polylogue.persistence.state import ConversationStateRepository
from polylogue.persistence.database import ConversationDatabase
from polylogue.archive import Archive
from tests.conftest import _configure_state


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

    state_home = _configure_state(monkeypatch, tmp_path)
    missing_conv_dir = defaults.render / "orphan"
    attachments_dir = missing_conv_dir / "attachments"
    attachments_dir.mkdir(parents=True, exist_ok=True)
    (attachments_dir / "sample.txt").write_text("orphan", encoding="utf-8")
    db_path = state_home / "polylogue.db"
    repo = ConversationStateRepository(database=ConversationDatabase(path=db_path))
    repo.upsert(
        "render",
        "conv-orphan",
        {
            "outputPath": str(missing_conv_dir / "conversation.md"),
            "attachmentsDir": str(attachments_dir),
        },
    )

    codex_dir = tmp_path / "codex"
    claude_dir = tmp_path / "claude"
    codex_dir.mkdir()
    claude_dir.mkdir()

    registrar = ConversationRegistrar(
        state_repo=repo,
        database=ConversationDatabase(path=db_path),
        archive=Archive(doctor_module.CONFIG),
    )
    service = ConversationService(registrar=registrar)

    report = run_doctor(codex_dir=codex_dir, claude_code_dir=claude_dir, limit=0, service=service)

    assert any(issue.provider == "state" and "Removed" in issue.message for issue in report.issues)
    assert repo.get("render", "conv-orphan") is None
    with db_module.open_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT 1 FROM conversations WHERE provider='render' AND conversation_id='conv-orphan'"
        ).fetchall()
    assert not rows
    assert not attachments_dir.exists()


def test_doctor_skips_external_attachments(tmp_path, monkeypatch):
    defaults = _fake_defaults(tmp_path)
    monkeypatch.setattr(doctor_module, "CONFIG", type("Cfg", (), {"defaults": defaults}))

    state_home = _configure_state(monkeypatch, tmp_path)
    db_path = state_home / "polylogue.db"
    external_dir = tmp_path / "external"
    external_dir.mkdir()
    missing_conv_dir = defaults.render / "ghost"
    repo = ConversationStateRepository(database=ConversationDatabase(path=db_path))
    repo.upsert(
        "render",
        "conv-external",
        {
            "outputPath": str(missing_conv_dir / "conversation.md"),
            "attachmentsDir": str(external_dir),
        },
    )

    codex_dir = tmp_path / "codex"
    claude_dir = tmp_path / "claude"
    codex_dir.mkdir()
    claude_dir.mkdir()

    registrar = ConversationRegistrar(
        state_repo=repo,
        database=ConversationDatabase(path=db_path),
        archive=Archive(doctor_module.CONFIG),
    )
    service = ConversationService(registrar=registrar)

    report = run_doctor(codex_dir=codex_dir, claude_code_dir=claude_dir, limit=0, service=service)

    assert external_dir.exists()
    assert any(
        issue.provider == "state" and "Skipped removing attachment path outside managed directories" in issue.message
        for issue in report.issues
    )


def test_doctor_reports_drive_failures(tmp_path, monkeypatch):
    runs = [
        {
            "cmd": "sync drive",
            "provider": "drive",
            "driveRequests": 5,
            "driveFailures": 2,
            "driveLastError": "timeout",
        }
    ]
    monkeypatch.setattr(doctor_module, "load_runs", lambda limit=None: runs)

    state_home = _configure_state(monkeypatch, tmp_path)
    db_path = state_home / "polylogue.db"

    codex_dir = tmp_path / "codex"
    claude_dir = tmp_path / "claude"
    codex_dir.mkdir()
    claude_dir.mkdir()

    registrar = ConversationRegistrar(
        state_repo=ConversationStateRepository(database=ConversationDatabase(path=db_path)),
        database=ConversationDatabase(path=db_path),
        archive=Archive(doctor_module.CONFIG),
    )
    service = ConversationService(registrar=registrar)

    report = run_doctor(codex_dir=codex_dir, claude_code_dir=claude_dir, limit=0, service=service)

    assert any(issue.provider == "drive" and "failures" in issue.message for issue in report.issues)


def test_doctor_skip_index_checks(tmp_path, monkeypatch):
    _configure_state(monkeypatch, tmp_path)
    codex_dir = tmp_path / "codex"
    claude_dir = tmp_path / "claude"
    codex_dir.mkdir()
    claude_dir.mkdir()

    def fail_sqlite(*_args, **_kwargs):
        raise AssertionError("verify_sqlite_indexes should be skipped")

    def fail_qdrant(*_args, **_kwargs):
        raise AssertionError("verify_qdrant_collection should be skipped")

    monkeypatch.setattr(doctor_module, "verify_sqlite_indexes", fail_sqlite)
    monkeypatch.setattr(doctor_module, "verify_qdrant_collection", fail_qdrant)

    report = run_doctor(
        codex_dir=codex_dir,
        claude_code_dir=claude_dir,
        limit=0,
        skip_index=True,
        skip_qdrant=True,
    )

    assert report.checked.get("codex") == 0
    assert report.checked.get("claude-code") == 0


def test_doctor_uses_config_drive_paths(tmp_path, monkeypatch):
    defaults = _fake_defaults(tmp_path)
    cred_path = tmp_path / "drive" / "credentials.json"
    token_path = tmp_path / "drive" / "token.json"
    cred_path.parent.mkdir(parents=True, exist_ok=True)
    cred_path.write_text("{}", encoding="utf-8")
    token_path.write_text("{}", encoding="utf-8")
    drive_cfg = type("DriveCfg", (), {"credentials_path": cred_path, "token_path": token_path})
    cfg = type("Cfg", (), {"defaults": defaults, "drive": drive_cfg})
    monkeypatch.setattr(doctor_module, "CONFIG", cfg)
    monkeypatch.delenv("POLYLOGUE_CREDENTIAL_PATH", raising=False)
    monkeypatch.delenv("POLYLOGUE_TOKEN_PATH", raising=False)

    state_home = _configure_state(monkeypatch, tmp_path)
    db_path = state_home / "polylogue.db"
    codex_dir = tmp_path / "codex"
    claude_dir = tmp_path / "claude"
    codex_dir.mkdir()
    claude_dir.mkdir()
    archive = Archive(cfg)
    registrar = ConversationRegistrar(
        state_repo=ConversationStateRepository(database=ConversationDatabase(path=db_path)),
        database=ConversationDatabase(path=db_path),
        archive=archive,
    )
    service = ConversationService(registrar=registrar)

    report = run_doctor(codex_dir=codex_dir, claude_code_dir=claude_dir, limit=0, service=service, archive=archive)

    assert report.credential_path == cred_path
    assert report.token_path == token_path
    assert report.credentials_present is True
    assert report.token_present is True
