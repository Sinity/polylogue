import json
from pathlib import Path

from polylogue.config import DriveConfig, IndexConfig, load_config, persist_config


def test_persist_config_includes_roots(tmp_path: Path) -> None:
    input_root = tmp_path / "inbox"
    output_root = tmp_path / "archive"
    roots = {"work": tmp_path / "work-archive"}

    config_path = persist_config(
        input_root=input_root,
        output_root=output_root,
        collapse_threshold=10,
        html_previews=True,
        html_theme="dark",
        index=IndexConfig(),
        path=tmp_path / "config.json",
        roots=roots,
    )

    data = json.loads(config_path.read_text(encoding="utf-8"))
    assert data["paths"]["roots"]["work"]["render"].endswith("/work-archive/render")


def test_load_config_populates_labeled_roots(tmp_path: Path, monkeypatch) -> None:
    work_root = tmp_path / "work-archive"
    payload = {
        "paths": {
            "input_root": str(tmp_path / "inbox"),
            "output_root": str(tmp_path / "archive"),
            "roots": {
                "work": {
                    "render": str(work_root / "render"),
                    "sync_drive": str(work_root / "gemini"),
                    "sync_codex": str(work_root / "codex"),
                    "sync_claude_code": str(work_root / "claude-code"),
                    "import_chatgpt": str(work_root / "chatgpt"),
                    "import_claude": str(work_root / "claude"),
                }
            },
        },
        "defaults": {"collapse_threshold": 10, "html_previews": True, "html_theme": "light"},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("POLYLOGUE_CONFIG", str(config_path))

    cfg = load_config()
    assert "work" in cfg.defaults.roots
    assert cfg.defaults.roots["work"].render == work_root / "render"
    assert cfg.defaults.roots["work"].import_claude == work_root / "claude"


def test_persist_config_includes_drive_settings(tmp_path: Path) -> None:
    input_root = tmp_path / "inbox"
    output_root = tmp_path / "archive"
    drive_cfg = DriveConfig(
        credentials_path=tmp_path / "creds.json",
        token_path=tmp_path / "token.json",
        retries=5,
        retry_base=1.25,
        from_config=True,
    )

    config_path = persist_config(
        input_root=input_root,
        output_root=output_root,
        collapse_threshold=10,
        html_previews=True,
        html_theme="dark",
        index=IndexConfig(),
        path=tmp_path / "config.json",
        drive=drive_cfg,
    )

    data = json.loads(config_path.read_text(encoding="utf-8"))
    assert data["drive"]["credentials_path"] == str(drive_cfg.credentials_path)
    assert data["drive"]["token_path"] == str(drive_cfg.token_path)
    assert data["drive"]["retries"] == 5
    assert data["drive"]["retry_base"] == 1.25


def test_load_config_populates_drive_settings(tmp_path: Path, monkeypatch) -> None:
    payload = {
        "paths": {
            "input_root": str(tmp_path / "inbox"),
            "output_root": str(tmp_path / "archive"),
        },
        "drive": {
            "credentials_path": str(tmp_path / "creds.json"),
            "token_path": str(tmp_path / "token.json"),
            "retries": 4,
            "retry_base": 0.75,
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("POLYLOGUE_CONFIG", str(config_path))

    cfg = load_config()
    assert cfg.drive.credentials_path == tmp_path / "creds.json"
    assert cfg.drive.token_path == tmp_path / "token.json"
    assert cfg.drive.retries == 4
    assert cfg.drive.retry_base == 0.75
    assert cfg.drive.from_config is True
