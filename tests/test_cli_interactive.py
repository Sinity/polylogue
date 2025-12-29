from __future__ import annotations

import json
from pathlib import Path

from tests.test_cli_plain_smoke import _run_polylogue, _common_env


def _write_prompt_file(path: Path, entries: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(entry) for entry in entries), encoding="utf-8")


def test_interactive_config_init_uses_prompt_file(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    env = _common_env(tmp_path, force_plain=False)

    responses = [
        {"type": "input", "value": ""},  # output dir default
        {"type": "input", "value": ""},  # inbox default
        {"type": "confirm", "value": True},  # enable HTML
        {"type": "choose", "value": "dark"},
        {"type": "input", "value": "25"},
        {"type": "choose", "value": "sqlite"},
        {"type": "confirm", "value": False},  # add labeled root?
        {"type": "confirm", "value": False},  # Drive setup
    ]
    prompt_file = tmp_path / "prompts.jsonl"
    _write_prompt_file(prompt_file, responses)
    env["POLYLOGUE_TEST_PROMPT_FILE"] = str(prompt_file)

    result = _run_polylogue(["--interactive", "config", "init", "--force"], cwd=repo_root, env=env)
    assert result.returncode == 0, result.stderr

    config_home = Path(env["XDG_CONFIG_HOME"]) / "polylogue"
    assert (config_home / "settings.json").exists()
    assert (config_home / "config.json").exists()


def test_interactive_browse_branches_uses_prompt_file(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]

    # Seed archive via plain sync.
    env_plain = _common_env(tmp_path, force_plain=True)
    codex_base = tmp_path / "codex"
    codex_base.mkdir(parents=True, exist_ok=True)
    fixture = repo_root / "tests" / "fixtures" / "golden" / "codex" / "codex-golden.jsonl"
    (codex_base / "session.jsonl").write_text(fixture.read_text(encoding="utf-8"), encoding="utf-8")
    seed = _run_polylogue(
        ["sync", "codex", "--base-dir", str(codex_base), "--all"],
        cwd=repo_root,
        env=env_plain,
    )
    assert seed.returncode == 0, seed.stderr

    env = _common_env(tmp_path, force_plain=False)
    responses = [
        {"type": "choose", "index": 0},
        {"type": "choose", "value": "Done"},
    ]
    prompt_file = tmp_path / "prompts-browse.jsonl"
    _write_prompt_file(prompt_file, responses)
    env["POLYLOGUE_TEST_PROMPT_FILE"] = str(prompt_file)

    result = _run_polylogue(["--interactive", "browse", "branches"], cwd=repo_root, env=env)
    assert result.returncode == 0, result.stderr
    assert "Provider: codex" in result.stdout


def test_interactive_config_edit_updates_output_root(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    env_plain = _common_env(tmp_path, force_plain=True)

    init = _run_polylogue(["config", "init", "--force"], cwd=repo_root, env=env_plain)
    assert init.returncode == 0, init.stderr

    new_root = tmp_path / "custom-archive"
    responses = [
        {"type": "choose", "value": "Output root"},
        {"type": "input", "value": str(new_root)},
        {"type": "choose", "value": "Quit"},
    ]
    prompt_file = tmp_path / "prompts-edit.jsonl"
    _write_prompt_file(prompt_file, responses)

    env = _common_env(tmp_path, force_plain=False)
    env["POLYLOGUE_TEST_PROMPT_FILE"] = str(prompt_file)

    result = _run_polylogue(["--interactive", "config", "edit"], cwd=repo_root, env=env)
    assert result.returncode == 0, result.stderr

    config_home = Path(env["XDG_CONFIG_HOME"]) / "polylogue"
    config_payload = json.loads((config_home / "config.json").read_text(encoding="utf-8"))
    assert config_payload["paths"]["output_root"] == str(new_root)
