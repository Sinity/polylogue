import json
from pathlib import Path

from polylogue.commands import CommandEnv, render_command
from polylogue.options import RenderOptions


class _SilentConsole:
    def print(self, *_args, **_kwargs):
        pass


class _SilentUI:
    plain = True
    console = _SilentConsole()

    def summary(self, *_args, **_kwargs):
        pass

    def banner(self, *_args, **_kwargs):
        pass


def _write_render_input(path: Path) -> None:
    payload = {
        "chunkedPrompt": {
            "chunks": [
                {"role": "user", "text": "Hello"},
                {"role": "model", "text": "Hi there!"},
            ]
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _run_render(tmp_path: Path, state_env) -> Path:
    _ = state_env  # ensure fixture applies environment paths
    src = tmp_path / "sample.json"
    _write_render_input(src)
    out_dir = tmp_path / "render"
    options = RenderOptions(
        inputs=[src],
        output_dir=out_dir,
        collapse_threshold=16,
        download_attachments=False,
        dry_run=False,
        force=False,
        html=False,
        html_theme="light",
        diff=False,
    )
    env = CommandEnv(ui=_SilentUI())
    render_command(options, env)
    return out_dir / "sample"


def test_render_branch_full_writes_branch_tree(tmp_path, state_env):
    conversation_dir = _run_render(tmp_path, state_env)
    branch_dir = conversation_dir / "branches" / "branch-000"
    assert branch_dir.exists()
    assert (branch_dir / "branch-000.md").exists()
