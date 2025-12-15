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


def _write_render_input_branched(path: Path) -> None:
    payload = {
        "chunkedPrompt": {
            "chunks": [
                {"role": "user", "text": "Hello", "messageId": "m1", "timestamp": "2024-01-01T00:00:00Z"},
                {"role": "model", "text": "Hi", "messageId": "m2", "parentId": "m1", "timestamp": "2024-01-01T00:01:00Z"},
                {"role": "user", "text": "Question", "messageId": "m3", "parentId": "m2", "timestamp": "2024-01-01T00:02:00Z"},
                {"role": "model", "text": "Answer A", "messageId": "m4", "parentId": "m3", "timestamp": "2024-01-01T00:03:00Z"},
                {"role": "model", "text": "Answer B", "messageId": "m5", "parentId": "m3", "timestamp": "2024-01-01T00:04:00Z"},
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


def test_render_branch_full_includes_branch_map_in_conversation_md(tmp_path, state_env):
    _ = state_env  # ensure fixture applies environment paths
    src = tmp_path / "branched.json"
    _write_render_input_branched(src)
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

    conversation_dir = out_dir / "branched"
    conversation_md = conversation_dir / "conversation.md"
    assert conversation_md.exists()
    rendered = conversation_md.read_text(encoding="utf-8")
    assert "## Branch map" in rendered
    assert "- [branch-000](./branches/branch-000/branch-000.md)" in rendered
    assert "- [branch-001](./branches/branch-001/branch-001.md)" in rendered
    assert (conversation_dir / "branches" / "branch-001" / "branch-001.md").exists()
