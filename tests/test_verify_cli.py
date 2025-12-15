from __future__ import annotations

import json
from contextlib import contextmanager
from types import SimpleNamespace

from polylogue.cli.verify import run_verify_cli
from polylogue.commands import CommandEnv, RenderOptions, render_command


class DummyUI:
    plain = True

    def __init__(self):
        self.console = self

    def print(self, *_args, **_kwargs):  # pragma: no cover - diagnostics only
        pass

    def summary(self, *_args, **_kwargs):  # pragma: no cover
        pass

    def banner(self, *_args, **_kwargs):  # pragma: no cover
        pass

    @contextmanager
    def progress(self, *_args, **_kwargs):  # pragma: no cover
        class DummyProgressTracker:
            def advance(self, *args, **kwargs):
                pass

            def update(self, *args, **kwargs):
                pass

        yield DummyProgressTracker()


def test_verify_cli_reports_ok(tmp_path, state_env, capsys):
    src = tmp_path / "render.json"
    payload = {
        "id": "conv-verify",
        "title": "Verify",
        "chunkedPrompt": {
            "chunks": [
                {"role": "user", "text": "Hello", "timestamp": "2024-01-01T00:00:00Z"},
                {"role": "model", "text": "Hi", "timestamp": "2024-01-01T00:01:00Z"},
            ]
        },
    }
    src.write_text(json.dumps(payload), encoding="utf-8")
    out_dir = tmp_path / "out"

    options = RenderOptions(
        inputs=[src],
        output_dir=out_dir,
        collapse_threshold=12,
        download_attachments=False,
        dry_run=False,
        force=False,
        html=False,
        html_theme=None,
        diff=False,
    )
    env = CommandEnv(ui=DummyUI())
    render_command(options, env)

    run_verify_cli(SimpleNamespace(provider="render", slug="render", conversation_ids=(), limit=None, json=True), env)
    payload = json.loads(capsys.readouterr().out)
    assert payload["errors"] == 0
    assert payload["verified"] == 1

