from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import frontmatter
import pytest

from polylogue.cli.verify import run_verify_cli
from polylogue.commands import CommandEnv, RenderOptions, render_command


class DummyUI:
    plain = True

    def __init__(self):
        self.console = self

    def print(self, *_args, **_kwargs):  # pragma: no cover
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


def test_verify_flags_noncanonical_and_unknown_keys(tmp_path, state_env, capsys):
    src = tmp_path / "render.json"
    payload = {
        "id": "conv-verify-canon",
        "title": "Verify Canon",
        "chunkedPrompt": {
            "chunks": [
                {"role": "user", "text": "Hello", "timestamp": "2024-01-01T00:00:00Z"},
                {"role": "model", "text": "Hi", "timestamp": "2024-01-01T00:01:00Z"},
            ]
        },
    }
    src.write_text(json.dumps(payload), encoding="utf-8")
    out_dir = tmp_path / "out"

    env = CommandEnv(ui=DummyUI())
    render_command(
        RenderOptions(
            inputs=[src],
            output_dir=out_dir,
            collapse_threshold=12,
            download_attachments=False,
            dry_run=False,
            force=False,
            html=False,
            html_theme=None,
            diff=False,
        ),
        env,
    )

    md_path = out_dir / "render" / "conversation.md"
    post = frontmatter.loads(md_path.read_text(encoding="utf-8"))
    polylogue_meta = dict(post.metadata.get("polylogue") or {})
    polylogue_meta["mysteryKey"] = "x"
    post.metadata["polylogue"] = polylogue_meta
    # Write without sorting/canonicalization.
    md_path.write_text(frontmatter.dumps(post), encoding="utf-8")

    with pytest.raises(SystemExit):
        run_verify_cli(
            SimpleNamespace(
                provider="render",
                slug="render",
                conversation_ids=(),
                limit=None,
                json=True,
                fix=False,
                strict=False,
                unknown_policy="error",
                allow_polylogue_keys=(),
            ),
            env,
        )
    payload = json.loads(capsys.readouterr().out)
    assert payload["errors"] >= 1
    messages = [issue["message"] for issue in payload["issues"]]
    assert any("Unknown polylogue metadata keys" in msg for msg in messages)


def test_verify_fix_rewrites_frontmatter(tmp_path, state_env):
    src = tmp_path / "render.json"
    payload = {
        "id": "conv-verify-fix",
        "title": "Verify Fix",
        "chunkedPrompt": {"chunks": [{"role": "user", "text": "Hello"}]},
    }
    src.write_text(json.dumps(payload), encoding="utf-8")
    out_dir = tmp_path / "out"

    env = CommandEnv(ui=DummyUI())
    render_command(
        RenderOptions(
            inputs=[src],
            output_dir=out_dir,
            collapse_threshold=12,
            download_attachments=False,
            dry_run=False,
            force=False,
            html=False,
            html_theme=None,
            diff=False,
        ),
        env,
    )

    md_path = out_dir / "render" / "conversation.md"
    # Make the YAML order non-canonical by reversing keys at top-level.
    post = frontmatter.loads(md_path.read_text(encoding="utf-8"))
    meta = dict(post.metadata)
    post.metadata = dict(reversed(list(meta.items())))
    md_path.write_text(frontmatter.dumps(post), encoding="utf-8")

    run_verify_cli(
        SimpleNamespace(
            provider="render",
            slug="render",
            conversation_ids=(),
            limit=None,
            json=True,
            fix=True,
            strict=False,
            unknown_policy="ignore",
            allow_polylogue_keys=(),
        ),
        env,
    )
    # second pass should not need to rewrite and should exit cleanly.
    run_verify_cli(
        SimpleNamespace(
            provider="render",
            slug="render",
            conversation_ids=(),
            limit=None,
            json=True,
            fix=False,
            strict=True,
            unknown_policy="ignore",
            allow_polylogue_keys=(),
        ),
        env,
    )

