from __future__ import annotations

import os
import shutil
import sqlite3
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


FIXED_NOW = "2000-01-01T00:00:00Z"

class _SimpleEncoding:
    def __init__(self, name: str):
        self.name = name

    def encode(self, text: str):
        if not text:
            return []
        return text.split()


def _install_tiktoken_stub() -> None:
    module = types.ModuleType("tiktoken")

    def _get_encoding(name: str) -> _SimpleEncoding:  # noqa: ANN001
        return _SimpleEncoding(name)

    def _encoding_for_model(model: str) -> _SimpleEncoding:  # noqa: ANN001
        return _SimpleEncoding(model)

    module.get_encoding = _get_encoding  # type: ignore[attr-defined]
    module.encoding_for_model = _encoding_for_model  # type: ignore[attr-defined]
    core_module = types.ModuleType("tiktoken.core")
    core_module.Encoding = _SimpleEncoding  # type: ignore[attr-defined]
    module.core = core_module  # type: ignore[attr-defined]
    sys.modules["tiktoken.core"] = core_module
    sys.modules["tiktoken"] = module


class DummyUI:
    plain = True

    def __init__(self):
        self.console = self

    def print(self, *_args, **_kwargs):
        pass

    def summary(self, *_args, **_kwargs):
        pass

    def banner(self, *_args, **_kwargs):
        pass

    def progress(self, *_args, **_kwargs):
        class _Tracker:
            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                return False

            def advance(self, *_args, **_kwargs):
                pass

            def update(self, *_args, **_kwargs):
                pass

        return _Tracker()


@dataclass(frozen=True)
class GoldenCase:
    name: str
    run: Callable[[Path], Tuple[str, str]]  # returns (slug, provider)


def _render_slug(db_path: Path, output_dir: Path, slug: str) -> Tuple[str, str]:
    conn = sqlite3.connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT provider, conversation_id FROM conversations WHERE slug = ?",
            (slug,),
        ).fetchone()
        if not row:
            raise RuntimeError(f"Slug not found in DB: {slug}")
        provider = row["provider"]
        conversation_id = row["conversation_id"]
    finally:
        conn.close()

    from polylogue.renderers.db_renderer import DatabaseRenderer

    renderer = DatabaseRenderer(db_path)
    renderer.render_conversation(provider, conversation_id, output_dir)
    return provider, conversation_id


def main() -> None:
    repo_root = REPO_ROOT
    _install_tiktoken_stub()
    os.environ["POLYLOGUE_FIXED_NOW"] = FIXED_NOW
    os.environ["XDG_STATE_HOME"] = str(repo_root / ".tmp" / "golden-state")
    os.environ["XDG_CACHE_HOME"] = str(repo_root / ".tmp" / "golden-cache")
    (repo_root / ".tmp").mkdir(parents=True, exist_ok=True)

    os.chdir(repo_root)
    golden_dir = repo_root / "tests" / "golden"
    golden_dir.mkdir(parents=True, exist_ok=True)
    golden_html_dir = repo_root / "tests" / "golden_html"
    golden_html_dir.mkdir(parents=True, exist_ok=True)

    tmp_root = repo_root / ".tmp" / "golden-work"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)

    from polylogue.commands import CommandEnv
    from polylogue.frontmatter_canonical import canonicalize_markdown
    from polylogue.html import HtmlRenderOptions, render_html
    from polylogue.importers.chatgpt import import_chatgpt_export
    from polylogue.importers.claude_ai import import_claude_export
    from polylogue.importers.claude_code import import_claude_code_session
    from polylogue.importers.codex import import_codex_session
    from polylogue.render import MarkdownDocument
    import frontmatter

    env = CommandEnv(ui=DummyUI())
    db_path = env.database.resolve_path()
    if db_path is None:
        raise RuntimeError("Database path unavailable.")

    def run_chatgpt(output_dir: Path) -> Tuple[str, str]:
        results = import_chatgpt_export(
            Path("tests/fixtures/golden/chatgpt"),
            output_dir=output_dir,
            collapse_threshold=10,
            html=False,
            html_theme="light",
            force=True,
        )
        if not results:
            raise RuntimeError("ChatGPT import produced no results.")
        slug = results[0].slug
        _render_slug(db_path, output_dir, slug)
        return slug, "chatgpt"

    def run_chatgpt_tool(output_dir: Path) -> Tuple[str, str]:
        results = import_chatgpt_export(
            Path("tests/fixtures/golden/chatgpt_tool"),
            output_dir=output_dir,
            collapse_threshold=10,
            html=False,
            html_theme="light",
            force=True,
        )
        if not results:
            raise RuntimeError("ChatGPT tool import produced no results.")
        slug = results[0].slug
        _render_slug(db_path, output_dir, slug)
        return slug, "chatgpt"

    def run_claude(output_dir: Path) -> Tuple[str, str]:
        results = import_claude_export(
            Path("tests/fixtures/golden/claude"),
            output_dir=output_dir,
            collapse_threshold=10,
            html=False,
            html_theme="light",
            force=True,
        )
        if not results:
            raise RuntimeError("Claude import produced no results.")
        slug = results[0].slug
        _render_slug(db_path, output_dir, slug)
        return slug, "claude"

    def run_claude_tool(output_dir: Path) -> Tuple[str, str]:
        results = import_claude_export(
            Path("tests/fixtures/golden/claude_tool"),
            output_dir=output_dir,
            collapse_threshold=10,
            html=False,
            html_theme="light",
            force=True,
        )
        if not results:
            raise RuntimeError("Claude tool import produced no results.")
        slug = results[0].slug
        _render_slug(db_path, output_dir, slug)
        return slug, "claude"

    def run_codex(output_dir: Path) -> Tuple[str, str]:
        result = import_codex_session(
            "tests/fixtures/golden/codex/codex-golden.jsonl",
            base_dir=repo_root,
            output_dir=output_dir,
            collapse_threshold=10,
            html=False,
            force=True,
        )
        slug = result.slug
        _render_slug(db_path, output_dir, slug)
        return slug, "codex"

    def run_claude_code(output_dir: Path) -> Tuple[str, str]:
        result = import_claude_code_session(
            "tests/fixtures/golden/claude_code/claude-code-golden.jsonl",
            base_dir=Path("."),
            output_dir=output_dir,
            collapse_threshold=10,
            html=False,
            html_theme="light",
            force=True,
        )
        slug = result.slug
        _render_slug(db_path, output_dir, slug)
        return slug, "claude-code"

    cases: List[GoldenCase] = [
        GoldenCase("chatgpt-basic.md", run_chatgpt),
        GoldenCase("chatgpt-tool.md", run_chatgpt_tool),
        GoldenCase("claude-basic.md", run_claude),
        GoldenCase("claude-tool.md", run_claude_tool),
        GoldenCase("codex-basic.md", run_codex),
        GoldenCase("claude-code-basic.md", run_claude_code),
    ]

    for case in cases:
        case_out = tmp_root / "out" / Path(case.name).stem
        case_out.mkdir(parents=True, exist_ok=True)
        slug, _provider = case.run(case_out)
        md_path = case_out / slug / "conversation.md"
        if not md_path.exists():
            raise RuntimeError(f"Expected markdown not found: {md_path}")
        target = golden_dir / case.name
        text = md_path.read_text(encoding="utf-8")
        canonical_md = canonicalize_markdown(text, repo_root=repo_root, scrub_paths=True)
        target.write_text(canonical_md, encoding="utf-8")

        post = frontmatter.loads(canonical_md)
        doc = MarkdownDocument(body=post.content, metadata=dict(post.metadata), attachments=[], stats={})
        html_text = render_html(doc, HtmlRenderOptions(theme="light")).replace("\r\n", "\n")
        html_target = golden_html_dir / Path(case.name).with_suffix(".html").name
        html_target.write_text(html_text, encoding="utf-8")


if __name__ == "__main__":
    main()
