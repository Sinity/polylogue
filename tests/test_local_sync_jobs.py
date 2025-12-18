from __future__ import annotations

from pathlib import Path

from polylogue.importers.base import ImportResult
from polylogue.local_sync import _sync_sessions
from polylogue.render import MarkdownDocument
from polylogue.ui import UI


def test_local_sync_jobs_preserves_deterministic_order(tmp_path):
    base = tmp_path / "sessions"
    base.mkdir(parents=True, exist_ok=True)
    sessions = []
    for idx in range(5):
        p = base / f"{idx:02d}.jsonl"
        p.write_text("{}", encoding="utf-8")
        sessions.append(p)

    out_dir = tmp_path / "out"

    def import_fn(session_id: str, *, output_dir: Path, **_kwargs):
        slug = Path(session_id).stem
        conv_dir = output_dir / slug
        conv_dir.mkdir(parents=True, exist_ok=True)
        md_path = conv_dir / "conversation.md"
        md_path.write_text("---\n---\n\nHello\n", encoding="utf-8")
        doc = MarkdownDocument(
            body="Hello",
            metadata={"attachmentBytes": 0},
            attachments=[],
            stats={"totalTokensApprox": 0, "totalWordsApprox": 0},
        )
        return ImportResult(markdown_path=md_path, html_path=None, attachments_dir=None, document=doc, slug=slug)

    ui = UI(plain=True)
    result = _sync_sessions(
        sessions,
        output_dir=out_dir,
        collapse_threshold=10,
        collapse_thresholds={"message": 10, "tool": 10},
        base_dir=base,
        html=False,
        html_theme="light",
        force=True,
        prune=False,
        diff=False,
        provider="codex",
        import_fn=import_fn,
        ui=ui,
        jobs=3,
    )

    assert [r.slug for r in result.written] == [f"{idx:02d}" for idx in range(5)]
    assert result.failures == 0

