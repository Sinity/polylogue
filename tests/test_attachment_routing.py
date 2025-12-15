from pathlib import Path

from polylogue.importers.utils import LINE_THRESHOLD, store_large_text
from polylogue.render import AttachmentInfo


def test_store_large_text_updates_routing_stats_skipped(tmp_path: Path) -> None:
    attachments: list[AttachmentInfo] = []
    per_chunk_links: dict[int, list[tuple[str, Path]]] = {}
    stats = {"routed": 0, "skipped": 0}

    out = store_large_text(
        "hello",
        chunk_index=0,
        attachments_dir=tmp_path / "attachments",
        markdown_dir=tmp_path,
        attachments=attachments,
        per_chunk_links=per_chunk_links,
        routing_stats=stats,
    )

    assert out == "hello"
    assert stats["skipped"] == 1
    assert stats["routed"] == 0
    assert not attachments


def test_store_large_text_updates_routing_stats_routed(tmp_path: Path) -> None:
    attachments: list[AttachmentInfo] = []
    per_chunk_links: dict[int, list[tuple[str, Path]]] = {}
    stats = {"routed": 0, "skipped": 0}
    text = "\n".join(f"line {i}" for i in range(LINE_THRESHOLD + 1))

    out = store_large_text(
        text,
        chunk_index=0,
        attachments_dir=tmp_path / "attachments",
        markdown_dir=tmp_path,
        attachments=attachments,
        per_chunk_links=per_chunk_links,
        routing_stats=stats,
    )

    assert "Full content saved to" in out
    assert stats["routed"] == 1
    assert stats["skipped"] == 0
    assert len(attachments) == 1
    assert (tmp_path / "attachments" / "chunk000.txt").exists()
