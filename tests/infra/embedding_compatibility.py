"""Synthetic archive and protocol fixtures for retained embedding compatibility."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, MaterialOrigin, Provider
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.live_ingest import write_index_session

_PAYLOAD = json.loads((Path(__file__).parents[1] / "fixtures" / "embedding_compatibility.json").read_text())
_TEXT = _PAYLOAD["retained"]
_NEW_TEXT = _PAYLOAD["missing"]


def _session(root: Path, *, extra: bool = False) -> tuple[str, tuple[str, ...]]:
    messages = [
        ParsedMessage(
            provider_message_id=f"m{i}",
            role=Role.USER,
            text=text,
            blocks=[ParsedContentBlock(type=BlockType.TEXT, text=text)],
            material_origin=MaterialOrigin.HUMAN_AUTHORED,
        )
        for i, text in enumerate([_TEXT, _TEXT] + ([_NEW_TEXT] if extra else []))
    ]
    with ArchiveStore(root) as store:
        sid = write_index_session(
            store, ParsedSession(source_name=Provider.CODEX, provider_session_id="compatibility", messages=messages)
        )
    with sqlite3.connect(root / "index.db") as conn:
        ids = tuple(str(r[0]) for r in conn.execute("SELECT message_id FROM messages ORDER BY message_id"))
    return sid, ids


class _Documents:
    dimension = 1024

    def __init__(self, model: str) -> None:
        self.model = model
        self.calls: list[tuple[str, ...]] = []

    def _get_embeddings(self, texts: list[str], input_type: str = "document") -> list[list[float]]:
        assert input_type == "document"
        self.calls.append(tuple(texts))
        return [[0.1] * self.dimension for _ in texts]

    def upsert(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("document protocol fixture uses the archive write owner")

    def query(self, *args: object, **kwargs: object) -> list[tuple[str, float]]:
        raise AssertionError("document protocol fixture does not query")

    def query_by_session(self, *args: object, **kwargs: object) -> list[tuple[str, float]]:
        raise AssertionError("document protocol fixture does not query")

    async def read_session_similarity(self, *args: object, **kwargs: object) -> dict[str, object]:
        raise AssertionError("document protocol fixture does not query")


def _rows(root: Path) -> tuple[list[tuple[object, ...]], list[tuple[object, ...]]]:
    with sqlite3.connect(root / "embeddings.db") as conn:
        return (
            conn.execute("SELECT * FROM message_embeddings_meta ORDER BY vector_derivation_hash").fetchall(),
            conn.execute("SELECT * FROM message_embedding_refs ORDER BY message_id").fetchall(),
        )
