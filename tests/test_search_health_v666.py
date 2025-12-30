from __future__ import annotations

import json
from pathlib import Path

from polylogue.config import default_config, load_config, write_config
from polylogue.health import get_health
from polylogue.index_v666 import rebuild_index
from polylogue.search_v666 import search_messages
from polylogue.ingest import IngestBundle, ingest_bundle
from polylogue.store import ConversationRecord, MessageRecord


def _seed_conversation():
    ingest_bundle(
        IngestBundle(
            conversation=ConversationRecord(
                conversation_id="conv:hash",
                provider_name="codex",
                provider_conversation_id="conv",
                title="Demo",
                created_at=None,
                updated_at=None,
                content_hash="hash",
                provider_meta=None,
            ),
            messages=[
                MessageRecord(
                    message_id="msg:hash",
                    conversation_id="conv:hash",
                    provider_message_id="msg",
                    role="user",
                    text="hello world",
                    timestamp=None,
                    content_hash="hash",
                    provider_meta=None,
                )
            ],
            attachments=[],
        )
    )


def test_search_after_index(workspace_env):
    _seed_conversation()
    rebuild_index()
    results = search_messages("hello", archive_root=workspace_env["archive_root"], limit=5)
    assert results.hits
    assert results.hits[0].conversation_id == "conv:hash"


def test_health_cached(workspace_env):
    config = default_config()
    write_config(config)
    loaded = load_config()
    first = get_health(loaded)
    second = get_health(loaded)
    assert second.get("cached") is True
    assert second.get("age_seconds") is not None
