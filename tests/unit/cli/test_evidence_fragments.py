"""CLI evidence views preserve advancing, partially delivered rows (07.F049)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from polylogue.cli.read_views import session_evidence


@pytest.mark.parametrize("kind,key", [("file_edits", "file_edits"), ("web_content", "web_content_constructs")])
def test_cli_keeps_fragment_bytes_and_continuation(monkeypatch: pytest.MonkeyPatch, kind: str, key: str) -> None:
    fragment = {
        "row_offset": 0,
        "complete": False,
        "fields": [{"field": "text", "encoding": "utf-8", "offset": 0, "total_bytes": 10, "data_base64": "YQ=="}],
    }
    window = {
        "rows": [],
        "row_fragment": fragment,
        "total": 1,
        "returned": 0,
        "limit": 1,
        "offset": 0,
        "next_offset": 0,
        "continuation": "advancing-byte-cursor",
        "complete": False,
    }
    monkeypatch.setattr(
        session_evidence, "_read_evidence_window", lambda *_args, **_kwargs: (window, "test-session", "test")
    )
    monkeypatch.setattr(session_evidence, "_echo_served_by", lambda *_args: None)
    deliver = MagicMock()
    monkeypatch.setattr(session_evidence, "_deliver_evidence_document", deliver)
    getattr(session_evidence, f"run_read_{kind}")(MagicMock(), MagicMock(), MagicMock())
    document = deliver.call_args.args[2]
    assert document[key] == []
    assert document["row_fragment"] == fragment
    assert document["returned"] == document["next_offset"] == 0
    assert document["continuation"] == "advancing-byte-cursor"
    assert document["complete"] is False
