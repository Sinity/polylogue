"""Unsealed prepared sessions held immutable reuse one decoded walk and report progress."""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.core.work_progress import work_progress
from polylogue.sources.parsers.base import ParsedMessage
from polylogue.sources.prepared_message_sink import SqliteMessageStore


def _messages() -> list[ParsedMessage]:
    return [
        ParsedMessage(provider_message_id=f"m{index}", role=Role.USER, text=f"text {index}", position=index)
        for index in range(5)
    ]


def test_held_walks_replay_the_decoded_session_and_refuse_edits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Red if a held walk re-decodes every pass, or if an edit inside the window is accepted.

    An accepted edit would be missed by every later replay of the spool.
    """
    import polylogue.sources.prepared_message_sink as sink_module

    store = SqliteMessageStore(tmp_path / "prepared.db")
    try:
        sink = store.new_sink()
        for message in _messages():
            sink.append(message)
        decoded = 0
        real_decode = sink_module._from_text_json

        def counting_decode(model: type[ParsedMessage], encoded: str) -> ParsedMessage:
            nonlocal decoded
            decoded += 1
            return real_decode(model, encoded)

        monkeypatch.setattr(sink_module, "_from_text_json", counting_decode)
        with sink.held_walks():
            first = list(sink)
            second = list(sink)
            suffix = list(sink.iter_from(3))
            assert first == second == _messages()
            assert suffix == _messages()[3:]
            # Replays yield fresh objects, as a decode does.
            assert first[0] is not second[0]
            assert decoded == len(first)
            with pytest.raises(RuntimeError):
                sink.append(_messages()[0])
            with pytest.raises(RuntimeError):
                sink[0] = _messages()[1]
        sink[0] = _messages()[1]
        assert next(iter(sink)) == _messages()[1]
    finally:
        store.close()


def test_walking_a_prepared_session_reports_work_progress(tmp_path: Path) -> None:
    """Red if appending or walking messages inside a unit of work counts nothing."""
    store = SqliteMessageStore(tmp_path / "prepared.db")
    try:
        with work_progress("test_preparation") as progress:
            sink = store.new_sink()
            for message in _messages():
                sink.append(message)
            assert progress.messages == 5
            assert len(list(sink)) == 5
            assert progress.messages == 10
    finally:
        store.close()
