from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest

from polylogue.config import Source
from polylogue.pipeline.services import acquisition_streams
from polylogue.sources.parsers.base import RawConversationData
from polylogue.types import Provider


async def test_iter_raw_record_stream_logs_make_raw_record_value_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _raw_stream(*args: object, **kwargs: object) -> AsyncIterator[RawConversationData]:
        del args, kwargs
        yield RawConversationData(
            raw_bytes=b'{"id":"broken"}',
            source_path=str(tmp_path / "broken.json"),
            provider_hint=Provider.CHATGPT,
        )

    def _raise(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise ValueError("boom")

    warnings: list[tuple[str, dict[str, object]]] = []

    def _record_warning(event: str, **kwargs: object) -> None:
        warnings.append((event, kwargs))

    monkeypatch.setattr(acquisition_streams, "iter_source_raw_stream", _raw_stream)
    monkeypatch.setattr(acquisition_streams, "make_raw_record", _raise)
    monkeypatch.setattr(acquisition_streams.logger, "warning", _record_warning)

    items = [item async for item in acquisition_streams.iter_raw_record_stream(Source(name="chatgpt", path=tmp_path))]

    assert items == []
    assert warnings == [
        (
            "Skipping raw payload",
            {
                "source": "chatgpt",
                "path": str(tmp_path / "broken.json"),
                "error": "boom",
            },
        )
    ]


@pytest.mark.asyncio
async def test_iter_raw_record_stream_forwards_source_status_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.pipeline.services import acquisition as acquisition_module

    def _iter_source_raw_data(*args: object, **kwargs: object) -> Iterator[RawConversationData]:
        del args
        status_callback = kwargs["status_callback"]
        assert callable(status_callback)
        status_callback("Scanning [chatgpt] reading export.json")
        yield RawConversationData(
            raw_bytes=b'{"mapping": {}, "id": "ok"}',
            source_path=str(tmp_path / "export.json"),
            provider_hint=Provider.CHATGPT,
        )

    progress_events: list[tuple[int, str | None]] = []

    def _record_progress(amount: int, desc: str | None = None) -> None:
        progress_events.append((amount, desc))

    monkeypatch.setattr(acquisition_module, "iter_source_raw_data", _iter_source_raw_data)

    items = [
        item
        async for item in acquisition_streams.iter_raw_record_stream(
            Source(name="chatgpt", path=tmp_path),
            progress_callback=_record_progress,
        )
    ]
    await asyncio.sleep(0)

    assert len(items) == 1
    assert progress_events == [(0, "Scanning [chatgpt] reading export.json")]
