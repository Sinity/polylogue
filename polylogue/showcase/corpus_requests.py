"""Authored corpus-request presets for showcase and QA workflows."""

from __future__ import annotations

from polylogue.scenarios import CorpusRequest, CorpusSourceKind


def showcase_corpus_request(
    *,
    count: int = 3,
    providers: tuple[str, ...] | None = None,
    source: CorpusSourceKind | str = CorpusSourceKind.DEFAULT,
    style: str = "showcase",
    messages_min: int = 6,
    messages_max: int = 19,
    seed: int = 42,
) -> CorpusRequest:
    """Build the canonical synthetic corpus request for showcase-style seeding."""
    return CorpusRequest(
        providers=providers,
        source=source,
        count=count,
        messages_min=messages_min,
        messages_max=messages_max,
        seed=seed,
        style=style,
    )


__all__ = ["showcase_corpus_request"]
