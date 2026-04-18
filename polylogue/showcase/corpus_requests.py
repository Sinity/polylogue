"""Authored corpus-request presets for showcase and QA workflows."""

from __future__ import annotations

from polylogue.scenarios import CorpusRequest, CorpusSourceKind

SHOWCASE_CORPUS_COUNT = 3
SHOWCASE_CORPUS_STYLE = "showcase"
SHOWCASE_MESSAGES_MIN = 6
SHOWCASE_MESSAGES_MAX = 19
SHOWCASE_CORPUS_SEED = 42


def showcase_corpus_request(
    *,
    count: int = SHOWCASE_CORPUS_COUNT,
    providers: tuple[str, ...] | None = None,
    source: CorpusSourceKind | str = CorpusSourceKind.DEFAULT,
    style: str = SHOWCASE_CORPUS_STYLE,
    messages_min: int = SHOWCASE_MESSAGES_MIN,
    messages_max: int = SHOWCASE_MESSAGES_MAX,
    seed: int = SHOWCASE_CORPUS_SEED,
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


__all__ = [
    "SHOWCASE_CORPUS_COUNT",
    "SHOWCASE_CORPUS_SEED",
    "SHOWCASE_CORPUS_STYLE",
    "SHOWCASE_MESSAGES_MAX",
    "SHOWCASE_MESSAGES_MIN",
    "showcase_corpus_request",
]
