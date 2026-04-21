"""Local typing surface for the dateparser API used by Polylogue."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime

LanguageDetector = Callable[[str, float], Sequence[str]]


def parse(
    date_string: str,
    date_formats: Sequence[str] | None = None,
    languages: Sequence[str] | None = None,
    locales: Sequence[str] | None = None,
    region: str | None = None,
    settings: Mapping[str, object] | None = None,
    detect_languages_function: LanguageDetector | None = None,
) -> datetime | None: ...
