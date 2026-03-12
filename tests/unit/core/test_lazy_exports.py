"""Focused tests for package-level lazy exports."""

from __future__ import annotations

import pytest


def test_lazy_import_conversation_repository_root() -> None:
    import polylogue

    assert polylogue.ConversationRepository.__name__ == "ConversationRepository"


def test_lazy_import_unknown_raises_root() -> None:
    import polylogue

    with pytest.raises(AttributeError, match="has no attribute"):
        _ = polylogue.NonExistentThing


def test_lazy_import_conversation_repository_lib() -> None:
    import polylogue.lib

    assert polylogue.lib.ConversationRepository.__name__ == "ConversationRepository"


def test_lazy_import_conversation_projection_lib() -> None:
    import polylogue.lib

    assert polylogue.lib.ConversationProjection.__name__ == "ConversationProjection"


def test_lazy_import_archive_stats_lib() -> None:
    import polylogue.lib

    assert polylogue.lib.ArchiveStats.__name__ == "ArchiveStats"


def test_lazy_import_unknown_raises_lib() -> None:
    import polylogue.lib

    with pytest.raises(AttributeError, match="has no attribute"):
        _ = polylogue.lib.NonExistentThing
