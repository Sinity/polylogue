"""Archive read mixin composed from narrower concern families."""

from __future__ import annotations

from polylogue.storage.repository.archive.queries import RepositoryArchiveQueryMixin
from polylogue.storage.repository.archive.search import RepositoryArchiveSearchMixin
from polylogue.storage.repository.archive.sessions import (
    RepositoryArchiveSessionMixin,
)
from polylogue.storage.repository.archive.tree import RepositoryArchiveTreeMixin


class RepositoryArchiveReadMixin(
    RepositoryArchiveSessionMixin,
    RepositoryArchiveQueryMixin,
    RepositoryArchiveTreeMixin,
    RepositoryArchiveSearchMixin,
):
    """Archive read surface composed from session/query/tree/search mixins."""


__all__ = ["RepositoryArchiveReadMixin"]
