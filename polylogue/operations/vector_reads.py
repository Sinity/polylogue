"""Retained-vector evidence boundary shared by daemon read presentation."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from typing import TypeVar

from polylogue.core.errors import DatabaseError, VectorReadUnavailableError, VectorRuntimeUnavailableError
from polylogue.core.sqlite_locking import is_corrupt_sqlite_database, is_transient_sqlite_lock

_T = TypeVar("_T")


def read_retained_vectors(read: Callable[[], _T]) -> _T:
    """Translate an unanswered stored-vector read into a visible typed refusal."""
    try:
        return read()
    except VectorReadUnavailableError:
        raise
    except (DatabaseError, ValueError, sqlite3.Error, OSError) as exc:
        causes: list[BaseException] = []
        cause: BaseException | None = exc
        while cause is not None and all(cause is not seen for seen in causes):
            causes.append(cause)
            cause = cause.__cause__
        if any(is_transient_sqlite_lock(cause) for cause in causes):
            reason = "sqlite_contention"
        elif any(is_corrupt_sqlite_database(cause) for cause in causes):
            reason = "embeddings_db_unreadable"
        elif any(isinstance(cause, VectorRuntimeUnavailableError) for cause in causes):
            reason = "sqlite_vec_not_loaded"
        else:
            reason = "embedding_read_failed"
        raise VectorReadUnavailableError("retained vector evidence is unavailable", reason=reason) from exc
