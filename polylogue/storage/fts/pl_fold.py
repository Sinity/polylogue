"""Shared deterministic Unicode folding for Polish lexical search recall.

FTS5's ``unicode61`` tokenizer with ``remove_diacritics=2`` folds combining-mark
diacritics (``ó`` -> ``o``, ``ą`` -> ``a``, ``ż`` -> ``z``, ...) automatically
and *symmetrically* for both indexed text and ``MATCH`` query text, because the
same tokenizer runs on both sides of a search. Verified locally (see
``tests/unit/storage/test_pl_fold.py``): ``rozowy`` finds seeded ``różowy``
under ``unicode61 remove_diacritics 2`` with no extra code.

It does **not** fold ``ł``/``Ł`` (Latin small/capital letter L with stroke).
Unicode assigns the stroke no canonical (NFD/NFKD) decomposition -- it is an
atomic code point, not a base letter plus a combining mark -- so neither
Unicode normalization nor FTS5's diacritic table can reach it. Confirmed
empirically: ``unicode61 remove_diacritics 2`` tokenizes ``żółty`` to
``zołty`` (ż and ó folded, ł untouched), so a plain-ASCII query like
``latwo`` never finds indexed ``łatwo`` content without an explicit fold.

``pl_fold`` closes that specific, narrow gap. It is expressed three ways from
one source of truth (:data:`PL_FOLD_TABLE`) so write-side (FTS trigger/insert
SQL) and read-side (``MATCH`` query normalization) folding can never drift
apart silently:

- :func:`pl_fold` -- the Python implementation. Used for query-side
  normalization (:func:`polylogue.storage.search.query_support.escape_fts5_query`)
  and available as the reference implementation for agreement tests.
- :func:`pl_fold_sql_expr` -- a SQL ``REPLACE`` chain over a column/parameter
  reference, computing the identical transform. Embedded directly into
  contentless-FTS trigger and bulk-insert DDL text
  (``polylogue/storage/sqlite/archive_tiers/index.py``,
  ``polylogue/storage/fts/fts_lifecycle.py``, ``polylogue/storage/fts/sql.py``)
  so write-side folding does not depend on every connection that fires a
  ``blocks``/``threads``/``session_work_events`` trigger having registered a
  custom function first -- SQLite trigger bodies execute using whichever
  connection performs the write, including test/tooling connections that
  never call :func:`register_pl_fold`.
- :func:`register_pl_fold` -- registers the identical transform as a SQL
  scalar function named ``pl_fold`` for ad-hoc queries, diagnostics, and the
  Python/SQL agreement test (``tests/unit/storage/test_pl_fold.py``).
  Production write and read connections register it in
  ``polylogue/storage/sqlite/connection.py`` and
  ``polylogue/storage/sqlite/async_sqlite.py`` before any write can reach a
  FTS-backing trigger on those connections.

Deliberately out of scope: :func:`polylogue.storage.search.query_support.
extract_match_terms` (the "why this matched" highlight-term extractor) is
unfolded on purpose -- it must echo the reader-facing literal token, and
highlighting matches against the *original* unfolded message text.
"""

from __future__ import annotations

import sqlite3

PL_FOLD_TABLE: tuple[tuple[str, str], ...] = (
    ("ł", "l"),
    ("Ł", "L"),
)
"""Characters ``unicode61 remove_diacritics`` cannot fold (no NFD decomposition)."""

_PL_FOLD_TRANSLATION: dict[int, str] = {ord(source): target for source, target in PL_FOLD_TABLE}


def pl_fold(text: str | None) -> str | None:
    """Fold characters ``unicode61 remove_diacritics`` cannot reach.

    Idempotent: every replacement target (``l``, ``L``) is plain ASCII and
    never itself a fold source, so re-folding an already-folded string is a
    no-op. ``None`` passes through unchanged, matching SQLite's NULL-in/
    NULL-out convention for scalar functions, so this is safe to register
    directly via :func:`register_pl_fold`.
    """
    if text is None:
        return None
    return text.translate(_PL_FOLD_TRANSLATION)


def pl_fold_sql_expr(column_expr: str) -> str:
    """Return a SQL ``REPLACE`` chain computing :func:`pl_fold` over ``column_expr``.

    Byte-identical output to :func:`pl_fold` by construction (both derive
    from :data:`PL_FOLD_TABLE`). Embeddable directly in trigger/insert DDL so
    write-side folding never depends on the executing connection having
    registered the ``pl_fold`` scalar function.
    """
    expr = column_expr
    for source, target in PL_FOLD_TABLE:
        expr = f"REPLACE({expr}, '{source}', '{target}')"
    return expr


def register_pl_fold(conn: sqlite3.Connection) -> None:
    """Register ``pl_fold`` as a deterministic SQL scalar function on ``conn``."""
    conn.create_function("pl_fold", 1, pl_fold, deterministic=True)


__all__ = [
    "PL_FOLD_TABLE",
    "pl_fold",
    "pl_fold_sql_expr",
    "register_pl_fold",
]
