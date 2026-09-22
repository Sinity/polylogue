"""One profile lifecycle, not two (polylogue-foour).

``session_profiles`` / ``session_latency_profiles`` used to have a second
lifecycle owner beside the convergence domain: an incremental "refresh" in
``polylogue/storage/derived/session/refresh.py``, reached from
``pipeline/services/ingest_batch`` as ``refresh_session_insights_bulk``.  It
composed a whole profile family *and* stamped ``input_content_hash`` -- and
stamping that column is a certification, because ``inspect_session_profiles``
reads a matching binding as proof the rows were computed from the current
inputs.

The predecessor made that certification without the checks the owning domain
makes.  ``publish_session_profile`` refuses to certify a profile whose
``session_usage_rollup`` prerequisite is not ``valid``
(``storage/derived/session/derivation.py``; covered by
``tests/unit/storage/test_session_usage_rollup_derivation.py``), because the
profile reads canonical ``session_model_usage`` values.  The refresh route had
no such refusal and reconciled no rollup, so it could stamp a binding over rows
built from superseded cost values.

This is the ratchet for the retirement.  It is a *census*, not a name check: a
new second composer under any new name lands in the census and turns it red.
Each module is frozen with the kinds of profile mutation it may perform, so
restoring the update side to ``refresh.py`` -- a module that legitimately still
appears here for its delete side -- is red too.

Anti-vacuity: call ``replace_session_profile`` from any module not declared
below, give a declared module a mutation kind it is not allowed, or reintroduce
``refresh_session_insights_bulk``, and this test fails.
"""

from __future__ import annotations

import io
import re
import tokenize
from pathlib import Path

import polylogue

_PACKAGE_ROOT = Path(polylogue.__file__).resolve().parent

_ROW_WRITE = "row-write"
_DELETE = "delete"
_UPDATE = "update-column"

#: Helper calls are matched against NAME tokens only, and SQL against string
#: literals only, so prose in a comment -- including this module's own -- can
#: neither add nor hide a writer.
_HELPER = re.compile(r"^replace_session_(latency_)?profiles?(_bulk)?(_sync)?$")
#: An f-string is not one STRING token on this interpreter; its literal pieces
#: arrive as FSTRING_MIDDLE, and two of the declared modules build their SQL
#: that way.
_STRING_TOKENS = frozenset({tokenize.STRING, getattr(tokenize, "FSTRING_MIDDLE", tokenize.STRING)})
_SQL: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(INSERT|REPLACE)\s+INTO\s+session_(latency_)?profiles\b", re.I), _ROW_WRITE),
    (re.compile(r"DELETE\s+FROM\s+session_(latency_)?profiles\b", re.I), _DELETE),
    (re.compile(r"UPDATE\s+session_(latency_)?profiles\b", re.I), _UPDATE),
)

#: module path (relative to the ``polylogue`` package) -> the profile mutations
#: it is allowed to perform, with the reason it is allowed to perform them.
_ALLOWED: dict[str, set[str]] = {
    # The one lifecycle owner: composes profile families and writes them. The
    # convergence domain drives it; nothing else may compose a family.
    "storage/derived/session/rebuild.py": {_ROW_WRITE},
    # The domain publisher: stamps/clears the binding column and drops an
    # excess key's rows. It composes nothing itself -- it calls rebuild.
    "storage/derived/session/derivation.py": {_DELETE, _UPDATE},
    # Delete-side upkeep only, inside the caller's session-delete transaction.
    "storage/derived/session/refresh.py": {_DELETE},
    # SQL helpers the writers above call. Not lifecycles.
    "storage/derived/session/storage.py": {_ROW_WRITE, _DELETE},
    "storage/sqlite/queries/session_insight_profile_writes.py": {_ROW_WRITE},
    "storage/sqlite/async_sqlite.py": {_ROW_WRITE},
    # Session deletion and full index replacement clear derived rows with the
    # session they belong to.
    "storage/sqlite/archive_tiers/write.py": {_DELETE},
    # Single-column touches, not family composition: the DDL triggers that
    # NULL the binding when an input moves, and the source-sort-key reset.
    "storage/sqlite/archive_tiers/index.py": {_UPDATE},
    "archive/write_effects.py": {_UPDATE},
    "demo/seed.py": {_UPDATE},
}


def _mutations(source: str) -> set[str]:
    kinds: set[str] = set()
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.NAME:
            if _HELPER.match(token.string):
                kinds.add(_ROW_WRITE)
        elif token.type in _STRING_TOKENS:
            for pattern, kind in _SQL:
                if pattern.search(token.string):
                    kinds.add(kind)
    return kinds


def _census() -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        kinds = _mutations(path.read_text(encoding="utf-8"))
        if kinds:
            found[path.relative_to(_PACKAGE_ROOT).as_posix()] = kinds
    return found


def test_profile_mutation_census_matches_the_declared_owners() -> None:
    """Exactly one module composes profile families; the rest are declared."""
    assert _census() == _ALLOWED


def test_the_predecessor_refresh_lifecycle_is_gone() -> None:
    """The retired update-side entry points do not exist under any route."""
    from polylogue.pipeline.services import ingest_batch
    from polylogue.storage.derived.session import refresh

    assert not hasattr(ingest_batch, "refresh_session_insights_bulk")
    assert sorted(refresh.__all__) == [
        "delete_session_insights_for_session_async",
        "refresh_thread_after_session_delete_async",
    ]
