"""One shared CLI archive-store double, bound to the production surface.

``tests/unit/cli/test_query_exec_laws.py`` exercises how a root query is
lowered into ``ArchiveStore`` calls.  That law is about the lowering, not about
storage, so the store itself is stubbed.  Since the CLI's local query executor
was retired, the lowering runs inside the declared read handlers
(``polylogue/operations/daemon_reads.py``) and the store is opened by
``operation_context.open_operation_read``, which is where the double installs.  Thirty-eight
independent hand-written stubs previously drifted from the production class
until they proved only that they matched themselves: every stub method took
``**kwargs``, so a renamed production parameter was absorbed silently; each
stub installed its own ``open_existing`` lambda that broke whenever the
production classmethod grew a keyword; and none of them carried the attributes
``open_operation_read`` reads off an opened store.

``ArchiveStoreDouble`` restores the missing law.  Everything the double
presents is derived from the real :class:`ArchiveStore`:

* every method a double defines must exist on ``ArchiveStore`` -- a renamed or
  deleted production method fails the double immediately;
* every parameter a double declares explicitly must be accepted by the
  production method of the same name -- a renamed production parameter fails
  the double instead of vanishing into ``**kwargs``;
* the operation-read and tier-path attributes are read out of the production
  class body, so a rename there fails at import rather than being patched in.

A double still names only the parameters whose values it asserts on; the
checking is what makes that safe.
"""

from __future__ import annotations

import ast
import functools
import inspect
import sqlite3
import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.query_unit_frame import INDEX_FRAME_RELATIONS

__all__ = ["ArchiveStoreDouble", "install_archive_store_double"]

_OPEN_EXISTING_TARGET = "polylogue.operations.operation_context.ArchiveStore.open_existing"

# Attributes ``open_operation_read`` and the read path set or read on an opened
# store. Production owns the names; the double only mirrors them.
_OPERATION_READ_ATTRIBUTES = (
    "operation_identity",
    "operation_vector_connection",
    "operation_schema_versions",
    "operation_degraded_components",
)
_ACTIVE_INDEX_ATTRIBUTE = "index_db_path"

#: A window wider than any double's transcript, so the derived summary counts
#: every message the double is willing to serve.
_DOUBLE_PAGE_CEILING = 10_000


def _production_class_body() -> ast.Module:
    return ast.parse(textwrap.dedent(inspect.getsource(ArchiveStore)))


def _self_assignments(tree: ast.Module) -> set[str]:
    assigned: set[str] = set()
    for node in ast.walk(tree):
        targets = [node.target] if isinstance(node, ast.AnnAssign) else getattr(node, "targets", [])
        for target in targets:
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                assigned.add(target.attr)
    return assigned


def _tier_path_filenames(tree: ast.Module) -> dict[str, str]:
    """Read ``self.<name>_db_path = archive_root / "<file>"`` out of production."""

    filenames: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target, value = node.targets[0], node.value
        if not (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self"):
            continue
        if (
            isinstance(value, ast.BinOp)
            and isinstance(value.op, ast.Div)
            and isinstance(value.left, ast.Name)
            and value.left.id == "archive_root"
            and isinstance(value.right, ast.Constant)
            and isinstance(value.right.value, str)
        ):
            filenames[target.attr] = value.right.value
    return filenames


def _resolve_production_surface() -> dict[str, str]:
    tree = _production_class_body()
    assigned = _self_assignments(tree)
    missing = [name for name in (*_OPERATION_READ_ATTRIBUTES, _ACTIVE_INDEX_ATTRIBUTE) if name not in assigned]
    if missing:
        raise AttributeError(
            f"ArchiveStore no longer declares {missing}; the opened-store attribute surface "
            "moved. Update ArchiveStoreDouble to the current names."
        )
    filenames = _tier_path_filenames(tree)
    if not filenames:
        raise AttributeError(
            "ArchiveStore no longer derives any '<tier>_db_path' from its archive root; "
            "the tier-path surface moved. Update ArchiveStoreDouble to the current shape."
        )
    return filenames


_TIER_PATH_FILENAMES = _resolve_production_surface()


def _production_signature(name: str) -> inspect.Signature:
    production = getattr(ArchiveStore, name, None)
    if production is None or not callable(production):
        raise AttributeError(
            f"ArchiveStoreDouble declares {name!r}, which production ArchiveStore does not expose. "
            "The production surface moved: update the double to the current method name."
        )
    return inspect.signature(production)


def _bind_to_production(owner: str, name: str, func: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a double method so each call must satisfy the production signature.

    The declaration check below only covers parameters a double names
    explicitly. A double that takes ``**kwargs`` and reads ``kwargs["origins"]``
    would otherwise keep passing after production renamed that parameter, since
    the caller's keyword and the subscript drift together. Binding the incoming
    call against production's own signature closes that: the call the read path
    makes must be one the real ArchiveStore could serve.
    """

    signature = _production_signature(name)

    @functools.wraps(func)
    def _checked(self: ArchiveStoreDouble, *args: object, **kwargs: object) -> Any:
        try:
            signature.bind(self, *args, **kwargs)
        except TypeError as exc:
            raise TypeError(
                f"{owner}.{name} was called with arguments production "
                f"ArchiveStore.{name}{signature} does not accept ({exc}). The production "
                "surface moved: update the double and the caller to the current shape."
            ) from exc
        return func(self, *args, **kwargs)

    return _checked


def _check_method(owner: str, name: str, func: Callable[..., Any]) -> None:
    signature = _production_signature(name)
    parameters = signature.parameters
    accepts_extra_keywords = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())
    for index, (declared, parameter) in enumerate(inspect.signature(func).parameters.items()):
        if index == 0 and declared == "self":
            continue
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if declared in parameters:
            continue
        if accepts_extra_keywords and parameter.kind is inspect.Parameter.KEYWORD_ONLY:
            continue
        raise TypeError(
            f"{owner}.{name} declares parameter {declared!r}, which production "
            f"ArchiveStore.{name}{signature} does not accept. The production surface moved: "
            "update the double to the current parameter name."
        )


def _query_unit_frame_connection() -> sqlite3.Connection:
    """A minimal ``_conn`` carrying the query-unit frame epoch surface.

    ``archive_snapshot_epoch`` reads the index tier's per-relation frame rows
    and the user tier's singleton off the opened store's connection to bind a windowed
    read's continuation to one snapshot. A double with no connection at all
    made every ``session.read`` window fail as unframeable, which is a missing
    double rather than a real refusal -- and the refusal it raises is the one
    production reserves for derived-tier schema drift.
    """

    connection = sqlite3.connect(":memory:")
    connection.execute("ATTACH DATABASE ':memory:' AS user_tier")
    connection.execute("CREATE TABLE main.query_unit_frame_state (relation TEXT PRIMARY KEY, epoch INTEGER)")
    connection.executemany(
        "INSERT INTO main.query_unit_frame_state VALUES (?, 1)",
        [(relation,) for relation in INDEX_FRAME_RELATIONS],
    )
    connection.execute("CREATE TABLE user_tier.query_unit_frame_state (singleton INTEGER PRIMARY KEY, epoch INTEGER)")
    connection.execute("INSERT INTO user_tier.query_unit_frame_state VALUES (1, 1)")
    return connection


class ArchiveStoreDouble:
    """Base class for the CLI query-execution store doubles.

    Supplies the context-manager protocol and the opened-store attributes every
    double needs, and checks each subclass against the production
    ``ArchiveStore`` surface at class-definition time.  ``opened_roots`` records
    the archive root of each ``open_existing`` call the double served, so a test
    can assert which root the CLI resolved without writing its own
    ``open_existing`` replacement.
    """

    def __init__(self) -> None:
        self.opened_roots: list[Path] = []
        self._conn = _query_unit_frame_connection()
        self.operation_identity: Any = None
        self.operation_vector_connection: Any = None
        self.operation_schema_versions: Any = None
        self.operation_degraded_components: tuple[str, ...] = ()

    def bind_archive_root(self, archive_root: Path, index_path: Path | None = None) -> None:
        """Derive the tier paths production would derive for ``archive_root``.

        ``index_path`` mirrors production's pinned-index open: when the caller
        pins an index, that path is the active one; otherwise it is resolved the
        way ``ArchiveStore`` resolves it.
        """

        from polylogue.storage.archive_identity import resolve_active_index_path

        self.opened_roots.append(archive_root)
        self.archive_root = archive_root
        for attribute, filename in _TIER_PATH_FILENAMES.items():
            setattr(self, attribute, archive_root / filename)
        active_index = index_path if index_path is not None else resolve_active_index_path(archive_root)
        setattr(self, _ACTIVE_INDEX_ATTRIBUTE, active_index)

    def count_sessions(self, **kwargs: object) -> int:
        """Report an empty archive unless the double says otherwise.

        The declared read handler asks the store for a page *and* its total;
        the CLI's retired local branch asked only for the page.  The count is
        not derived from the double's own page: ``count_sessions`` is called
        with the page-shaping keywords stripped, so deriving it would re-enter
        a ``list_summaries`` double that asserts on exactly those keywords.  A
        double that returns rows must therefore say how many the store holds,
        which is the honest thing for it to state anyway -- an ``empty``
        outcome over a non-empty page is a real defect this keeps visible.
        """
        return 0

    def count_search_sessions(self, query: str, **kwargs: object) -> int:
        """Report no ranked matches unless the double says otherwise."""
        return 0

    def begin_read_snapshot(self) -> None:
        return None

    def read_summary(self, session_id: str) -> Any:
        """Summarise whatever page this double serves for ``session_id``.

        ``session.read`` reads the summary for the window's ``total`` and
        stops when the window covers it.  Deriving that from the double's own
        page is what makes a single-window double render as a *complete*
        transcript instead of looping forever on a continuation.
        """
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveSessionSummary

        page = self.read_session_page(session_id, limit=_DOUBLE_PAGE_CEILING, offset=0)  # type: ignore[attr-defined]
        return ArchiveSessionSummary(
            session_id=session_id,
            native_id=page.native_id,
            origin=page.origin,
            title=page.title,
            created_at=page.created_at,
            updated_at=page.updated_at,
            message_count=len(page.messages),
            word_count=0,
            tags=(),
        )

    def __enter__(self) -> ArchiveStoreDouble:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        for name, member in vars(cls).items():
            if name.startswith("__"):
                continue
            func = member.__func__ if isinstance(member, (classmethod, staticmethod)) else member
            if not callable(func) or isinstance(member, (classmethod, staticmethod)):
                continue
            _check_method(cls.__qualname__, name, func)
            setattr(cls, name, _bind_to_production(cls.__qualname__, name, func))


def install_archive_store_double(
    monkeypatch: pytest.MonkeyPatch,
    store: ArchiveStoreDouble,
    *,
    target: str = _OPEN_EXISTING_TARGET,
) -> ArchiveStoreDouble:
    """Route ``ArchiveStore.open_existing`` to ``store`` for one test.

    The replacement accepts exactly the production ``open_existing`` signature,
    so a call the real classmethod could not serve fails here too, and a new
    production keyword needs no edit in any test.
    """

    if not isinstance(store, ArchiveStoreDouble):
        raise TypeError("install_archive_store_double requires an ArchiveStoreDouble instance")
    signature = inspect.signature(ArchiveStore.open_existing)

    def _open_existing(_cls: type[ArchiveStore], /, *args: object, **kwargs: object) -> ArchiveStoreDouble:
        bound = signature.bind(*args, **kwargs)
        pinned = bound.arguments.get("index_path")
        store.bind_archive_root(
            Path(str(bound.arguments["archive_root"])),
            None if pinned is None else Path(str(pinned)),
        )
        return store

    monkeypatch.setattr(target, classmethod(_open_existing))
    return store
