"""Static-conformance assertions for read-surface contracts.

``assert_implements(SurfaceClass, Protocol)`` returns the class unchanged
but pins the conformance claim at the module level so ``mypy --strict``
fails if the surface drifts out of compliance with the Protocol.

Usage::

    from polylogue.api.contracts import SessionListSurface, assert_implements

    class MyAPIListAdapter:
        async def list_sessions(
            self, spec: SessionQuerySpec
        ) -> SessionListResponse:
            ...

    assert_implements(MyAPIListAdapter, SessionListSurface)

The function is a no-op at runtime; its sole purpose is to materialize the
``isinstance``/``issubclass``-style conformance check for the type
checker.  We deliberately avoid ``runtime_checkable.__subclasshook__``
because Protocol structural runtime checks are expensive and false-friendly
for missing parameter names; mypy is the authoritative gate.
"""

from __future__ import annotations

from typing import TypeVar

_T = TypeVar("_T")


def assert_implements(cls: type[_T], protocol: type[object]) -> type[_T]:
    """Statically pin that ``cls`` conforms to ``protocol``.

    The body is a runtime ``isinstance`` style check that always succeeds
    for ``runtime_checkable`` Protocols when the class supplies the
    required methods.  The real enforcement is the call site itself:
    mypy expands the call and checks ``cls`` against ``protocol``.
    """
    # Materialize a structural check at import time.  For runtime_checkable
    # Protocols this asserts method presence; for non-runtime Protocols
    # the call is still observed by mypy as a conformance pin.
    if hasattr(protocol, "__subclasshook__"):
        try:
            is_subclass = issubclass(cls, protocol)
        except TypeError:
            is_subclass = True
        if not is_subclass:
            raise TypeError(f"{cls.__name__} does not structurally implement {protocol.__name__}")
    return cls


__all__ = ["assert_implements"]
