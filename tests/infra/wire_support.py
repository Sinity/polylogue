"""One full-catalog wire-support receipt per worker process.

``build_wire_support_receipt`` runs the synthetic corpus generator and the
production parser over every catalogued package element. That is ~27s of work
in a quiet process, and ``verify --all`` runs eight workers, so each test that
builds the whole catalog for itself pays it again under contention -- which is
what pushed this family past its 120s timeout.

The receipt is a pure function of the registry it reads, so a test that only
reads one can share it. A test whose subject is the *building* (determinism,
catalog ordering, a mutated route or handler) must still build its own.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polylogue.schemas.synthetic.wire_formats import WireSupportReceipt

__all__ = ["shared_wire_support_receipt"]


@lru_cache(maxsize=4)
def _receipt(storage_root: str) -> WireSupportReceipt:
    from polylogue.schemas.runtime_registry import SchemaRegistry
    from polylogue.schemas.synthetic.wire_formats import build_wire_support_receipt

    return build_wire_support_receipt(registry=SchemaRegistry(storage_root=Path(storage_root)))


def shared_wire_support_receipt(*, storage_root: Path | None = None) -> WireSupportReceipt:
    """Return the full-catalog receipt for ``storage_root``, built once here.

    The default is the packaged catalog, never the ambient user schema
    directory, so every caller of the default shares one build.
    """
    from polylogue.schemas.runtime_registry import SCHEMA_DIR

    return _receipt(str(SCHEMA_DIR if storage_root is None else storage_root))
