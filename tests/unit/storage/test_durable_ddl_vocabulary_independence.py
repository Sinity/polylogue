"""Durable-tier DDL is independent of every persisted vocabulary's size.

``source.db``, ``user.db`` and ``audit.db`` evolve only by additive numbered
migration behind a verified backup. If any of their DDL rendered an enum's
members, adding a token to that enum would be a durable migration. Membership
is validated at the write boundary instead (see
``test_vocabulary_write_boundary.py``), so the DDL must not move when a
vocabulary grows.

This is the rendering property, not a scan of the DDL text: a hand-typed list
that happens to spell a vocabulary does not vary with the enum and so passes
here. ``devtools gate durable-enum-checks`` owns that failure, with the two
measured waivers.

Anti-vacuity: the ``check()``-rendered sentinel module travels the identical
extend-then-re-render path and must change. If the extension does not reach
the freshly executed module, the sentinel is unchanged and every assertion
below is meaningless -- so the sentinel assertion runs first, in the same test.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest

from polylogue.core import enums as enums_module
from polylogue.core.enums import PolylogueStrEnum

#: Durable tier module -> the DDL constant it renders at import time.
DURABLE_TIER_DDL: dict[str, str] = {
    "polylogue.storage.sqlite.archive_tiers.source": "SOURCE_DDL",
    "polylogue.storage.sqlite.archive_tiers.user": "USER_DDL",
    "polylogue.storage.sqlite.archive_tiers.audit": "AUDIT_DDL",
}

#: A member value no vocabulary declares, so its appearance in rendered DDL is
#: unambiguous evidence that the DDL varies with the enum.
SYNTHETIC_MEMBER_VALUE = "synthetic-vocabulary-member"

_SENTINEL_MODULE_SOURCE = """
from polylogue.core.enums import Origin
from polylogue.storage.sqlite.archive_tiers.common import check

SENTINEL_DDL = f"CREATE TABLE sentinel (origin TEXT NOT NULL CHECK ({check('origin', Origin)})) STRICT;"
"""


def _persisted_vocabularies() -> tuple[tuple[str, type[PolylogueStrEnum]], ...]:
    """Every closed vocabulary ``polylogue.core.enums`` publishes, by binding name.

    Keyed on the binding rather than ``__name__`` so a re-exported alias is
    extended under the name a durable module would import it by.
    """
    return tuple(
        (name, value)
        for name, value in vars(enums_module).items()
        if isinstance(value, type)
        and issubclass(value, PolylogueStrEnum)
        and value is not PolylogueStrEnum
        and len(value) > 0
    )


def _extended(vocabulary: type[PolylogueStrEnum]) -> type[PolylogueStrEnum]:
    """Return *vocabulary* plus one synthetic member, as a token addition would."""
    members = {member.name: member.value for member in vocabulary}
    members["SYNTHETIC_VOCABULARY_MEMBER"] = SYNTHETIC_MEMBER_VALUE
    extended = PolylogueStrEnum(vocabulary.__name__, members)  # type: ignore[call-arg]
    return cast("type[PolylogueStrEnum]", extended)


def _execute_fresh(module_name: str, origin_path: str) -> ModuleType:
    """Execute *origin_path* as a throwaway module, resolving imports live.

    Deliberately not registered in ``sys.modules``: the real module graph must
    keep its own instances, while this copy re-runs the import-time rendering
    against whatever the enums module currently holds.
    """
    spec = importlib.util.spec_from_file_location(f"_vocabulary_probe_{module_name}", origin_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _durable_tier_sources() -> dict[str, tuple[str, str]]:
    """Map each durable tier module to its ``(file path, DDL attribute)``."""
    sources: dict[str, tuple[str, str]] = {}
    for module_name, attribute in DURABLE_TIER_DDL.items():
        spec = importlib.util.find_spec(module_name)
        assert spec is not None and spec.origin is not None
        sources[module_name] = (spec.origin, attribute)
    return sources


def test_durable_ddl_does_not_change_when_every_vocabulary_grows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Adding a member to any persisted vocabulary renders byte-identical durable DDL."""
    sentinel_path = tmp_path / "sentinel_ddl.py"
    sentinel_path.write_text(_SENTINEL_MODULE_SOURCE)
    sources = _durable_tier_sources()

    before_sentinel = _execute_fresh("sentinel", str(sentinel_path)).SENTINEL_DDL
    before = {
        module_name: getattr(_execute_fresh(module_name, path), attribute)
        for module_name, (path, attribute) in sources.items()
    }

    vocabularies = _persisted_vocabularies()
    assert vocabularies, "polylogue.core.enums publishes no vocabulary; the probe would be vacuous"
    for name, vocabulary in vocabularies:
        monkeypatch.setattr(enums_module, name, _extended(vocabulary))

    after_sentinel = _execute_fresh("sentinel", str(sentinel_path)).SENTINEL_DDL
    assert SYNTHETIC_MEMBER_VALUE in after_sentinel, (
        "the extended vocabulary never reached a freshly executed module, so this test proves nothing"
    )
    assert after_sentinel != before_sentinel

    after = {
        module_name: getattr(_execute_fresh(module_name, path), attribute)
        for module_name, (path, attribute) in sources.items()
    }
    assert after == before
