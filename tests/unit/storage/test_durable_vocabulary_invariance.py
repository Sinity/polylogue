"""Adding a vocabulary member must change zero durable DDL.

The durable-enum-checks gate tests a *signature* of this property: it flags a
durable literal list whose member set equals some enum's value set. That is
unevadable but it cannot tell a real regression from a hand-written list that
coincides, which is why it carries waivers. This file tests the property
itself — render the durable DDL, extend every persisted vocabulary with a
synthetic member, re-render, assert byte equality — so an enum-derived CHECK
reaching durable DDL by any route (``check()``, an inlined ``sql_check_in``,
an f-string over enum values) is red with no waiver consulted.
"""

from __future__ import annotations

import importlib
import inspect
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import cast

from polylogue.core import enums as enums_module
from polylogue.core.enums import PolylogueStrEnum, sql_check_in

_DURABLE_DDL_MODULES = {
    "polylogue.storage.sqlite.archive_tiers.source": "SOURCE_DDL",
    "polylogue.storage.sqlite.archive_tiers.user": "USER_DDL",
    "polylogue.storage.sqlite.archive_tiers.audit": "AUDIT_DDL",
}

_SYNTHETIC_MEMBER = ("SYNTHETIC_VOCABULARY_PROBE", "synthetic-vocabulary-probe")


def _persisted_vocabularies() -> dict[tuple[str, str], type[PolylogueStrEnum]]:
    """Every loaded ``PolylogueStrEnum`` subclass, keyed by defining module and name."""
    found: dict[tuple[str, str], type[PolylogueStrEnum]] = {}
    for module_name, module in list(sys.modules.items()):
        if not module_name.startswith("polylogue") or module is None:
            continue
        for attribute, value in list(vars(module).items()):
            if (
                inspect.isclass(value)
                and issubclass(value, PolylogueStrEnum)
                and value is not PolylogueStrEnum
                and value.__module__ == module_name
            ):
                found[(module_name, attribute)] = value
    return found


def _extended(enum_type: type[PolylogueStrEnum]) -> type[PolylogueStrEnum]:
    members = {member.name: member.value for member in enum_type}
    members[_SYNTHETIC_MEMBER[0]] = _SYNTHETIC_MEMBER[1]
    extended = PolylogueStrEnum(enum_type.__name__, members)  # type: ignore[call-arg]
    return cast(type[PolylogueStrEnum], extended)


@contextmanager
def _every_vocabulary_extended() -> Iterator[None]:
    """Extend every persisted vocabulary, leaving the real modules untouched after."""
    for module_name in _DURABLE_DDL_MODULES:
        importlib.import_module(module_name)
    originals = _persisted_vocabularies()
    saved_modules = {name: sys.modules.pop(name) for name in _DURABLE_DDL_MODULES}
    try:
        for (module_name, attribute), enum_type in originals.items():
            setattr(sys.modules[module_name], attribute, _extended(enum_type))
        yield
    finally:
        for (module_name, attribute), enum_type in originals.items():
            setattr(sys.modules[module_name], attribute, enum_type)
        for name, module in saved_modules.items():
            sys.modules[name] = module


def _render_durable_ddl() -> dict[str, str]:
    return {
        module_name: getattr(importlib.import_module(module_name), attribute)
        for module_name, attribute in _DURABLE_DDL_MODULES.items()
    }


def test_extending_every_vocabulary_changes_no_durable_ddl() -> None:
    """Anti-vacuity: put ``check(column, SomeEnum)`` on any durable column and this goes red."""
    before = _render_durable_ddl()

    with _every_vocabulary_extended():
        after = _render_durable_ddl()

    assert after == before


def test_the_probe_reaches_the_renderers_it_claims_to_cover() -> None:
    """The extension must actually move a rendered CHECK, or the invariance test is vacuous."""
    with _every_vocabulary_extended():
        rendered = sql_check_in("status", enums_module.OperationStatus)

    assert _SYNTHETIC_MEMBER[1] in rendered


def test_the_durable_ddl_modules_are_restored_after_probing() -> None:
    """The probe cannot leak a synthetic vocabulary into the rest of the session."""
    before = _render_durable_ddl()

    with _every_vocabulary_extended():
        pass

    assert _render_durable_ddl() == before
    assert _SYNTHETIC_MEMBER[1] not in sql_check_in("status", enums_module.OperationStatus)
