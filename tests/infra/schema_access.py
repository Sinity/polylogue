"""Test helpers for narrowing generated schema payloads."""

from __future__ import annotations

import os
from typing import NoReturn

import pytest

from polylogue.core.json import JSONDocument, JSONValue, json_document

#: Escape hatch for lanes that intentionally exclude packaged schema data
#: (e.g. a distribution/installed-package smoke test against a wheel built
#: without the ``polylogue/schemas/**/providers`` data files). In an ordinary
#: repo checkout, packaged provider schemas are always present, so a missing
#: schema there is real breakage (deleted/renamed schema file, packaging
#: error), never a legitimate environmental variation.
ALLOW_MISSING_SCHEMAS_ENV = "POLYLOGUE_TEST_ALLOW_MISSING_SCHEMAS"


def fail_missing_schema(message: str) -> NoReturn:
    """Fail loudly on a missing packaged provider schema, in a normal checkout.

    Packaged provider schemas ship in-repo and must always be present. A
    missing schema can only mean something is broken. This raises a hard
    test failure naming what's missing, unless
    ``POLYLOGUE_TEST_ALLOW_MISSING_SCHEMAS=1`` is set, in which case it skips
    instead -- reserved for lanes that intentionally run without schema data.
    """
    if os.environ.get(ALLOW_MISSING_SCHEMAS_ENV) == "1":
        pytest.skip(message)
    pytest.fail(
        f"{message} -- packaged provider schemas are always present in a full "
        "repo checkout; a missing schema here means something is broken "
        "(deleted/renamed schema file, packaging error), not an environmental "
        f"limitation. Set {ALLOW_MISSING_SCHEMAS_ENV}=1 to allow a skip in "
        "contexts that intentionally exclude schema data (e.g. distribution "
        "smoke tests against an installed wheel)."
    )


def schema_node(value: object) -> JSONDocument:
    return json_document(value)


def schema_properties(schema: object) -> JSONDocument:
    return json_document(schema_node(schema).get("properties"))


def schema_property(schema: object, name: str) -> JSONDocument:
    return json_document(schema_properties(schema).get(name))


def schema_items(schema: object) -> JSONDocument:
    return json_document(schema_node(schema).get("items"))


def schema_values(schema: object) -> list[JSONValue]:
    values = schema_node(schema).get("x-polylogue-values")
    return list(values) if isinstance(values, list) else []


__all__ = [
    "ALLOW_MISSING_SCHEMAS_ENV",
    "fail_missing_schema",
    "schema_items",
    "schema_node",
    "schema_properties",
    "schema_property",
    "schema_values",
]
