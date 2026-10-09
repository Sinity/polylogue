"""Explicit eager fixture boundary for values borrowed from JSON owners."""

from polylogue.core.json import JSONValue


def materialize_json(value: JSONValue) -> JSONValue:
    if isinstance(value, dict):
        return {key: materialize_json(child) for key, child in value.items()}
    if isinstance(value, list):
        return [materialize_json(child) for child in value]
    return value
