"""Immutable JSON answers shared by the Python client."""

from dataclasses import dataclass
from types import MappingProxyType
from collections.abc import Mapping

__all__ = ["Answer"]


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _copy(value):
    if isinstance(value, Mapping):
        return {key: _copy(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_copy(item) for item in value]
    return value


@dataclass(frozen=True)
class Answer:
    """A daemon answer, with nulls, integer precision and every field kept."""

    data: object

    def __post_init__(self):
        object.__setattr__(self, "data", _freeze(self.data))

    def json(self):
        """An independent, ordinary JSON value for export or comparison."""
        return _copy(self.data)
