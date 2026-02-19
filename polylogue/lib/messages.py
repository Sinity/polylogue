"""Eager message collection for conversation handling.

This module provides `MessageCollection`, an eager container for messages that
supports iteration, len(), boolean coercion, indexing, and Pydantic
serialization. All storage is async and eager — lazy/streaming loading is not
used in production.
"""

from __future__ import annotations

from collections.abc import Iterator, Sized
from typing import TYPE_CHECKING, Any

from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler
from pydantic_core import core_schema

if TYPE_CHECKING:
    from pydantic.json_schema import JsonSchemaValue

    from polylogue.lib.models import Message


class MessageCollection(Sized):
    """Eager message collection — always backed by a pre-loaded list.

    Supports standard sequence operations (len, bool, iteration, indexing)
    and integrates with Pydantic V2 for serialization.

    Usage::

        coll = MessageCollection(messages=[msg1, msg2])
        for msg in coll:
            print(msg.text)
        print(len(coll))            # O(1)
        first = coll[0]             # Direct index
        all_msgs = coll.to_list()   # Returns a copy
    """

    def __init__(self, *, messages: list[Message]) -> None:
        self._messages: list[Message] = messages

    @property
    def is_lazy(self) -> bool:
        """Always False — collections are always eager."""
        return False

    def __iter__(self) -> Iterator[Message]:
        yield from self._messages

    def __len__(self) -> int:
        return len(self._messages)

    def __getitem__(self, index: int | slice) -> Message | list[Message]:
        return self._messages[index]

    def __bool__(self) -> bool:
        return len(self._messages) > 0

    def __repr__(self) -> str:
        return f"MessageCollection(eager, {len(self._messages)} messages)"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MessageCollection):
            return NotImplemented
        return self.to_list() == other.to_list()

    def __hash__(self) -> int:
        return id(self)

    def to_list(self) -> list[Message]:
        """Return a copy of the messages list."""
        return list(self._messages)

    def materialize(self) -> MessageCollection:
        """No-op — collection is always eager. Returns self for chaining."""
        return self

    @classmethod
    def empty(cls) -> MessageCollection:
        """Create an empty MessageCollection."""
        return cls(messages=[])

    # --- Pydantic serialization support ---

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        """Pydantic V2 schema: serialize as list[Message], accept MessageCollection or list."""
        from polylogue.lib.models import Message

        message_schema = _handler.generate_schema(Message)

        def validate_messages(v: Any) -> MessageCollection:
            if isinstance(v, MessageCollection):
                return v
            if isinstance(v, list):
                return MessageCollection(messages=v)
            raise ValueError(f"Expected MessageCollection or list, got {type(v)}")

        def serialize_messages(v: MessageCollection) -> list[Any]:
            return v.to_list()

        return core_schema.no_info_plain_validator_function(
            validate_messages,
            serialization=core_schema.plain_serializer_function_ser_schema(
                serialize_messages,
                info_arg=False,
                return_schema=core_schema.list_schema(message_schema),
            ),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        _core_schema: core_schema.CoreSchema,
        handler: GetJsonSchemaHandler,
    ) -> JsonSchemaValue:
        """JSON schema: array of Message objects."""
        from polylogue.lib.models import Message

        return {"type": "array", "items": handler.resolve_ref_schema(handler.generate(Message))}  # type: ignore[attr-defined]


__all__ = ["MessageCollection"]
