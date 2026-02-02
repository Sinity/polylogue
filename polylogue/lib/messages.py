"""Lazy message collection for memory-efficient conversation handling.

This module provides `MessageCollection`, a lazy container for messages that:
- Streams from the database on iteration (never loads all at once)
- Uses COUNT(*) query for len() instead of loading messages
- Supports both lazy mode (from DB) and eager mode (from filter operations)

This architecture reduces memory usage from O(n) to O(1) for iteration,
critical for conversations with thousands of messages.
"""

from __future__ import annotations

from collections.abc import Iterator, Sized
from typing import TYPE_CHECKING, Annotated, Any, Protocol, overload

from pydantic import GetJsonSchemaHandler, GetCoreSchemaHandler
from pydantic_core import core_schema

if TYPE_CHECKING:
    from polylogue.lib.models import Message
    from pydantic.json_schema import JsonSchemaValue


class MessageSource(Protocol):
    """Protocol for lazy message loading.

    Implementations provide streaming access to messages and count queries
    without loading all messages into memory.
    """

    def iter_messages(self, conversation_id: str) -> Iterator[Message]:
        """Stream messages for a conversation.

        Args:
            conversation_id: ID of the conversation

        Yields:
            Message objects one at a time
        """
        ...

    def count_messages(self, conversation_id: str) -> int:
        """Get message count without loading messages.

        Args:
            conversation_id: ID of the conversation

        Returns:
            Total number of messages in the conversation
        """
        ...


class MessageCollection(Sized):
    """Lazy message collection - streams from DB, never loads all at once.

    This collection supports two modes:

    1. **Lazy mode** (default from repository):
       - Created with `MessageCollection(conversation_id=..., source=...)`
       - Iterates by streaming from the database
       - len() uses COUNT(*) query
       - Memory: O(1) regardless of conversation size

    2. **Eager mode** (from filter operations):
       - Created with `MessageCollection(messages=[...])`
       - Messages already loaded in memory
       - Iterates over the cached list
       - Memory: O(n) but necessary for filtered views

    Usage:
        # Iteration works in both modes (lazy streaming in lazy mode)
        for msg in conv.messages:
            print(msg.text)

        # len() is O(1) in lazy mode (COUNT query), O(1) in eager mode
        print(f"Conversation has {len(conv.messages)} messages")

        # Explicit materialization when needed
        all_messages = conv.messages.to_list()

        # Indexing materializes the list on first access (for backward compatibility)
        first_msg = conv.messages[0]  # Materializes if lazy

    Performance characteristics:
        - Iteration: O(1) memory in lazy mode, streams from DB
        - len(): O(1) in both modes (COUNT query vs len(list))
        - Indexing: Materializes on first access, then O(1)
        - to_list(): Always O(n) - explicit full load
    """

    @overload
    def __init__(self, *, conversation_id: str, source: MessageSource) -> None:
        """Create a lazy collection that streams from the database."""
        ...

    @overload
    def __init__(self, *, messages: list[Message]) -> None:
        """Create an eager collection with pre-loaded messages."""
        ...

    def __init__(
        self,
        *,
        conversation_id: str | None = None,
        source: MessageSource | None = None,
        messages: list[Message] | None = None,
    ) -> None:
        """Initialize MessageCollection in lazy or eager mode.

        Args:
            conversation_id: ID for lazy loading (requires source)
            source: MessageSource for lazy loading (requires conversation_id)
            messages: Pre-loaded messages for eager mode

        Raises:
            ValueError: If arguments are inconsistent
        """
        if messages is not None:
            # Eager mode - messages already loaded
            if conversation_id is not None or source is not None:
                raise ValueError("Cannot specify both messages and conversation_id/source")
            self._messages: list[Message] | None = messages
            self._conversation_id: str | None = None
            self._source: MessageSource | None = None
            self._is_lazy = False
        elif conversation_id is not None and source is not None:
            # Lazy mode - will stream from source
            self._messages = None
            self._conversation_id = conversation_id
            self._source = source
            self._is_lazy = True
        else:
            raise ValueError(
                "Must specify either messages=... for eager mode, "
                "or both conversation_id=... and source=... for lazy mode"
            )

        # Cached count for lazy mode (populated on first len() call)
        self._cached_count: int | None = None

    @property
    def is_lazy(self) -> bool:
        """True if this collection is in lazy mode (not yet materialized)."""
        return self._is_lazy and self._messages is None

    def __iter__(self) -> Iterator[Message]:
        """Iterate over messages.

        In lazy mode, streams from the database without loading all messages.
        In eager mode (or after materialization), iterates over the cached list.
        """
        if self._messages is not None:
            # Already materialized or eager mode
            yield from self._messages
        elif self._source is not None and self._conversation_id is not None:
            # Lazy mode - stream from source
            yield from self._source.iter_messages(self._conversation_id)
        else:
            # Edge case: empty collection
            return

    def __len__(self) -> int:
        """Get message count.

        In lazy mode, uses COUNT(*) query (O(1)).
        In eager mode, returns len(list) (O(1)).
        """
        if self._messages is not None:
            # Already materialized or eager mode
            return len(self._messages)

        # Lazy mode - use cached count or query
        if self._cached_count is not None:
            return self._cached_count

        if self._source is not None and self._conversation_id is not None:
            self._cached_count = self._source.count_messages(self._conversation_id)
            return self._cached_count

        return 0

    @overload
    def __getitem__(self, index: int) -> Message: ...

    @overload
    def __getitem__(self, index: slice) -> list[Message]: ...

    def __getitem__(self, index: int | slice) -> Message | list[Message]:
        """Access messages by index.

        WARNING: This materializes the entire collection on first access.
        For iteration, use `for msg in collection` instead.

        This is provided for backward compatibility with tests that use
        `conv.messages[0]` style access.
        """
        # Materialize if lazy
        if self._messages is None:
            self._messages = list(self)
            self._is_lazy = False

        return self._messages[index]

    def __bool__(self) -> bool:
        """Check if collection is non-empty.

        Uses efficient len() check which is O(1) in lazy mode.
        """
        return len(self) > 0

    def __repr__(self) -> str:
        mode = "lazy" if self.is_lazy else "eager"
        count = len(self)
        return f"MessageCollection({mode}, {count} messages)"

    def __eq__(self, other: object) -> bool:
        """Compare MessageCollections by their contents.

        Note: This materializes both collections if comparing lazy to lazy.
        """
        if not isinstance(other, MessageCollection):
            return NotImplemented
        # Compare by materializing to lists
        return self.to_list() == other.to_list()

    def __hash__(self) -> int:
        """Hash based on id (MessageCollection is mutable, use object id)."""
        return id(self)

    def to_list(self) -> list[Message]:
        """Explicitly materialize all messages to a list.

        Use this when you need the full list, e.g., for indexing,
        slicing, or passing to functions that expect a list.

        Returns:
            List of all Message objects
        """
        if self._messages is not None:
            return list(self._messages)  # Return a copy
        return list(self)

    def materialize(self) -> MessageCollection:
        """Materialize lazy collection and return self.

        After calling this, iteration will use the cached list
        instead of streaming from the database.

        Returns:
            self (for chaining)
        """
        if self._messages is None:
            self._messages = list(self)
            self._is_lazy = False
        return self

    @classmethod
    def empty(cls) -> MessageCollection:
        """Create an empty MessageCollection.

        Useful for error cases or placeholder values.
        """
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

        # Import Message schema handler for proper typing
        message_schema = _handler.generate_schema(Message)

        def validate_messages(v: Any) -> MessageCollection:
            """Validate input: accept list[Message] or MessageCollection."""
            if isinstance(v, MessageCollection):
                return v
            if isinstance(v, list):
                return MessageCollection(messages=v)
            raise ValueError(f"Expected MessageCollection or list, got {type(v)}")

        def serialize_messages(v: MessageCollection) -> list[Any]:
            """Serialize to list for JSON output."""
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

        # Return schema for list of messages
        return {"type": "array", "items": handler.resolve_ref_schema(handler.generate(Message))}


__all__ = ["MessageCollection", "MessageSource"]
