"""Runtime behavior helpers for ``Message`` models."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from datetime import datetime
from functools import cached_property

from polylogue.archive.attachment.models import Attachment
from polylogue.archive.message.roles import Role
from polylogue.archive.message.types import MessageType
from polylogue.types import Provider


def _block_texts(blocks: Iterable[Mapping[str, object]], *, block_type: str) -> list[str]:
    texts: list[str] = []
    for block in blocks:
        if block.get("type") != block_type:
            continue
        text = block.get("text")
        if isinstance(text, str) and text:
            texts.append(text)
    return texts


class MessageRuntimeMixin:
    """Runtime behavior for the hydrated domain ``Message``.

    Canonical archive storage keeps message semantics in content-block rows
    and typed columns on ``messages``. Runtime decisions read content blocks,
    the persisted ``message_type``, and the precomputed ``has_*`` flags.
    Provider-shaped raw payloads remain available through ``raw_sessions`` /
    the blob store, never through the hydrated domain model.
    """

    id: str
    role: Role
    text: str | None
    timestamp: datetime | None
    provider: Provider | None
    attachments: list[Attachment]
    content_blocks: list[dict[str, object]]
    message_type: MessageType
    parent_id: str | None
    branch_index: int

    @property
    def is_branch(self) -> bool:
        return self.branch_index > 0

    @property
    def is_user(self) -> bool:
        return self.role == Role.USER

    @property
    def is_assistant(self) -> bool:
        return self.role == Role.ASSISTANT

    @property
    def is_system(self) -> bool:
        return self.role == Role.SYSTEM

    @property
    def is_dialogue(self) -> bool:
        return self.is_user or self.is_assistant

    @cached_property
    def is_tool_use(self) -> bool:
        message_type = MessageType.normalize(getattr(self, "message_type", MessageType.MESSAGE))
        if message_type in {MessageType.TOOL_USE, MessageType.TOOL_RESULT}:
            return True

        if any(block.get("type") in {"tool_use", "tool_result"} for block in self.content_blocks):
            return True

        if self.role == Role.TOOL:
            # ChatGPT thinking messages use role=TOOL; the persisted
            # ``THINKING`` content block disambiguates them from real
            # tool calls. Pre-#839 rows without typed blocks are
            # reclassified through the backfill path, not at read time.
            return not any(block.get("type") == "thinking" for block in self.content_blocks)

        return False

    @cached_property
    def is_thinking(self) -> bool:
        return any(block.get("type") == "thinking" for block in self.content_blocks)

    @cached_property
    def is_context_dump(self) -> bool:
        """Returns True iff the persisted ``message_type`` is CONTEXT.

        Stored ``message_type`` is the only source of truth (issue #839 AC #3).
        Pre-#839 rows without a persisted CONTEXT type are not recognized at
        read time; backfill (issue #839 AC #2 / #971) reclassifies them.
        """
        return self.message_type == MessageType.CONTEXT

    @cached_property
    def is_protocol_artifact(self) -> bool:
        """Returns True iff the persisted ``message_type`` is PROTOCOL.

        Stored ``message_type`` is the only source of truth (issue #839 AC #3).
        """
        return self.message_type == MessageType.PROTOCOL

    @cached_property
    def is_noise(self) -> bool:
        return self.is_tool_use or self.is_context_dump or self.is_protocol_artifact or self.is_system

    @cached_property
    def is_substantive(self) -> bool:
        if not self.is_dialogue or self.is_noise or self.is_thinking:
            return False
        text = self.text
        return text is not None and len(text.strip()) > 10

    @cached_property
    def word_count(self) -> int:
        text = self.text
        if not text:
            return 0
        return len(text.split())

    def extract_thinking(self) -> str | None:
        direct_texts = _block_texts(self.content_blocks, block_type="thinking")
        if direct_texts:
            return "\n\n".join(direct_texts).strip() or None

        text = self.text
        if text:
            match = re.search(r"<(?:antml:)?thinking>(.*?)</(?:antml:)?thinking>", text, re.DOTALL)
            if match:
                return match.group(1).strip()

        return None


__all__ = ["MessageRuntimeMixin"]
