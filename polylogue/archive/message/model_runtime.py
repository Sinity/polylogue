"""Runtime behavior helpers for ``Message`` models."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from datetime import datetime
from functools import cached_property
from typing import TYPE_CHECKING

from polylogue.archive.attachment.models import Attachment
from polylogue.archive.message.artifacts import classify_text_message_type
from polylogue.archive.message.roles import Role
from polylogue.archive.message.types import MessageType
from polylogue.core.json import JSONDocument, json_document, json_document_list
from polylogue.logging import get_logger
from polylogue.types import Provider

if TYPE_CHECKING:
    from polylogue.schemas.unified.models import HarmonizedMessage


def _coerce_optional_float(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _coerce_optional_int(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return None
    return None


def _mapping(value: object) -> Mapping[str, object] | None:
    return value if isinstance(value, Mapping) else None


def _provider_meta_record(provider_meta: dict[str, object] | None) -> JSONDocument:
    return json_document(provider_meta)


def _content_block_documents(value: object) -> list[JSONDocument]:
    return json_document_list(value)


def _block_texts(blocks: Iterable[Mapping[str, object]], *, block_type: str) -> list[str]:
    texts: list[str] = []
    for block in blocks:
        if block.get("type") != block_type:
            continue
        text = block.get("text")
        if isinstance(text, str) and text:
            texts.append(text)
    return texts


logger = get_logger(__name__)


class MessageRuntimeMixin:
    id: str
    role: Role
    text: str | None
    timestamp: datetime | None
    provider: Provider | None
    attachments: list[Attachment]
    provider_meta: dict[str, object] | None
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
    def harmonized(self) -> HarmonizedMessage | None:
        if self.provider is None or self.provider_meta is None:
            return None
        try:
            from polylogue.schemas.unified.unified import extract_from_provider_meta

            return extract_from_provider_meta(
                self.provider,
                _provider_meta_record(self.provider_meta),
                message_id=self.id,
                role=self.role,
                text=self.text,
                timestamp=self.timestamp,
            )
        except Exception:
            logger.warning(
                "Failed to extract harmonized viewports for message %s (provider=%s)",
                self.id,
                self.provider,
                exc_info=True,
            )
            return None

    def _is_chatgpt_thinking(self) -> bool:
        """ChatGPT-specific thinking detection via provider_meta.

        PARSER-ONLY compatibility path. After hydration, provider_meta is
        None and this always returns False. Hydrated messages have ChatGPT
        thinking preserved as THINKING-type content blocks, which
        ``is_thinking`` checks directly.
        """
        if self.provider_meta is None:
            return False
        provider_meta = _provider_meta_record(self.provider_meta)
        raw = json_document(provider_meta.get("raw"))
        if not raw:
            return False

        content = json_document(raw.get("content"))
        if content:
            content_type = content.get("content_type")
            if isinstance(content_type, str) and content_type in {"thoughts", "reasoning_recap"}:
                return True

        if self.role == Role.TOOL:
            metadata = _mapping(raw.get("metadata"))
            if metadata is not None and "finished_text" in metadata:
                return True

        return False

    @cached_property
    def is_tool_use(self) -> bool:
        message_type = MessageType.normalize(getattr(self, "message_type", MessageType.MESSAGE))
        if message_type in {MessageType.TOOL_USE, MessageType.TOOL_RESULT}:
            return True

        if any(block.get("type") in {"tool_use", "tool_result"} for block in self.content_blocks):
            return True

        # PARSER-ONLY: provider_meta fallbacks are unreachable for hydrated
        # messages (provider_meta is None after hydration). They serve
        # direct parser consumers with provider_meta populated.
        provider_meta = self.provider_meta
        if provider_meta is not None:
            provider_meta_record = _provider_meta_record(provider_meta)
            if any(
                block.get("type") in {"tool_use", "tool_result"}
                for block in _content_block_documents(provider_meta_record.get("content_blocks"))
            ):
                return True
            if bool(provider_meta_record.get("isSidechain")) or bool(provider_meta_record.get("isMeta")):
                return True
            harmonized = self.harmonized
            if harmonized is not None and harmonized.tool_calls:
                return True

        if self.role == Role.TOOL:
            # ChatGPT thinking messages use role=TOOL; avoid misclassifying
            # them as tool_use. Check canonical content_blocks first
            # (works for hydrated messages), then fall back to the
            # parser-only provider_meta heuristic.
            if any(block.get("type") == "thinking" for block in self.content_blocks):
                return False
            return not self._is_chatgpt_thinking()

        return False

    @cached_property
    def is_thinking(self) -> bool:
        if any(block.get("type") == "thinking" for block in self.content_blocks):
            return True

        # PARSER-ONLY: these fallbacks read provider_meta which is None for
        # hydrated messages. The canonical content_blocks check above covers
        # all hydrated cases.
        provider_meta = self.provider_meta
        if provider_meta is not None:
            provider_meta_record = _provider_meta_record(provider_meta)
            if any(
                block.get("type") == "thinking"
                for block in _content_block_documents(provider_meta_record.get("content_blocks"))
            ):
                return True
            if bool(provider_meta_record.get("isThought")):
                return True
            raw = json_document(provider_meta_record.get("raw"))
            if raw and bool(raw.get("isThought")):
                return True
            harmonized = self.harmonized
            if harmonized is not None and harmonized.reasoning_traces:
                return True

        return self._is_chatgpt_thinking()

    @cached_property
    def is_context_dump(self) -> bool:
        # Authoritative: stored message_type from materialization.
        if self.message_type == MessageType.CONTEXT:
            return True

        # Resilience: text-based classification for pre-#839 archive rows.
        text = self.text
        if not text:
            return False
        if classify_text_message_type(text) == MessageType.CONTEXT:
            return True
        # Narrow fallbacks that require context beyond message text.
        if self.attachments and len(text) < 100:
            return True
        return bool("```" in text and text.count("```") >= 6)

    @cached_property
    def is_protocol_artifact(self) -> bool:
        # Authoritative: stored message_type from materialization.
        if self.message_type == MessageType.PROTOCOL:
            return True
        # Resilience: text-based classification for pre-#839 archive rows.
        return classify_text_message_type(self.text) == MessageType.PROTOCOL

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

    @property
    def cost_usd(self) -> float | None:
        """PARSER-ONLY: reads provider_meta, None for hydrated messages.

        Hydrated-message cost data is provided by #803's typed model.
        """
        provider_meta = self.provider_meta
        if provider_meta is None:
            return None
        provider_meta_record = _provider_meta_record(provider_meta)
        raw = json_document(provider_meta_record.get("raw")) or provider_meta_record
        return _coerce_optional_float(raw.get("costUSD"))

    @property
    def duration_ms(self) -> int | None:
        """PARSER-ONLY: reads provider_meta, None for hydrated messages."""
        provider_meta = self.provider_meta
        if provider_meta is None:
            return None
        provider_meta_record = _provider_meta_record(provider_meta)
        raw = json_document(provider_meta_record.get("raw")) or provider_meta_record
        return _coerce_optional_int(raw.get("durationMs"))

    def extract_thinking(self) -> str | None:
        direct_texts = _block_texts(self.content_blocks, block_type="thinking")
        if direct_texts:
            return "\n\n".join(direct_texts).strip() or None

        harmonized = self.harmonized
        if harmonized is not None and harmonized.reasoning_traces:
            reasoning_texts = [trace.text for trace in harmonized.reasoning_traces if trace.text]
            if reasoning_texts:
                return "\n\n".join(reasoning_texts).strip() or None

        provider_meta = self.provider_meta
        if provider_meta is not None:
            provider_meta_record = _provider_meta_record(provider_meta)
            provider_texts = _block_texts(
                _content_block_documents(provider_meta_record.get("content_blocks")),
                block_type="thinking",
            )
            if provider_texts:
                return "\n\n".join(provider_texts).strip() or None

        text = self.text
        if text:
            match = re.search(r"<(?:antml:)?thinking>(.*?)</(?:antml:)?thinking>", text, re.DOTALL)
            if match:
                return match.group(1).strip()

        if text and (
            self._is_chatgpt_thinking()
            or (provider_meta is not None and bool(_provider_meta_record(provider_meta).get("isThought")))
        ):
            return text.strip() or None

        return None


__all__ = ["MessageRuntimeMixin"]
