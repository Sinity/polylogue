"""Message and dialogue-pair domain models."""

from __future__ import annotations

import re
from datetime import datetime
from functools import cached_property

from pydantic import BaseModel, Field, field_validator, model_validator

from polylogue.lib.attachment_models import Attachment
from polylogue.lib.model_support import (
    _CONTEXT_PATTERNS,
    _coerce_optional_float,
    _coerce_optional_int,
)
from polylogue.lib.roles import Role
from polylogue.logging import get_logger
from polylogue.types import Provider

logger = get_logger(__name__)


class Message(BaseModel):
    id: str
    role: Role
    text: str | None = None
    timestamp: datetime | None = None
    provider: Provider | None = None
    attachments: list[Attachment] = Field(default_factory=list)
    provider_meta: dict[str, object] | None = None
    content_blocks: list[dict[str, object]] = Field(default_factory=list)
    parent_id: str | None = None
    branch_index: int = 0

    @field_validator("role", mode="before")
    @classmethod
    def coerce_role(cls, v: object) -> Role:
        if isinstance(v, Role):
            return v
        raw = (str(v) if v is not None else "").strip() or "unknown"
        return Role.normalize(raw)

    @field_validator("provider", mode="before")
    @classmethod
    def coerce_provider(cls, v: object) -> Provider | None:
        if v is None:
            return None
        if isinstance(v, Provider):
            return v
        return Provider.from_string(str(v))

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
    def harmonized(self):
        """Harmonized viewports extracted from provider_meta when provider is known."""
        if not self.provider or not self.provider_meta:
            return None
        try:
            from polylogue.schemas.unified import extract_from_provider_meta

            return extract_from_provider_meta(
                self.provider,
                self.provider_meta,
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
        if not self.provider_meta:
            return False
        raw = self.provider_meta.get("raw", {})
        if not isinstance(raw, dict):
            return False

        content = raw.get("content", {})
        if isinstance(content, dict):
            content_type = content.get("content_type", "")
            if content_type in ("thoughts", "reasoning_recap"):
                return True

        if self.role == Role.TOOL:
            metadata = raw.get("metadata", {})
            if isinstance(metadata, dict) and "finished_text" in metadata:
                return True

        return False

    @property
    def is_tool_use(self) -> bool:
        if any(block.get("type") in ("tool_use", "tool_result") for block in self.content_blocks):
            return True

        if self.provider_meta:
            provider_meta = self.provider_meta
            if any(
                isinstance(block, dict) and block.get("type") in ("tool_use", "tool_result")
                for block in provider_meta.get("content_blocks") or []
            ):
                return True
            if provider_meta.get("isSidechain") or provider_meta.get("isMeta"):
                return True
            harmonized = self.harmonized
            if harmonized and harmonized.tool_calls:
                return True

        if self.role == Role.TOOL:
            return not self._is_chatgpt_thinking()

        return False

    @property
    def is_thinking(self) -> bool:
        if any(block.get("type") == "thinking" for block in self.content_blocks):
            return True

        if self.provider_meta:
            provider_meta = self.provider_meta
            if any(
                isinstance(block, dict) and block.get("type") == "thinking"
                for block in provider_meta.get("content_blocks") or []
            ):
                return True
            if provider_meta.get("isThought"):
                return True
            raw = provider_meta.get("raw")
            if isinstance(raw, dict) and raw.get("isThought"):
                return True
            harmonized = self.harmonized
            if harmonized and harmonized.reasoning_traces:
                return True

        return bool(self._is_chatgpt_thinking())

    @property
    def is_context_dump(self) -> bool:
        if not self.text:
            return False
        stripped = self.text.lstrip()
        if stripped.startswith(("<environment_context>", "<subagent_notification>", "<permissions instructions>")):
            return True
        if self.attachments and len(self.text) < 100:
            return True
        if "<system>" in self.text and "</system>" in self.text:
            return True
        if self.text.count("```") >= 6:
            return True
        return any(re.search(pattern, self.text, re.MULTILINE) for pattern in _CONTEXT_PATTERNS)

    @property
    def is_noise(self) -> bool:
        return self.is_tool_use or self.is_context_dump or self.is_system

    @property
    def is_substantive(self) -> bool:
        if not self.is_dialogue or self.is_noise or self.is_thinking:
            return False
        return bool(self.text and len(self.text.strip()) > 10)

    @property
    def word_count(self) -> int:
        if not self.text:
            return 0
        return len(self.text.split())

    @property
    def cost_usd(self) -> float | None:
        if not self.provider_meta:
            return None
        raw = self.provider_meta.get("raw", self.provider_meta)
        return _coerce_optional_float(raw.get("costUSD"))

    @property
    def duration_ms(self) -> int | None:
        if not self.provider_meta:
            return None
        raw = self.provider_meta.get("raw", self.provider_meta)
        return _coerce_optional_int(raw.get("durationMs"))

    def extract_thinking(self) -> str | None:
        db_texts = [
            block["text"]
            for block in self.content_blocks
            if block.get("type") == "thinking" and isinstance(block.get("text"), str)
        ]
        if db_texts:
            return "\n\n".join(db_texts).strip() or None

        harmonized = self.harmonized
        if harmonized and harmonized.reasoning_traces:
            texts = [trace.text for trace in harmonized.reasoning_traces if trace.text]
            if texts:
                return "\n\n".join(texts).strip() or None

        if self.provider_meta:
            blocks = self.provider_meta.get("content_blocks", [])
            if isinstance(blocks, list):
                thinking_texts = [
                    block["text"]
                    for block in blocks
                    if isinstance(block, dict) and block.get("type") == "thinking" and isinstance(block.get("text"), str)
                ]
                if thinking_texts:
                    return "\n\n".join(thinking_texts).strip() or None

        if self.text:
            match = re.search(r"<(?:antml:)?thinking>(.*?)</(?:antml:)?thinking>", self.text, re.DOTALL)
            if match:
                return match.group(1).strip()

        if self.text and (self._is_chatgpt_thinking() or (self.provider_meta and self.provider_meta.get("isThought"))):
            return self.text.strip() or None

        return None


class DialoguePair(BaseModel):
    """A user message followed by assistant response."""

    user: Message
    assistant: Message

    @model_validator(mode="after")
    def validate_roles(self) -> DialoguePair:
        if not self.user.is_user:
            raise ValueError(f"user message must have user role, got {self.user.role}")
        if not self.assistant.is_assistant:
            raise ValueError(f"assistant message must have assistant role, got {self.assistant.role}")
        return self

    @property
    def exchange(self) -> str:
        return f"User: {self.user.text or ''}\n\nAssistant: {self.assistant.text or ''}"


__all__ = ["DialoguePair", "Message"]
