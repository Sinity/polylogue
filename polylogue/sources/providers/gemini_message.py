"""Gemini message model and viewport extraction."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from polylogue.lib.roles import Role
from polylogue.lib.timestamps import parse_timestamp
from polylogue.lib.viewports import (
    ContentBlock,
    ContentType,
    MessageMeta,
    ReasoningTrace,
    TokenUsage,
    ToolCall,
)
from polylogue.types import Provider

from .gemini_models import GeminiBranchParent, GeminiGrounding, GeminiPart, GeminiThoughtSignature


class GeminiMessage(BaseModel):
    """A single Gemini AI Studio message."""

    model_config = ConfigDict(extra="allow")

    text: str = ""
    role: str
    createTime: str | None = None
    timestamp: str | None = None
    tokenCount: int | None = None
    finishReason: str | None = None
    isThought: bool = False
    thinkingBudget: int | None = None
    thoughtSignatures: list[GeminiThoughtSignature | dict[str, Any] | str] = Field(default_factory=list)
    parts: list[GeminiPart | dict[str, Any]] = Field(default_factory=list)
    grounding: GeminiGrounding | dict[str, Any] | None = None
    branchParent: GeminiBranchParent | dict[str, Any] | None = None
    branchChildren: list[dict[str, Any]] = Field(default_factory=list)
    safetyRatings: list[dict[str, Any]] = Field(default_factory=list)
    executableCode: dict[str, Any] | None = None
    codeExecutionResult: dict[str, Any] | None = None
    driveDocument: dict[str, Any] | str | None = None
    inlineFile: dict[str, Any] | None = None
    youtubeVideo: dict[str, Any] | None = None
    errorMessage: str | None = None
    isEdited: bool = False

    @property
    def role_normalized(self) -> Role:
        role = self.role if self.role else "unknown"
        try:
            return Role.normalize(role)
        except ValueError:
            return Role.UNKNOWN

    @property
    def parsed_timestamp(self) -> datetime | None:
        return parse_timestamp(self.createTime or self.timestamp)

    @property
    def text_content(self) -> str:
        if self.text:
            return self.text

        texts = []
        for part in self.parts:
            if isinstance(part, GeminiPart) and part.text:
                texts.append(part.text)
            elif isinstance(part, dict) and part.get("text"):
                val = part["text"]
                texts.append(val if isinstance(val, str) else str(val))
        return "\n".join(texts)

    def to_meta(self) -> MessageMeta:
        tokens = None
        if self.tokenCount is not None:
            tokens = TokenUsage(output_tokens=self.tokenCount)

        return MessageMeta(
            timestamp=self.parsed_timestamp,
            role=self.role_normalized,
            tokens=tokens,
            provider=Provider.GEMINI,
        )

    def extract_reasoning_traces(self) -> list[ReasoningTrace]:
        traces = []
        if self.isThought and self.text:
            sigs: list[str | dict[str, Any]] = []
            for signature in self.thoughtSignatures:
                if isinstance(signature, str):
                    sigs.append(signature)
                elif isinstance(signature, BaseModel):
                    sigs.append(signature.model_dump())
                else:
                    sigs.append(signature)

            traces.append(
                ReasoningTrace(
                    text=self.text,
                    token_count=self.thinkingBudget,
                    provider=Provider.GEMINI,
                    raw={
                        "isThought": True,
                        "thinkingBudget": self.thinkingBudget,
                        "thoughtSignatures": sigs,
                    },
                )
            )
        return traces

    def extract_content_blocks(self) -> list[ContentBlock]:
        blocks = []

        if self.isThought:
            blocks.append(
                ContentBlock(
                    type=ContentType.THINKING,
                    text=self.text,
                    raw={"isThought": True},
                )
            )
        elif self.text:
            blocks.append(
                ContentBlock(
                    type=ContentType.TEXT,
                    text=self.text,
                    raw={"role": self.role},
                )
            )

        for part in self.parts:
            if isinstance(part, GeminiPart):
                if part.text:
                    blocks.append(
                        ContentBlock(
                            type=ContentType.TEXT,
                            text=part.text,
                            raw=part.model_dump(),
                        )
                    )
                elif getattr(part, "inlineData", None) is not None or getattr(part, "fileData", None) is not None:
                    blocks.append(
                        ContentBlock(
                            type=ContentType.FILE,
                            raw=part.model_dump(),
                        )
                    )
            elif isinstance(part, dict):
                if part.get("text"):
                    blocks.append(
                        ContentBlock(
                            type=ContentType.TEXT,
                            text=part["text"],
                            raw=part,
                        )
                    )
                elif "inlineData" in part or "fileData" in part:
                    blocks.append(
                        ContentBlock(
                            type=ContentType.FILE,
                            raw=part,
                        )
                    )

        if self.executableCode:
            language = self.executableCode.get("language", "")
            code = self.executableCode.get("code", "")
            if code:
                blocks.append(
                    ContentBlock(
                        type=ContentType.CODE,
                        text=code,
                        language=language if isinstance(language, str) else None,
                        raw=self.executableCode,
                    )
                )

        if self.codeExecutionResult:
            outcome = self.codeExecutionResult.get("outcome", "")
            output = self.codeExecutionResult.get("output", "")
            if output or outcome:
                blocks.append(
                    ContentBlock(
                        type=ContentType.TOOL_RESULT,
                        text=str(output) if output else f"[{outcome}]",
                        raw=self.codeExecutionResult,
                    )
                )

        if self.driveDocument:
            raw_doc = self.driveDocument if isinstance(self.driveDocument, dict) else {"id": self.driveDocument}
            blocks.append(
                ContentBlock(
                    type=ContentType.FILE,
                    raw={"driveDocument": raw_doc},
                )
            )

        if self.inlineFile:
            mime_type = self.inlineFile.get("mimeType")
            inline_raw = dict(self.inlineFile)
            inline_raw.pop("data", None)
            blocks.append(
                ContentBlock(
                    type=ContentType.FILE,
                    mime_type=mime_type if isinstance(mime_type, str) else None,
                    raw={"inlineFile": inline_raw},
                )
            )

        if self.youtubeVideo:
            video_id = self.youtubeVideo.get("id")
            url = f"https://www.youtube.com/watch?v={video_id}" if isinstance(video_id, str) and video_id else None
            blocks.append(
                ContentBlock(
                    type=ContentType.VIDEO,
                    url=url,
                    raw={"youtubeVideo": self.youtubeVideo},
                )
            )

        return blocks

    def extract_tool_calls(self) -> list[ToolCall]:
        return []
