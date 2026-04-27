"""Gemini message model and viewport extraction."""

from __future__ import annotations

from datetime import datetime
from typing import TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from polylogue.lib.json import json_document
from polylogue.lib.roles import Role
from polylogue.lib.timestamps import parse_timestamp
from polylogue.lib.viewport.viewports import (
    ContentBlock,
    ContentType,
    MessageMeta,
    ReasoningTrace,
    TokenUsage,
    ToolCall,
)
from polylogue.types import Provider

from .gemini_models import GeminiBranchParent, GeminiGrounding, GeminiPart, GeminiThoughtSignature

ThoughtSignatureValue: TypeAlias = GeminiThoughtSignature | dict[str, object] | str
GeminiPartValue: TypeAlias = GeminiPart | dict[str, object]
GeminiDictValue: TypeAlias = dict[str, object]
GeminiBranchChildren: TypeAlias = list[GeminiDictValue]


def _json_object(value: object) -> GeminiDictValue:
    result: GeminiDictValue = {}
    result.update(json_document(value))
    return result


def _normalize_mapping(payload: BaseModel | GeminiDictValue) -> GeminiDictValue:
    if isinstance(payload, BaseModel):
        return _json_object(payload.model_dump())
    return payload


def _to_text(value: object) -> str | None:
    if isinstance(value, str):
        return value
    if value is None:
        return None
    return str(value)


def _part_text(part: GeminiPartValue) -> str | None:
    if isinstance(part, GeminiPart):
        return _get_str_or_none(part.text)
    return _to_text(part.get("text"))


def _part_has_file_payload(part: GeminiPartValue) -> bool:
    if isinstance(part, GeminiPart):
        return getattr(part, "inlineData", None) is not None or getattr(part, "fileData", None) is not None
    return "inlineData" in part or "fileData" in part


def _get_str_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _string_or_empty_dict(value: object) -> GeminiDictValue:
    return _json_object(value)


def _part_record(part: GeminiPartValue) -> GeminiDictValue:
    if isinstance(part, GeminiPart):
        return _json_object(part.model_dump())
    return part


def _drive_document_record(value: GeminiDictValue | str | None) -> GeminiDictValue | None:
    if value is None:
        return None
    if isinstance(value, str):
        return {"id": value}
    return value


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
    thoughtSignatures: list[ThoughtSignatureValue] = Field(default_factory=list)
    parts: list[GeminiPartValue] = Field(default_factory=list)
    grounding: GeminiGrounding | GeminiDictValue | None = None
    branchParent: GeminiBranchParent | GeminiDictValue | None = None
    branchChildren: GeminiBranchChildren = Field(default_factory=list)
    safetyRatings: list[GeminiDictValue] = Field(default_factory=list)
    executableCode: GeminiDictValue | None = None
    codeExecutionResult: GeminiDictValue | None = None
    driveDocument: GeminiDictValue | str | None = None
    inlineFile: GeminiDictValue | None = None
    youtubeVideo: GeminiDictValue | None = None
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

        texts: list[str] = []
        for part in self.parts:
            text = _part_text(part)
            if text:
                texts.append(text)
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
            sigs: list[str | GeminiDictValue] = []
            for signature in self.thoughtSignatures:
                if isinstance(signature, str):
                    sigs.append(signature)
                else:
                    sigs.append(_normalize_mapping(signature))

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
            raw_part = _part_record(part)
            part_text = _part_text(part)
            if part_text:
                blocks.append(
                    ContentBlock(
                        type=ContentType.TEXT,
                        text=part_text,
                        raw=raw_part,
                    )
                )
            elif _part_has_file_payload(part):
                blocks.append(
                    ContentBlock(
                        type=ContentType.FILE,
                        raw=raw_part,
                    )
                )

        if self.executableCode:
            executable_raw = self.executableCode
            language = executable_raw.get("language", "")
            code = executable_raw.get("code", "")
            code_text = _to_text(code)
            if code_text:
                blocks.append(
                    ContentBlock(
                        type=ContentType.CODE,
                        text=code_text,
                        language=language if isinstance(language, str) else None,
                        raw=self.executableCode,
                    )
                )

        if self.codeExecutionResult:
            result_raw = self.codeExecutionResult
            outcome = result_raw.get("outcome", "")
            output = result_raw.get("output", "")
            output_text = _to_text(output)
            if output or outcome:
                blocks.append(
                    ContentBlock(
                        type=ContentType.TOOL_RESULT,
                        text=output_text if output_text else f"[{outcome}]",
                        raw=result_raw,
                    )
                )

        if self.driveDocument:
            raw_doc = _drive_document_record(self.driveDocument)
            if raw_doc is not None:
                blocks.append(
                    ContentBlock(
                        type=ContentType.FILE,
                        raw={"driveDocument": raw_doc},
                    )
                )

        if self.inlineFile:
            mime_type = _get_str_or_none(self.inlineFile.get("mimeType"))
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
            video_id = _get_str_or_none(self.youtubeVideo.get("id"))
            url = f"https://www.youtube.com/watch?v={video_id}" if video_id else None
            blocks.append(
                ContentBlock(
                    type=ContentType.VIDEO,
                    url=url,
                    raw={"youtubeVideo": _string_or_empty_dict(self.youtubeVideo)},
                )
            )

        return blocks

    def extract_tool_calls(self) -> list[ToolCall]:
        return []
