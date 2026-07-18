"""Gemini message model and viewport extraction."""

from __future__ import annotations

from datetime import datetime
from typing import TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from polylogue.archive.message.roles import Role
from polylogue.archive.viewport.viewports import (
    ContentBlock,
    ContentType,
    MessageMeta,
    ReasoningTrace,
    TokenUsage,
    ToolCall,
)
from polylogue.core.enums import Provider
from polylogue.core.json import json_document
from polylogue.core.sources import origin_from_provider
from polylogue.core.timestamps import parse_timestamp

from ..parsers.drive_support_attachments import DRIVE_LIVE_FETCH_DATA_KEY
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
        return _json_object(payload.model_dump(exclude_none=True, exclude_defaults=True))
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


def _get_str_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _string_or_empty_dict(value: object) -> GeminiDictValue:
    return _json_object(value)


def _part_record(part: GeminiPartValue) -> GeminiDictValue:
    if isinstance(part, GeminiPart):
        return _json_object(part.model_dump(exclude_none=True, exclude_defaults=True))
    return part


def _drive_document_record(value: GeminiDictValue | str | None) -> GeminiDictValue | None:
    if value is None:
        return None
    if isinstance(value, str):
        return {"id": value}
    return value


def _without_inline_bytes(payload: GeminiDictValue) -> GeminiDictValue:
    sanitized = dict(payload)
    sanitized.pop("data", None)
    return sanitized


def _thought_raw(
    *,
    thinking_budget: int | None,
    thought_signatures: list[str | GeminiDictValue],
) -> GeminiDictValue:
    raw: GeminiDictValue = {"isThought": True}
    if thinking_budget is not None:
        raw["thinkingBudget"] = thinking_budget
    if thought_signatures:
        raw["thoughtSignatures"] = thought_signatures
    return raw


def _code_content_block(payload: GeminiDictValue) -> ContentBlock | None:
    code_text = _to_text(payload.get("code"))
    if not code_text:
        return None
    language = payload.get("language")
    return ContentBlock(
        type=ContentType.CODE,
        text=code_text,
        language=language if isinstance(language, str) else None,
        raw=payload,
    )


def _tool_result_content_block(payload: GeminiDictValue) -> ContentBlock | None:
    output = payload.get("output")
    outcome = payload.get("outcome")
    output_text = _to_text(output)
    outcome_text = _to_text(outcome)
    if not output_text and not outcome_text:
        return None
    return ContentBlock(
        type=ContentType.TOOL_RESULT,
        text=output_text if output_text else f"[{outcome_text}]",
        raw=payload,
    )


def _drive_media_content_block(
    field_name: str,
    value: GeminiDictValue | str | None,
    content_type: ContentType,
) -> ContentBlock | None:
    record = _drive_document_record(value)
    if record is None:
        return None
    mime_type = _get_str_or_none(record.get("mimeType"))
    # polylogue-83u.2: live Drive attachment acquisition injects fetched bytes
    # into this same record as base64 under DRIVE_LIVE_FETCH_DATA_KEY before
    # attachment_from_doc decodes it into ParsedAttachment.inline_bytes. That
    # sidecar must never reach block metadata (unlike attachment rows, blocks
    # are not content-addressed) -- strip it here the same way
    # _without_inline_bytes strips inlineData/inlineFile/inlineImage's `data`.
    if DRIVE_LIVE_FETCH_DATA_KEY in record:
        record = {key: value for key, value in record.items() if key != DRIVE_LIVE_FETCH_DATA_KEY}
    return ContentBlock(
        type=content_type,
        mime_type=mime_type,
        raw={field_name: record},
    )


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
    driveImage: GeminiDictValue | str | None = None
    driveAudio: GeminiDictValue | str | None = None
    driveVideo: GeminiDictValue | str | None = None
    inlineFile: GeminiDictValue | None = None
    inlineImage: GeminiDictValue | None = None
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
            origin=origin_from_provider(Provider.GEMINI),
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
                    origin=origin_from_provider(Provider.GEMINI),
                    raw=_thought_raw(
                        thinking_budget=self.thinkingBudget,
                        thought_signatures=sigs,
                    ),
                )
            )
        return traces

    def extract_content_blocks(self) -> list[ContentBlock]:
        blocks = []

        if self.isThought:
            sigs: list[str | GeminiDictValue] = []
            for signature in self.thoughtSignatures:
                sigs.append(signature if isinstance(signature, str) else _normalize_mapping(signature))
            blocks.append(
                ContentBlock(
                    type=ContentType.THINKING,
                    text=self.text,
                    raw=_thought_raw(
                        thinking_budget=self.thinkingBudget,
                        thought_signatures=sigs,
                    ),
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
                        type=ContentType.THINKING if raw_part.get("thought") is True else ContentType.TEXT,
                        text=part_text,
                        raw=raw_part,
                    )
                )
            inline_data = _json_object(raw_part.get("inlineData"))
            if inline_data:
                mime_type = _get_str_or_none(inline_data.get("mimeType"))
                blocks.append(
                    ContentBlock(
                        type=ContentType.FILE,
                        mime_type=mime_type,
                        raw={"inlineData": _without_inline_bytes(inline_data)},
                    )
                )
            file_data = _json_object(raw_part.get("fileData"))
            if file_data:
                mime_type = _get_str_or_none(file_data.get("mimeType"))
                blocks.append(
                    ContentBlock(
                        type=ContentType.FILE,
                        mime_type=mime_type,
                        url=_get_str_or_none(file_data.get("fileUri")),
                        raw={"fileData": file_data},
                    )
                )
            executable_code = _json_object(raw_part.get("executableCode"))
            if executable_code and (code_block := _code_content_block(executable_code)) is not None:
                blocks.append(code_block)
            execution_result = _json_object(raw_part.get("codeExecutionResult"))
            if execution_result and (result_block := _tool_result_content_block(execution_result)) is not None:
                blocks.append(result_block)

        if self.executableCode:
            code_block = _code_content_block(self.executableCode)
            if code_block is not None:
                blocks.append(code_block)

        if self.codeExecutionResult:
            result_block = _tool_result_content_block(self.codeExecutionResult)
            if result_block is not None:
                blocks.append(result_block)

        for field_name, value, content_type in (
            ("driveDocument", self.driveDocument, ContentType.FILE),
            ("driveImage", self.driveImage, ContentType.IMAGE),
            ("driveAudio", self.driveAudio, ContentType.AUDIO),
            ("driveVideo", self.driveVideo, ContentType.VIDEO),
        ):
            drive_block = _drive_media_content_block(field_name, value, content_type)
            if drive_block is not None:
                blocks.append(drive_block)

        if self.inlineFile:
            mime_type = _get_str_or_none(self.inlineFile.get("mimeType"))
            blocks.append(
                ContentBlock(
                    type=ContentType.FILE,
                    mime_type=mime_type if isinstance(mime_type, str) else None,
                    raw={"inlineFile": _without_inline_bytes(self.inlineFile)},
                )
            )

        if self.inlineImage:
            mime_type = _get_str_or_none(self.inlineImage.get("mimeType"))
            blocks.append(
                ContentBlock(
                    type=ContentType.IMAGE,
                    mime_type=mime_type,
                    raw={"inlineImage": _without_inline_bytes(self.inlineImage)},
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

        if self.errorMessage:
            blocks.append(
                ContentBlock(
                    type=ContentType.ERROR,
                    text=self.errorMessage,
                    raw={"errorMessage": self.errorMessage},
                )
            )

        return blocks

    def extract_tool_calls(self) -> list[ToolCall]:
        return []
