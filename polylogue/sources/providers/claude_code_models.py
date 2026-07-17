"""Claude Code typed block and message support models."""

from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from polylogue.archive.viewport.viewports import ReasoningTrace, TokenUsage, ToolCall, classify_tool
from polylogue.core.enums import Provider
from polylogue.core.json import json_document
from polylogue.core.sources import origin_from_provider

ClaudeCodeBlockRecord: TypeAlias = dict[str, object]
ClaudeCodeContentBlocks: TypeAlias = list[ClaudeCodeBlockRecord]

_BACKGROUND_COMMAND_SUCCESS_RE = re.compile(r'^Background command ".+" completed \(exit code (?P<exit_code>[0-9]+)\)$')
_BACKGROUND_COMMAND_FAILURE_RE = re.compile(r'^Background command ".+" failed with exit code (?P<exit_code>[0-9]+)$')
_TASK_NOTIFICATION_REQUIRED_FIELDS = frozenset({"task-id", "output-file", "status", "summary"})


class _TaskNotificationEnvelopeParser(HTMLParser):
    """Strictly collect direct fields from Claude Code's task-notification envelope."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.fields: dict[str, list[str]] = {}
        self._stack: list[str] = []
        self._saw_root = False
        self._closed_root = False
        self.malformed = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if not self._stack:
            if tag != "task-notification" or self._saw_root or self._closed_root:
                self.malformed = True
                return
            self._saw_root = True
        self._stack.append(tag)
        if len(self._stack) == 2 and self._stack[0] == "task-notification":
            self.fields.setdefault(tag, [])

    def handle_endtag(self, tag: str) -> None:
        if not self._stack or self._stack[-1] != tag:
            self.malformed = True
            return
        self._stack.pop()
        if not self._stack:
            self._closed_root = True

    def handle_data(self, data: str) -> None:
        if not self._stack:
            if data.strip() and not self._closed_root:
                self.malformed = True
            return
        if len(self._stack) == 2 and self._stack[0] == "task-notification":
            self.fields[self._stack[-1]].append(data)


class ClaudeCodeBackgroundTaskNotification(BaseModel):
    """Typed Claude Code background-command completion protocol evidence.

    The numeric exit code is populated only for the observed, version-specific
    success and failed-command summary templates.
    Other structurally valid notifications deliberately retain ``None`` so
    callers cannot mistake arbitrary provider prose for a process outcome.
    """

    task_id: str
    tool_use_id: str | None = None
    output_file: str
    status: str
    summary: str
    exit_code: int | None = None

    @classmethod
    def from_protocol_text(cls, text: str | None) -> ClaudeCodeBackgroundTaskNotification | None:
        if not text:
            return None
        parser = _TaskNotificationEnvelopeParser()
        parser.feed(text)
        parser.close()
        if parser.malformed or parser._stack or not parser._saw_root or not parser._closed_root:
            return None
        if set(parser.fields) & _TASK_NOTIFICATION_REQUIRED_FIELDS != _TASK_NOTIFICATION_REQUIRED_FIELDS:
            return None
        if any(len(parser.fields[field]) != 1 for field in _TASK_NOTIFICATION_REQUIRED_FIELDS):
            return None

        values = {field: "".join(parser.fields[field]).strip() for field in _TASK_NOTIFICATION_REQUIRED_FIELDS}
        if any(not value for value in values.values()):
            return None
        tool_use_values = parser.fields.get("tool-use-id")
        if tool_use_values is not None and len(tool_use_values) != 1:
            return None
        tool_use_id = "".join(tool_use_values).strip() if tool_use_values is not None else None
        if tool_use_values is not None and not tool_use_id:
            return None
        summary = values["summary"]
        match = (
            _BACKGROUND_COMMAND_SUCCESS_RE.fullmatch(summary)
            if values["status"] == "completed"
            else _BACKGROUND_COMMAND_FAILURE_RE.fullmatch(summary)
            if values["status"] == "failed"
            else None
        )
        exit_code = int(match.group("exit_code")) if match else None
        return cls(
            task_id=values["task-id"],
            tool_use_id=tool_use_id,
            output_file=values["output-file"],
            status=values["status"],
            summary=summary,
            exit_code=exit_code,
        )


class ClaudeCodeToolUse(BaseModel):
    """A tool_use content block from Claude Code."""

    model_config = ConfigDict(extra="allow")

    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: ClaudeCodeBlockRecord = Field(default_factory=dict)

    def to_tool_call(self) -> ToolCall:
        return ToolCall(
            name=self.name,
            id=self.id,
            input=self.input,
            category=classify_tool(self.name, json_document(self.input)),
            origin=origin_from_provider(Provider.CLAUDE_CODE),
            raw=self.model_dump(),
        )


class ClaudeCodeToolResult(BaseModel):
    """A tool_result content block from Claude Code."""

    model_config = ConfigDict(extra="allow")

    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str | list[object] = ""
    is_error: bool = False


class ClaudeCodeTextBlock(BaseModel):
    """A text content block from Claude Code."""

    model_config = ConfigDict(extra="allow")

    type: Literal["text"] = "text"
    text: str


class ClaudeCodeThinkingBlock(BaseModel):
    """A thinking content block from Claude Code."""

    model_config = ConfigDict(extra="allow")

    type: Literal["thinking"] = "thinking"
    thinking: str

    def to_reasoning_trace(self) -> ReasoningTrace:
        return ReasoningTrace(
            text=self.thinking,
            origin=origin_from_provider(Provider.CLAUDE_CODE),
            raw=self.model_dump(),
        )


class ClaudeCodeUsage(BaseModel):
    """Token usage from Claude Code response."""

    model_config = ConfigDict(extra="allow")

    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None

    def to_token_usage(self) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cache_read_tokens=self.cache_read_input_tokens,
            cache_write_tokens=self.cache_creation_input_tokens,
        )


class ClaudeCodeMessageContent(BaseModel):
    """Message content from Claude Code (assistant turn)."""

    model_config = ConfigDict(extra="allow")

    id: str | None = None
    type: str = "message"
    role: str
    model: str | None = None
    content: ClaudeCodeContentBlocks = Field(default_factory=list)
    stop_reason: str | None = None
    stop_sequence: str | None = None
    usage: ClaudeCodeUsage | None = None


class ClaudeCodeUserMessage(BaseModel):
    """User message content from Claude Code."""

    model_config = ConfigDict(extra="allow")

    role: Literal["user"] = "user"
    content: str | ClaudeCodeContentBlocks = ""
