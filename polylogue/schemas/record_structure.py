"""Public provider record grammar shared by inference and synthetic driving.

Only tokens already declared in the source adapters can enter a published
record kind. Observed private discriminator strings are never a vocabulary.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from functools import cache
from pathlib import Path

from polylogue.core.tool_result_pointer import PERSISTED_OUTPUT_POINTER_PATTERN

CODEX_PUBLIC_MESSAGE_ROLES = frozenset({"user", "assistant", "developer", "system", "tool"})

IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_\-]{0,47}$")


PUBLIC_CONTENT_BODY_FIELDS = ("text", "thinking", "content", "input")
NATIVE_CODEX_TOOL_VALUE_FIELDS = ("name", "execution", "arguments", "input", "action", "output", "tools", "result")
PUBLIC_CLAUDE_TOOL_NAMES = frozenset({"Read", "Edit", "Write", "TodoWrite", "Bash", "Agent"})
PUBLIC_CLAUDE_TODO_TOKENS = {
    "status": frozenset({"pending", "in_progress", "completed"}),
    "priority": frozenset({"low", "medium", "high"}),
}


def claude_todo_token_counts(value: object, fields: tuple[str, ...]) -> dict[str, dict[str, int]]:
    """Count fixed TodoWrite protocol tokens; unknown string values stay anonymous."""
    counts: dict[str, Counter[str]] = {}
    if not isinstance(value, Mapping):
        return {}
    for field in fields:
        items = value.get(field)
        if not isinstance(items, Sequence) or isinstance(items, str | bytes | bytearray):
            continue
        for item in items:
            if not isinstance(item, Mapping):
                continue
            for slot, vocabulary in PUBLIC_CLAUDE_TODO_TOKENS.items():
                token = item.get(slot)
                if isinstance(token, str):
                    counts.setdefault(field + "[*]." + slot, Counter())[token if token in vocabulary else ""] += 1
    return {path: dict(sorted(weights.items())) for path, weights in sorted(counts.items())}


PUBLIC_CLAUDE_TOOL_INPUT_FIELDS = (
    "file_path",
    "old_string",
    "new_string",
    "content",
    "todos",
    "command",
    "run_in_background",
    "prompt",
    "subagent_type",
)
PUBLIC_CLAUDE_TOOL_RESULT_FIELDS = (
    "filePath",
    "originalFile",
    "oldString",
    "newString",
    "replaceAll",
    "userModified",
    "structuredPatch",
    "oldTodos",
    "newTodos",
    "stdout",
    "stderr",
    "interrupted",
    "backgroundTaskId",
    "agentId",
)


def _safe_token(value: object) -> str | None:
    return value if isinstance(value, str) and IDENTIFIER_PATTERN.match(value) else None


_SOURCES_ROOT = Path(__file__).resolve().parents[1] / "sources"


@cache
def published_kind_tokens() -> frozenset[str]:
    """Record-type vocabulary already public in Polylogue's own source adapters.

    A record ``type``/subtype value is published into a profile only when it
    appears as a string literal in ``polylogue/sources``; any other value is
    source data, not vocabulary, and is folded away.
    """
    import ast

    tokens: set[str] = set()
    for path in sorted(_SOURCES_ROOT.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and IDENTIFIER_PATTERN.match(node.value):
                tokens.add(node.value)
    return frozenset(tokens)


def _kind(base: object, subtype: object) -> str:
    vocabulary = published_kind_tokens()
    base_token = _safe_token(base)
    if base_token is None or base_token not in vocabulary:
        return "record:other"
    subtype_token = _safe_token(subtype)
    return f"record:{base_token}:{subtype_token}" if subtype_token in vocabulary else f"record:{base_token}"


def classify_claude_code_record(record: Mapping[str, object]) -> str:
    """Return the workload kind of one Claude Code transcript record."""
    record_type = record.get("type")
    message = record.get("message")
    if record_type in {"user", "assistant"} and isinstance(message, Mapping):
        content = message.get("content")
        kind = "user_text" if record_type == "user" else "assistant_text"
        for block in iter_public_content_kinds(content):
            kind = content_prefix_kind("claude-code", kind, kind, block)
        return kind
    data = record.get("data")
    attachment = record.get("attachment")
    subtype = (
        (data.get("type") if isinstance(data, Mapping) else None)
        or (attachment.get("type") if isinstance(attachment, Mapping) else None)
        or record.get("subtype")
    )
    return _kind(record_type, subtype)


def codex_session_header(record: Mapping[str, object]) -> Mapping[str, object] | None:
    """The parser-declared envelope or flat first-header wire carrier."""
    record_type = record.get("type")
    if record_type == "session_meta":
        payload = record.get("payload")
        return payload if isinstance(payload, Mapping) else None
    identity = record.get("id")
    timestamp = record.get("timestamp")
    if (
        not (record_type if isinstance(record_type, str) else None)
        and isinstance(identity, str)
        and identity
        and isinstance(timestamp, str | int | float)
        and timestamp
    ):
        return record
    return None


def codex_message_record(record: Mapping[str, object]) -> bool:
    """The parser-declared flat or response-item message carrier."""
    if isinstance(record.get("payload"), Mapping):
        payload = record["payload"]
        assert isinstance(payload, Mapping)
        return record.get("type") == "response_item" and payload.get("type") == "message"
    return record.get("type") == "message" or isinstance(record.get("role"), str)


def classify_codex_record(record: Mapping[str, object]) -> str:
    """Return the workload kind of one Codex rollout record.

    Flat messages and rollout envelopes retain their declared wire form.
    Their roles share a public record kind; the profile records the form
    separately rather than mixing the two shapes in one stream.
    """
    record_type = record.get("type")
    payload = record.get("payload")
    if codex_session_header(record) is not None:
        return "session_meta"
    if not isinstance(payload, Mapping):
        role = record.get("role") if codex_message_record(record) else None
        return f"{role}_message" if role in CODEX_PUBLIC_MESSAGE_ROLES else "record:other"
    payload_type = payload.get("type")
    if record_type in {"session_meta", "turn_context"}:
        return str(record_type)
    if record_type == "response_item":
        if payload_type == "message":
            role = payload.get("role")
            return f"{role}_message" if role in CODEX_PUBLIC_MESSAGE_ROLES else "record:other"
        if payload_type in {
            "reasoning",
            "function_call",
            "function_call_output",
            "custom_tool_call",
            "custom_tool_call_output",
        }:
            return str(payload_type)
    if record_type == "event_msg" and payload_type == "token_count":
        return "event_token_count"
    return _kind(record_type, payload_type)


def iter_provider_records(provider: str, sample: Mapping[str, object]) -> Iterator[Mapping[str, object]]:
    """Traverse only the provider's declared record container.

    Mapping order is wire order, not inferred chronological order. A document
    tree's parent links remain explicit lineage evidence. User argument maps
    and sidecar contents never become transcript records through recursion.
    """
    if provider == "chatgpt":
        mapping = sample.get("mapping")
        if isinstance(mapping, Mapping):
            for node in mapping.values():
                if isinstance(node, Mapping):
                    yield node
        return
    if provider == "claude-ai":
        messages = sample.get("chat_messages")
        if isinstance(messages, list):
            for message in messages:
                if isinstance(message, Mapping):
                    yield message
        return
    if provider == "gemini":
        prompt = sample.get("chunkedPrompt")
        chunks = prompt.get("chunks") if isinstance(prompt, Mapping) else None
        if isinstance(chunks, list):
            for chunk in chunks:
                if isinstance(chunk, Mapping):
                    yield chunk
        return
    # JSONL samples already are individual physical records. Unsupported
    # document grammars are classified as other instead of inspecting keys.
    yield sample


def record_kind(provider: str, record: Mapping[str, object]) -> str:
    if provider == "claude-code":
        return classify_claude_code_record(record)
    if provider == "codex":
        return classify_codex_record(record)
    message = record.get("message") if provider == "chatgpt" else record
    if not isinstance(message, Mapping):
        return "record:empty"
    author = message.get("author")
    role = (
        author.get("role")
        if isinstance(author, Mapping)
        else message.get("sender")
        if provider == "claude-ai"
        else message.get("role")
    )
    if role == "human" and provider == "claude-ai":
        role = "user"
    if role == "model" and provider == "gemini":
        role = "assistant"
    return f"{role}_message" if role in {"user", "assistant", "system", "developer", "tool"} else "record:other"


def iter_tool_records(provider: str, record: Mapping[str, object]) -> Iterator[Mapping[str, object]]:
    """Borrow tool operands at declared envelope and content locations only."""
    if provider == "codex":
        payload = record.get("payload")
        envelope = payload if isinstance(payload, Mapping) else record
        yield envelope
        # Flat/enveloped Codex messages also admit typed tool blocks. These
        # are the parser's declared content coordinates, never argument maps.
        content = envelope.get("content")
        if isinstance(content, Sequence) and not isinstance(content, str | bytes | bytearray):
            for block in content:
                if isinstance(block, Mapping):
                    yield block
        return
    if provider == "claude-code" and record_form(provider, record) != "claude-message":
        return
    message = record.get("message") if provider in {"claude-code", "chatgpt"} else record
    if not isinstance(message, Mapping):
        return
    content = message.get("content")
    if isinstance(content, Sequence) and not isinstance(content, str | bytes | bytearray):
        for block in content:
            if isinstance(block, Mapping):
                yield block


def tool_result_content_mode(record: Mapping[str, object]) -> str:
    if record.get("type") in {"function_call_output", "custom_tool_call_output"}:
        for field in ("output", "tools", "result"):
            if field in record:
                value = record[field]
                shape = "null" if value is None else content_representation(value)
                return f"native-{field}:{shape}"
    content = record.get("content")
    shape = "absent" if "content" not in record else "null" if content is None else content_representation(content)
    if shape not in {"absent", "null", "string", "array", "object"}:
        shape = "other"
    truncated = isinstance(content, str) and PERSISTED_OUTPUT_POINTER_PATTERN.search(content) is not None
    return shape + (":truncated" if truncated else ":inline")


def tool_identity_operand(record: Mapping[str, object]) -> tuple[str, str, str, object] | None:
    """Declared tool event/identity coordinates, shared by measurement and drawing."""
    kind = record.get("type") or record.get("kind")
    if not isinstance(kind, str):
        return None
    fields: tuple[str, ...]
    calls = {"tool_use", "function_call", "tool_call", "function", "custom_tool_call"}
    results = {"tool_result", "function_call_output", "tool_call_output", "function_result", "custom_tool_call_output"}
    if kind in calls:
        event, fields = "call", ("id", "call_id", "tool_call_id")
    elif kind in results:
        event, fields = "result", ("tool_use_id", "call_id", "tool_call_id", "id")
    else:
        return None
    assert isinstance(kind, str)
    selected = next((field for field in fields if isinstance(record.get(field), str) and record[field]), None)
    if selected is None:
        selected = next((field for field in fields if field in record), None)
    if selected is None:
        selected = "id" if kind == "tool_use" else "tool_use_id" if kind == "tool_result" else "call_id"
    return kind, event, selected, record.get(selected)


def record_timestamp(provider: str, record: Mapping[str, object]) -> object:
    message = record.get("message") if provider == "chatgpt" else record
    if not isinstance(message, Mapping):
        return None
    if provider == "chatgpt":
        return message.get("create_time")
    if provider == "claude-ai":
        return message.get("created_at")
    return message.get("timestamp")


def record_content(provider: str, record: Mapping[str, object]) -> object:
    """Borrow only the declared provider content envelope, never argument maps."""
    envelope: object
    if provider == "codex":
        payload = record.get("payload")
        envelope = payload if isinstance(payload, Mapping) else record
    elif provider == "claude-code":
        if record_form(provider, record) != "claude-message":
            return None
        envelope = record.get("message")
    elif provider == "chatgpt":
        envelope = record.get("message")
    else:
        envelope = record
    return envelope.get("content") if isinstance(envelope, Mapping) else None


def content_representation(content: object) -> str:
    if isinstance(content, str):
        return "string"
    if isinstance(content, Sequence) and not isinstance(content, bytes | bytearray):
        return "array"
    if isinstance(content, Mapping):
        return "object"
    return "absent" if content is None else "other"


def iter_public_content_kinds(content: object) -> Iterator[str]:
    """Observe public block types in exact physical order without a codebook."""
    if isinstance(content, str):
        yield "block:text"
    elif isinstance(content, Mapping):
        yield _public_block_kind(content.get("type", content.get("content_type")))
    elif isinstance(content, Sequence) and not isinstance(content, bytes | bytearray):
        for block in content:
            yield _public_block_kind(block.get("type") if isinstance(block, Mapping) else None)


def _public_block_kind(value: object) -> str:
    token = _safe_token(value)
    return f"block:{token}" if token in published_kind_tokens() else "block:other"


def content_prefix_kind(provider: str, record: str, previous: str, block: str) -> str:
    """Finite public prefix state needed by the provider's record classifier."""
    if provider == "claude-code":
        if record.startswith("user_"):
            return "user_tool_result" if block == "block:tool_result" else previous
        if record.startswith("assistant_"):
            if block == "block:tool_use" or previous == "assistant_tool_use":
                return "assistant_tool_use"
            if block == "block:thinking" or previous == "assistant_thinking":
                return "assistant_thinking"
            return "assistant_text"
    return "content"


def initial_content_prefix(provider: str, record: str) -> str:
    if provider == "claude-code":
        if record.startswith("user_"):
            return "user_text"
        if record.startswith("assistant_"):
            return "assistant_text"
    return "content"


def public_role_carriers(record: Mapping[str, object]) -> tuple[str, str]:
    """Finite Claude role carriers, including absence, without private values."""
    from polylogue.core.enums import ROLE_SYNONYMS

    vocabulary = frozenset(value for aliases in ROLE_SYNONYMS.values() for value in aliases)

    def carrier(parent: Mapping[str, object], key: str) -> str:
        if key not in parent:
            return "absent"
        value = parent[key]
        if value is None:
            return "null"
        if isinstance(value, str) and value in vocabulary:
            return "role:" + value
        return "unpublished"

    message = record.get("message")
    return carrier(record, "role"), carrier(message if isinstance(message, Mapping) else {}, "role")


def record_form(provider: str, record: Mapping[str, object]) -> str:
    """The finite envelope alternatives already read by the provider adapter."""
    if provider == "chatgpt" and not isinstance(record.get("message"), Mapping):
        if "message" not in record:
            return "chatgpt-message-absent"
        return "chatgpt-message-null" if record["message"] is None else "chatgpt-message-unpublished"
    if provider == "codex":
        if codex_session_header(record) is not None and record.get("type") != "session_meta":
            suffix = (
                ""
                if "type" not in record
                else "-null"
                if record["type"] is None
                else "-empty"
                if record["type"] == ""
                else "-unpublished"
            )
            return "codex-flat-header" + suffix
        if not isinstance(record.get("payload"), Mapping) and isinstance(record.get("role"), str):
            if "type" not in record:
                return "codex-role-message"
            if record["type"] is None:
                return "codex-role-message-null"
        return "codex-envelope" if isinstance(record.get("payload"), Mapping) else "codex-flat"
    if provider == "claude-code":
        if record.get("type") in {"user", "assistant"}:
            return "claude-message"
        for name in ("data", "attachment"):
            child = record.get(name)
            if isinstance(child, Mapping) and child.get("type"):
                return f"claude-record-{name}"
        return "claude-record-subtype" if record.get("subtype") else "claude-record"
    return "document-record"


# These are provider field coordinates, not observed names or identity values.
CODEX_PARENT_FIELDS = ("forked_from_id", "parent_thread_id", "source")


def codex_parent_field_forms(header: Mapping[str, object]) -> dict[str, str]:
    """Retain first-header presence/types without retaining parent spellings."""
    forms: dict[str, str] = {}

    def visit(node: Mapping[str, object], fields: tuple[str, ...], prefix: str) -> None:
        for field in fields:
            path = prefix + field
            if field not in node:
                forms[path] = "absent"
                continue
            value = node[field]
            forms[path] = (
                "null"
                if value is None
                else "object"
                if isinstance(value, Mapping)
                else "empty"
                if value == ""
                else "blank"
                if isinstance(value, str) and not value.strip()
                else "string"
                if isinstance(value, str)
                else "boolean"
                if isinstance(value, bool)
                else "integer"
                if isinstance(value, int)
                else "number"
                if isinstance(value, float)
                else "array"
                if isinstance(value, list)
                else "unsupported"
            )
            nested = {
                "source": ("subagent",),
                "subagent": ("thread_spawn",),
                "thread_spawn": ("parent_thread_id",),
            }.get(field)
            if nested is not None and isinstance(value, Mapping):
                visit(value, nested, path + ".")

    visit(header, CODEX_PARENT_FIELDS, "")
    return forms


def claude_dispatch_field_forms(record: Mapping[str, object]) -> dict[str, str]:
    """Public dispatcher carrier presence; values remain schema-owned."""

    def form(parent: Mapping[str, object], field: str) -> str:
        if field not in parent:
            return "absent"
        value = parent[field]
        if value is None:
            return "null"
        if isinstance(value, Mapping):
            return "object"
        if isinstance(value, str):
            return "empty" if not value else "string"
        if isinstance(value, bool):
            return "boolean"
        if isinstance(value, int):
            return "integer"
        if isinstance(value, float):
            return "number"
        if isinstance(value, list):
            return "array"
        return "unsupported"

    forms = {"toolUseResult": form(record, "toolUseResult")}
    result = record.get("toolUseResult")
    if isinstance(result, Mapping):
        forms["toolUseResult.agentId"] = form(result, "agentId")
    return forms


def claude_agent_metadata_field_forms(record: Mapping[str, object]) -> dict[str, str]:
    """Fixed Source alias coordinates; the schema retains all field values."""
    from polylogue.core.claude_code_children import CLAUDE_AGENT_ID_FIELDS

    forms: dict[str, str] = {}
    for field in CLAUDE_AGENT_ID_FIELDS:
        if field not in record:
            forms[field] = "absent"
            continue
        value = record[field]
        forms[field] = (
            "null"
            if value is None
            else "object"
            if isinstance(value, Mapping)
            else "empty"
            if value == ""
            else "string"
            if isinstance(value, str)
            else "boolean"
            if isinstance(value, bool)
            else "integer"
            if isinstance(value, int)
            else "number"
            if isinstance(value, float)
            else "array"
            if isinstance(value, list)
            else "unsupported"
        )
    return forms
