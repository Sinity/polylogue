"""Read-surface output fact extraction for proof surfaces."""

from __future__ import annotations

import csv
import io
import json
import re
from collections import Counter
from typing import Any

from polylogue.lib.semantic_facts import normalized_role_label, sorted_counts

_SUMMARY_TEXT_RE = re.compile(
    r"^(?P<id>.{1,24})\s{2,}(?P<date>\S*)\s{2,}(?P<provider>\S+)\s{2,}(?P<title>.*) \((?P<messages>\d+) msgs\)$"
)


def _summary_output_facts(payload: dict[str, Any]) -> dict[str, Any]:
    tags = payload.get("tags")
    if not isinstance(tags, list):
        tags = []
    return {
        "conversation_id": str(payload.get("id") or ""),
        "provider": str(payload.get("provider") or ""),
        "title": payload.get("title"),
        "date": payload.get("date"),
        "messages": int(payload.get("messages") or 0),
        "tags": [str(tag) for tag in tags],
        "summary": payload.get("summary"),
    }


def _summary_csv_output_facts(csv_text: str) -> list[dict[str, Any]]:
    reader = csv.DictReader(io.StringIO(csv_text))
    rows: list[dict[str, Any]] = []
    for row in reader:
        tags_text = str(row.get("tags") or "")
        rows.append(
            {
                "conversation_id": str(row.get("id") or ""),
                "provider": str(row.get("provider") or ""),
                "title": row.get("title"),
                "date": row.get("date") or None,
                "messages": int(row.get("messages") or 0),
                "tags": [tag for tag in tags_text.split(",") if tag],
                "summary": row.get("summary") or None,
            }
        )
    return rows


def _summary_text_output_facts(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        match = _SUMMARY_TEXT_RE.match(line.rstrip())
        if match is None:
            continue
        rows.append(
            {
                "conversation_id_prefix": match.group("id").strip(),
                "provider": match.group("provider").strip(),
                "date": match.group("date").strip() or None,
                "title": match.group("title").strip(),
                "messages": int(match.group("messages")),
            }
        )
    return rows


def _stream_markdown_output_facts(text: str) -> dict[str, Any]:
    title: str | None = None
    provider: str | None = None
    has_date = False
    message_sections = 0
    role_counts: Counter[str] = Counter()
    for line in text.splitlines():
        if title is None and line.startswith("# "):
            title = line[2:].strip()
        elif line.startswith("**Provider**: "):
            provider = line.split(": ", 1)[1].strip()
        elif line.startswith("**Date**: "):
            has_date = True
        elif line.startswith("## "):
            role = line[3:].strip()
            message_sections += 1
            role_counts[normalized_role_label(role)] += 1
    footer_match = re.search(r"_Streamed (\d+) messages_", text)
    return {
        "title": title,
        "provider": provider,
        "has_date": has_date,
        "message_sections": message_sections,
        "role_counts": sorted_counts(dict(role_counts)),
        "footer_count": int(footer_match.group(1)) if footer_match else 0,
    }


def _stream_plaintext_output_facts(text: str) -> dict[str, Any]:
    role_counts: Counter[str] = Counter()
    message_sections = 0
    for line in text.splitlines():
        if line.startswith("[") and line.endswith("]"):
            role = line[1:-1].strip()
            message_sections += 1
            role_counts[normalized_role_label(role)] += 1
    return {
        "message_sections": message_sections,
        "role_counts": sorted_counts(dict(role_counts)),
        "title": None,
        "provider": None,
        "has_date": False,
    }


def _stream_json_lines_output_facts(text: str) -> dict[str, Any]:
    header: dict[str, Any] = {}
    footer: dict[str, Any] = {}
    message_ids: list[str] = []
    role_counts: Counter[str] = Counter()
    timestamped_messages = 0

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        record_type = payload.get("type")
        if record_type == "header":
            header = payload
        elif record_type == "footer":
            footer = payload
        elif record_type == "message":
            message_ids.append(str(payload.get("id") or ""))
            role_counts[normalized_role_label(payload.get("role"))] += 1
            if payload.get("timestamp"):
                timestamped_messages += 1

    return {
        "conversation_id": str(header.get("conversation_id") or ""),
        "title": header.get("title"),
        "provider": str(header.get("provider") or ""),
        "date": header.get("date"),
        "message_ids": message_ids,
        "message_sections": len(message_ids),
        "role_counts": sorted_counts(dict(role_counts)),
        "timestamped_messages": timestamped_messages,
        "footer_count": int(footer.get("message_count") or 0),
    }


def _mcp_summary_output_facts(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "conversation_id": str(payload.get("id") or ""),
        "provider": str(payload.get("provider") or ""),
        "title": payload.get("title"),
        "messages": int(payload.get("message_count") or 0),
        "created_at": payload.get("created_at"),
        "updated_at": payload.get("updated_at"),
    }


def _mcp_detail_output_facts(payload: dict[str, Any]) -> dict[str, Any]:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        messages = []
    role_counts: Counter[str] = Counter()
    message_ids: list[str] = []
    timestamped_messages = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        message_ids.append(str(message.get("id") or ""))
        role_counts[normalized_role_label(message.get("role"))] += 1
        if message.get("timestamp"):
            timestamped_messages += 1
    return {
        "conversation_id": str(payload.get("id") or ""),
        "provider": str(payload.get("provider") or ""),
        "title": payload.get("title"),
        "created_at": payload.get("created_at"),
        "updated_at": payload.get("updated_at"),
        "messages": len(message_ids),
        "message_ids": message_ids,
        "role_counts": sorted_counts(dict(role_counts)),
        "timestamped_messages": timestamped_messages,
    }


__all__ = [
    "_mcp_detail_output_facts",
    "_mcp_summary_output_facts",
    "_stream_json_lines_output_facts",
    "_stream_markdown_output_facts",
    "_stream_plaintext_output_facts",
    "_summary_csv_output_facts",
    "_summary_output_facts",
    "_summary_text_output_facts",
]
