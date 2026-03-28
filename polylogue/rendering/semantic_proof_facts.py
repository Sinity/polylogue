"""Fact extraction and primitive semantic checks for proof surfaces."""

from __future__ import annotations

import csv
import io
import json
import re
from collections import Counter
from html import unescape as html_unescape
from typing import Any

from polylogue.lib.semantic_facts import normalized_role_label, sorted_counts
from polylogue.rendering.semantic_proof_models import SemanticMetricCheck

_HTML_ROLE_RE = re.compile(r'class="role-label">([^<]+)</span>')
_HTML_TIMESTAMP_RE = re.compile(r'class="timestamp">[^<]+</time>')
_HTML_BRANCH_RE = re.compile(r'class="branch-label">Branch\s+\d+</div>')
_HTML_TITLE_RE = re.compile(r"<h1>(.*?)</h1>", re.DOTALL)
_HTML_BADGE_RE = re.compile(r'class="badge">([^<]+)</span>')
_HTML_META_RE = re.compile(r'class="meta-item">([^<]+)</span>')
_SUMMARY_TEXT_RE = re.compile(
    r"^(?P<id>.{1,24})\s{2,}(?P<date>\S*)\s{2,}(?P<provider>\S+)\s{2,}(?P<title>.*) \((?P<messages>\d+) msgs\)$"
)

def _critical_or_preserved(*, metric: str, policy: str, input_value: Any, output_value: Any) -> SemanticMetricCheck:
    return SemanticMetricCheck(
        metric=metric,
        status="preserved" if input_value == output_value else "critical_loss",
        policy=policy,
        input_value=input_value,
        output_value=output_value,
    )


def _declared_loss_or_preserved(
    *,
    metric: str,
    policy: str,
    input_value: int,
    output_value: Any = 0,
) -> SemanticMetricCheck:
    return SemanticMetricCheck(
        metric=metric,
        status="declared_loss" if input_value else "preserved",
        policy=policy,
        input_value=input_value,
        output_value=output_value,
    )


def _presence_check(
    *,
    metric: str,
    policy: str,
    input_value: object,
    output_present: bool,
) -> SemanticMetricCheck:
    expected_present = input_value is not None
    return SemanticMetricCheck(
        metric=metric,
        status="preserved" if expected_present == output_present else "critical_loss",
        policy=policy,
        input_value=input_value,
        output_value=output_present,
    )

def _canonical_markdown_output_facts(markdown_text: str) -> dict[str, Any]:
    message_sections = 0
    timestamp_lines = 0
    attachment_lines = 0
    role_section_counts: Counter[str] = Counter()

    for line in markdown_text.splitlines():
        if line.startswith("## "):
            section = line[3:].strip()
            if section.lower() != "attachments":
                message_sections += 1
                role_section_counts[normalized_role_label(section)] += 1
        elif line.startswith("_Timestamp: "):
            timestamp_lines += 1
        elif line.startswith("- Attachment: "):
            attachment_lines += 1

    return {
        "message_sections": message_sections,
        "timestamp_lines": timestamp_lines,
        "attachment_lines": attachment_lines,
        "typed_thinking_markers": 0,
        "typed_tool_markers": 0,
        "role_section_counts": sorted_counts(dict(role_section_counts)),
    }


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


def _json_like_output_facts(payload: dict[str, Any]) -> dict[str, Any]:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        messages = []
    role_counts: Counter[str] = Counter()
    message_ids: list[str] = []
    timestamped_messages = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        message_ids.append(str(message.get("id", "")))
        role_counts[normalized_role_label(message.get("role"))] += 1
        if message.get("timestamp"):
            timestamped_messages += 1
    return {
        "conversation_id": str(payload.get("id") or ""),
        "provider": str(payload.get("provider") or ""),
        "title": payload.get("title"),
        "date": payload.get("date"),
        "messages": len(message_ids),
        "message_ids": message_ids,
        "role_counts": sorted_counts(dict(role_counts)),
        "timestamped_messages": timestamped_messages,
    }


def _csv_output_facts(csv_text: str) -> dict[str, Any]:
    reader = csv.DictReader(io.StringIO(csv_text))
    message_ids: list[str] = []
    role_counts: Counter[str] = Counter()
    timestamped_messages = 0
    conversation_ids: Counter[str] = Counter()
    for row in reader:
        conversation_id = str(row.get("conversation_id") or "")
        if conversation_id:
            conversation_ids[conversation_id] += 1
        message_ids.append(str(row.get("message_id") or ""))
        role_counts[normalized_role_label(row.get("role"))] += 1
        if row.get("timestamp"):
            timestamped_messages += 1
    conversation_id = ""
    if conversation_ids:
        conversation_id = conversation_ids.most_common(1)[0][0]
    return {
        "conversation_id": conversation_id,
        "messages": len(message_ids),
        "message_ids": message_ids,
        "role_counts": sorted_counts(dict(role_counts)),
        "timestamped_messages": timestamped_messages,
        "provider": None,
        "title": None,
        "date": None,
    }


def _markdown_doc_output_facts(text: str) -> dict[str, Any]:
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
    return {
        "title": title,
        "provider": provider,
        "has_date": has_date,
        "message_sections": message_sections,
        "role_counts": sorted_counts(dict(role_counts)),
        "timestamp_lines": 0,
        "branch_labels": 0,
        "conversation_id": None,
    }


def _obsidian_output_facts(text: str) -> dict[str, Any]:
    frontmatter: dict[str, Any] = {}
    body = text
    if text.startswith("---\n"):
        parts = text.split("\n---\n", 1)
        if len(parts) == 2:
            _, body = parts
            try:
                import yaml

                parsed = yaml.safe_load(parts[0].replace("---\n", "", 1))
                if isinstance(parsed, dict):
                    frontmatter = parsed
            except Exception:
                frontmatter = {}
    body_facts = _markdown_doc_output_facts(body)
    body_facts["conversation_id"] = str(frontmatter.get("id") or "")
    body_facts["provider"] = str(frontmatter.get("provider") or "")
    body_facts["has_date"] = frontmatter.get("date") is not None
    return body_facts


def _org_output_facts(text: str) -> dict[str, Any]:
    title: str | None = None
    provider: str | None = None
    has_date = False
    message_sections = 0
    role_counts: Counter[str] = Counter()
    for line in text.splitlines():
        if line.startswith("#+TITLE: "):
            title = line.split(": ", 1)[1].strip()
        elif line.startswith("#+DATE: "):
            has_date = True
        elif line.startswith("#+PROPERTY: provider "):
            provider = line.split("#+PROPERTY: provider ", 1)[1].strip()
        elif line.startswith("* "):
            role = line[2:].strip()
            message_sections += 1
            role_counts[normalized_role_label(role)] += 1
    return {
        "title": title,
        "provider": provider,
        "has_date": has_date,
        "message_sections": message_sections,
        "role_counts": sorted_counts(dict(role_counts)),
        "timestamp_lines": 0,
        "branch_labels": 0,
        "conversation_id": None,
    }


def _html_output_facts(text: str) -> dict[str, Any]:
    title_match = _HTML_TITLE_RE.search(text)
    title = html_unescape(title_match.group(1).strip()) if title_match else None
    badge_match = _HTML_BADGE_RE.search(text)
    provider = badge_match.group(1).strip() if badge_match else None
    meta_items = [html_unescape(item).strip() for item in _HTML_META_RE.findall(text)]
    has_date = len(meta_items) >= 2
    role_counts: Counter[str] = Counter(
        normalized_role_label(role.strip()) for role in _HTML_ROLE_RE.findall(text)
    )
    return {
        "title": title,
        "provider": provider,
        "has_date": has_date,
        "message_sections": sum(role_counts.values()),
        "role_counts": sorted_counts(dict(role_counts)),
        "timestamp_lines": len(_HTML_TIMESTAMP_RE.findall(text)),
        "branch_labels": len(_HTML_BRANCH_RE.findall(text)),
        "conversation_id": None,
    }


__all__ = [
    "_canonical_markdown_output_facts",
    "_critical_or_preserved",
    "_csv_output_facts",
    "_declared_loss_or_preserved",
    "_html_output_facts",
    "_json_like_output_facts",
    "_markdown_doc_output_facts",
    "_mcp_detail_output_facts",
    "_mcp_summary_output_facts",
    "_obsidian_output_facts",
    "_org_output_facts",
    "_presence_check",
    "_stream_json_lines_output_facts",
    "_stream_markdown_output_facts",
    "_stream_plaintext_output_facts",
    "_summary_csv_output_facts",
    "_summary_output_facts",
    "_summary_text_output_facts",
]
