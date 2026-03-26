"""Export-oriented output fact extraction for proof surfaces."""

from __future__ import annotations

import csv
import io
import json
import re
from collections import Counter
from html import unescape as html_unescape
from typing import Any

from polylogue.lib.semantic_facts import normalized_role_label, sorted_counts

_HTML_ROLE_RE = re.compile(r'class="role-label">([^<]+)</span>')
_HTML_TIMESTAMP_RE = re.compile(r'class="timestamp">[^<]+</time>')
_HTML_BRANCH_RE = re.compile(r'class="branch-label">Branch\s+\d+</div>')
_HTML_TITLE_RE = re.compile(r"<h1>(.*?)</h1>", re.DOTALL)
_HTML_BADGE_RE = re.compile(r'class="badge">([^<]+)</span>')
_HTML_META_RE = re.compile(r'class="meta-item">([^<]+)</span>')


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
    conversation_id = conversation_ids.most_common(1)[0][0] if conversation_ids else ""
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
    "_csv_output_facts",
    "_html_output_facts",
    "_json_like_output_facts",
    "_markdown_doc_output_facts",
    "_obsidian_output_facts",
    "_org_output_facts",
]
