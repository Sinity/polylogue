"""Semantic preservation proofing for canonical render and export surfaces."""

from __future__ import annotations

import asyncio
import csv
import io
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from html import unescape as html_unescape
from pathlib import Path
from typing import TYPE_CHECKING, Any

from polylogue.lib.roles import Role
from polylogue.paths import archive_root as default_archive_root
from polylogue.paths import db_path as default_db_path
from polylogue.rendering.core import ConversationFormatter
from polylogue.rendering.formatting import format_conversation
from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.repository import ConversationRepository
from polylogue.storage.store import ConversationRenderProjection, MessageRecord

if TYPE_CHECKING:
    from polylogue.lib.models import Conversation, Message


DEFAULT_SEMANTIC_SURFACES: tuple[str, ...] = (
    "canonical_markdown_v1",
    "export_json_v1",
    "export_yaml_v1",
    "export_csv_v1",
    "export_markdown_v1",
    "export_html_v1",
    "export_obsidian_v1",
    "export_org_v1",
)

_SURFACE_ALIASES: dict[str, tuple[str, ...]] = {
    "all": DEFAULT_SEMANTIC_SURFACES,
    "canonical": ("canonical_markdown_v1",),
    "canonical_markdown": ("canonical_markdown_v1",),
    "canonical_markdown_v1": ("canonical_markdown_v1",),
    "json": ("export_json_v1",),
    "yaml": ("export_yaml_v1",),
    "csv": ("export_csv_v1",),
    "markdown": ("export_markdown_v1",),
    "html": ("export_html_v1",),
    "obsidian": ("export_obsidian_v1",),
    "org": ("export_org_v1",),
    "export_all": tuple(surface for surface in DEFAULT_SEMANTIC_SURFACES if surface != "canonical_markdown_v1"),
}

_EXPORT_SURFACE_FORMATS: dict[str, str] = {
    "export_json_v1": "json",
    "export_yaml_v1": "yaml",
    "export_csv_v1": "csv",
    "export_markdown_v1": "markdown",
    "export_html_v1": "html",
    "export_obsidian_v1": "obsidian",
    "export_org_v1": "org",
}

_HTML_ROLE_RE = re.compile(r'class="role-label">([^<]+)</span>')
_HTML_TIMESTAMP_RE = re.compile(r'class="timestamp">[^<]+</time>')
_HTML_BRANCH_RE = re.compile(r'class="branch-label">Branch\s+\d+</div>')
_HTML_TITLE_RE = re.compile(r"<h1>(.*?)</h1>", re.DOTALL)
_HTML_BADGE_RE = re.compile(r'class="badge">([^<]+)</span>')
_HTML_META_RE = re.compile(r'class="meta-item">([^<]+)</span>')


def _sorted_counts(counts: dict[str, int]) -> dict[str, int]:
    return dict(sorted(counts.items()))


def _empty_metric_counts() -> dict[str, int]:
    return {"preserved": 0, "declared_loss": 0, "critical_loss": 0}


def _normalized_role_label(value: object) -> str:
    if isinstance(value, Role):
        return str(value)
    if value:
        return str(Role.normalize(str(value)))
    return "message"


def _is_text_message(message: Message | MessageRecord) -> bool:
    return bool((message.text or "").strip())


def _critical_or_preserved(*, metric: str, policy: str, input_value: Any, output_value: Any) -> SemanticMetricCheck:
    return SemanticMetricCheck(
        metric=metric,
        status="preserved" if input_value == output_value else "critical_loss",
        policy=policy,
        input_value=input_value,
        output_value=output_value,
    )


def _declared_loss_or_preserved(*, metric: str, policy: str, input_value: int, output_value: Any = 0) -> SemanticMetricCheck:
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


def _input_facts(projection: ConversationRenderProjection) -> dict[str, Any]:
    attachment_counts: Counter[str] = Counter(
        attachment.message_id for attachment in projection.attachments if attachment.message_id
    )
    renderable_messages = 0
    timestamped_renderable_messages = 0
    empty_messages = 0
    thinking_messages = 0
    tool_messages = 0
    role_counts: Counter[str] = Counter()

    for message in projection.messages:
        has_attachments = attachment_counts.get(message.message_id, 0) > 0
        has_text = _is_text_message(message)
        if has_text or has_attachments:
            renderable_messages += 1
            role_counts[_normalized_role_label(message.role)] += 1
            if message.sort_key is not None:
                timestamped_renderable_messages += 1
        else:
            empty_messages += 1
        if int(message.has_thinking or 0) > 0:
            thinking_messages += 1
        if int(message.has_tool_use or 0) > 0:
            tool_messages += 1

    return {
        "total_messages": len(projection.messages),
        "renderable_messages": renderable_messages,
        "timestamped_renderable_messages": timestamped_renderable_messages,
        "attachment_count": len(projection.attachments),
        "empty_messages": empty_messages,
        "thinking_messages": thinking_messages,
        "tool_messages": tool_messages,
        "renderable_role_counts": _sorted_counts(dict(role_counts)),
    }


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
                role_section_counts[_normalized_role_label(section)] += 1
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
        "role_section_counts": _sorted_counts(dict(role_section_counts)),
    }


def _conversation_input_facts(conversation: Conversation) -> dict[str, Any]:
    role_counts: Counter[str] = Counter()
    message_ids: list[str] = []
    text_message_ids: list[str] = []
    timestamped_text_messages = 0
    attachment_count = 0
    thinking_messages = 0
    tool_messages = 0
    branch_messages = 0

    for message in conversation.messages:
        message_ids.append(str(message.id))
        attachment_count += len(message.attachments)
        if message.is_thinking:
            thinking_messages += 1
        if message.is_tool_use:
            tool_messages += 1
        if message.branch_index > 0:
            branch_messages += 1
        if _is_text_message(message):
            text_message_ids.append(str(message.id))
            role_counts[_normalized_role_label(message.role)] += 1
            if message.timestamp is not None:
                timestamped_text_messages += 1

    display_date = conversation.display_date.isoformat() if conversation.display_date else None
    return {
        "conversation_id": str(conversation.id),
        "provider": str(conversation.provider),
        "title": conversation.display_title,
        "date": display_date,
        "total_messages": len(conversation.messages),
        "text_messages": len(text_message_ids),
        "message_ids": message_ids,
        "text_message_ids": text_message_ids,
        "text_role_counts": _sorted_counts(dict(role_counts)),
        "timestamped_text_messages": timestamped_text_messages,
        "attachment_count": attachment_count,
        "thinking_messages": thinking_messages,
        "tool_messages": tool_messages,
        "branch_messages": branch_messages,
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
        role_counts[_normalized_role_label(message.get("role"))] += 1
        if message.get("timestamp"):
            timestamped_messages += 1
    return {
        "conversation_id": str(payload.get("id") or ""),
        "provider": str(payload.get("provider") or ""),
        "title": payload.get("title"),
        "date": payload.get("date"),
        "messages": len(message_ids),
        "message_ids": message_ids,
        "role_counts": _sorted_counts(dict(role_counts)),
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
        role_counts[_normalized_role_label(row.get("role"))] += 1
        if row.get("timestamp"):
            timestamped_messages += 1
    conversation_id = ""
    if conversation_ids:
        conversation_id = conversation_ids.most_common(1)[0][0]
    return {
        "conversation_id": conversation_id,
        "messages": len(message_ids),
        "message_ids": message_ids,
        "role_counts": _sorted_counts(dict(role_counts)),
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
            role_counts[_normalized_role_label(role)] += 1
    return {
        "title": title,
        "provider": provider,
        "has_date": has_date,
        "message_sections": message_sections,
        "role_counts": _sorted_counts(dict(role_counts)),
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
            role_counts[_normalized_role_label(role)] += 1
    return {
        "title": title,
        "provider": provider,
        "has_date": has_date,
        "message_sections": message_sections,
        "role_counts": _sorted_counts(dict(role_counts)),
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
        _normalized_role_label(role.strip()) for role in _HTML_ROLE_RE.findall(text)
    )
    return {
        "title": title,
        "provider": provider,
        "has_date": has_date,
        "message_sections": sum(role_counts.values()),
        "role_counts": _sorted_counts(dict(role_counts)),
        "timestamp_lines": len(_HTML_TIMESTAMP_RE.findall(text)),
        "branch_labels": len(_HTML_BRANCH_RE.findall(text)),
        "conversation_id": None,
    }


@dataclass(frozen=True)
class SemanticMetricCheck:
    """One measurable preservation or declared-loss claim."""

    metric: str
    status: str
    policy: str
    input_value: Any
    output_value: Any

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "status": self.status,
            "policy": self.policy,
            "input_value": self.input_value,
            "output_value": self.output_value,
        }


@dataclass(frozen=True)
class SemanticConversationProof:
    """Semantic proof result for one rendered/exported conversation surface."""

    conversation_id: str
    provider: str
    surface: str
    input_facts: dict[str, Any]
    output_facts: dict[str, Any]
    checks: list[SemanticMetricCheck] = field(default_factory=list)

    @property
    def critical_loss_checks(self) -> list[SemanticMetricCheck]:
        return [check for check in self.checks if check.status == "critical_loss"]

    @property
    def declared_loss_checks(self) -> list[SemanticMetricCheck]:
        return [check for check in self.checks if check.status == "declared_loss"]

    @property
    def preserved_checks(self) -> list[SemanticMetricCheck]:
        return [check for check in self.checks if check.status == "preserved"]

    @property
    def metric_summary(self) -> dict[str, dict[str, int]]:
        summary: dict[str, dict[str, int]] = {}
        for check in self.checks:
            counts = summary.setdefault(check.metric, _empty_metric_counts())
            counts[check.status] += 1
        return dict(sorted(summary.items()))

    @property
    def is_clean(self) -> bool:
        return not self.critical_loss_checks

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "provider": self.provider,
            "surface": self.surface,
            "input_facts": self.input_facts,
            "output_facts": self.output_facts,
            "summary": {
                "preserved_checks": len(self.preserved_checks),
                "declared_loss_checks": len(self.declared_loss_checks),
                "critical_loss_checks": len(self.critical_loss_checks),
                "metric_summary": self.metric_summary,
                "clean": self.is_clean,
            },
            "checks": [check.to_dict() for check in self.checks],
        }


@dataclass(frozen=True)
class ProviderSemanticProof:
    """Per-provider semantic-preservation proof summary."""

    provider: str
    total_conversations: int = 0
    clean_conversations: int = 0
    critical_conversations: int = 0
    preserved_checks: int = 0
    declared_loss_checks: int = 0
    critical_loss_checks: int = 0
    metric_summary: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def is_clean(self) -> bool:
        return self.critical_conversations == 0

    @property
    def clean(self) -> bool:
        return self.is_clean

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "total_conversations": self.total_conversations,
            "clean_conversations": self.clean_conversations,
            "critical_conversations": self.critical_conversations,
            "preserved_checks": self.preserved_checks,
            "declared_loss_checks": self.declared_loss_checks,
            "critical_loss_checks": self.critical_loss_checks,
            "metric_summary": dict(sorted(self.metric_summary.items())),
            "clean": self.clean,
        }


def _build_provider_reports(
    conversations: list[SemanticConversationProof],
) -> dict[str, ProviderSemanticProof]:
    provider_totals: dict[str, dict[str, Any]] = {}
    for proof in conversations:
        state = provider_totals.setdefault(
            proof.provider,
            {
                "total_conversations": 0,
                "clean_conversations": 0,
                "critical_conversations": 0,
                "preserved_checks": 0,
                "declared_loss_checks": 0,
                "critical_loss_checks": 0,
                "metric_summary": {},
            },
        )
        state["total_conversations"] += 1
        if proof.is_clean:
            state["clean_conversations"] += 1
        else:
            state["critical_conversations"] += 1
        state["preserved_checks"] += len(proof.preserved_checks)
        state["declared_loss_checks"] += len(proof.declared_loss_checks)
        state["critical_loss_checks"] += len(proof.critical_loss_checks)
        for metric, counts in proof.metric_summary.items():
            metric_counts = state["metric_summary"].setdefault(metric, _empty_metric_counts())
            for status, count in counts.items():
                metric_counts[status] += count

    return {
        provider: ProviderSemanticProof(
            provider=provider,
            total_conversations=state["total_conversations"],
            clean_conversations=state["clean_conversations"],
            critical_conversations=state["critical_conversations"],
            preserved_checks=state["preserved_checks"],
            declared_loss_checks=state["declared_loss_checks"],
            critical_loss_checks=state["critical_loss_checks"],
            metric_summary=dict(sorted(state["metric_summary"].items())),
        )
        for provider, state in sorted(provider_totals.items())
    }


@dataclass(frozen=True)
class SemanticProofReport:
    """Aggregate semantic proof report for one output surface."""

    surface: str
    conversations: list[SemanticConversationProof]
    provider_reports: dict[str, ProviderSemanticProof]
    record_limit: int | None = None
    record_offset: int = 0
    provider_filters: list[str] = field(default_factory=list)

    @property
    def total_conversations(self) -> int:
        return len(self.conversations)

    @property
    def provider_count(self) -> int:
        return len(self.provider_reports)

    @property
    def providers(self) -> dict[str, ProviderSemanticProof]:
        return self.provider_reports

    @property
    def clean_conversations(self) -> int:
        return sum(1 for proof in self.conversations if proof.is_clean)

    @property
    def critical_conversations(self) -> int:
        return sum(1 for proof in self.conversations if not proof.is_clean)

    @property
    def preserved_checks(self) -> int:
        return sum(len(proof.preserved_checks) for proof in self.conversations)

    @property
    def declared_loss_checks(self) -> int:
        return sum(len(proof.declared_loss_checks) for proof in self.conversations)

    @property
    def critical_loss_checks(self) -> int:
        return sum(len(proof.critical_loss_checks) for proof in self.conversations)

    @property
    def metric_summary(self) -> dict[str, dict[str, int]]:
        summary: dict[str, dict[str, int]] = {}
        for proof in self.conversations:
            for check in proof.checks:
                metric = summary.setdefault(check.metric, _empty_metric_counts())
                metric[check.status] += 1
        return dict(sorted(summary.items()))

    @property
    def is_clean(self) -> bool:
        return self.critical_conversations == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "surface": self.surface,
            "record_limit": self.record_limit if self.record_limit is not None else "all",
            "record_offset": self.record_offset,
            "provider_filters": list(self.provider_filters),
            "summary": {
                "total_conversations": self.total_conversations,
                "provider_count": self.provider_count,
                "clean_conversations": self.clean_conversations,
                "critical_conversations": self.critical_conversations,
                "preserved_checks": self.preserved_checks,
                "declared_loss_checks": self.declared_loss_checks,
                "critical_loss_checks": self.critical_loss_checks,
                "metric_summary": self.metric_summary,
                "clean": self.is_clean,
            },
            "providers": {
                provider: stats.to_dict() for provider, stats in sorted(self.provider_reports.items())
            },
            "conversations": [proof.to_dict() for proof in self.conversations],
        }


@dataclass(frozen=True)
class SemanticProofSuiteReport:
    """Aggregate semantic proof report spanning multiple output surfaces."""

    surface_reports: dict[str, SemanticProofReport]
    record_limit: int | None = None
    record_offset: int = 0
    provider_filters: list[str] = field(default_factory=list)
    surface_filters: list[str] = field(default_factory=list)

    @property
    def surfaces(self) -> dict[str, SemanticProofReport]:
        return self.surface_reports

    @property
    def surface_count(self) -> int:
        return len(self.surface_reports)

    @property
    def clean_surfaces(self) -> int:
        return sum(1 for report in self.surface_reports.values() if report.is_clean)

    @property
    def critical_surfaces(self) -> int:
        return sum(1 for report in self.surface_reports.values() if not report.is_clean)

    @property
    def total_conversations(self) -> int:
        return sum(report.total_conversations for report in self.surface_reports.values())

    @property
    def preserved_checks(self) -> int:
        return sum(report.preserved_checks for report in self.surface_reports.values())

    @property
    def declared_loss_checks(self) -> int:
        return sum(report.declared_loss_checks for report in self.surface_reports.values())

    @property
    def critical_loss_checks(self) -> int:
        return sum(report.critical_loss_checks for report in self.surface_reports.values())

    @property
    def metric_summary(self) -> dict[str, dict[str, int]]:
        summary: dict[str, dict[str, int]] = {}
        for report in self.surface_reports.values():
            for metric, counts in report.metric_summary.items():
                metric_counts = summary.setdefault(metric, _empty_metric_counts())
                for status, count in counts.items():
                    metric_counts[status] += count
        return dict(sorted(summary.items()))

    @property
    def is_clean(self) -> bool:
        return self.critical_surfaces == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_limit": self.record_limit if self.record_limit is not None else "all",
            "record_offset": self.record_offset,
            "provider_filters": list(self.provider_filters),
            "surface_filters": list(self.surface_filters),
            "summary": {
                "surface_count": self.surface_count,
                "clean_surfaces": self.clean_surfaces,
                "critical_surfaces": self.critical_surfaces,
                "total_conversations": self.total_conversations,
                "preserved_checks": self.preserved_checks,
                "declared_loss_checks": self.declared_loss_checks,
                "critical_loss_checks": self.critical_loss_checks,
                "metric_summary": self.metric_summary,
                "clean": self.is_clean,
            },
            "surfaces": {
                surface: report.to_dict()
                for surface, report in sorted(self.surface_reports.items())
            },
        }


def resolve_semantic_surfaces(surfaces: list[str] | tuple[str, ...] | None) -> list[str]:
    """Normalize semantic-proof surface filters to canonical surface names."""
    if not surfaces:
        return list(DEFAULT_SEMANTIC_SURFACES)

    resolved: list[str] = []
    seen: set[str] = set()
    for surface in surfaces:
        token = str(surface).strip().lower().replace("-", "_")
        aliases = _SURFACE_ALIASES.get(token)
        if aliases is None:
            raise ValueError(
                "Unknown semantic surface "
                f"{surface!r}. Valid values: {', '.join(sorted(_SURFACE_ALIASES))}"
            )
        for alias in aliases:
            if alias not in seen:
                seen.add(alias)
                resolved.append(alias)
    return resolved


def prove_markdown_projection_semantics(
    projection: ConversationRenderProjection,
    markdown_text: str,
) -> SemanticConversationProof:
    """Compare a repository render projection to canonical markdown output."""
    input_facts = _input_facts(projection)
    output_facts = _canonical_markdown_output_facts(markdown_text)

    checks = [
        _critical_or_preserved(
            metric="renderable_messages",
            policy="canonical markdown must preserve every renderable message section",
            input_value=input_facts["renderable_messages"],
            output_value=output_facts["message_sections"],
        ),
        _critical_or_preserved(
            metric="attachment_lines",
            policy="canonical markdown must preserve attachment presence as attachment lines",
            input_value=input_facts["attachment_count"],
            output_value=output_facts["attachment_lines"],
        ),
        _critical_or_preserved(
            metric="timestamp_lines",
            policy="canonical markdown must preserve timestamps for renderable messages that have them",
            input_value=input_facts["timestamped_renderable_messages"],
            output_value=output_facts["timestamp_lines"],
        ),
        _critical_or_preserved(
            metric="role_sections",
            policy="canonical markdown must preserve renderable message role sections",
            input_value=input_facts["renderable_role_counts"],
            output_value=output_facts["role_section_counts"],
        ),
        _declared_loss_or_preserved(
            metric="empty_messages",
            policy="canonical markdown intentionally omits messages with no text and no attachments",
            input_value=input_facts["empty_messages"],
        ),
        _declared_loss_or_preserved(
            metric="thinking_semantics",
            policy="canonical markdown preserves display text but not typed thinking markers",
            input_value=input_facts["thinking_messages"],
            output_value=output_facts["typed_thinking_markers"],
        ),
        _declared_loss_or_preserved(
            metric="tool_semantics",
            policy="canonical markdown preserves display text but not typed tool markers",
            input_value=input_facts["tool_messages"],
            output_value=output_facts["typed_tool_markers"],
        ),
    ]

    return SemanticConversationProof(
        conversation_id=projection.conversation.conversation_id,
        provider=projection.conversation.provider_name or "unknown",
        surface="canonical_markdown_v1",
        input_facts=input_facts,
        output_facts=output_facts,
        checks=checks,
    )


def _prove_export_json_like_surface(
    *,
    conversation: Conversation,
    rendered_text: str,
    surface: str,
) -> SemanticConversationProof:
    try:
        if surface == "export_yaml_v1":
            import yaml

            parsed = yaml.safe_load(rendered_text)
        else:
            parsed = json.loads(rendered_text)
    except Exception:
        parsed = {}
    payload = parsed if isinstance(parsed, dict) else {}
    input_facts = _conversation_input_facts(conversation)
    output_facts = _json_like_output_facts(payload)
    checks = [
        _critical_or_preserved(
            metric="conversation_id",
            policy=f"{surface} must preserve the conversation identifier",
            input_value=input_facts["conversation_id"],
            output_value=output_facts["conversation_id"],
        ),
        _critical_or_preserved(
            metric="provider_identity",
            policy=f"{surface} must preserve provider identity",
            input_value=input_facts["provider"],
            output_value=output_facts["provider"],
        ),
        _critical_or_preserved(
            metric="title_metadata",
            policy=f"{surface} must preserve the display title",
            input_value=input_facts["title"],
            output_value=output_facts["title"],
        ),
        _critical_or_preserved(
            metric="date_metadata",
            policy=f"{surface} must preserve the display date value when present",
            input_value=input_facts["date"],
            output_value=output_facts["date"],
        ),
        _critical_or_preserved(
            metric="message_entries",
            policy=f"{surface} must preserve every message entry",
            input_value=input_facts["total_messages"],
            output_value=output_facts["messages"],
        ),
        _critical_or_preserved(
            metric="message_ids",
            policy=f"{surface} must preserve message identifiers",
            input_value=input_facts["message_ids"],
            output_value=output_facts["message_ids"],
        ),
        _critical_or_preserved(
            metric="role_entries",
            policy=f"{surface} must preserve message role distribution",
            input_value=input_facts["text_role_counts"],
            output_value=output_facts["role_counts"],
        ),
        _critical_or_preserved(
            metric="timestamp_values",
            policy=f"{surface} must preserve message timestamps",
            input_value=input_facts["timestamped_text_messages"],
            output_value=output_facts["timestamped_messages"],
        ),
        _declared_loss_or_preserved(
            metric="attachment_semantics",
            policy=f"{surface} intentionally omits attachment payload semantics",
            input_value=input_facts["attachment_count"],
        ),
        _declared_loss_or_preserved(
            metric="thinking_semantics",
            policy=f"{surface} preserves display text but not typed thinking markers",
            input_value=input_facts["thinking_messages"],
        ),
        _declared_loss_or_preserved(
            metric="tool_semantics",
            policy=f"{surface} preserves display text but not typed tool markers",
            input_value=input_facts["tool_messages"],
        ),
        _declared_loss_or_preserved(
            metric="branch_structure",
            policy=f"{surface} intentionally omits explicit branch topology",
            input_value=input_facts["branch_messages"],
        ),
    ]
    return SemanticConversationProof(
        conversation_id=input_facts["conversation_id"],
        provider=input_facts["provider"],
        surface=surface,
        input_facts=input_facts,
        output_facts=output_facts,
        checks=checks,
    )


def _prove_export_csv_surface(
    *,
    conversation: Conversation,
    rendered_text: str,
) -> SemanticConversationProof:
    input_facts = _conversation_input_facts(conversation)
    output_facts = _csv_output_facts(rendered_text)
    checks = [
        _critical_or_preserved(
            metric="conversation_id",
            policy="export_csv_v1 must preserve the conversation identifier per row",
            input_value=input_facts["conversation_id"],
            output_value=output_facts["conversation_id"],
        ),
        _critical_or_preserved(
            metric="text_messages",
            policy="export_csv_v1 must preserve one row per text-bearing message",
            input_value=input_facts["text_messages"],
            output_value=output_facts["messages"],
        ),
        _critical_or_preserved(
            metric="text_message_ids",
            policy="export_csv_v1 must preserve identifiers for text-bearing messages",
            input_value=input_facts["text_message_ids"],
            output_value=output_facts["message_ids"],
        ),
        _critical_or_preserved(
            metric="role_entries",
            policy="export_csv_v1 must preserve roles for text-bearing messages",
            input_value=input_facts["text_role_counts"],
            output_value=output_facts["role_counts"],
        ),
        _critical_or_preserved(
            metric="timestamp_values",
            policy="export_csv_v1 must preserve timestamps for text-bearing messages",
            input_value=input_facts["timestamped_text_messages"],
            output_value=output_facts["timestamped_messages"],
        ),
        _declared_loss_or_preserved(
            metric="provider_identity",
            policy="export_csv_v1 intentionally omits conversation-level provider metadata",
            input_value=1 if input_facts["provider"] else 0,
            output_value=0,
        ),
        _declared_loss_or_preserved(
            metric="title_metadata",
            policy="export_csv_v1 intentionally omits conversation-level title metadata",
            input_value=1 if input_facts["title"] else 0,
            output_value=0,
        ),
        _declared_loss_or_preserved(
            metric="date_metadata",
            policy="export_csv_v1 intentionally omits conversation-level date metadata",
            input_value=1 if input_facts["date"] else 0,
            output_value=0,
        ),
        _declared_loss_or_preserved(
            metric="attachment_semantics",
            policy="export_csv_v1 intentionally omits attachment payload semantics",
            input_value=input_facts["attachment_count"],
        ),
        _declared_loss_or_preserved(
            metric="thinking_semantics",
            policy="export_csv_v1 preserves display text but not typed thinking markers",
            input_value=input_facts["thinking_messages"],
        ),
        _declared_loss_or_preserved(
            metric="tool_semantics",
            policy="export_csv_v1 preserves display text but not typed tool markers",
            input_value=input_facts["tool_messages"],
        ),
        _declared_loss_or_preserved(
            metric="branch_structure",
            policy="export_csv_v1 intentionally omits explicit branch topology",
            input_value=input_facts["branch_messages"],
        ),
    ]
    return SemanticConversationProof(
        conversation_id=input_facts["conversation_id"],
        provider=input_facts["provider"],
        surface="export_csv_v1",
        input_facts=input_facts,
        output_facts=output_facts,
        checks=checks,
    )


def _prove_export_markdown_surface(
    *,
    conversation: Conversation,
    rendered_text: str,
) -> SemanticConversationProof:
    input_facts = _conversation_input_facts(conversation)
    output_facts = _markdown_doc_output_facts(rendered_text)
    checks = [
        _critical_or_preserved(
            metric="title_metadata",
            policy="export_markdown_v1 must preserve the display title",
            input_value=input_facts["title"],
            output_value=output_facts["title"],
        ),
        _critical_or_preserved(
            metric="provider_identity",
            policy="export_markdown_v1 must preserve provider identity at document level",
            input_value=input_facts["provider"],
            output_value=output_facts["provider"],
        ),
        _presence_check(
            metric="date_metadata",
            policy="export_markdown_v1 must preserve conversation date presence at document level",
            input_value=input_facts["date"],
            output_present=bool(output_facts["has_date"]),
        ),
        _critical_or_preserved(
            metric="text_messages",
            policy="export_markdown_v1 must preserve one section per text-bearing message",
            input_value=input_facts["text_messages"],
            output_value=output_facts["message_sections"],
        ),
        _critical_or_preserved(
            metric="role_sections",
            policy="export_markdown_v1 must preserve role sections for text-bearing messages",
            input_value=input_facts["text_role_counts"],
            output_value=output_facts["role_counts"],
        ),
        _declared_loss_or_preserved(
            metric="timestamp_values",
            policy="export_markdown_v1 intentionally omits per-message timestamps",
            input_value=input_facts["timestamped_text_messages"],
            output_value=output_facts["timestamp_lines"],
        ),
        _declared_loss_or_preserved(
            metric="attachment_semantics",
            policy="export_markdown_v1 intentionally omits attachment payload semantics",
            input_value=input_facts["attachment_count"],
        ),
        _declared_loss_or_preserved(
            metric="thinking_semantics",
            policy="export_markdown_v1 preserves display text but not typed thinking markers",
            input_value=input_facts["thinking_messages"],
        ),
        _declared_loss_or_preserved(
            metric="tool_semantics",
            policy="export_markdown_v1 preserves display text but not typed tool markers",
            input_value=input_facts["tool_messages"],
        ),
        _declared_loss_or_preserved(
            metric="branch_structure",
            policy="export_markdown_v1 intentionally omits explicit branch topology",
            input_value=input_facts["branch_messages"],
            output_value=output_facts["branch_labels"],
        ),
    ]
    return SemanticConversationProof(
        conversation_id=input_facts["conversation_id"],
        provider=input_facts["provider"],
        surface="export_markdown_v1",
        input_facts=input_facts,
        output_facts=output_facts,
        checks=checks,
    )


def _prove_export_obsidian_surface(
    *,
    conversation: Conversation,
    rendered_text: str,
) -> SemanticConversationProof:
    input_facts = _conversation_input_facts(conversation)
    output_facts = _obsidian_output_facts(rendered_text)
    checks = [
        _critical_or_preserved(
            metric="conversation_id",
            policy="export_obsidian_v1 must preserve conversation identity in frontmatter",
            input_value=input_facts["conversation_id"],
            output_value=output_facts["conversation_id"],
        ),
        _critical_or_preserved(
            metric="title_metadata",
            policy="export_obsidian_v1 must preserve the display title",
            input_value=input_facts["title"],
            output_value=output_facts["title"],
        ),
        _critical_or_preserved(
            metric="provider_identity",
            policy="export_obsidian_v1 must preserve provider identity in frontmatter",
            input_value=input_facts["provider"],
            output_value=output_facts["provider"],
        ),
        _presence_check(
            metric="date_metadata",
            policy="export_obsidian_v1 must preserve conversation date presence in frontmatter",
            input_value=input_facts["date"],
            output_present=bool(output_facts["has_date"]),
        ),
        _critical_or_preserved(
            metric="text_messages",
            policy="export_obsidian_v1 must preserve one section per text-bearing message",
            input_value=input_facts["text_messages"],
            output_value=output_facts["message_sections"],
        ),
        _critical_or_preserved(
            metric="role_sections",
            policy="export_obsidian_v1 must preserve role sections for text-bearing messages",
            input_value=input_facts["text_role_counts"],
            output_value=output_facts["role_counts"],
        ),
        _declared_loss_or_preserved(
            metric="timestamp_values",
            policy="export_obsidian_v1 intentionally omits per-message timestamps",
            input_value=input_facts["timestamped_text_messages"],
            output_value=output_facts["timestamp_lines"],
        ),
        _declared_loss_or_preserved(
            metric="attachment_semantics",
            policy="export_obsidian_v1 intentionally omits attachment payload semantics",
            input_value=input_facts["attachment_count"],
        ),
        _declared_loss_or_preserved(
            metric="thinking_semantics",
            policy="export_obsidian_v1 preserves display text but not typed thinking markers",
            input_value=input_facts["thinking_messages"],
        ),
        _declared_loss_or_preserved(
            metric="tool_semantics",
            policy="export_obsidian_v1 preserves display text but not typed tool markers",
            input_value=input_facts["tool_messages"],
        ),
        _declared_loss_or_preserved(
            metric="branch_structure",
            policy="export_obsidian_v1 intentionally omits explicit branch topology",
            input_value=input_facts["branch_messages"],
            output_value=output_facts["branch_labels"],
        ),
    ]
    return SemanticConversationProof(
        conversation_id=input_facts["conversation_id"],
        provider=input_facts["provider"],
        surface="export_obsidian_v1",
        input_facts=input_facts,
        output_facts=output_facts,
        checks=checks,
    )


def _prove_export_org_surface(
    *,
    conversation: Conversation,
    rendered_text: str,
) -> SemanticConversationProof:
    input_facts = _conversation_input_facts(conversation)
    output_facts = _org_output_facts(rendered_text)
    checks = [
        _critical_or_preserved(
            metric="title_metadata",
            policy="export_org_v1 must preserve the display title",
            input_value=input_facts["title"],
            output_value=output_facts["title"],
        ),
        _critical_or_preserved(
            metric="provider_identity",
            policy="export_org_v1 must preserve provider identity at document level",
            input_value=input_facts["provider"],
            output_value=output_facts["provider"],
        ),
        _presence_check(
            metric="date_metadata",
            policy="export_org_v1 must preserve conversation date presence at document level",
            input_value=input_facts["date"],
            output_present=bool(output_facts["has_date"]),
        ),
        _critical_or_preserved(
            metric="text_messages",
            policy="export_org_v1 must preserve one heading per text-bearing message",
            input_value=input_facts["text_messages"],
            output_value=output_facts["message_sections"],
        ),
        _critical_or_preserved(
            metric="role_sections",
            policy="export_org_v1 must preserve role headings for text-bearing messages",
            input_value=input_facts["text_role_counts"],
            output_value=output_facts["role_counts"],
        ),
        _declared_loss_or_preserved(
            metric="timestamp_values",
            policy="export_org_v1 intentionally omits per-message timestamps",
            input_value=input_facts["timestamped_text_messages"],
            output_value=output_facts["timestamp_lines"],
        ),
        _declared_loss_or_preserved(
            metric="attachment_semantics",
            policy="export_org_v1 intentionally omits attachment payload semantics",
            input_value=input_facts["attachment_count"],
        ),
        _declared_loss_or_preserved(
            metric="thinking_semantics",
            policy="export_org_v1 preserves display text but not typed thinking markers",
            input_value=input_facts["thinking_messages"],
        ),
        _declared_loss_or_preserved(
            metric="tool_semantics",
            policy="export_org_v1 preserves display text but not typed tool markers",
            input_value=input_facts["tool_messages"],
        ),
        _declared_loss_or_preserved(
            metric="branch_structure",
            policy="export_org_v1 intentionally omits explicit branch topology",
            input_value=input_facts["branch_messages"],
            output_value=output_facts["branch_labels"],
        ),
    ]
    return SemanticConversationProof(
        conversation_id=input_facts["conversation_id"],
        provider=input_facts["provider"],
        surface="export_org_v1",
        input_facts=input_facts,
        output_facts=output_facts,
        checks=checks,
    )


def _prove_export_html_surface(
    *,
    conversation: Conversation,
    rendered_text: str,
) -> SemanticConversationProof:
    input_facts = _conversation_input_facts(conversation)
    output_facts = _html_output_facts(rendered_text)
    checks = [
        _critical_or_preserved(
            metric="title_metadata",
            policy="export_html_v1 must preserve the display title",
            input_value=input_facts["title"],
            output_value=output_facts["title"],
        ),
        _critical_or_preserved(
            metric="provider_identity",
            policy="export_html_v1 must preserve provider identity at document level",
            input_value=input_facts["provider"],
            output_value=output_facts["provider"],
        ),
        _presence_check(
            metric="date_metadata",
            policy="export_html_v1 must preserve conversation date presence at document level",
            input_value=input_facts["date"],
            output_present=bool(output_facts["has_date"]),
        ),
        _critical_or_preserved(
            metric="text_messages",
            policy="export_html_v1 must preserve visible message sections for text-bearing messages",
            input_value=input_facts["text_messages"],
            output_value=output_facts["message_sections"],
        ),
        _critical_or_preserved(
            metric="role_sections",
            policy="export_html_v1 must preserve visible role labels for text-bearing messages",
            input_value=input_facts["text_role_counts"],
            output_value=output_facts["role_counts"],
        ),
        _critical_or_preserved(
            metric="timestamp_values",
            policy="export_html_v1 must preserve visible message timestamps",
            input_value=input_facts["timestamped_text_messages"],
            output_value=output_facts["timestamp_lines"],
        ),
        _critical_or_preserved(
            metric="branch_structure",
            policy="export_html_v1 must preserve visible branch groupings for branched messages",
            input_value=input_facts["branch_messages"],
            output_value=output_facts["branch_labels"],
        ),
        _declared_loss_or_preserved(
            metric="attachment_semantics",
            policy="export_html_v1 intentionally omits attachment payload semantics",
            input_value=input_facts["attachment_count"],
        ),
        _declared_loss_or_preserved(
            metric="thinking_semantics",
            policy="export_html_v1 preserves display text but not typed thinking markers",
            input_value=input_facts["thinking_messages"],
        ),
        _declared_loss_or_preserved(
            metric="tool_semantics",
            policy="export_html_v1 preserves display text but not typed tool markers",
            input_value=input_facts["tool_messages"],
        ),
    ]
    return SemanticConversationProof(
        conversation_id=input_facts["conversation_id"],
        provider=input_facts["provider"],
        surface="export_html_v1",
        input_facts=input_facts,
        output_facts=output_facts,
        checks=checks,
    )


def prove_export_surface_semantics(
    conversation: Conversation,
    surface: str,
    rendered_text: str,
) -> SemanticConversationProof:
    """Compare a conversation export surface to the canonical conversation semantics."""
    if surface == "export_json_v1":
        return _prove_export_json_like_surface(
            conversation=conversation,
            rendered_text=rendered_text,
            surface=surface,
        )
    if surface == "export_yaml_v1":
        return _prove_export_json_like_surface(
            conversation=conversation,
            rendered_text=rendered_text,
            surface=surface,
        )
    if surface == "export_csv_v1":
        return _prove_export_csv_surface(conversation=conversation, rendered_text=rendered_text)
    if surface == "export_markdown_v1":
        return _prove_export_markdown_surface(conversation=conversation, rendered_text=rendered_text)
    if surface == "export_obsidian_v1":
        return _prove_export_obsidian_surface(conversation=conversation, rendered_text=rendered_text)
    if surface == "export_org_v1":
        return _prove_export_org_surface(conversation=conversation, rendered_text=rendered_text)
    if surface == "export_html_v1":
        return _prove_export_html_surface(conversation=conversation, rendered_text=rendered_text)
    raise ValueError(f"Unsupported semantic surface: {surface}")


def _empty_surface_report(
    surface: str,
    *,
    record_limit: int | None,
    record_offset: int,
    provider_filters: list[str],
) -> SemanticProofReport:
    return SemanticProofReport(
        surface=surface,
        conversations=[],
        provider_reports={},
        record_limit=record_limit,
        record_offset=record_offset,
        provider_filters=provider_filters,
    )


async def _prove_semantic_surface_suite_async(
    *,
    db_path: Path,
    archive_root: Path,
    providers: list[str] | None,
    surfaces: list[str],
    record_limit: int | None,
    record_offset: int,
) -> SemanticProofSuiteReport:
    backend = SQLiteBackend(db_path=db_path)
    repository = ConversationRepository(backend=backend)
    formatter = ConversationFormatter(archive_root=archive_root, db_path=db_path, backend=backend)
    try:
        summaries = await repository.list_summaries(
            limit=record_limit,
            offset=record_offset,
            providers=providers,
        )
        proofs_by_surface: dict[str, list[SemanticConversationProof]] = {surface: [] for surface in surfaces}
        need_export_surfaces = any(surface != "canonical_markdown_v1" for surface in surfaces)

        for summary in summaries:
            conversation_id = str(summary.id)
            conversation = await repository.view(conversation_id) if need_export_surfaces else None
            projection = None
            for surface in surfaces:
                if surface == "canonical_markdown_v1":
                    if projection is None:
                        projection = await repository.get_render_projection(conversation_id)
                    if projection is None:
                        continue
                    formatted = formatter.format_projection(projection)
                    proofs_by_surface[surface].append(
                        prove_markdown_projection_semantics(projection, formatted.markdown_text)
                    )
                    continue

                if conversation is None:
                    continue
                rendered_text = format_conversation(conversation, _EXPORT_SURFACE_FORMATS[surface], None)
                proofs_by_surface[surface].append(
                    prove_export_surface_semantics(conversation, surface, rendered_text)
                )

        provider_filters = list(providers or [])
        return SemanticProofSuiteReport(
            surface_reports={
                surface: SemanticProofReport(
                    surface=surface,
                    conversations=proofs_by_surface[surface],
                    provider_reports=_build_provider_reports(proofs_by_surface[surface]),
                    record_limit=record_limit,
                    record_offset=record_offset,
                    provider_filters=provider_filters,
                )
                for surface in surfaces
            },
            record_limit=record_limit,
            record_offset=record_offset,
            provider_filters=provider_filters,
            surface_filters=list(surfaces),
        )
    finally:
        await backend.close()


def prove_semantic_surface_suite(
    *,
    db_path: Path | None = None,
    archive_root: Path | None = None,
    providers: list[str] | None = None,
    surfaces: list[str] | tuple[str, ...] | None = None,
    record_limit: int | None = None,
    record_offset: int = 0,
) -> SemanticProofSuiteReport:
    """Run semantic preservation proof across canonical render and export surfaces."""
    effective_db_path = db_path or default_db_path()
    bounded_limit = max(1, int(record_limit)) if record_limit is not None else None
    bounded_offset = max(0, int(record_offset))
    resolved_surfaces = resolve_semantic_surfaces(surfaces)
    provider_filters = list(providers or [])

    if not effective_db_path.exists():
        return SemanticProofSuiteReport(
            surface_reports={
                surface: _empty_surface_report(
                    surface,
                    record_limit=bounded_limit,
                    record_offset=bounded_offset,
                    provider_filters=provider_filters,
                )
                for surface in resolved_surfaces
            },
            record_limit=bounded_limit,
            record_offset=bounded_offset,
            provider_filters=provider_filters,
            surface_filters=list(resolved_surfaces),
        )

    return asyncio.run(
        _prove_semantic_surface_suite_async(
            db_path=effective_db_path,
            archive_root=archive_root or default_archive_root(),
            providers=providers,
            surfaces=resolved_surfaces,
            record_limit=bounded_limit,
            record_offset=bounded_offset,
        )
    )


def prove_markdown_render_semantics(
    *,
    db_path: Path | None = None,
    archive_root: Path | None = None,
    providers: list[str] | None = None,
    record_limit: int | None = None,
    record_offset: int = 0,
) -> SemanticProofReport:
    """Run semantic preservation proof over canonical markdown rendering."""
    suite = prove_semantic_surface_suite(
        db_path=db_path,
        archive_root=archive_root,
        providers=providers,
        surfaces=["canonical_markdown_v1"],
        record_limit=record_limit,
        record_offset=record_offset,
    )
    return suite.surfaces["canonical_markdown_v1"]


__all__ = [
    "DEFAULT_SEMANTIC_SURFACES",
    "ProviderSemanticProof",
    "SemanticConversationProof",
    "SemanticMetricCheck",
    "SemanticProofReport",
    "SemanticProofSuiteReport",
    "prove_export_surface_semantics",
    "prove_markdown_projection_semantics",
    "prove_markdown_render_semantics",
    "prove_semantic_surface_suite",
    "resolve_semantic_surfaces",
]
