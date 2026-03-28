"""Semantic preservation proofing for rendered output surfaces."""

from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from polylogue.lib.roles import Role
from polylogue.paths import archive_root as default_archive_root
from polylogue.paths import db_path as default_db_path
from polylogue.rendering.core import ConversationFormatter
from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.repository import ConversationRepository
from polylogue.storage.store import ConversationRenderProjection, MessageRecord


def _sorted_counts(counts: dict[str, int]) -> dict[str, int]:
    return dict(sorted(counts.items()))


def _empty_metric_counts() -> dict[str, int]:
    return {"preserved": 0, "declared_loss": 0, "critical_loss": 0}


def _normalized_role(message: MessageRecord) -> str:
    role = message.role
    if isinstance(role, Role):
        return str(role)
    if role:
        return str(Role.normalize(str(role)))
    return "message"


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
        has_text = bool((message.text or "").strip())
        if has_text or has_attachments:
            renderable_messages += 1
            role_counts[_normalized_role(message)] += 1
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


def _output_facts(markdown_text: str) -> dict[str, Any]:
    message_sections = 0
    timestamp_lines = 0
    attachment_lines = 0
    role_section_counts: Counter[str] = Counter()

    for line in markdown_text.splitlines():
        if line.startswith("## "):
            section = line[3:].strip()
            if section.lower() != "attachments":
                message_sections += 1
                role_section_counts[section] += 1
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
    """Semantic proof result for one rendered conversation surface."""

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


def prove_markdown_projection_semantics(
    projection: ConversationRenderProjection,
    markdown_text: str,
) -> SemanticConversationProof:
    """Compare a repository render projection to canonical markdown output."""
    input_facts = _input_facts(projection)
    output_facts = _output_facts(markdown_text)

    checks = [
        SemanticMetricCheck(
            metric="renderable_messages",
            status=(
                "preserved"
                if input_facts["renderable_messages"] == output_facts["message_sections"]
                else "critical_loss"
            ),
            policy="canonical markdown must preserve every renderable message section",
            input_value=input_facts["renderable_messages"],
            output_value=output_facts["message_sections"],
        ),
        SemanticMetricCheck(
            metric="attachment_lines",
            status=(
                "preserved"
                if input_facts["attachment_count"] == output_facts["attachment_lines"]
                else "critical_loss"
            ),
            policy="canonical markdown must preserve attachment presence as attachment lines",
            input_value=input_facts["attachment_count"],
            output_value=output_facts["attachment_lines"],
        ),
        SemanticMetricCheck(
            metric="timestamp_lines",
            status=(
                "preserved"
                if input_facts["timestamped_renderable_messages"] == output_facts["timestamp_lines"]
                else "critical_loss"
            ),
            policy="canonical markdown must preserve timestamps for renderable messages that have them",
            input_value=input_facts["timestamped_renderable_messages"],
            output_value=output_facts["timestamp_lines"],
        ),
        SemanticMetricCheck(
            metric="role_sections",
            status=(
                "preserved"
                if input_facts["renderable_role_counts"] == output_facts["role_section_counts"]
                else "critical_loss"
            ),
            policy="canonical markdown must preserve renderable message role sections",
            input_value=input_facts["renderable_role_counts"],
            output_value=output_facts["role_section_counts"],
        ),
        SemanticMetricCheck(
            metric="empty_messages",
            status="declared_loss" if input_facts["empty_messages"] else "preserved",
            policy="canonical markdown intentionally omits messages with no text and no attachments",
            input_value=input_facts["empty_messages"],
            output_value=0,
        ),
        SemanticMetricCheck(
            metric="thinking_semantics",
            status="declared_loss" if input_facts["thinking_messages"] else "preserved",
            policy="canonical markdown preserves display text but not typed thinking markers",
            input_value=input_facts["thinking_messages"],
            output_value=output_facts["typed_thinking_markers"],
        ),
        SemanticMetricCheck(
            metric="tool_semantics",
            status="declared_loss" if input_facts["tool_messages"] else "preserved",
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


async def _prove_markdown_render_semantics_async(
    *,
    db_path: Path,
    archive_root: Path,
    providers: list[str] | None,
    record_limit: int | None,
    record_offset: int,
) -> SemanticProofReport:
    backend = SQLiteBackend(db_path=db_path)
    repository = ConversationRepository(backend=backend)
    formatter = ConversationFormatter(archive_root=archive_root, db_path=db_path, backend=backend)
    try:
        summaries = await repository.list_summaries(
            limit=record_limit,
            offset=record_offset,
            providers=providers,
        )
        proofs: list[SemanticConversationProof] = []
        for summary in summaries:
            projection = await repository.get_render_projection(str(summary.id))
            if projection is None:
                continue
            formatted = formatter.format_projection(projection)
            proofs.append(prove_markdown_projection_semantics(projection, formatted.markdown_text))
        return SemanticProofReport(
            surface="canonical_markdown_v1",
            conversations=proofs,
            provider_reports=_build_provider_reports(proofs),
            record_limit=record_limit,
            record_offset=record_offset,
            provider_filters=list(providers or []),
        )
    finally:
        await backend.close()


def prove_markdown_render_semantics(
    *,
    db_path: Path | None = None,
    archive_root: Path | None = None,
    providers: list[str] | None = None,
    record_limit: int | None = None,
    record_offset: int = 0,
) -> SemanticProofReport:
    """Run semantic preservation proof over canonical markdown rendering."""
    effective_db_path = db_path or default_db_path()
    bounded_limit = max(1, int(record_limit)) if record_limit is not None else None
    bounded_offset = max(0, int(record_offset))
    if not effective_db_path.exists():
        return SemanticProofReport(
            surface="canonical_markdown_v1",
            conversations=[],
            provider_reports={},
            record_limit=bounded_limit,
            record_offset=bounded_offset,
            provider_filters=list(providers or []),
        )
    return asyncio.run(
        _prove_markdown_render_semantics_async(
            db_path=effective_db_path,
            archive_root=archive_root or default_archive_root(),
            providers=providers,
            record_limit=bounded_limit,
            record_offset=bounded_offset,
        )
    )


__all__ = [
    "ProviderSemanticProof",
    "SemanticConversationProof",
    "SemanticMetricCheck",
    "SemanticProofReport",
    "prove_markdown_projection_semantics",
    "prove_markdown_render_semantics",
]
