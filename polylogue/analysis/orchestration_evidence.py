"""Read-only orchestration evidence over retained session records.

This projection preserves the distinction between requested settings, recorded
message models, cumulative usage counters, and account quota snapshots. It does
not inspect source files or infer identities from agent names or dialogue.
"""

from __future__ import annotations

import itertools
import re
import shlex
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal, cast

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from polylogue.analysis.topology import SessionTopology
    from polylogue.archive.message.models import Message
    from polylogue.archive.session.events import SessionEvent
    from polylogue.storage.sqlite.archive_tiers.archive_query_reads import ArchiveDelegationQueryRow

_LIMIT = 1000
_BEAD_ID = re.compile(r"[A-Za-z][A-Za-z0-9_-]*-[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)*\Z")
_LAUNCH_TOOLS = {"Agent", "Task", "spawn_agent"}
# polylogue-w4hfb: ``Message``'s four per-message token lanes are nullable and
# ``None`` means "the provider reported no counter at this grain", which is a
# different fact from a measured ``0`` (archive/message/models.py, polylogue-
# qgyuj). This projection therefore never orders a lane against an int without
# first separating the unmeasured case, and never folds unmeasured into zero.
_MESSAGE_TOKEN_LANES = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")
_SHELL_TOOLS = {"Bash", "exec_command", "shell_command", "shell"}


class SessionOrchestrationEvidence(BaseModel):
    """Versioned owner response; absent measurements remain JSON null."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    version: Literal[1] = 1
    outcome: Literal["ok", "degraded"]
    session_id: str
    children: list[dict[str, object]]
    topology: dict[str, object] | None
    launches: list[dict[str, object]]
    model_segments: list[dict[str, object]]
    bead_mentions: list[dict[str, object]]
    rate_limits: list[dict[str, object]]
    usage: dict[str, object]
    coverage: dict[str, object]
    gaps: list[str]


def _mapping(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, dict) else {}


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _numbers(value: object) -> dict[str, int]:
    return {
        key: item
        for key, item in _mapping(value).items()
        if key.endswith("tokens") and isinstance(item, int) and not isinstance(item, bool) and item >= 0
    }


def _event_ref(event: SessionEvent) -> dict[str, object]:
    return {
        "event_id": str(event.id),
        "event_index": event.event_index,
        "event_type": event.event_type,
        "event_time": event.timestamp.isoformat() if event.timestamp else None,
        "source_message_id": event.source_message_id,
        "source_message_native_id": event.source_message_provider_id,
        "raw_id": event.raw_id,
        "source_index": event.payload.get("source_index"),
    }


def _bead_ids(tool_name: str, arguments: Mapping[str, object]) -> list[str]:
    values: list[str] = []
    if "beads" in tool_name.lower() or tool_name.startswith("bd_"):
        for key in ("id", "issue_id", "bead_id", "ids", "issue_ids", "bead_ids"):
            value = arguments.get(key)
            if isinstance(value, str):
                values.append(value)
            elif isinstance(value, list):
                values.extend(item for item in value if isinstance(item, str))
    elif tool_name.rsplit(".", 1)[-1] in _SHELL_TOOLS:
        command = arguments.get("command", arguments.get("cmd"))
        if isinstance(command, str):
            try:
                tokens = shlex.split(command)
            except ValueError:
                return []
            # Only direct bd invocations. Quoted prose, scripts and arbitrary
            # shell composition do not establish a task-tool interaction.
            if (
                len(tokens) > 2
                and tokens[0].rsplit("/", 1)[-1] == "bd"
                and tokens[1] in {"show", "update", "close", "reopen", "claim", "delete", "note"}
            ):
                for token in tokens[2:]:
                    if not _BEAD_ID.fullmatch(token):
                        break
                    values.append(token)
    return sorted({value for value in values if _BEAD_ID.fullmatch(value)})


class _Section:
    """One evidence section: its full count and its first ``_LIMIT`` rows."""

    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []
        self.count = 0

    def append(self, row: dict[str, object]) -> None:
        self.count += 1
        if len(self.rows) < _LIMIT:
            self.rows.append(row)


class _TimeSpan:
    def __init__(self) -> None:
        self.start: datetime | None = None
        self.end: datetime | None = None

    def observe(self, stamp: datetime | None) -> None:
        if stamp is None:
            return
        stamp = stamp.replace(tzinfo=UTC) if stamp.tzinfo is None else stamp.astimezone(UTC)
        self.start = stamp if self.start is None or stamp < self.start else self.start
        self.end = stamp if self.end is None or stamp > self.end else self.end


def build_session_orchestration(
    session_id: str,
    topology: SessionTopology | None,
    *,
    messages: Iterable[Message] = (),
    events: Iterable[SessionEvent] = (),
    acquisition: Mapping[str, object] | None = None,
    delegations: Sequence[ArchiveDelegationQueryRow] = (),
    usage_rows: Iterable[Mapping[str, object]] = (),
) -> SessionOrchestrationEvidence:
    """Project this session's own records, excluding inherited dialogue.

    ``messages`` (transcript order), ``events`` (``event_index`` order) and
    ``usage_rows`` (``position`` order) are consumed once as streams: each
    section keeps its full count but retains only its first ``_LIMIT`` rows,
    so memory does not grow with the length of the session.
    """
    launches = _Section()
    segments = _Section()
    mentions = _Section()
    quotas = _Section()
    usage_observations = _Section()
    last_segment: dict[str, object] | None = None
    span = _TimeSpan()
    message_count = 0
    event_count = 0
    message_tokens: dict[str, int] = {}
    messages_with_tokens = 0
    messages_with_unmeasured_lanes = 0
    unmeasured_lane_messages: dict[str, int] = {}
    gaps = ["archive_freshness_unverified", "quota_consumed_tokens_unavailable"]
    watermark = None
    if acquisition:
        watermark = {key: acquisition.get(key) for key in ("raw_id", "acquired_at", "parsed_at", "validation_status")}
        watermark["scope"] = "session_primary_raw_revision"
    if not watermark or not watermark.get("parsed_at"):
        gaps.append("ingestion_watermark_unavailable")

    delegations_by_block: dict[str | None, list[ArchiveDelegationQueryRow]] = {}
    for delegation in delegations:
        delegations_by_block.setdefault(delegation.instruction_tool_use_block_id, []).append(delegation)
    matched_blocks: set[str | None] = set()

    def apply_delegation(row: dict[str, object], delegation: ArchiveDelegationQueryRow) -> None:
        row.update(
            {
                "child_session_id": delegation.child_session_id,
                "message_id": delegation.instruction_message_id,
                "block_id": delegation.instruction_tool_use_block_id,
                "mapping_state": delegation.mapping_state,
                "link_method": delegation.link_method,
                "link_confidence": delegation.link_confidence,
                "requested_model": delegation.requested_model,
                "result_status": delegation.result_status,
                "child_recorded_dominant_model": delegation.child_session_dominant_model,
                "child_tokens": delegation.child_tokens,
                "actual_model": None,
            }
        )

    for message in messages:
        if not message.id.startswith(session_id + ":"):
            continue
        message_count += 1
        span.observe(message.timestamp)
        timestamp = message.timestamp.isoformat() if message.timestamp else None
        lanes = {key: getattr(message, key) for key in _MESSAGE_TOKEN_LANES}
        positive_tokens = {key: value for key, value in lanes.items() if value is not None and value > 0}
        unmeasured_lanes = [key for key, value in lanes.items() if value is None]
        if positive_tokens:
            messages_with_tokens += 1
            for key, value in positive_tokens.items():
                message_tokens[key] = message_tokens.get(key, 0) + value
        if unmeasured_lanes:
            messages_with_unmeasured_lanes += 1
            for key in unmeasured_lanes:
                unmeasured_lane_messages[key] = unmeasured_lane_messages.get(key, 0) + 1
        if str(message.role) == "assistant" and message.model_name:
            if last_segment is not None and last_segment["model"] == message.model_name:
                last_segment["end_message_id"] = message.id
                last_segment["end_event_time"] = timestamp
                last_segment["message_count"] = cast(int, last_segment["message_count"]) + 1
            else:
                last_segment = {
                    "model": message.model_name,
                    "basis": "recorded_message_model",
                    "actual_model_verified": None,
                    "start_message_id": message.id,
                    "end_message_id": message.id,
                    "start_event_time": timestamp,
                    "end_event_time": timestamp,
                    "message_count": 1,
                }
                segments.append(last_segment)
        for block in message.blocks:
            if block.get("type") != "tool_use":
                continue
            name = _text(block.get("tool_name"))
            if name is None:
                continue
            arguments = _mapping(block.get("tool_input"))
            evidence: dict[str, object] = {
                "message_id": message.id,
                "block_id": block.get("block_id", block.get("id")),
                "tool_id": block.get("tool_id"),
                "tool_name": name,
                "event_time": timestamp,
            }
            if name.rsplit(".", 1)[-1] in _LAUNCH_TOOLS:
                launch: dict[str, object] = {
                    **evidence,
                    "basis": "tool_request",
                    "child_session_id": None,
                    "requested_model": _text(arguments.get("model")),
                    "requested_effort": _text(arguments.get("reasoning_effort")),
                    "agent_type": _text(arguments.get("subagent_type", arguments.get("agent_type"))),
                    "task_name": _text(arguments.get("task_name", arguments.get("name"))),
                    "actual_model": None,
                }
                block_id = cast("str | None", evidence["block_id"])
                if block_id and block_id in delegations_by_block:
                    matched_blocks.add(block_id)
                    for delegation in delegations_by_block[block_id]:
                        apply_delegation(launch, delegation)
                launches.append(launch)
            for bead_id in _bead_ids(name, arguments):
                mentions.append({**evidence, "bead_id": bead_id, "bead_revision": None, "attempt_id": None})

    for delegation in delegations:
        if delegation.instruction_tool_use_block_id in matched_blocks:
            continue
        stored: dict[str, object] = {"basis": "stored_delegation", "event_time": None}
        apply_delegation(stored, delegation)
        launches.append(stored)

    usage_iter = iter(usage_rows)
    first_usage_row = next(usage_iter, None)
    has_usage_rows = first_usage_row is not None
    latest_total: dict[str, int] | None = None
    previous_total: dict[str, int] | None = None
    counter_reset = False
    capture_gap = False
    for event in events:
        event_count += 1
        span.observe(event.timestamp)
        capture_gap = capture_gap or event.event_type == "capture_gap"
        payload = event.payload
        reference = _event_ref(event)
        if event.event_type == "collab_agent_spawn_end":
            launches.append(
                {
                    **reference,
                    "basis": "native_spawn_event",
                    "child_session_id": None,
                    "child_native_id": _text(payload.get("new_thread_id")),
                    "requested_model": None,
                    "actual_model": None,
                    "status": payload.get("status"),
                }
            )
            if not payload.get("new_thread_id"):
                gaps.append("native_spawn_child_identity_not_retained")
        if event.event_type in {"turn_context", "thread_settings_applied"}:
            model = _text(payload.get("model", payload.get("model_name")))
            if model:
                segments.append(
                    {
                        **reference,
                        "basis": "configured_turn_model",
                        "model": model,
                        "effort": payload.get("effort", payload.get("reasoning_effort")),
                        "actual_model_verified": None,
                    }
                )
        windows = _mapping(payload.get("rate_limits"))
        if windows:
            quotas.append({**reference, "windows": windows, "consumed_tokens": None})
        if event.event_type != "token_count" or has_usage_rows:
            continue
        total = _numbers(payload.get("total_token_usage"))
        last = _numbers(payload.get("last_token_usage"))
        if total or last:
            usage_observations.append({**reference, "cumulative": total or None, "last": last or None})
        if total:
            if previous_total and any(total[key] < value for key, value in previous_total.items() if key in total):
                counter_reset = True
            previous_total = total
            latest_total = total
    for usage_row in itertools.chain(() if first_usage_row is None else (first_usage_row,), usage_iter):
        occurred = usage_row.get("occurred_at_ms")
        stamp = datetime.fromtimestamp(occurred / 1000, tz=UTC) if isinstance(occurred, int) else None
        span.observe(stamp)
        timestamp = stamp.isoformat() if stamp else None
        total = {
            key: value
            for column, key in (
                ("total_input_tokens", "input_tokens"),
                ("total_output_tokens", "output_tokens"),
                ("total_cached_input_tokens", "cached_input_tokens"),
                ("total_cache_write_tokens", "cache_write_tokens"),
                ("total_reasoning_output_tokens", "reasoning_output_tokens"),
                ("total_tokens", "total_tokens"),
            )
            if isinstance(value := usage_row.get(column), int) and value > 0
        }
        last = {
            key: value
            for column, key in (
                ("last_input_tokens", "input_tokens"),
                ("last_output_tokens", "output_tokens"),
                ("last_cached_input_tokens", "cached_input_tokens"),
                ("last_cache_write_tokens", "cache_write_tokens"),
                ("last_reasoning_output_tokens", "reasoning_output_tokens"),
                ("last_total_tokens", "total_tokens"),
            )
            if isinstance(value := usage_row.get(column), int) and value > 0
        }
        reference = {
            "usage_row": {"session_id": session_id, "position": usage_row["position"]},
            "event_time": timestamp,
            "source_message_id": usage_row.get("source_message_id"),
            "event_type": usage_row.get("provider_event_type"),
            "model": usage_row.get("model_name"),
        }
        usage_observations.append({**reference, "cumulative": total or None, "last": last or None})
        if total and usage_row.get("provider_event_type") == "token_count":
            if previous_total and any(total.get(key, 0) < value for key, value in previous_total.items()):
                counter_reset = True
            previous_total = total
            latest_total = total
    if has_usage_rows:
        gaps.append("stored_usage_zero_and_missing_indistinguishable")
        if not quotas.count:
            gaps.append("rate_limit_windows_not_retained_in_usage_rows")
    if counter_reset:
        gaps.append("cumulative_usage_counter_reset")
    if latest_total is None:
        gaps.append("cumulative_token_usage_unavailable")
    if message_tokens:
        gaps.append("message_usage_missing_and_zero_indistinguishable")
    if unmeasured_lane_messages:
        gaps.append("message_token_lanes_unmeasured")
    if not segments.count:
        gaps.append("model_evidence_unavailable")
    if not quotas.count:
        gaps.append("rate_limit_observations_unavailable")
    if capture_gap:
        gaps.append("native_capture_gap")

    children = _Section()
    if topology:
        nodes = {str(node.session_id): node for node in topology.nodes}
        for edge in topology.edges:
            if str(edge.parent_id) == session_id:
                child = nodes.get(str(edge.child_id))
                children.append(
                    {
                        "session_id": str(edge.child_id),
                        "origin": child.origin if child else None,
                        "branch_type": edge.kind.value,
                        "basis": "stored_topology_edge",
                    }
                )
        if topology.cycle_detected:
            gaps.append("topology_cycle")
        if topology.unresolved_edges():
            gaps.append("unresolved_topology_edges")
    else:
        gaps.append("topology_unavailable")

    sections = {
        "children": children,
        "launches": launches,
        "model_segments": segments,
        "bead_mentions": mentions,
        "rate_limits": quotas,
        "usage_observations": usage_observations,
    }
    truncated = [key for key, section in sections.items() if section.count > _LIMIT]
    if truncated:
        gaps.append("observation_limit")
    topology_payload = topology.model_dump(mode="json") if topology else None
    if topology_payload and topology and (len(topology.nodes) > _LIMIT or len(topology.edges) > _LIMIT):
        topology_payload["nodes"] = topology_payload["nodes"][:_LIMIT]
        retained_ids = {node["session_id"] for node in topology_payload["nodes"]}
        topology_payload["edges"] = [
            edge
            for edge in topology_payload["edges"][:_LIMIT]
            # An unresolved edge has no parent node to retain; it survives
            # with its retained child so the unresolved relationship stays visible.
            if edge["child_id"] in retained_ids and (edge["parent_id"] is None or edge["parent_id"] in retained_ids)
        ]
        children.rows = [child for child in children.rows if child["session_id"] in retained_ids]
        truncated.append("topology")
        gaps.append("observation_limit")
    return SessionOrchestrationEvidence(
        outcome="degraded" if gaps else "ok",
        session_id=session_id,
        children=children.rows,
        topology=topology_payload,
        launches=launches.rows,
        model_segments=segments.rows,
        bead_mentions=mentions.rows,
        rate_limits=quotas.rows,
        usage={
            "tokens": latest_total if not counter_reset else None,
            "basis": "latest_stored_cumulative_counter" if latest_total and not counter_reset else None,
            "counter_semantics": "origin_native_fields_with_owner_lineage_normalization",
            "latest_cumulative": latest_total,
            "message_tokens_lower_bound": message_tokens or None,
            "messages_with_positive_tokens": messages_with_tokens,
            # The two buckets below are what keeps ``message_tokens_lower_bound``
            # honest: it sums measured-positive lanes only, so a reader needs the
            # count of messages that carried an unmeasured lane -- and which lanes
            # those were -- to know the denominator the lower bound was taken over.
            # A measured ``0`` is excluded from both: it is a measurement.
            "messages_with_unmeasured_token_lanes": messages_with_unmeasured_lanes,
            "unmeasured_token_lane_messages": unmeasured_lane_messages or None,
            "message_token_lane_scope": message_count,
            "observations": usage_observations.rows,
            "quota_consumed_tokens": None,
            "includes_children": None,
            "includes_inherited_usage": None,
        },
        coverage={
            "scope": "stored_session_records",
            "inherited_messages_excluded": True,
            "atomic_snapshot": False,
            "observed_at": acquisition.get("acquired_at") if acquisition else None,
            "ingestion_watermark": watermark,
            "event_time_start": span.start.isoformat() if span.start else None,
            "event_time_end": span.end.isoformat() if span.end else None,
            "message_count": message_count,
            "event_count": event_count,
            "section_counts": {key: section.count for key, section in sections.items()},
            "truncated_sections": truncated,
            "limit_per_section": _LIMIT,
            "complete": False,
        },
        gaps=sorted(set(gaps)),
    )
