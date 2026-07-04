"""Human-readable projections for coordination envelopes."""

from __future__ import annotations

from polylogue.coordination.payloads import AgentCoordinationPayload


def _work_item_label(payload: AgentCoordinationPayload) -> str:
    ref = payload.work_item.ref or "none"
    title = f" - {payload.work_item.title}" if payload.work_item.title else ""
    return f"{ref}{title} ({payload.work_item.source}, confidence={payload.work_item.confidence:.2f})"


def _beads_summary(payload: AgentCoordinationPayload) -> str:
    if not payload.beads:
        return "not available"
    hook_state = (
        "unknown"
        if payload.beads.hooks_all_installed is None
        else ("ok" if payload.beads.hooks_all_installed else "incomplete")
    )
    merge = payload.beads.merge_slot
    merge_label = (
        "n/a" if merge is None else f"{merge.id or 'merge-slot'}:{merge.status or merge.available or merge.error}"
    )
    return (
        f"hooks={hook_state}"
        f" outdated={payload.beads.hooks_outdated_count}"
        f" gates={payload.beads.open_gate_count}"
        f" merge={merge_label}"
    )


def render_coordination_text(payload: AgentCoordinationPayload) -> str:
    """Render a compact text view for humans; JSON remains the primary contract."""

    lines = [
        f"Agent coordination ({payload.view})",
        f"  repo: {payload.repo.root or payload.repo.cwd}",
        f"  branch: {payload.repo.branch or 'n/a'} head={payload.repo.head or 'n/a'} dirty={payload.repo.dirty}",
        f"  self: {payload.self.agent_kind} pid={payload.self.pid}",
        "  work item: " + _work_item_label(payload),
    ]
    if payload.peers:
        lines.append(f"  peers: {len(payload.peers)}")
        for peer in payload.peers[: payload.limits.peer_limit]:
            lines.append(f"    - {peer.kind} pid={peer.pid} cwd={peer.cwd or 'n/a'}")
    if payload.resource_episodes:
        lines.append(f"  resources: {len(payload.resource_episodes)}")
        for episode in payload.resource_episodes[: payload.limits.resource_limit]:
            lines.append(f"    - {episode.kind} pid={episode.pid} {episode.command}")
    if payload.overlaps:
        lines.append(f"  overlaps: {len(payload.overlaps)}")
        for overlap in payload.overlaps:
            blocker = "blocking" if overlap.blocking else "non-blocking"
            lines.append(f"    - [{overlap.severity}/{blocker}] {overlap.summary}")
    if payload.handoff:
        lines.append("  handoff refs:")
        for ref in payload.handoff:
            state = "present" if ref.exists else "missing"
            lines.append(f"    - {ref.kind}: {state} {ref.path}")
    if payload.archive:
        lines.append(
            "  archive: "
            f"{payload.archive.archive_root} index_exists={payload.archive.index_exists} "
            f"index_schema={payload.archive.index_user_version}"
        )
    if payload.beads:
        lines.append(f"  beads: {_beads_summary(payload)}")
    if payload.advisories:
        lines.append("  advisories:")
        for advisory in payload.advisories:
            lines.append(f"    - {advisory}")
    return "\n".join(lines)


def render_coordination_markdown(payload: AgentCoordinationPayload) -> str:
    """Render a readable mission-control packet from the shared envelope."""

    lines = [
        "# Agent Coordination Mission Control",
        "",
        f"- View: `{payload.view}`",
        f"- Generated: `{payload.generated_at}`",
        f"- Repo: `{payload.repo.root or payload.repo.cwd}`",
        f"- Branch: `{payload.repo.branch or 'n/a'}` at `{payload.repo.head or 'n/a'}`",
        f"- Dirty: `{str(payload.repo.dirty).lower()}`",
        f"- Agent: `{payload.self.agent_kind}` pid `{payload.self.pid}`",
        f"- Work item: `{_work_item_label(payload)}`",
        f"- Beads: `{_beads_summary(payload)}`",
    ]
    if payload.archive:
        lines.extend(
            [
                f"- Archive root: `{payload.archive.archive_root}`",
                f"- Index schema: `{payload.archive.index_user_version}`",
            ]
        )
    lines.append("")
    lines.append("## Active Agents")
    if payload.peers:
        for peer in payload.peers:
            lines.append(f"- `{peer.kind}` pid `{peer.pid}` cwd `{peer.cwd or 'n/a'}`")
    else:
        lines.append("- None detected.")
    lines.append("")
    lines.append("## Resource Episodes")
    if payload.resource_episodes:
        for episode in payload.resource_episodes:
            lines.append(f"- `{episode.kind}` pid `{episode.pid}` status `{episode.status}` - `{episode.command}`")
    else:
        lines.append("- None detected.")
    lines.append("")
    lines.append("## Overlap Awareness")
    if payload.overlaps:
        for overlap in payload.overlaps:
            blocking = "blocking" if overlap.blocking else "awareness"
            lines.append(f"- **{overlap.severity} / {blocking}**: {overlap.summary}")
    else:
        lines.append("- No overlap warnings.")
    lines.append("")
    lines.append("## Handoff Refs")
    if payload.handoff:
        for ref in payload.handoff:
            state = "present" if ref.exists else "missing"
            lines.append(f"- `{ref.kind}` {state}: `{ref.path}`")
    else:
        lines.append("- No handoff refs in this projection.")
    if payload.advisories:
        lines.append("")
        lines.append("## Advisories")
        for advisory in payload.advisories:
            lines.append(f"- {advisory}")
    return "\n".join(lines) + "\n"


def render_coordination_tree(payload: AgentCoordinationPayload) -> str:
    """Render a compact terminal tree for the shared envelope."""

    lines = [
        "coordination",
        f"+- repo {payload.repo.branch or 'n/a'}@{payload.repo.head or 'n/a'} dirty={payload.repo.dirty}",
        f"+- self {payload.self.agent_kind} pid={payload.self.pid}",
        f"+- work {payload.work_item.ref or 'none'} source={payload.work_item.source} confidence={payload.work_item.confidence:.2f}",
        f"+- beads {_beads_summary(payload)}",
    ]
    if payload.archive:
        lines.append(f"+- archive schema={payload.archive.index_user_version} root={payload.archive.archive_root}")
    lines.append(f"+- peers {len(payload.peers)}")
    for peer in payload.peers:
        lines.append(f"|  +- {peer.kind} pid={peer.pid} cwd={peer.cwd or 'n/a'}")
    lines.append(f"+- resources {len(payload.resource_episodes)}")
    for episode in payload.resource_episodes:
        lines.append(f"|  +- {episode.kind} pid={episode.pid} status={episode.status}")
    lines.append(f"+- overlaps {len(payload.overlaps)}")
    for overlap in payload.overlaps:
        blocking = "blocking" if overlap.blocking else "awareness"
        lines.append(f"   +- {overlap.severity}/{blocking}: {overlap.summary}")
    return "\n".join(lines)


__all__ = ["render_coordination_markdown", "render_coordination_text", "render_coordination_tree"]
