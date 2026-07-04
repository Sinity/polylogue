"""Human-readable projections for coordination envelopes."""

from __future__ import annotations

from polylogue.coordination.payloads import AgentCoordinationPayload


def render_coordination_text(payload: AgentCoordinationPayload) -> str:
    """Render a compact text view for humans; JSON remains the primary contract."""

    lines = [
        f"Agent coordination ({payload.view})",
        f"  repo: {payload.repo.root or payload.repo.cwd}",
        f"  branch: {payload.repo.branch or 'n/a'} head={payload.repo.head or 'n/a'} dirty={payload.repo.dirty}",
        f"  self: {payload.self.agent_kind} pid={payload.self.pid}",
        "  work item: "
        + (
            f"{payload.work_item.ref or 'none'} [{payload.work_item.source}, confidence={payload.work_item.confidence:.2f}]"
        ),
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
    if payload.advisories:
        lines.append("  advisories:")
        for advisory in payload.advisories:
            lines.append(f"    - {advisory}")
    return "\n".join(lines)


__all__ = ["render_coordination_text"]
