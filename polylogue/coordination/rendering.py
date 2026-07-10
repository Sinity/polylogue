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


def _projection_summary(payload: AgentCoordinationPayload) -> str:
    projection = payload.projection
    omissions = ", ".join(f"{name}={count}" for name, count in projection.omitted_counts.items()) or "none"
    return (
        f"detail={str(projection.detail).lower()} bytes={projection.serialized_bytes or 'unknown'} omitted={omissions}"
    )


def render_coordination_text(payload: AgentCoordinationPayload) -> str:
    """Render a compact text view for humans; JSON remains the primary contract."""

    lines = [
        f"Agent coordination ({payload.view})",
        f"  repo: {payload.repo.root or payload.repo.cwd}",
        f"  branch: {payload.repo.branch or 'n/a'} head={payload.repo.head or 'n/a'} dirty={payload.repo.dirty}",
        f"  self: {payload.self.agent_kind} pid={payload.self.pid}",
        "  work item: " + _work_item_label(payload),
        "  projection: " + _projection_summary(payload),
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
        for handoff_ref in payload.handoff:
            state = "present" if handoff_ref.exists else "missing"
            lines.append(f"    - {handoff_ref.kind}: {state} {handoff_ref.ref or handoff_ref.path}")
    if payload.archive:
        lines.append(
            "  archive: "
            f"{payload.archive.archive_root} index_exists={payload.archive.index_exists} "
            f"index_schema={payload.archive.index_user_version}"
        )
        hook_states = ", ".join(
            f"{harness}={state}" for harness, state in sorted(payload.archive.hook_flow_states.items())
        )
        if hook_states:
            lines.append(f"  hook health: healthy={payload.archive.hook_flow_healthy} {hook_states}")
        for gap in payload.archive.hook_flow_gaps:
            lines.append(f"    - hook gap: {gap}")
    if payload.session_trees:
        tree = payload.session_trees[0]
        lines.append(
            "  session tree: "
            f"target={tree.target_session_id} root={tree.root_session_id} "
            f"nodes={len(tree.nodes)} edges={len(tree.edges)}"
        )
    if payload.activity_episodes:
        lines.append(f"  archive activity: {len(payload.activity_episodes)}")
        for activity_episode in payload.activity_episodes[: payload.limits.resource_limit]:
            lines.append(
                f"    - {activity_episode.kind} {activity_episode.ref} {activity_episode.summary or ''}".rstrip()
            )
    if payload.subagent_exchanges:
        lines.append(f"  subagent exchanges: {len(payload.subagent_exchanges)}")
        for exchange in payload.subagent_exchanges[: payload.limits.resource_limit]:
            status = f" status={exchange.status}" if exchange.status else ""
            prompt = f" prompt={exchange.dispatch_prompt}" if exchange.dispatch_prompt else ""
            final = f" final={exchange.returned_final_message}" if exchange.returned_final_message else ""
            lines.append(f"    - {exchange.run_ref}{status}{prompt}{final}".rstrip())
    if payload.proof_refs:
        lines.append(f"  proof refs: {len(payload.proof_refs)}")
        for proof in payload.proof_refs[: payload.limits.resource_limit]:
            status = f" status={proof.status}" if proof.status else ""
            lines.append(f"    - {proof.kind} {proof.ref}{status} {proof.summary or ''}".rstrip())
    if payload.context_flow_refs:
        lines.append(f"  context flow refs: {len(payload.context_flow_refs)}")
        for context_ref in payload.context_flow_refs[: payload.limits.resource_limit]:
            lines.append(f"    - {context_ref.boundary} {context_ref.ref} segments={len(context_ref.segment_refs)}")
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
        f"- Projection: `{_projection_summary(payload)}`",
    ]
    if payload.archive:
        lines.extend(
            [
                f"- Archive root: `{payload.archive.archive_root}`",
                f"- Index schema: `{payload.archive.index_user_version}`",
            ]
        )
    lines.append("")
    lines.append("## Archive Session Tree")
    if payload.session_trees:
        for tree in payload.session_trees:
            lines.append(f"- Target `{tree.target_session_id}` root `{tree.root_session_id}`")
            for node in tree.nodes:
                marker = " target" if node.is_target else ""
                title = f" - {node.title}" if node.title else ""
                lines.append(f"  - depth `{node.depth}` `{node.session_id}`{marker}{title}")
    else:
        lines.append("- No archive session tree in this projection.")
    lines.append("")
    lines.append("## Archive Activity")
    if payload.activity_episodes:
        for activity_episode in payload.activity_episodes:
            summary = f" - {activity_episode.summary}" if activity_episode.summary else ""
            lines.append(
                f"- `{activity_episode.kind}` `{activity_episode.ref}` session `{activity_episode.session_id}`{summary}"
            )
    else:
        lines.append("- No archive activity refs in this projection.")
    lines.append("")
    lines.append("## Subagent Exchanges")
    if payload.subagent_exchanges:
        for exchange in payload.subagent_exchanges:
            lines.append(f"- Run `{exchange.run_ref}` status `{exchange.status or 'n/a'}`")
            if exchange.dispatch_prompt:
                lines.append(f"  - Dispatch: {exchange.dispatch_prompt}")
            if exchange.returned_final_message:
                lines.append(f"  - Returned: {exchange.returned_final_message}")
            if exchange.context_snapshot_ref:
                lines.append(f"  - Context: `{exchange.context_snapshot_ref}`")
    else:
        lines.append("- No subagent dispatch/return refs in this projection.")
    lines.append("")
    lines.append("## Proof / Outcome Refs")
    if payload.proof_refs:
        for proof in payload.proof_refs:
            status = f" status `{proof.status}`" if proof.status else ""
            summary = f" - {proof.summary}" if proof.summary else ""
            lines.append(f"- `{proof.kind}` `{proof.ref}`{status}{summary}")
    else:
        lines.append("- No proof refs in this projection.")
    lines.append("")
    lines.append("## Context Flow Refs")
    if payload.context_flow_refs:
        for context_ref in payload.context_flow_refs:
            lines.append(
                f"- `{context_ref.boundary}` `{context_ref.ref}` session `{context_ref.session_id}` "
                f"segments `{len(context_ref.segment_refs)}` evidence `{len(context_ref.evidence_refs)}`"
            )
    else:
        lines.append("- No context-flow refs in this projection.")
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
        for resource_episode in payload.resource_episodes:
            lines.append(
                f"- `{resource_episode.kind}` pid `{resource_episode.pid}` "
                f"status `{resource_episode.status}` - `{resource_episode.command}`"
            )
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
        for handoff_ref in payload.handoff:
            state = "present" if handoff_ref.exists else "missing"
            lines.append(f"- `{handoff_ref.kind}` {state}: `{handoff_ref.ref or handoff_ref.path}`")
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
        f"+- projection {_projection_summary(payload)}",
    ]
    if payload.archive:
        lines.append(f"+- archive schema={payload.archive.index_user_version} root={payload.archive.archive_root}")
    lines.append(f"+- session-trees {len(payload.session_trees)}")
    for tree in payload.session_trees:
        lines.append(f"|  +- target={tree.target_session_id} root={tree.root_session_id} nodes={len(tree.nodes)}")
        for node in tree.nodes:
            marker = " target" if node.is_target else ""
            lines.append(f"|     +- depth={node.depth} {node.session_id}{marker}")
    lines.append(f"+- archive-activity {len(payload.activity_episodes)}")
    for activity_episode in payload.activity_episodes:
        lines.append(f"|  +- {activity_episode.kind} {activity_episode.ref} session={activity_episode.session_id}")
    lines.append(f"+- subagent-exchanges {len(payload.subagent_exchanges)}")
    for exchange in payload.subagent_exchanges:
        lines.append(f"|  +- {exchange.run_ref} status={exchange.status or 'n/a'}")
        if exchange.dispatch_prompt:
            lines.append(f"|     +- dispatch {exchange.dispatch_prompt}")
        if exchange.returned_final_message:
            lines.append(f"|     +- returned {exchange.returned_final_message}")
    lines.append(f"+- proof-refs {len(payload.proof_refs)}")
    for proof in payload.proof_refs:
        lines.append(f"|  +- {proof.kind} {proof.ref} status={proof.status or 'n/a'}")
    lines.append(f"+- context-flow {len(payload.context_flow_refs)}")
    for context_ref in payload.context_flow_refs:
        lines.append(f"|  +- {context_ref.boundary} {context_ref.ref} session={context_ref.session_id}")
    lines.append(f"+- peers {len(payload.peers)}")
    for peer in payload.peers:
        lines.append(f"|  +- {peer.kind} pid={peer.pid} cwd={peer.cwd or 'n/a'}")
    lines.append(f"+- resources {len(payload.resource_episodes)}")
    for resource_episode in payload.resource_episodes:
        lines.append(f"|  +- {resource_episode.kind} pid={resource_episode.pid} status={resource_episode.status}")
    lines.append(f"+- overlaps {len(payload.overlaps)}")
    for overlap in payload.overlaps:
        blocking = "blocking" if overlap.blocking else "awareness"
        lines.append(f"   +- {overlap.severity}/{blocking}: {overlap.summary}")
    return "\n".join(lines)


__all__ = ["render_coordination_markdown", "render_coordination_text", "render_coordination_tree"]
