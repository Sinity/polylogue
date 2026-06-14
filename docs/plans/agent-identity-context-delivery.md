# Agent identity, context, and delivery semantics

## Purpose

Give Polylogue enough structure to answer: which agent ran, under what context, what it inherited, what outside material was delivered to it, and what evidence supports its claims.

## Identity layers

RoleSpec is a behavior recipe or prompt artifact. It is not identity.

AgentPath is durable addressability: the name a user or other agent can send work or mail to.

AgentInstance is the concrete context-bearing session or incarnation. This is the strict cognitive identity.

AgentRun is an execution interval for an instance: wallclock, cwd, worktree, branch, harness, model, tools, spans.

## Example naming

- role://polylogue/dsl-fixer@v2
- agent://plg.orch.i1873-dslfix
- instance://plg.orch.i1873-dslfix@20260614T022516Z.claude-code.agent-def456
- run://plg.orch.i1873-dslfix@20260614T022516Z.claude-code.agent-def456/r01

## ContextEnvelope

A ContextEnvelope records what was loaded or made visible: RoleSpec, system/developer/user instructions, AGENTS/CLAUDE files, tool schemas, MCP config, hook config, permission mode, model, prompt-cache inheritance, injected memory, current work state, and compact summaries.

## Communication and delivery

A CommunicationEvent records that something existed: user prompt, hook payload, GitHub comment, PR review, Beads message, subagent report, or mailbox entry.

A DeliveryEvent records whether that communication was queued, selected for injection, read, injected, acknowledged, acted on, superseded, or expired.

This distinction is critical: a GitHub review existing is not the same as the agent having seen it before merge.

## Context inheritance

Every child or successor session should record one inheritance mode:

- clean: role, task, and selected evidence only.
- summary: selected evidence plus parent RunState.
- prefix: parent prefix up to a boundary plus child delta.
- snapshot: exact parent context image fork.

## First slices

1. Define ref syntax and small DTOs for RoleSpec, AgentPath, AgentInstance, AgentRun, ContextEnvelope, CommunicationEvent, and DeliveryEvent.
2. Map Claude Code and Codex native session/subagent IDs into AgentInstance refs.
3. Add WorkPacket fields for agent path, instance, run, and context envelope.
4. Add delivery state for GitHub reviews/comments and Beads messages.
5. Make continuation prompts say when they are successor sessions rather than the original context.

## Acceptance criteria

- Native harness session IDs map to stable Polylogue refs.
- A WorkPacket can name who acted, under which session, on which run.
- A successor session is represented explicitly.
- A PR/review report can say whether review material was seen before a merge decision.
- Context inheritance is visible in recovery/digest views.