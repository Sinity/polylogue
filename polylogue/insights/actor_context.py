"""Derive real ``ActorRef``/``ExecutionContextRef`` identity from archive evidence.

``polylogue.core.refs`` declares the ActorRef/ExecutionContextRef/WorkerProfileRef
shapes (polylogue-h6r): a durable actor (human, service, agent persona, or model
family) is never the same identity as the specific prompt/config/runtime an
attempt ran under. Before this module, every production caller only *threaded*
those refs through as optional pass-through parameters (see
``polylogue.insights.work_evidence.node_from_projected_run`` and
``polylogue.insights.incident_evidence_materialization``'s own docstring,
which named actor/execution-context population as "deliberately out of
scope"). Nothing actually looked at real session evidence to build one.

This module is that adapter. Two independent evidence shapes already flow
through the archive and both carry real, provider-reported signal:

- :class:`~polylogue.sources.parsers.base_models.ParsedSession` (pre-storage
  parser output): ``source_name`` (provider/harness), ``models_used``, and
  ``instructions_text`` are real fields every provider parser already
  populates from its own export -- not guessed from prose.
- :class:`~polylogue.insights.run_projection.ProjectedRun` (post-storage
  projection): ``harness``, ``provider_origin``, and ``agent_ref`` (an
  explicit subagent persona name when the runtime reports dispatch, e.g.
  Claude Code's ``Task(subagent_type=...)``) are the analogous real fields.

Design rules enforced here, matching the h6r contract:

1. **Actor identity excludes configuration.** A model/persona name is stable
   identity; the prompt/instructions text that happened to be in force is not
   part of it. Two sessions with the same model under different
   ``instructions_text`` collapse to ONE ``ActorRef`` and yield two distinct
   ``ExecutionContextRef``s (h6r AC1).
2. **Execution-context identity never depends on a session/attempt id.** A
   session or attempt may *cite* an actor/context; nothing here folds the
   session id into either ref (h6r AC2).
3. **Missing evidence is an explicit unknown field, never a fabricated
   value or a silently-dropped field.** A session with no reported model or
   instructions still yields a stable, usable ``agent:unknown`` actor and a
   content-addressed context whose ``unknown_fields`` names exactly what
   wasn't captured (h6r AC3). Fields this module can never observe yet --
   tool/MCP profile, permissions, runtime build, sampling parameters -- are
   *always* declared unknown rather than omitted; omission would silently
   read as "not applicable" instead of "not captured" (that capture is
   polylogue-7aw's job).
4. **Working directory / git branch are deliberately excluded from context
   fields.** They describe where a run executed, not the behavioral
   environment (prompt/tools/permissions/runtime) ExecutionContextRef
   content-addresses -- folding them in would be exactly the "fabricate a
   context from bare cwd/git_branch strings without a positive-evidence
   rule" anti-pattern ``incident_evidence_materialization`` already warns
   against.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from polylogue.core.refs import ActorRef, ExecutionContextRef

if TYPE_CHECKING:
    from polylogue.insights.run_projection import ProjectedRun
    from polylogue.sources.parsers.base_models import ParsedSession

#: Behavioral-environment fields the h6r design names (harness/version,
#: prompt/context receipts, skills/tools/MCP profile, configuration
#: artifacts, runtime/build, permissions, effort/sampling parameters) that
#: neither ``ParsedSession`` nor ``ProjectedRun`` capture today. Declared
#: unknown on every context this module builds -- polylogue-7aw's
#: configuration-artifact ingestion is the only thing allowed to resolve
#: these to real evidence.
_UNCAPTURED_CONTEXT_FIELDS: tuple[str, ...] = (
    "tools_profile",
    "mcp_profile",
    "permissions",
    "runtime_build",
    "sampling_params",
)

#: Explicit stand-in identity for "this evidence source reported no model or
#: persona name" -- never fabricated from the session id or harness, which
#: would silently promote a runtime/session identity into actor identity.
_UNKNOWN_ACTOR_IDENTITY = "unknown"


def _instructions_hash(instructions_text: str) -> str:
    return hashlib.sha256(instructions_text.encode("utf-8")).hexdigest()


def actor_ref_from_session(session: ParsedSession) -> ActorRef:
    """Derive a durable actor identity from a session's own reported model.

    Keyed on ``models_used[0]`` only -- the first (and typically only)
    model family a provider export reports for the session. Deliberately
    ignores ``instructions_text``: the same model under two different
    instruction sets is one actor operating under two execution contexts,
    not two actors (h6r AC1). A session with no reported model (real for
    some browser-capture/export shapes) yields the explicit
    ``agent:unknown`` actor rather than keying off ``provider_session_id``,
    which would make every session its own actor (h6r AC2/AC6).
    """

    if session.models_used:
        return ActorRef(kind="agent", identity=session.models_used[0])
    return ActorRef(kind="agent", identity=_UNKNOWN_ACTOR_IDENTITY)


def execution_context_ref_from_session(session: ParsedSession) -> ExecutionContextRef:
    """Content-address the behavioral environment a session's evidence supports.

    Known fields are limited to what ``ParsedSession`` actually carries as
    positive evidence: ``harness`` (the reporting provider) and
    ``instructions_hash`` (a hash of the system/instructions text actually
    delivered, when the parser captured one -- the raw text itself is not
    embedded in the id to keep the identity a fingerprint, not a payload
    dump). Everything ``polylogue-7aw`` would add is declared unknown, never
    omitted or guessed.
    """

    fields: dict[str, object] = {"harness": str(session.source_name)}
    unknown_fields = list(_UNCAPTURED_CONTEXT_FIELDS)

    if session.instructions_text is not None and session.instructions_text.strip():
        fields["instructions_hash"] = _instructions_hash(session.instructions_text)
    else:
        unknown_fields.append("instructions_hash")

    return ExecutionContextRef.from_observation(fields, unknown_fields=tuple(unknown_fields))


def actor_ref_from_run(run: ProjectedRun) -> ActorRef:
    """Derive a durable actor identity from a projected run's own evidence.

    ``build_run_projection`` (the production ``ProjectedRun`` constructor)
    always reports an ``agent_ref`` -- ``"{harness}/main"`` for the top-level
    run, ``"{harness}/{subagent_type}"`` for a structurally-dispatched
    subagent (e.g. Claude Code ``Task(subagent_type=...)``) -- real,
    provider-observed identity, reused verbatim rather than re-derived. A run
    built without one (the field is optional for hand-built fixtures and any
    future producer) yields the explicit ``agent:unknown`` actor:
    ``run.harness`` alone describes the *runtime*, which is
    execution-context evidence, not actor identity, so it is never used as
    an actor-identity fallback (that would be exactly the
    context-collapsed-into-actor anti-pattern h6r AC6 rejects).
    """

    if run.agent_ref is not None:
        return ActorRef(kind="agent", identity=run.agent_ref.object_id)
    return ActorRef(kind="agent", identity=_UNKNOWN_ACTOR_IDENTITY)


def execution_context_ref_from_run(run: ProjectedRun) -> ExecutionContextRef:
    """Content-address a projected run's behavioral environment.

    Limited to ``harness`` and ``provider_origin`` -- both provider-reported
    facts about how the run executed. ``cwd``/``git_branch`` are
    deliberately excluded (see module docstring); everything
    polylogue-7aw would add is declared unknown.
    """

    fields: dict[str, object] = {"harness": str(run.harness), "provider_origin": run.provider_origin}
    return ExecutionContextRef.from_observation(fields, unknown_fields=_UNCAPTURED_CONTEXT_FIELDS)


__all__ = [
    "actor_ref_from_run",
    "actor_ref_from_session",
    "execution_context_ref_from_run",
    "execution_context_ref_from_session",
]
