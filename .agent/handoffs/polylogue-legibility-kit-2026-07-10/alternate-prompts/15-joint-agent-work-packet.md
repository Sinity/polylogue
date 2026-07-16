# Fork 15 — Joint “Resume This Bead” Agent Work Packet demo

Work across the supplied Polylogue and Sinex repositories. Use Beads as roadmap authority, especially `sinex-a4w.3.9`, the Polylogue coordination/context programs, and `sinex-4j2`.

## Mission

Design and, as far as the current substrates permit, implement the combined flagship demonstration:

> Resume one unit of agent work from intent, transcript evidence, machine effects, verification, and reviewed lessons.

## Domain rule

There is no intrinsic `session_commit` object. Sessions remain Polylogue objects; commits remain Git objects; checks remain telemetry or source-domain objects; Beads remain task objects. The Agent Work Packet is a replayable derived relation over native refs.

## Fixture

Create or adapt one private-data-free change episode containing:

- a Bead with intent, dependencies, and status transition;
- one parent agent session and one subagent/fork;
- typed tool calls/results including one failed and one successful verification;
- repository branch, file changes, commit(s), and check result;
- optional browser or terminal evidence outside the transcript;
- one accepted lesson and one unreviewed candidate;
- a context delivery to a resumed agent;
- one deliberately missing leg to exercise honest gaps.

## Packet contract

The output should include:

- packet ID and semantics version;
- intent refs;
- session and agent topology;
- repository/worktree/branch refs;
- toil and verification observations;
- artifact and source-material refs;
- outcome class and support vector;
- reviewed lessons versus candidates;
- coverage/freshness caveats;
- context image and delivery snapshot;
- provenance for every derived relation.

## Proof

Use independent fixture manifests and direct Beads/Git/command-state oracles. The demo should support a later blinded resumption duel but must not claim uplift yet.

## Owned scope

Prefer schemas, fixture world, derivation, packet renderer, and focused tests. Avoid broad changes to either generic query language or unrelated source parsers.

## Deliverables

Produce cross-repo patches or a standalone protocol/fixture package if implementation boundaries are not ready, one human-readable packet, machine-readable packet, evidence graph, context snapshot, tests, and a list of missing product primitives discovered by the exercise.
