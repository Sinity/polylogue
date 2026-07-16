# Fork 13 — Sinex self-diagnosing capture-outage demo

Work directly on the supplied Sinex repository. Use Beads as roadmap authority, especially `sinex-cem.2`, `sinex-jdp`, and `sinex-r6d`.

## Mission

Prove that Sinex can recognize when its own evidence is incomplete and propagate that fact into user-facing claims.

## Fault model

Choose one deterministic, meaningful capture fault such as:

- source cursor advanced before durable admission;
- sequence gap in a stream;
- stale runtime binding;
- interrupted staged material;
- inotify overflow or dropped segment;
- projection frontier behind admitted events.

Inject the fault through a supported test/fault boundary rather than editing database rows after the fact.

## Claim

The source-health and coverage surfaces identify the fault, quantify only what can actually be measured, and cause dependent queries to carry an explicit caveat or unavailable state.

Do not turn “sources with at least one error” into an event-loss percentage.

## Owned scope

Own the fault fixture/harness, coverage result, propagation into one query/read surface, focused tests, and proof packet. Avoid broad source cleanup unrelated to the selected fault.

## Independent oracle

Use known source sequence numbers, material manifest counts, or checkpoint barriers. The expected gap must be independent of the production coverage query.

## Negative controls

- healthy source produces no false outage;
- source disabled by policy is not labeled failed;
- an empty but healthy interval remains distinguishable from missing coverage;
- recovery does not erase the historical gap;
- query cannot silently downgrade the caveat.

## Deliverables

Produce patch, source-health view, affected-query view, machine-readable gap packet, recovery result, tests, and a concise public explanation of what the coverage metric means and does not mean.
