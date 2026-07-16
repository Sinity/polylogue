# Fork 12 — Sinex interpretation-revision and replay demo

Work directly on the supplied Sinex repository. Use Beads as roadmap authority, especially `sinex-cem.3`, `sinex-cem.14`, `sinex-908`, and derivation-control work.

## Mission

Build a deterministic proof that Sinex can change its interpretation without rewriting the observed occurrence or erasing prior history.

## Scenario

Use one small source record whose parser-v1 interpretation is intentionally wrong or incomplete. Parser/semantics v2 corrects it.

The demo must show:

- source material identity and exact anchor remain stable;
- `ts_orig` remains the source-domain occurrence time;
- the new interpretation receives a new event identity and coining time;
- old interpretation lifecycle is explicit;
- current projection moves to the corrected state;
- a semantic diff explains what changed;
- replay does not create another live occurrence;
- public refs can resolve current and historical interpretations.

## Owned scope

Own the fixture source/parser pair, replay invocation path, semantic-diff packet, focused lifecycle/projection tests, and generated artifacts. Avoid changes to unrelated source runtimes.

## Independent oracle

A hand-authored manifest specifies the source occurrence, expected v1 interpretation, expected v2 interpretation, and fields that must remain invariant.

## Negative controls

- overwrite the v1 row in place: test must fail;
- alter occurrence time on replay: test must fail;
- emit two live occurrences: test must fail;
- reuse event ID across replay: test must fail;
- lose material reachability: test must fail.

## Deliverables

Produce patch, before/after projection, interpretation history, semantic diff, source-material drill-down, machine-readable packet, command transcript, and focused test output.
