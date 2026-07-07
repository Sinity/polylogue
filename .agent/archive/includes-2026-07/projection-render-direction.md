# Projection And Rendering Direction

Polylogue should make export, read, context, recovery-like, and demo behavior
emerge from a common query/projection/render model rather than from siloed
commands or bespoke DTOs.

## Target Model

Use two composable layers:

1. A query expression selects sessions or query units.
2. A projection/render expression describes what to attach, how to window it,
   and how to render it.

Selection examples:

```text
sessions where repo:polylogue and has:paste
actions where tool:bash and text:pytest
messages where role:user and time >= 2026-06-01
```

Projection/render examples:

```text
with messages window around matches limit tokens 4000
with actions grouped by tool
with assertions attached to sessions
render layout:context-image timestamps:include-available
render format:markdown destination:browser
```

## Working Split

- Keep **selection** in the query DSL: origin, repo, time, text, lineage,
  session ids, query-unit predicates, sort, and selected-session cardinality.
- Keep **projection policy** in a typed projection expression/spec: attached
  units, body/window/edge limits, token budgets, omission policy, evidence
  families, and per-view computation knobs.
- Keep **rendering** in `RenderSpec`: layout, output format, timestamp policy,
  redaction, destination, and machine-vs-readable shape.
- Allow CLI flags only as ergonomic shorthands that compile into those typed
  contracts. A flag that cannot be represented in the spec is either a substrate
  gap or compatibility debt.

The product does not need literal `--projection` and `--render` flags before
the model is useful. It does need every new read capability to be explainable
as:

```text
SelectionSpec + ProjectionSpec + RenderSpec -> payload + readable rendering
```

## Useful Slices

1. Inventory live `read --help`, `read --views`, and representative
   `read --spec` output. Classify every option as selection, projection,
   rendering, delivery, or compatibility debt.
2. Attach one related unit through projection substrate, likely assertions or
   actions, without per-view bespoke loops.
3. Clarify query-set cardinality versus projection body/window/edge limits.
4. Make render layout explicit in context-image, temporal/chronicle, dialogue,
   raw, and export-like outputs.
5. Remove obsolete public recovery/export/context-pack flags and DTOs once their
   behavior is represented by query + projection + render contracts.

## Slice Acceptance Criteria

A projection/render slice is good only if it:

- uses the canonical active archive or a clearly named fixture;
- states root, schema version, and relevant counts when quoting data;
- can be inspected outside chat;
- improves or removes a user-facing command, spec, payload, doc, or demo;
- avoids new one-off DTOs unless the DTO is the general projection/render
  contract;
- records what the proof does not show.

