# Projection And Render Spec

Polylogue's read/export/report surfaces should converge on one composition
shape:

1. selection chooses archive evidence;
2. projection chooses evidence families and body policy;
3. rendering chooses format, layout, and destination.

The code contract lives in `polylogue.surfaces.projection_spec`. It is
storage-free on purpose so CLI, MCP, daemon, demos, and future temporal
analysis can share the same vocabulary.

## Why This Exists

The legacy cleanup work removed public recovery/dialogue/tool-output flags, but
`read --view` still mixes several concepts in one option namespace:

- view selection;
- export formatting;
- output destination;
- view-specific filters;
- content body policy.

The projection/render spec gives those concepts stable names without
reintroducing special recovery/export/read silos.

## Contract

- `SelectionSpec` carries refs, query text, origin, time bounds, and limit.
- `ProjectionSpec` carries evidence families, field selection, body policy,
  role/block filters, token budget, and redaction policy.
- `RenderSpec` carries format, destination, layout, and file output path.
- `QueryProjectionSpec` composes all three.

Body policies replace ad hoc flags:

- `full`
- `omit-tool-outputs`
- `dialogue-only`
- `metadata-only`

`READ_VIEW_PROJECTION_FAMILIES` maps executable read views into this vocabulary
and is tested against the CLI read-view registry. Additional named projections,
such as `timeline`, live in `NAMED_PROJECTION_FAMILIES` so non-read projections
do not weaken read-view parity. The compatibility bridge
`projection_from_legacy_view()` accepts the named projection set and rejects
`recovery`; recovery remains an operational failure-recovery concept, not a
read view.

## Next Wiring

The next implementation steps should route existing `read --view` behavior
through this spec incrementally:

1. build a spec from current read CLI options;
2. pass the spec to existing handlers without changing their output;
3. move view-specific option clusters into projection/render fields;
4. express export as query + projection + render rather than as a separate
   command family.
