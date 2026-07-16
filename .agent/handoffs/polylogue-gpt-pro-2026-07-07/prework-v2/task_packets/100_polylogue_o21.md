# 100. polylogue-o21 — Extension-point ergonomics: declare-once registries, scaffolds, actionable completeness errors

Priority/type/status: **P2 / feature / open**. Lane: **11-interoperability-origin**. Release: **K-interop-origin**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

Every extension today is a scavenger hunt across parallel registration sites, each failing opaquely when missed — the accumulated tribal knowledge lives in bd memories: a new MCP tool needs EXPECTED_TOOL_NAMES + TOOL_CONTRACT + role gating + render openapi + render cli-output-schemas (four separate opaque failures); a new golden-path workflow must be in REQUIRED_WORKFLOW_IDS or CLI startup crashes with an unrelated error; a new AssertionKind breaks two renders plus the every-kind-has-a-surface test; a new module fails topology verify until two regens run; a new origin touches detector, parser, enum, schema package, usage-coverage, and completeness matrix. This is the single biggest tax on future expansion: the cost is not writing the feature, it is discovering the registration constellation.

## Existing design note

Three legs, applied per extension point (MCP tool, CLI verb/command, DSL unit/stage, insight, origin, assertion kind, devtools command, workflow): (1) DECLARE-ONCE: each point gets a single declaration object carrying ALL metadata the parallel sites currently hold (name, contract, role gating, schema, docs blurb, owning surface) — the parallel lists become derivations: EXPECTED_TOOL_NAMES is generated FROM tool declarations, REQUIRED_WORKFLOW_IDS from workflow specs, render inputs read the declarations. Where a hard second site must remain (generated OpenAPI), the deriver owns it. (2) ACTIONABLE ERRORS: every registration validator, when it fails, names the missing step and the command that fixes it ('assertion kind X has no surface entry: add to user_audit surface map at <path>; then run devtools render openapi') — turn the four opaque failures into one checklist error. The registration-traps bd memory becomes obsolete BY CONSTRUCTION, which is the acceptance test: a new agent adds a tool end-to-end without the memory. (3) SCAFFOLDS: devtools new tool|origin|insight|command generates the declaration + stub + test skeleton in the right places (repo already generates surfaces; generating starting points is the same machinery pointed forward). Sequencing: pilot on ONE extension point (MCP tools — highest trap density), extract the pattern, then sweep the rest one per PR. Relates t46 (contracts own surfaces — this is the authoring-side complement) and utf (devtools catalog lint rides the same declaration).

## Acceptance criteria

- Slice 1 (size:S, unblocks dependents): a DeclarationSpec dataclass + registry protocol are defined and one pilot extension point (MCP tools) is migrated to declare-once, with a published pattern doc; dependents can build against the protocol immediately.
- DECLARE-ONCE (pilot): EXPECTED_TOOL_NAMES is derived FROM the tool declarations (grep shows the parallel list is a derivation, not hand-maintained); where a hard second site remains (generated OpenAPI) the deriver owns it.
- ACTIONABLE ERRORS: the MCP-tool registration validator, on failure, names the missing step and the exact fixing command (e.g. 'add to <path>; then run devtools render openapi'). Acceptance test: a new agent adds an MCP tool end-to-end WITHOUT consulting the registration-traps memory — the memory becomes obsolete by construction.
- SCAFFOLD: `devtools new tool` generates the declaration + stub + test skeleton in the correct places.
- Slices 2..n sweep the remaining extension points one per PR (child beads); `devtools verify` and `devtools render all --check` are green after each.

## Static mechanism / likely defect

Issue description localizes the mechanism: Every extension today is a scavenger hunt across parallel registration sites, each failing opaquely when missed — the accumulated tribal knowledge lives in bd memories: a new MCP tool needs EXPECTED_TOOL_NAMES + TOOL_CONTRACT + role gating + render openapi + render cli-output-schemas (four separate opaque failures); a new golden-path workflow must be in REQUIRED_WORKFLOW_IDS or CLI startup crashes with an unrelated error; a new AssertionKind breaks two renders plus the every-kind-has-a-surface test; a new module f… Design direction: Three legs, applied per extension point (MCP tool, CLI verb/command, DSL unit/stage, insight, origin, assertion kind, devtools command, workflow): (1) DECLARE-ONCE: each point gets a single declaration object carrying ALL metadata the parallel sites currently hold (name, contract, role gating, schema, docs blurb, owning surface) — the parallel lists become derivations: EXPECTED_TOOL_NAMES is generated FROM tool decl…

## Source anchors to inspect first

- `polylogue/sources/dispatch.py` — Current origin/source dispatch logic; target for OriginSpec consolidation.
- `polylogue/sources/import_preflight.py` — Preflight/readiness should report origin strictness and ambiguity.
- `polylogue/sources/provider_completeness.py` — Provider completeness is adjacent to OriginSpec readiness.
- `polylogue/sources/parsers/base.py` — Parser base contracts should be folded into OriginSpec.
- `polylogue/core/refs.py` — Existing ref model should be extended, not bypassed.
- `polylogue/storage/sqlite/archive_tiers/user_write.py:901` — Findings should reuse assertion/candidate lifecycle.
- `polylogue/surfaces/payloads.py:747` — Surface payloads currently expose branch/message refs; expand consistently.
- `polylogue/mcp/payloads.py:377` — MCP message payloads carry variant/branch metadata today.

## Implementation plan

1. Three legs, applied per extension point (MCP tool, CLI verb/command, DSL unit/stage, insight, origin, assertion kind, devtools command, workflow): (1) DECLARE-ONCE: each point gets a single declaration object carrying ALL metadata the parallel sites currently hold (name, contract, role gating, schema, docs blurb, owning surface) — the parallel lists become derivations: EXPECTED_TOOL_NAMES is generated FROM tool de…
2. Where a hard second site must remain (generated OpenAPI), the deriver owns it.
3. (2) ACTIONABLE ERRORS: every registration validator, when it fails, names the missing step and the command that fixes it ('assertion kind X has no surface entry: add to user_audit surface map at <path>
4. then run devtools render openapi') — turn the four opaque failures into one checklist error.
5. The registration-traps bd memory becomes obsolete BY CONSTRUCTION, which is the acceptance test: a new agent adds a tool end-to-end without the memory.
6. (3) SCAFFOLDS: devtools new tool|origin|insight|command generates the declaration + stub + test skeleton in the right places (repo already generates surfaces
7. generating starting points is the same machinery pointed forward).

## Tests to add

- Acceptance proof: Slice 1 (size:S, unblocks dependents): a DeclarationSpec dataclass + registry protocol are defined and one pilot extension point (MCP tools) is migrated to declare-once, with a published pattern doc
- Acceptance proof: dependents can build against the protocol immediately.
- Acceptance proof: DECLARE-ONCE (pilot): EXPECTED_TOOL_NAMES is derived FROM the tool declarations (grep shows the parallel list is a derivation, not hand-maintained)
- Acceptance proof: where a hard second site remains (generated OpenAPI) the deriver owns it.
- Acceptance proof: ACTIONABLE ERRORS: the MCP-tool registration validator, on failure, names the missing step and the exact fixing command (e.g.
- Acceptance proof: 'add to <path>
- Acceptance proof: then run devtools render openapi').
- Acceptance proof: Acceptance test: a new agent adds an MCP tool end-to-end WITHOUT consulting the registration-traps memory — the memory becomes obsolete by construction.

## Verification commands

- `devtools test <focused tests added for this bead>`

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
