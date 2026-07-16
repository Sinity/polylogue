# 120. polylogue-212.7 — Demo Finding Packet contract + prompt runner + registry manifest

Priority/type/status: **P2 / task / open**. Lane: **05-analysis-provenance-citations**. Release: **C-read-contract**. Readiness: **blocked-hard**.

Hard blockers: polylogue-212.9

## What the bead says

Convert 212 from a shelf of named demos into a PORTFOLIO CONTRACT: every demo is an executable PROMPT.md handed to a coding agent, and every prompt emits the identical Demo Finding Packet: PROMPT.md, finding.yaml (five-part provenance stanza per 3tl.4: archive cursor, measure/query version, commit SHA, sample-frame predicate, run date), report.md (fixed section order: claim, corpus, method, findings, specimens, counterexamples, limits, reproduce), evidence.ndjson (one row per cited ref), queries.ndjson (text + lowered spec), annotations.ndjson (optional), checks.json (pass/fail + unsupported claims + coverage notes), run.log. The registry manifest lists every prompt file, expected packet path, public/private mode, and required primitives — so the portfolio is enumerable and CI-checkable. Compositionality rule inherited from 212: steps are product primitives (polylogue argv), shell/python is glue only.

## Existing design note

Anchor: .agent/demos/ (existing shelf: agent-forensics, claim-vs-evidence, degraded-archive-proof, CURATED_CATALOG.md as the manifest seed). Contract: every demo directory gains PROMPT.md (executable instructions a coding agent runs cold) and emits an identical Demo Finding Packet: finding.yaml (five-part provenance stanza per 3tl.4), rendered artifact, and the exact reproduction commands. Build a registry manifest (extend CURATED_CATALOG.md or a demos.yaml) listing id, claim, packet path, substrate features exercised, last-regenerated. A prompt runner (thin script or devtools lab command) executes one demo prompt end-to-end and validates packet shape. Pitfall: demos run against the LIVE archive — packet outputs must be private-data-audited before any publication lane (3tl.4 owns publishing).

## Acceptance criteria

Packet schema documented + validated by the runner; one existing demo (D1 receipts) re-emitted through the runner produces a conforming packet on the seeded corpus; registry manifest lint catches a missing packet. Verify: runner fixture test + manifest check.

## Static mechanism / likely defect

Issue description localizes the mechanism: Convert 212 from a shelf of named demos into a PORTFOLIO CONTRACT: every demo is an executable PROMPT.md handed to a coding agent, and every prompt emits the identical Demo Finding Packet: PROMPT.md, finding.yaml (five-part provenance stanza per 3tl.4: archive cursor, measure/query version, commit SHA, sample-frame predicate, run date), report.md (fixed section order: claim, corpus, method, findings, specimens, counterexamples, limits, reproduce), evidence.ndjson (one row per cited ref), queries.ndjson (text + low… Design direction: Anchor: .agent/demos/ (existing shelf: agent-forensics, claim-vs-evidence, degraded-archive-proof, CURATED_CATALOG.md as the manifest seed). Contract: every demo directory gains PROMPT.md (executable instructions a coding agent runs cold) and emits an identical Demo Finding Packet: finding.yaml (five-part provenance stanza per 3tl.4), rendered artifact, and the exact reproduction commands. Build a registry manifest …

## Source anchors to inspect first

- `polylogue/core/refs.py` — Existing ref model should be extended, not bypassed.
- `polylogue/storage/sqlite/archive_tiers/user_write.py:901` — Findings should reuse assertion/candidate lifecycle.
- `polylogue/surfaces/payloads.py:747` — Surface payloads currently expose branch/message refs; expand consistently.
- `polylogue/mcp/payloads.py:377` — MCP message payloads carry variant/branch metadata today.
- `polylogue/archive/query/spec.py:440` — SessionQuerySpec is the core read selection object.
- `polylogue/archive/query/spec.py:498` — from_params centralizes parameter lowering.
- `polylogue/cli/query_contracts.py:70` — QueryOutputSpec is current output/render contract seam.
- `polylogue/cli/archive_query.py:164` — CLI archive query execution remains a dense orchestration point.
- `polylogue/daemon/http.py:1734` — Daemon split-archive route already has from_params path for some cases.
- `polylogue/daemon/http.py:3296` — Another daemon path builds SessionQuerySpec from raw query params.
- `polylogue/mcp/query_contracts.py:54` — MCP query normalization has a distinct lowering path.
- `README.md` — Public claims should be grounded through the claims ledger.
- `docs/agent-forensics.md` — Existing forensics docs are a pattern for proof artifacts.
- `docs/demo.md` — Demo docs should depend on evidence/citation machinery.

## Implementation plan

1. Anchor: .agent/demos/ (existing shelf: agent-forensics, claim-vs-evidence, degraded-archive-proof, CURATED_CATALOG.md as the manifest seed).
2. Contract: every demo directory gains PROMPT.md (executable instructions a coding agent runs cold) and emits an identical Demo Finding Packet: finding.yaml (five-part provenance stanza per 3tl.4), rendered artifact, and the exact reproduction commands.
3. Build a registry manifest (extend CURATED_CATALOG.md or a demos.yaml) listing id, claim, packet path, substrate features exercised, last-regenerated.
4. A prompt runner (thin script or devtools lab command) executes one demo prompt end-to-end and validates packet shape.
5. Pitfall: demos run against the LIVE archive — packet outputs must be private-data-audited before any publication lane (3tl.4 owns publishing).

## Tests to add

- Acceptance proof: Packet schema documented + validated by the runner
- Acceptance proof: one existing demo (D1 receipts) re-emitted through the runner produces a conforming packet on the seeded corpus
- Acceptance proof: registry manifest lint catches a missing packet.
- Acceptance proof: Verify: runner fixture test + manifest check.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit tests/integration -k "query or contract or citation or insight"`

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
