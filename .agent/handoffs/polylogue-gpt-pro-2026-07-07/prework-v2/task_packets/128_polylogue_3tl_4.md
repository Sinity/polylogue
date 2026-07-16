# 128. polylogue-3tl.4 — Findings publishing lane: campaign artifacts on the docs site

Priority/type/status: **P2 / task / open**. Lane: **12-external-legibility-demos**. Release: **L-legibility-demos**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

The Pages pipeline already builds and deploys the docs site on master push; give campaign artifacts (claim-vs-evidence finding, forensics report) a publishing lane there — rendered report + reproduction instructions, regenerated from the seeded corpus so nothing private ships. The finding needs a URL before anything external can cite it.

## Existing design note

Publishing lane = a devtools render surface, not an ad-hoc workflow. (1) SOURCE: each finding lives at docs/findings/<slug>/finding.yaml carrying a five-part PROVENANCE STANZA (archive cursor id/position at measurement, measure+query-DSL/code version, git commit SHA, sample-frame predicate = the exact population query, run date) plus its structural body. (2) RENDER: add devtools/render_findings.py + a CommandSpec 'render findings' in devtools/command_catalog.py (model on render_pages.py entry at command_catalog.py:191); wire it into devtools/render_all.py so 'devtools render all --check' fails on drift. It renders docs/findings/<slug>/index.md (+ a per-finding CHANGELOG section). (3) PUBLISH: fold docs/findings/** into the existing site build (render_pages.py / pages_builder.py) so pages.yml (already deploys docs site on master push) serves each finding at a STABLE citeable URL /findings/<slug>/ — no per-run paths (per-PR/versioned trees are deferred under #1307; a finding is a living page at a fixed slug, that fixed slug is the URL 3tl's acceptance requires). (4) PROVENANCE GATE (enforcement point for the cpf finding-provenance doctrine): the lane REFUSES to render any finding missing any of the five stanza fields — non-zero exit with a named error. The stanza schema is defined once and shared with cpf (cpf lands the doctrine TEXT + deny-lexicon; 3tl.4 lands the executable refusal). (5) LIVING PAGES: re-running a finding does NOT mint a new URL and does NOT silently replace numbers — it appends a dated CHANGELOG entry attributing each changed number to the provenance delta (new cursor/commit/run date) and supersedes in place. (6) NO PRIVATE DATA: the published body regenerates its numbers from the deterministic demo corpus (polylogue demo seed, seed 1843) so nothing private ships; any live-archive figures shown are labeled and bounded by the documented sample-frame predicate (structural aggregates only, never raw private rows). PITFALLS: keep render output deterministic so render all --check stays clean; do not couple the ship of 3tl.4 to cpf's full doctrine landing — ship the gate function with an inline stanza schema that cpf's text then points at; never publish raw archive rows, only seeded-corpus-reproducible aggregates.

## Acceptance criteria

1. devtools render findings exists, registered in devtools/command_catalog.py, wired into devtools render all with a working --check. 2. At least one real finding (the base claim-vs-evidence finding) renders to docs/findings/<slug>/index.md and is served by pages.yml at a stable citeable URL /findings/<slug>/ (this is the 'published finding URL' 3tl acceptance clause 3 depends on). 3. Provenance gate: a finding source missing any of the five stanza fields makes the lane exit non-zero with a named error; a focused test covers the refusal. 4. Living-page changelog: re-rendering with a changed provenance stanza appends a dated changelog entry with attributed number deltas at the SAME slug/URL (no silent replace); a test covers supersede-with-delta. 5. The published finding regenerates its numbers from the seeded demo corpus (seed 1843); no private-archive rows appear in output. Verify: devtools render findings --check; devtools render all --check; devtools verify doc-commands; focused test for the provenance refusal + changelog supersede.

## Static mechanism / likely defect

Issue description localizes the mechanism: The Pages pipeline already builds and deploys the docs site on master push; give campaign artifacts (claim-vs-evidence finding, forensics report) a publishing lane there — rendered report + reproduction instructions, regenerated from the seeded corpus so nothing private ships. The finding needs a URL before anything external can cite it. Design direction: Publishing lane = a devtools render surface, not an ad-hoc workflow. (1) SOURCE: each finding lives at docs/findings/<slug>/finding.yaml carrying a five-part PROVENANCE STANZA (archive cursor id/position at measurement, measure+query-DSL/code version, git commit SHA, sample-frame predicate = the exact population query, run date) plus its structural body. (2) RENDER: add devtools/render_findings.py + a CommandSpec 'r…

## Source anchors to inspect first

- `README.md` — Public claims should be grounded through the claims ledger.
- `docs/agent-forensics.md` — Existing forensics docs are a pattern for proof artifacts.
- `docs/demo.md` — Demo docs should depend on evidence/citation machinery.
- `polylogue/core/refs.py` — Existing ref model should be extended, not bypassed.
- `polylogue/storage/sqlite/archive_tiers/user_write.py:901` — Findings should reuse assertion/candidate lifecycle.
- `polylogue/surfaces/payloads.py:747` — Surface payloads currently expose branch/message refs; expand consistently.
- `polylogue/mcp/payloads.py:377` — MCP message payloads carry variant/branch metadata today.
- `polylogue/archive/actions/followup.py` — Action/followup classification is a real structural analytics input.
- `polylogue/archive/actions/fields.py` — Action fields determine what can be measured without prose heuristics.
- `polylogue/insights/registry.py:294` — Insight registry should become measure/product registry input.
- `scripts/agent_forensics.py` — Existing forensics script is a proof artifact and candidate product surface.

## Implementation plan

1. Publishing lane = a devtools render surface, not an ad-hoc workflow.
2. (1) SOURCE: each finding lives at docs/findings/<slug>/finding.yaml carrying a five-part PROVENANCE STANZA (archive cursor id/position at measurement, measure+query-DSL/code version, git commit SHA, sample-frame predicate = the exact population query, run date) plus its structural body.
3. (2) RENDER: add devtools/render_findings.py + a CommandSpec 'render findings' in devtools/command_catalog.py (model on render_pages.py entry at command_catalog.py:191)
4. wire it into devtools/render_all.py so 'devtools render all --check' fails on drift.
5. It renders docs/findings/<slug>/index.md (+ a per-finding CHANGELOG section).
6. (3) PUBLISH: fold docs/findings/** into the existing site build (render_pages.py / pages_builder.py) so pages.yml (already deploys docs site on master push) serves each finding at a STABLE citeable URL /findings/<slug>/ — no per-run paths (per-PR/versioned trees are deferred under #1307
7. a finding is a living page at a fixed slug, that fixed slug is the URL 3tl's acceptance requires).

## Tests to add

- Acceptance proof: 1.
- Acceptance proof: devtools render findings exists, registered in devtools/command_catalog.py, wired into devtools render all with a working --check.
- Acceptance proof: 2.
- Acceptance proof: At least one real finding (the base claim-vs-evidence finding) renders to docs/findings/<slug>/index.md and is served by pages.yml at a stable citeable URL /findings/<slug>/ (this is the 'published finding URL' 3tl acceptance clause 3 depends on).
- Acceptance proof: 3.
- Acceptance proof: Provenance gate: a finding source missing any of the five stanza fields makes the lane exit non-zero with a named error
- Acceptance proof: a focused test covers the refusal.
- Acceptance proof: 4.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools verify --docs --demo`

## Pitfalls

- Do not publish a number or success claim until evidence tier, denominator, and caveat rendering exist.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
