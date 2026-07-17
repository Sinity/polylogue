Title: "Six-tool migration evidence: old-to-new equivalence map, parity-golden design, and client migration guide"

Result ZIP: `mcp-03-migration-parity-r01.zip`

## Mission

Analysis + design job (analysis contract). The six-tool cutover deletes ~97
MCP tools with NO runtime aliases (no external users; hard-rename policy).
What preserves correctness through that deletion is build-time equivalence
evidence: for every retired tool, proof that its externally-meaningful
capability is expressible through the new surface, plus goldens that pin
representative old outputs to new-route outputs. Design that evidence layer
completely so the cutover lane can execute it mechanically.

Inputs (snapshot):

- The live 103-tool inventory: `tests/infra/mcp.py` EXPECTED_TOOL_NAMES +
  per-tool contracts (`tests/unit/mcp/test_tool_contracts.py`), and
  `docs/mcp-reference.md`.
- Observed usage telemetry: ops.db MCP call logs exist as a concept — find
  the call-log schema (`storage/sqlite/archive_tiers/ops.py`) and specify
  how to rank tools by actual observed use so parity effort is spent where
  usage was real (the long tail of never-called tools needs only a
  capability-mapping row, not goldens).
- Bead `polylogue-t46.8.1` (equivalence-map declaration fields) and
  `t46.8.2`/`t46.8.2.1` (already-sentenced duplicates) and `t46.8.3`
  (context/assertion/judgment/maintenance family migration).

Deliver:

1. **The equivalence map**, complete: 103 rows — old tool → new-surface
   expression (six-tool call shape or role-gated family verb or URI
   resource or prompt), semantic deltas (pagination model change, field
   renames, result-ref changes), and disposition (parity-golden / mapping-
   only / retired-no-successor with justification).
2. **Parity-golden design**: fixture archive requirements, how goldens are
   captured from the OLD surface pre-deletion (exact harness invocation),
   normalization rules (timestamps, volatile ids), storage layout in-repo,
   and the shadow-call comparison harness shape for the transition branch.
3. **Client migration guide**: every consumer on the operator's machine —
   sinnix MCP profiles (`flake/data/mcp-registry.nix`, client-profiles),
   the polylogue skill, recipe prompts, SessionStart hooks, any agent docs —
   with the exact config/doc changes the six-tool era requires (you cannot
   read sinnix; list what to check and the expected shape of each change).
4. **Cutover sequencing risk table**: what breaks mid-branch (tests pinning
   EXPECTED_TOOL_NAMES, rendered openapi/output schemas, docs coverage
   ratchets), with the exact regeneration command order.

## Deliverable emphasis

REPORT.md (equivalence map — the complete table is the deliverable, no
sampling), EVIDENCE.md (usage-ranking method + contract citations),
DECISIONS.md (dispositions + normalization rules), NEXT-ACTIONS.md (golden
capture → cutover → regeneration → client-update checklist in exact order).


---

## Context and authority

You are a long-running ChatGPT Pro analysis worker. A recent, complete
Polylogue project-state archive will be attached. Retrieve and inspect it
broadly; attachment size alone is not a reason to ignore evidence. This prompt
defines the question. The snapshot's current source, repository instructions,
complete relevant Beads records, and cited history are the evidence authority,
in that order when older plans drift.

## Working contract

- Investigate the actual source and tracker state before recommending changes.
- Separate observed facts, source-supported inference, unresolved uncertainty,
  and recommendation. Quote paths/symbols/Bead ids precisely but do not fill the
  report with copied source.
- Adjudicate contradictions and duplicates; do not create a parallel product
  model or generic architecture merely to make the report look complete.
- Translate findings into decision-ready actions: exact owning areas, ordering,
  acceptance criteria, falsification evidence, and what a local implementer
  should verify.
- Do not claim live browser, daemon, archive, deployment, or test evidence you
  cannot access.

## Deliverable

Create the exact `Result ZIP` named near the top under `/mnt/data/`. It must
contain `REPORT.md`, `EVIDENCE.md`, `DECISIONS.md`, and `NEXT-ACTIONS.md`.
Include compact machine-readable tables as JSON/CSV only when they add genuine
integration value. Do not copy the input archive into the result. Attach the
finished ZIP to the conversation through a working user-clickable link; files
left only in an internal temporary directory are not delivered.

Reopen and validate the ZIP, then report its SHA-256, size, and members. The
final chat answer must itself explain the important conclusions and decisions,
limitations, missing evidence, and the likely value of another iteration before
linking the package.

Do not perform an adversarial review unless explicitly requested. On an
ordinary **iterate/continue** request, preserve sound findings, resolve the
highest-value remaining uncertainty, and regenerate a complete package
revision. On an explicit **adversarial review** request, try to falsify the
prior report: seek contrary source/history evidence, unsupported certainty,
missed stakeholders/call sites, duplicate or incompatible designs, weak
acceptance criteria, and recommendations that do not survive current code.
Repair legitimate findings, regenerate the cohesive package, and report the
delta, residual disputes, and expected value of another pass.
