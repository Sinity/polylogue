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
