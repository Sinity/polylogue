Title: "Lower Codex functions.exec child operations into typed, provenance-linked actions (j2zz)"

Result ZIP: `lin-04-codex-delegations-r01.zip`

## Mission

Implement bead `polylogue-j2zz` (P1 — read its full record). Modern Codex
embeds typed operations INSIDE `functions.exec` JavaScript envelopes. Live
evidence from the newest 100-session sample: every session had nested tool
calls; 14,004 envelopes held child operations; 19,180 results yielded ZERO
structured paths or outcomes even though 1,444 result texts contained
`exit_code`. Polylogue currently retains only the outer exec/shell
semantics — so for modern Codex work, the actions relation (the archive's
core forensic value: what tools ran, what failed) is nearly blind. This
also silently biases the claim-vs-evidence analysis (Codex structured
failures undercounted).

Build the lowering (bead design is authority):

1. **Typed child registry**: lower `functions.exec` children —
   `exec_command`, `apply_patch`, `write_stdin`, `update_plan`, `wait`,
   `web`, `image`, MCP calls, and unknown shapes — into provenance-linked
   child actions while RETAINING the outer call as transport (evidence
   preserved, interpretation added; never destroy the envelope).
2. **Structural promotion only**: promote only structural result fields
   (exit codes, paths, byte counts) — outcome fields are structural or
   `unknown`, never regex-guessed from prose (the tool_result_is_error
   discipline). Commands and patches expose normalized command strings and
   touched paths.
3. **Ordering + pairing**: preserve ordering and repeated calls; child
   use/result pairing is deterministic (Nth-use↔Nth-result within
   envelope), continuations pair without inventing recovery; malformed and
   unknown children retain raw evidence with typed unknown state.
4. **Feed the bounded relation**: children land in the `action_pairs`
   derived relation that `polylogue-z9gh.2` landed (PR #3018 — read the
   current `action_pairs` schema/rebuild path in
   `storage/sqlite/archive_tiers/` and the actions compatibility view).
   Parent/child linkage: child actions carry the transport action's
   identity as provenance so delegation queries can distinguish outer
   transport from inner operations.
5. **Fixtures + live-sample report**: fixtures lowering single and
   multiple children (bead AC enumerates the cases); plus a snapshot-
   runnable census script reporting child/parent counts and structured-
   outcome coverage over any Codex session corpus — the integrator runs it
   on the live archive to produce the before/after coverage numbers
   (14,004 envelopes → N typed children, 0 → M structured outcomes).

## Constraints

- Parser layer changes live in `polylogue/sources/parsers/` (codex parser)
  + normalization; keep detection tightness order intact
  (`sources/dispatch.py`).
- Reparse semantics: this is semantic-reparse-affecting for Codex sessions
  — content hash includes blocks, so lowering that CHANGES block structure
  changes import identity. Read `pipeline/ids.py` hashing rules and state
  clearly in HANDOFF whether your lowering alters content hashes (and thus
  triggers re-import of Codex sessions on next ingest) — if yes, that is
  acceptable but must be explicit, with the expected re-ingest cost.
- Index-tier effects (new action rows) are rebuild-route only; no durable
  tier changes.

## Deliverable emphasis

HANDOFF.md: registry design, per-child-type field promotion table,
pairing/ordering semantics, content-hash impact statement, census-script
usage, and the exact claim-vs-evidence interaction (which undercounts this
fixes — coordinate wording with the parallel demo lane's economy work).
