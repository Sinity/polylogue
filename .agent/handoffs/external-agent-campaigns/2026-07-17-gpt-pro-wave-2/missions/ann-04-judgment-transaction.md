Title: "Finish the canonical judgment transaction: evidence disclosure, queue health, TOCTOU-safe writes, one operator lifecycle"

Result ZIP: `ann-04-judgment-transaction-r01.zip`

## Mission

Complete beads `polylogue-37t.12` (P1) + `polylogue-41ow` (P1 bug) + prepare
`polylogue-mrxt`'s canary. PR #2791 already landed the core: candidate
lifecycle transition authority, bulk SAVEPOINT semantics, explicit injection
authorization, immutable retry/conflict behavior, MCP review capability, and
the root `polylogue judge` workflow. Treat that merged lifecycle as the SOLE
transaction authority — audit and extend, never reimplement (37t.12 design
is explicit). Read all three beads fully in `.beads/issues.jsonl`.

Work items:

1. **41ow first** (it corrupts trust in everything else):
   `polylogue/storage/sqlite/archive_tiers/user_write.py::upsert_assertion`
   has a reproduced TOCTOU race that silently reverts operator judgments
   when an automated candidate update interleaves. Decided fix: one
   BEGIN IMMEDIATE preserve/write transaction — operator judgments must
   never be silently reverted. Also reconcile the `tilk` finding (upsert_*
   identity semantics inconsistent: content-hash append vs stable-identity
   update — classify each upsert_ helper and document/align its identity
   rule). Regression test = the reproduced interleaving (the bead cites the
   reproduction; find it in dogfood-2 investigation notes under
   `.agent/scratch/dogfood-2/investigations/` if present in the snapshot).
2. **37t.12 residuals**: evidence disclosure (a judgment presents the
   evidence refs it judged — surface them through the judge workflow and
   MCP review capability); queue health (pending-candidate counts/ages as a
   queryable product, wired into status); retire the duplicate `mark
   candidates` public workflow after porting any still-useful presentation
   onto root `polylogue judge` (parity first, then removal — bead design
   names `cli/query_verbs.py` as the duplicate's home and
   `click_command_registration.py` as the canonical registration).
3. **The mrxt canary, prepared not faked**: mrxt requires a GENUINE
   operator-authored action through the production route — you cannot
   perform it (no live archive). Deliver the exact operator script: the
   commands, the expected durable effects at each step (candidate row →
   verdict → assertion + evidence ref + judgment receipt + context policy →
   surface visibility), and the verification queries — so the operator
   executes the canary in minutes and any friction becomes child beads.
4. Tests: 41ow interleaving regression; lifecycle invariants from 37t.12's
   AC (machine candidates stay candidate/inject=false; default accept
   promotes one active inject=false assertion; explicit authenticated
   review may inject; reject/defer do not promote; exact retries
   idempotent; changed decisions conflict); bulk semantics
   (valid/idempotent/malformed/conflicting refs).

## Constraints

- user.db is durable-irreplaceable: no schema changes unless additive
  numbered migration + backup-manifest rules demand it — prefer none.
- The judgment lifecycle is the authority the annotation program (parallel
  jobs ann-01/02/03) builds on — keep interfaces stable and document them.

## Deliverable emphasis

HANDOFF.md: what PR #2791 already guaranteed vs what you added (exact),
the 41ow fix mechanism + proof, upsert identity-semantics table, duplicate-
workflow retirement diff, the operator canary script, and the queue-health
query surface.
