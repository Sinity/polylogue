# Remaining operator decisions

`ARCH-04` denotes `.agent/handoffs/external-agent-campaigns/2026-07-16-gpt-pro-wave/testdiet/context/testsuite_diet/architecture/04-destructive-and-authentication-boundaries.md`.

The architecture direction is already decided. These are the five genuine product/operator choices that remain; each default is implementable without introducing a parallel policy system.

## 1. Is `judge` a role or a capability?

**Recommended default:** `assertion:judge` is a distinct capability. Keep `review` as a deployable MCP profile/discovery label during migration; it contains read + write + judgment. `admin` contains it. Ordinary `write` does not.

**Rationale:** Current source already has `read < write < review < admin` (`polylogue/mcp/declarations/models.py:11-18`) and registers judgment at review (`polylogue/mcp/server_tools.py:1424-1434`). Bead `polylogue-37t.12` requires an ordinary write role not to see judgment and says `actor_ref` is not authority. Modeling the distinction as a capability lets CLI/API/daemon/system callers use the same executor without turning MCP role names into universal policy.

**Consequence:** Discovery is profile-filtered; apply authorization checks the capability. Removing `review` later is a packaging choice, not a security-model migration.

**Reverse only if:** The product deliberately permits every write-capable receiver to make operator judgments. That would weaken the current judgment gate and requires explicit acceptance of the injection consequences.

## 2. What does legacy `delete_session` mean?

**Recommended default:** retire the ambiguous verb. Compatibility maps it to `session.prune` only, with an explicit response that source/user evidence remains. A user-visible hide is the separate reversible `session.suppress` write. Irreversible destruction is only `session.excise` under CB; archive-wide/broad excision is D3 and requires CB+.

**Rationale:** `ArchiveStore.delete_sessions` deletes rebuildable index rows and intentionally leaves user overlays (`polylogue/storage/sqlite/archive_tiers/archive.py:6328-6348`). Reset combines suppression and prune (`polylogue/cli/commands/reset.py:380-452`). Excision destroys evidence across tiers (`polylogue/security/excision.py:1-60,577-642`). The decided architecture requires suppression and excision to remain distinct (`ARCH-04:59-66`).

**Consequence:** Existing clients receive a deprecation/semantic-warning field and a `session.prune` receipt. No adapter may claim evidence was deleted unless it has an excision receipt.

**Reverse only if:** Compatibility data proves clients universally expect hide semantics rather than prune. In that case map the alias to a composite `session.suppress + session.prune` operation, still never to excision.

## 3. Do user-state remove/delete verbs retain interactive confirmation?

**Recommended default:** not in the final protocol. Tags, marks, annotations, saved views, recall packs, workspaces, metadata, and corrections are R2 assertion transitions. They require capability, exact target/generation, idempotency, durable before-image/restore receipt, and CAS where replacing state; they do not require interactive confirmation. Keep `polylogue-jn40` booleans until those receipts and inverse routes exist.

**Rationale:** The architecture explicitly gives reversible low-risk writes no interactive confirmation and reserves bound tokens for scoped/broad destruction (`ARCH-04:44-53`). The unified `user.db` assertion model preserves lifecycle/supersession semantics (`CLAUDE.md:99-107`). Removing a tag/mark is not evidence excision. The interim bead intentionally over-gates because the structural executor is absent.

**Consequence:** The migration has a measurable exit condition: no boolean removal until the operation receipt can reconstruct the prior active assertion and an inverse is executable. `correction.clear` remains CB during migration if all affected rows cannot be restored.

**Reverse only if:** Product policy intentionally treats deletion of a user-owned assertion as irreversible even while history remains. That would be a UX choice, not a storage-safety requirement.

## 4. Who may invoke `run`, and how does a recipe acquire authority?

**Recommended default:** expose `run` to write/review/admin profiles through `run:execute`. A saved query that is purely read remains executable through `query`/`read`. A recipe is authorized only after immutable expansion; effective capability is the union of all steps, and destructive confirmation binds the complete expanded DAG/target/effect digest. Saved content grants no capability.

**Rationale:** The generated target algebra assigns `run` to write and declares saved-query/recipe objects (`docs/generated/mcp-equivalence.json`, `target_algebra`). The architecture forbids saved recipes from acquiring authority (`ARCH-04:76-79`). Step-by-step authorization after partial execution would permit an unauthorized prefix effect; whole-plan authorization avoids that.

**Consequence:** Recipe versions are immutable for a run. Any changed ref/version/expansion requires a new preview. Resume uses durable step receipts and cannot widen scope.

**Reverse only if:** The product chooses not to support mutating recipes in this cutover. Then `run` is read-only initially and mutating recipe specs remain undiscoverable until the whole-plan executor exists.

## 5. May daemon/system automation use standing authorization?

**Recommended default:** yes, but only for declared, bounded R1 and selected D1 convergence operations. The system principal has explicit narrow capabilities and every run still goes through the executor with a real preview/effect digest and durable receipt. R1 uses C0; selected bounded D1 may use a system-bound CB minted and consumed under the declared standing policy. D2 operations such as scoped excision and blob/raw cleanup require a human-bound CB preview; D3 durable reset, destructive migration, archive-wide destruction, and live raw-authority repair require CB+, even when a daemon performs the eventual apply.

**Rationale:** Polylogue is local/single-writer, not multi-user RBAC (`CLAUDE.md:3-7,140-160`; `ARCH-04:68-79`). Internal repair must use the same executor (bead `polylogue-t46.9`), but requiring a human click for every bounded insight/FTS convergence pass would make normal operation impossible. The destructive-class boundary is the stable distinction. Current live ingest automatically invokes raw-retention deletion (`polylogue/sources/live/batch.py:2256-2284`), but that deletes durable raw rows/blob bytes (`polylogue/storage/raw_retention.py:1146-1248`) and therefore does not qualify for a standing D1 grant on present evidence.

**Consequence:** Standing grants specify operation id/version, archive/receiver, maximum target/count/byte/time budget, allowed tiers, and expiry/rotation policy. They do not erase confirmation class: a selected D1 run still records a system-bound CB. Scope widening fails closed and becomes an operator-preview request. Superseded raw-snapshot cleanup remains disabled or human-bound D2/CB until an executable spec proves byte-exact reconstruction and explicitly reclassifies it.

**Reverse only if:** Operators require all D1 deletion, including derived cleanup, to be interactive. That is stricter and safe, but should be accepted as an availability/maintenance cost.

## Decision record summary

| Decision | Default |
| --- | --- |
| Judgment authority | Distinct `assertion:judge` capability; `review` retained as profile |
| Legacy session delete | Retire; compatibility means prune (or explicitly composite suppress+prune), never excision |
| User assertion deletion confirmation | Interim booleans now; final C0/CAS with restorable receipts |
| Run authority | `run:execute` plus union of expanded step capabilities; whole-plan confirmation |
| System automation | Standing bounded R1 C0 / selected D1 system-bound CB only; D2 human-bound CB, D3 human-bound CB+ |
