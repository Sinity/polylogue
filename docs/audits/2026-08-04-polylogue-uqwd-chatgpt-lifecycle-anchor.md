# polylogue-uqwd evidence packet: ChatGPT lifecycle-anchor drift

Date: 2026-08-04. Worktree: `feature/fix/chatgpt-anchor-audit`. Scope: record a reproducible current-corpus census for the `generation_lifecycle` moved-anchor conflict without changing Beads, source.db, index.db, the blob store, backups, daemon state, or services.

## Verdict

The prior census was only an untracked `/realm/tmp` report, so its current-corpus conclusion was not independently auditable. This packet is now backed by the committed, reproducible `devtools workspace chatgpt-lifecycle-anchor-audit` command and its sanitized receipt at `docs/audits/2026-08-04-polylogue-uqwd-chatgpt-lifecycle-anchor-receipt.json`.

The historical semantic risk remains real in principle. `tests/unit/sources/test_parsers_chatgpt.py` now carries an end-to-end regression from two mapping orders through the ChatGPT parser, revision projection, relation, and classifier. The same fixture applies the pre-`b1e01d878` position tiebreak and asserts a `conflict` relation with both raws `ambiguous`, then asserts the current parser produces one accepted and one equivalent raw. The direct `ParsedSession` test in `tests/unit/archive/test_session_revision_membership.py` remains only as a classifier-only different-content guard.

The receipt does **not** de-gate `uqwd`, `xselt`, or `818fy`. A zero result from the current parser and corpus is insufficient to establish the bead's required historical replay fixture. No graph state, archive state, blob store, daemon state, or service state was written by this work.

## Reproducible receipt

The command opens `source.db` and `index.db` with SQLite `mode=ro`, scans and verifies the blob namespace without writes, then invokes `_parse_one`, `session_revision_projection`, `_relation`, and `classify_membership_revisions`. It selects persisted `raw_session_memberships.logical_source_key` cohorts, reads current `raw_revision_heads` as classifier heads, and does not call a replay or writeback function. A newly generated version 2 receipt will bind the result to the exact producer `HEAD`, working-tree cleanliness and status digest, plus a deterministic full blob namespace snapshot and content-integrity digest whose aggregate identity includes each observed blob digest. Receipt output must resolve outside the archive root. Those fields are not present in the checked-in historical receipt below.

The historical receipt checked into this packet predates the version 2 provenance contract. It is retained as historical evidence only and must not be presented as a post-repair rerun. The repaired command deliberately contains no raw ids, native ids, source paths, blob hashes, titles, or payload content. A fresh receipt records the same aggregate fields together with the producer and blob identities needed to make a rerun meaningful.

The historical receipt's pairwise counts were `equal=7,348`, `a_contains_b=126`, `b_contains_a=82`, and `conflict=71`. Its target predicate count was zero under the pre-repair implementation. That value is not evidence that the repaired predicate has no target pair. The repaired target predicate counts exactly one `generation_lifecycle` event per side, permits other session events, compares normalized lifecycle content after removing the anchor, requires all other normalized event content to match, and requires the production conflict relation. This does not imply that the remaining conflicts are harmless or that historical gates can be removed.

## Regression safeguards

The end-to-end regression constructs two otherwise identical ChatGPT export mappings with different insertion orders, then invokes the real parser, projection, relation, and classifier. Under a test-only mutation that restores the historical position-based tiebreak, it asserts different anchors, a `conflict` relation, and two `ambiguous` raws. With the current parser it asserts a common anchor, an equal relation, one accepted raw, and one equivalent raw.

The retained direct classifier guard constructs `ParsedSession` values with different lifecycle `state` content and moved anchors. It asserts conflict and ambiguity with an existing head. Its scope is limited to classifier behavior and it intentionally does not cover parser ordering.

## Acceptance match

| Acceptance criterion | Status | Evidence |
| --- | --- | --- |
| Run the real classifier and projection path against current quarantined ChatGPT data | Command ready, receipt pending | The repaired command binds parser, projection, relation, classifier, SQL selection, producer checkout and blob integrity. The checked-in receipt is pre-repair evidence only |
| Confirm moved lifecycle anchors still produce conflicts | Satisfied in historical mutation fixture | The end-to-end regression asserts the real parser-to-classifier conflict and ambiguous result under the pre-fix tiebreak |
| Implement the narrow comparison exception if reproduced | Satisfied | `_matches_target` compares normalized lifecycle content after removing `source_message_provider_id`, includes normalized timing semantics, and rejects unrelated event changes |
| Preserve different-content moved-anchor behavior | Satisfied | The direct `ParsedSession` guard keeps changed lifecycle content conflicting; the parser-to-classifier regression owns ordering behavior |
| Reclassify the current blocker edge honestly | Not satisfied | This packet makes no reclassification or de-gating recommendation until the retained historical replay fixture exists |

## Graph disposition and residual uncertainty

No graph-edge conclusion is made here. The historical mutation fixture establishes the classifier shape and the active parser remains protected against the known ordering bug, but the packet does not provide a retained pre-fix historical replay of the live corpus. The packet therefore preserves the no-de-gating stance for `uqwd`, `xselt`, and `818fy`.

Residual uncertainty is limited to provenance of the archive transition between the 2026-08-03 negative lookup and this 2026-08-04 snapshot, and the absence of an archived pre-fix parser output for the original 10/136 sample. The live durable membership decisions remain unchanged because this lane performed no writeback.
