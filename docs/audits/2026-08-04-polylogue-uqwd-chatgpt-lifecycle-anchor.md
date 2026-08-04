# polylogue-uqwd evidence packet: ChatGPT lifecycle-anchor drift

Date: 2026-08-04. Worktree: `feature/fix/chatgpt-anchor-audit`. Scope: record a reproducible current-corpus census for the `generation_lifecycle` moved-anchor conflict without changing Beads, source.db, index.db, the blob store, backups, daemon state, or services.

## Verdict

The prior census was only an untracked `/realm/tmp` report, so its current-corpus conclusion was not independently auditable. This packet is now backed by the committed, reproducible `devtools workspace chatgpt-lifecycle-anchor-audit` command and its sanitized receipt at `docs/audits/2026-08-04-polylogue-uqwd-chatgpt-lifecycle-anchor-receipt.json`.

The historical semantic risk remains real in principle. `tests/unit/sources/test_parsers_chatgpt.py` now carries an end-to-end regression from two mapping orders through the ChatGPT parser, revision projection, relation, and classifier. The direct `ParsedSession` test in `tests/unit/archive/test_session_revision_membership.py` remains only as a classifier-only different-content guard.

The receipt does **not** de-gate `uqwd`, `xselt`, or `818fy`. A zero result from the current parser and corpus is insufficient to establish the bead's required historical replay fixture. No graph state, archive state, blob store, daemon state, or service state was written by this work.

## Reproducible receipt

The command opens `source.db` and `index.db` with SQLite `mode=ro`, reads blobs, then invokes `_parse_one`, `session_revision_projection`, `_relation`, and `classify_membership_revisions`. It selects persisted `raw_session_memberships.logical_source_key` cohorts, reads current `raw_revision_heads` as classifier heads, and does not call a replay or writeback function. The exact SQL, predicate, code revision, schema versions, file sizes and mtimes are in the receipt.

The receipt deliberately contains no raw ids, native ids, source paths, blob hashes, titles, or payload content. It records 7,498 selected quarantined ChatGPT raws, 2,570 persisted cohorts, 15 singleton cohorts, 2,555 multi-candidate cohorts, and 7,483 raws in those multi-candidate cohorts. All 7,498 selected raws were parsed and projected.

Across pairwise comparisons the production relation counts were `equal=7,348`, `a_contains_b=126`, `b_contains_a=82`, and `conflict=71`. The target predicate count was zero: no pair had exactly one `generation_lifecycle` event per side, different anchors, equal transcript and attachment content, equal non-anchor lifecycle content, and a conflict relation. This does not imply that the remaining conflicts are harmless or that historical gates can be removed.

## Regression safeguards

The end-to-end regression constructs two otherwise identical ChatGPT export mappings with different insertion orders, then invokes the real parser, projection, relation, and classifier. It asserts a common anchor, an equal relation, one accepted raw, one equivalent raw, and no ambiguity. It fails against the historical position-based tie-break.

The retained direct classifier guard constructs `ParsedSession` values with different lifecycle `state` content and moved anchors. It asserts conflict and ambiguity with an existing head. Its scope is limited to classifier behavior and it intentionally does not cover parser ordering.

## Acceptance match

| Acceptance criterion | Status | Evidence |
| --- | --- | --- |
| Run the real classifier and projection path against current quarantined ChatGPT data | Reproducible current-corpus evidence | The committed command and sanitized receipt make the parser, projection, relation, classifier, SQL selection, denominators, and archive provenance reviewable |
| Confirm moved lifecycle anchors still produce conflicts | Historical fixture still required | The end-to-end regression reproduces the mapping-order mechanism, but the current snapshot does not substitute for the required archived pre-fix replay |
| Implement the narrow comparison exception if reproduced | Not applicable | No production code change made |
| Preserve different-content moved-anchor behavior | Satisfied, classifier scope only | The direct `ParsedSession` guard keeps changed lifecycle content conflicting; the parser-to-classifier regression owns ordering behavior |
| Reclassify the current blocker edge honestly | Not satisfied | This packet makes no reclassification or de-gating recommendation until the retained historical replay fixture exists |

## Graph disposition and residual uncertainty

No graph-edge conclusion is made here. The current-corpus receipt and regression establish that the active parser is protected against the known ordering bug, but they do not provide a retained pre-`b1e01d878` historical replay fixture. The packet therefore preserves the no-de-gating stance for `uqwd`, `xselt`, and `818fy`.

Residual uncertainty is limited to provenance of the archive transition between the 2026-08-03 negative lookup and this 2026-08-04 snapshot, and the absence of an archived pre-fix parser output for the original 10/136 sample. The live durable membership decisions remain unchanged because this lane performed no writeback.
