# polylogue-uqwd evidence packet: ChatGPT lifecycle-anchor drift

Date: 2026-08-04. Worktree: `feature/investigate/chatgpt-lifecycle-anchor` at `f7b721ab38538c82021d417faea363ba152c8758`. Scope: resolve whether the current quarantined ChatGPT population still reproduces the `generation_lifecycle` moved-anchor conflict described by polylogue-uqwd, without changing Beads, source.db, index.db, the blob store, backups, daemon state, or services.

## Verdict

The blocker does not reproduce in the current corpus. The full production-route run found zero pairs with exactly one `generation_lifecycle` event on each side, different anchors, equal non-anchor lifecycle content, equal message content, equal attachment content, and an event-axis conflict. No comparison-layer exception is justified by this snapshot.

The historical semantic risk remains real in principle and is preserved by the red-twin regression in `tests/unit/archive/test_session_revision_membership.py`. That test keeps a moved-anchor pair with different lifecycle state as a conflict. The existing parser fix `b1e01d878` already makes the ChatGPT generation-timing anchor deterministic, which is the upstream failure mode described by the bead.

## Exact proof route

The run used SQLite `mode=ro` connections to the current `source.db` and resolved `index.db`, read immutable blob files from the content-addressed blob store, then called the production functions directly in this order:

```text
polylogue.sources.revision_backfill._parse_one(Provider.CHATGPT, payload, source_path, ...)
polylogue.pipeline.ids.session_revision_projection(session)
polylogue.archive.session_revision_membership._relation(left, right)
polylogue.archive.session_revision_membership.classify_membership_revisions(revisions, existing_accepted_raw_id=head)
```

The population was selected from `raw_sessions` with `origin='chatgpt-export'` and `revision_authority='quarantined'`. Cohorts were taken from persisted `raw_session_memberships.logical_source_key`, not reconstructed by SQL content heuristics. Singleton logical keys were retained in the population count and excluded only from pairwise comparison because a singleton cannot conflict. Current `raw_revision_heads` were read from the resolved index generation and passed to the classifier as `existing_accepted_raw_id`; no writeback or replay apply function was called.

The exact population query was:

```sql
SELECT raw_id, native_id, source_path, source_index,
       lower(hex(blob_hash)) AS blob_hash, blob_size,
       revision_kind, logical_source_key, revision_authority
FROM raw_sessions
WHERE origin = 'chatgpt-export' AND revision_authority = 'quarantined'
ORDER BY raw_id;
```

Membership keys came from:

```sql
SELECT m.raw_id, m.logical_source_key
FROM raw_session_memberships AS m
JOIN raw_sessions AS r ON r.raw_id = m.raw_id
WHERE r.origin = 'chatgpt-export' AND r.revision_authority = 'quarantined'
ORDER BY m.logical_source_key, m.raw_id;
```

The transient full cohort report was `/realm/tmp/work/polylogue-uqwd-real-classifier-20260804.json`, SHA-256 `a00d15d0bcf49e6eee18613d419536a2f7217bc06b4de95bd7c6a479e629a49c`. The tracked packet below records the report's identity and decision-bearing results.

## Corpus identity and coverage

| Measure | Result |
| --- | ---: |
| Quarantined ChatGPT rows | 7,498 |
| Membership rows | 7,498 |
| Logical keys | 2,570 |
| Singleton logical keys | 15 |
| Multi-candidate logical keys | 2,555 |
| Rows in multi-candidate keys | 7,483 |
| Distinct non-null native ids | 2,464 |
| Rows with null native id | 5,021 |
| Candidate raws expected and parsed | 7,483 / 7,483 |
| Sessions returned and projected | 7,483 / 7,483 |
| Missing blobs | 0 |
| Parse errors | 0 |
| Parsed-key mismatches | 0 |
| Cohorts classified | 2,555 |

Source identity at the run boundary: `/realm/db/polylogue/source.db`, resolved to the same path, device 63, inode 1515046, size 1,891,467,264 bytes, mtime ns 1785817591883760785, `user_version=24`, `schema_version=157`, WAL mode.

Index identity at the run boundary: `/realm/db/polylogue/index.db` resolved to `/realm/db/polylogue/.index-generations/gen-1785377665711-06297b00/index.db`, device 63, inode 3789735, size 40,554,500,096 bytes, mtime ns 1785748371749663866, `user_version=46`, `schema_version=309`, WAL mode.

## Classifier result

Across all pairwise comparisons in the 2,555 multi-candidate cohorts, the production relation counts were `equal=7,348`, `a_contains_b=126`, `b_contains_a=82`, and `conflict=71`. Those 71 conflict pairs were distributed across 18 cohorts. The 18 conflict cohorts were not lifecycle-anchor drift: their shapes were message conflicts, attachment conflicts, event growth with missing lifecycle events, or many lifecycle events on both sides. The detailed shape census contained zero pairs with one lifecycle event on each side and moved anchor plus equal non-anchor content.

The classifier returned an accepted raw for 2,554 cohorts, an ambiguous raw set for 7 cohorts, and equivalent raws for 2,538 cohorts. These are read-only simulated verdicts. The durable `raw_session_memberships.decision` values were not changed.

As a cross-check of the narrower grouping used by the earlier evidence note, the current non-null `native_id` population has 5 multi-row cohorts and 18 rows. The same production parser, projection, and relation calls found `equal=8`, `a_contains_b=14`, `b_contains_a=5`, `conflict=0`, and lifecycle-anchor candidates `0`.

## Historical cohorts named by the bead

The premise that the named cohorts disappeared is false in the current snapshot. A read-only fragment query finds all three native ids in `raw_sessions`, and a membership query finds three rows for each logical key:

```sql
SELECT m.raw_id, m.logical_source_key, m.provider_session_id,
       m.decision, r.native_id, r.source_path, r.source_index
FROM raw_session_memberships AS m
JOIN raw_sessions AS r ON r.raw_id = m.raw_id
WHERE m.logical_source_key LIKE '%687f4424%'
   OR m.logical_source_key LIKE '%6898a012%'
   OR m.logical_source_key LIKE '%689b90d9%'
ORDER BY m.logical_source_key, m.raw_id;
```

Current real-parser outcomes for those three cohorts are one accepted raw plus two equivalent raws each, with no pairwise conflict:

| Logical key fragment | Current parsed shape | Lifecycle-anchor result | Classifier result |
| --- | --- | --- | --- |
| `687f4424` | 46 messages, 1 lifecycle event, 3 revisions | all three anchor `986a3a7e-6e2b-4fba-851f-ab308fe52fda`; identical event-content fingerprint | 1 accepted, 2 equivalent |
| `6898a012` | 1,372 messages, 60 lifecycle events, 3 revisions | all three revisions have the same 60-event content fingerprint; not the one-event target shape | 1 accepted, 2 equivalent |
| `689b90d9` | 19 messages, 1 lifecycle event, 3 revisions | all three anchor `7c7fad67-9702-4a8b-8538-489a7b345e5a`; identical event-content fingerprint | 1 accepted, 2 equivalent |

The current data therefore shows the original conflict has disappeared through stable parser output, not through a comparison exception. Commit `b1e01d878` changed `_extract_generation_timings` to use the candidate message id instead of mapping iteration position as the final tiebreak and added a parser regression for mapping-order stability. That is the recorded upstream explanation for why the moved anchor no longer changes across equivalent exports. The exact reason the 2026-08-03 read-only note failed to locate these rows cannot be established from this snapshot alone. The current evidence supersedes that negative lookup: the rows are now present and clean under the current parser.

## Red-twin safeguard

The added test constructs two ChatGPT-shaped revisions with one lifecycle event each, different anchors, and different `state` content. It calls `session_revision_projection`, `_relation`, and `classify_membership_revisions` with an existing head. It asserts equal message content, different event content, a `conflict` relation, no accepted replacement, and both raws remaining ambiguous. This is the minimal conditional regression needed if a future change revisits the proposed exactly-one-orphan comparison exception.

## Acceptance match

| Acceptance criterion | Status | Evidence |
| --- | --- | --- |
| Run the real classifier and projection path against current quarantined ChatGPT data | Satisfied | 7,483/7,483 candidate raws parsed and projected through production functions; 0 missing blobs, parse errors, or key mismatches |
| Confirm moved lifecycle anchors still produce conflicts | Not reproduced | 0 target-shaped pairs across 2,555 logical-key cohorts; named historical cohorts are clean |
| Implement the narrow comparison exception if reproduced | Not applicable | No production code change made |
| Preserve different-content moved-anchor behavior | Satisfied | Red-twin production-route regression added |
| Reclassify the current blocker edge honestly | Recommended | Current measurable blocker claim is unsupported; retain the historical semantic-risk note until a pre-fix replay artifact exists |

## Graph disposition and residual uncertainty

The `xselt/818fy` blocker edge is not justified as a current reindex blocker by this corpus. It is reasonable to retain it only as a historical semantic-risk reference until the graph owner reclassifies it. Objective evidence for removing the blocker edge is: the full current production-route census remains zero for the target predicate, the three named cohorts remain accepted/equivalent with stable anchors, the red twin remains green, and an archived pre-`b1e01d878` replay or fixture demonstrates the old moved-anchor conflict and the current parser's correction. The current packet does not fabricate that pre-fix replay because historical parser execution and backup access were outside this lane.

Residual uncertainty is limited to provenance of the archive transition between the 2026-08-03 negative lookup and this 2026-08-04 snapshot, and the absence of an archived pre-fix parser output for the original 10/136 sample. The live durable membership decisions remain unchanged because this lane performed no writeback.
