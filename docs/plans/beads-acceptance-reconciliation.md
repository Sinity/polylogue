# Acceptance contract reconciliation protocol

`devtools lab policy acceptance-contract-reconcile` is a file-level, fail-closed boundary between the canonical repository JSONL and a read-only live Beads export. It never invokes `bd`, never writes Dolt, never parses acceptance prose, and never chooses an authority for a changed source record.

## Lane output

The command compares rows by Bead ID and recomputes the source digest with the merged acceptance-contract validator. The report has separate ID sets for `master_only`, `live_only`, `master_newer`, `live_newer`, `same_timestamp_different`, and `contract_refused`. It also records the contract denominator, refused IDs and reasons, the targeted IDs, and equality digests.

The targeted JSONL contains only guarded records. Each record starts as the live row and changes only `acceptance_criteria` and `metadata.acceptance_contract_v1`. Dependencies, comments, status, notes, timestamps, and every other live field remain equal in the non-contract projection; the equality digest removes only the contract key and an empty metadata container, while retaining every non-contract metadata key. A live source digest must equal the contract's `source_digest`; otherwise that ID is refused and is absent from the wave. Malformed live metadata or timestamps are refused explicitly. Live-newer contract rows are separately deferred and are never put in the wave, even when their source digest matches.

The current supplied snapshots are expected to report three master-only IDs, two live-only IDs, twelve live-newer IDs, ten master-newer IDs, and one contract refusal for `polylogue-7rds` after its committed digest is refreshed. The live-only records `polylogue-5bxpy` and `polylogue-g8v5z`, and all twelve live-newer records, remain outside this wave.

## Coordinator-only apply sequence

The implementation lane stops before this sequence. The coordinator must retain every output and receipt under a named run directory.

1. Confirm the six-PR merge frontier is no longer full and identify the exact repository head and Beads database identity. Take a Dolt backup using the site-approved Beads backup procedure, verify that backup, and retain its manifest or receipt before any import.
2. Acquire the site-approved exclusive Beads/Dolt writer lease, or stop/quiesce every Beads writer for the whole operation. The lease must cover the live export, generator, dry-run, real import, and post-import export. If exclusive writer ownership cannot be established, stop; `--allow-stale` is not a concurrency lock and the after-check is not permission to discover a race after mutation.
3. Capture a read-only live export from the schema-compatible client. The installed client for the current v63 database is `/etc/profiles/per-user/sinity/bin/bd`; the devshell `bd` is v62 and must not be used. Do not run either client from this lane.
4. Run the generator against the committed repository JSONL and that export:

   ```text
   devtools lab policy acceptance-contract-reconcile --repository .beads/issues.jsonl --live RUN/live-before.jsonl --wave RUN/targeted.jsonl --report RUN/reconciliation.json --json
   ```

5. Review the report. Require the contract refusal and deferred denominators and every refused/deferred ID to be named. Adjudicate master-only, live-only, master-newer, live-newer, and same-timestamp-different rows separately. Do not add `polylogue-5bxpy`, `polylogue-g8v5z`, or any live-newer row to the wave. Do not regenerate timestamps. Record `live_equality_digest` as the exact pre-import live population binding.
6. Run the schema-compatible client's dry-run targeted import with `--allow-stale`, using only `RUN/targeted.jsonl`. The dry run must report exactly the generated guarded IDs. Do not use the normal stale-snapshot wrapper, because equal timestamps are intentionally crossed by this explicit, targeted `--allow-stale` operation.
7. After reviewing the dry-run receipt, run the real targeted import with the same client, the same wave, and `--allow-stale`, while still holding the exclusive writer lease. The coordinator must not add flags that permit unrelated rows, candidate-only rows, or source-digest bypasses.
8. Capture a post-import read-only export as `RUN/live-after.jsonl` with the same schema-compatible client. Verify the targeted wave and the unchanged remainder:

   ```text
   devtools lab policy acceptance-contract-reconcile --verify-before RUN/live-before.jsonl --verify-after RUN/live-after.jsonl --verify-wave RUN/targeted.jsonl --json
   ```

   This checks that the targeted records match the wave, that their non-contract equality digest is unchanged, and that no record outside the wave changed.
9. Run the merged contract validator against the post-import repository export and then run the graph-policy check. The graph check is a live-state check and must use the schema-compatible client. A green validator does not adjudicate live-only or live-newer records.
10. Reconcile the repository JSONL from the post-import export only after the live export and graph-policy checks pass. Review the resulting diff for exactly the guarded contract fields. Retain the before export, after export, wave, report, backup receipt, dry-run receipt, real-import receipt, validator output, and graph-policy output together.
11. Compare the equality digest in the post-import report with the pre-import report and require the post-import record universe to be identical. A mismatch blocks publication and requires restoring from the retained backup or reopening the reconciliation. No direct SQL, hand-edited JSONL, ordinary `bd` invocation, or second wave is permitted. Release the writer lease only after all receipts are durable.

## Residual boundary

This branch refreshes only `polylogue-7rds`'s source digest after checking its closed implementation in `1cb33c351` and its explicit `polylogue-93xe` verification residual. It does not absorb `polylogue-5bxpy`, `polylogue-g8v5z`, or any live-newer record. Those records require post-merge adjudication across the six-PR train. The lane performs no Dolt backup, Beads export, dry-run import, real import, graph-policy check, or live mutation.
