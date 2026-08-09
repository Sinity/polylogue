# Acceptance contract reconciliation protocol

`devtools lab policy acceptance-contract-reconcile` is a file-level, fail-closed boundary between the canonical repository JSONL and a read-only live Beads export. It never invokes `bd`, never writes Dolt, never parses acceptance prose, and never chooses an authority for a changed source record.

## Lane output

The command first loads the ratcheted 218-ID manifest inside `reconcile`; every manifest ID must exist in the canonical repository JSONL and carry a valid contract before any wave is generated. The fixed `contract_denominator` is therefore always 218, never the count of contracts discovered in a partial input. It then compares rows by Bead ID and recomputes the source and dependency digests with the merged acceptance-contract validator. Reconciliation is blocked when any canonical contract is invalid or stale. The reconciliation report has separate ID sets for `master_only`, `live_only`, `master_newer`, `live_newer`, `same_timestamp_different`, and `contract_refused`. It also records the fixed contract denominator, manifest digest, full canonical and live population digests, refused IDs and reasons, the ordered targeted IDs, per-row wave digests, and the exact guarded-wave digest.

The targeted JSONL contains only guarded records. Each record starts as the live row and changes only `acceptance_criteria` and `metadata.acceptance_contract_v1`. Dependencies, comments, status, notes, timestamps, and every other live field remain equal in the non-contract projection; the equality digest removes only the contract key and an empty metadata container, while retaining every non-contract metadata key. A live source digest must equal the contract's `source_digest`; otherwise that ID is refused and is absent from the wave. Both `updated_at` values are parsed as canonical Beads RFC3339 timestamps before classification; arbitrary strings never authorize a wave. Malformed live metadata or timestamps are refused explicitly. Live-newer contract rows are separately deferred and are never put in the wave, even when their source digest matches.

Acceptance contracts carry route authority as a structured named `route_spec.identifier` plus a type-compatible dispatch class. The identifier must resolve in `docs/plans/beads-acceptance-route-registry.json`, bind to the same Bead and target list, and agree with the registered contract class. Evidence completeness is not inferred from prose or a self-attested `complete` flag: every evidence item requires an exact typed `evidence_spans` carrier with the UTF-8 source snapshot, its SHA-256 digest, a half-open byte range, and the SHA-256 digest of the exact range text. The range must decode and equal the evidence item; extra or missing carrier fields, truncated snapshots, and digest mismatches are rejected.

The exact-head self-reconciliation against the committed canonical snapshot is an idempotent no-op: all 218 canonical records are already guarded, with no targeted rows, refusals, or deferred rows. A real live export is still required for an operational import; this local check does not invent live authority or adjudicate live-only and timestamp-conflict rows.

## Coordinator-only apply sequence

The implementation lane stops before this sequence. The coordinator must retain every output and receipt under a named run directory.

1. Confirm the six-PR merge frontier is no longer full and identify the exact repository head and Beads database identity. Take a Dolt backup using the site-approved Beads backup procedure, verify that backup, and retain its manifest or receipt before any import.
2. Acquire the site-approved exclusive Beads/Dolt writer lease, or stop/quiesce every Beads writer for the whole operation. The lease must cover the live export, generator, dry-run, real import, and post-import export. If exclusive writer ownership cannot be established, stop; `--allow-stale` is not a concurrency lock and the after-check is not permission to discover a race after mutation.
3. Capture a read-only live export from the schema-compatible client. The installed client for the current v63 database is `/etc/profiles/per-user/sinity/bin/bd`; the devshell `bd` is v62 and must not be used. Do not run either client from this lane.
4. Run the generator against the committed repository JSONL and that export:

   ```text
   devtools lab policy acceptance-contract-reconcile --repository .beads/issues.jsonl --live RUN/live-before.jsonl --wave RUN/targeted.jsonl --report RUN/reconciliation.json --json
   ```

5. Review the report. Require the contract refusal and deferred denominators and every refused/deferred ID to be named. Adjudicate master-only, live-only, master-newer, live-newer, and same-timestamp-different rows separately. Do not add `polylogue-5bxpy`, `polylogue-g8v5z`, or any live-newer row to the wave. Do not regenerate timestamps. Record `live_population_digest` as the exact pre-import live population binding.
6. Run the schema-compatible client's dry-run targeted import with `--allow-stale`, using only `RUN/targeted.jsonl`. The dry run must report exactly the generated guarded IDs. Do not use the normal stale-snapshot wrapper, because equal timestamps are intentionally crossed by this explicit, targeted `--allow-stale` operation.
7. After reviewing the dry-run receipt, run the real targeted import with the same client, the same wave, and `--allow-stale`, while still holding the exclusive writer lease. The coordinator must not add flags that permit unrelated rows, candidate-only rows, or source-digest bypasses.
8. Capture a post-import read-only export as `RUN/live-after.jsonl` with the same schema-compatible client. Verify the exact report, ordered wave, every wave row against the canonical source, and the unchanged remainder:

   ```text
   devtools lab policy acceptance-contract-reconcile --verify-repository .beads/issues.jsonl --verify-report RUN/reconciliation.json --verify-before RUN/live-before.jsonl --verify-after RUN/live-after.jsonl --verify-wave RUN/targeted.jsonl --json
   ```

   This checks that the report and wave digests match, the wave order is unchanged, every wave row still equals the canonical guarded row, the full before and after population digests are recorded, and no record outside the wave changed.
9. Run the merged contract validator against the post-import repository export and then run the graph-policy check. The graph check is a live-state check and must use the schema-compatible client. A green validator does not adjudicate live-only or live-newer records.
10. Reconcile the repository JSONL from the post-import export only after the live export and graph-policy checks pass. Review the resulting diff for exactly the guarded contract fields. Retain the before export, after export, wave, report, backup receipt, dry-run receipt, real-import receipt, validator output, and graph-policy output together.
11. Compare the full before and after population digests in the post-import receipt with the exact report and require the post-import record universe to be identical. A mismatch blocks publication and requires restoring from the retained backup or reopening the reconciliation. No direct SQL, hand-edited JSONL, ordinary `bd` invocation, or second wave is permitted. Release the writer lease only after all receipts are durable.

The local guarded actuator is `devtools lab policy acceptance-contract-apply`. It consumes the exact canonical repository, before export, reconciliation report, and wave, writes only a file copy, refuses stale or altered inputs, and treats an identical existing output as an idempotent reimport. It never invokes `bd`.

## Residual boundary

This branch regenerates and validates all 218 canonical manifest records against the exact route registry and typed carrier shape. It does not perform a live Beads export, Dolt backup, dry-run import, real import, graph-policy check, or live mutation. The old source bundle and applier are operationally incompatible because their source and route digests predate this exact-head authority; the committed registry, regenerated JSONL, reconciliation report, and guarded file applier supersede them.
