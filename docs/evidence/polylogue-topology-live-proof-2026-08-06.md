# Topology live-proof residue, 2026-08-06

## Scope

This report records the proof surface implemented for `polylogue-topology-live-proof`. The census reuses `devtools workspace lineage-validation` and the production topology write/read seams. The candidate evidence is a frozen test index populated by `write_parsed_session_to_archive`; it is not a claim about the operator's live archive.

## Candidate proof

The candidate fixture contains two resolved links and one unresolved native-parent link, all written through the production writer. The production writer supplies a non-empty method for all three rows. The census derives the ordinary `resolved` and `unresolved` states from `resolved_dst_session_id`, while preserving the nullable raw `status` column contract. The bounded unresolved-parent read sample exercises `read_archive_session_envelope` and proves the child remains child-local: no parent session is composed, and the served message count equals the child-owned count. Each receipt binds the report to the database and any SQLite sidecars by content digest, file identity, and a held read transaction. With a fixed capture time, an unchanged source reproduces the receipt, while a source mutation changes its binding.

| Evidence | Result |
| --- | ---: |
| effective topology states | `resolved=2`, `unresolved=1` |
| empty effective states | `0` |
| empty methods | `0` |
| raw nullable status values | `3` ordinary NULLs, reported transparently |
| unresolved-parent reads sampled | `1` |
| unresolved-parent reads safe | `true` |
| cycle-quarantine evidence in candidate | `0` |
| candidate snapshot stable during census | `true` |

The production-route cycle fixture separately proves a `quarantined` closing edge with `cycle_rejected` evidence. Its census has `resolved=1`, `quarantined=1`, zero empty effective states, zero empty methods, one valid cycle-evidence row, and zero malformed quarantine-evidence rows. Mutations that blank a method, introduce malformed quarantine JSON, or give a quarantined row a resolved parent each make the census fail, and the reader leaves the contradictory quarantined row uncomposed.

## Live residue

No live archive was opened or mutated in this lane. The live database path is outside the assigned worktree and is excluded by the repository operating boundary. Therefore this report does not claim live zero-empty counts, archive convergence, or a post-reindex status distribution. The remaining named follow-up is `polylogue-topology-live-proof`: run the read-only census against the approved live or activated candidate index, retain the generated receipt, and compare `effective_status_counts`, `empty_effective_status_count`, `empty_method_count`, `cycle_evidence_count`, and `unresolved_read_sample`.

## Verification

```text
devtools test tests/unit/devtools/test_lineage_validation.py tests/unit/storage/test_topology_cycle_quarantine_live.py
```

The tests include mutations that blank a method, introduce an unknown status, make an unresolved child claim a parent in `sessions.parent_session_id`, introduce malformed quarantine JSON, and make a quarantined row resolve a parent; each mutation makes the relevant proof fail. The live receipt step was not run, so the live census remains explicitly not observed.
