# Artifact publication contract

The cache and reusable-fixture routes use the same filesystem contract:

| Caller | Shared primitive | Authenticated object | Consumer binding |
| --- | --- | --- | --- |
| `tests.infra.workload_artifacts.build_seeded_archive` | `_publish_sealed_staging` + pinned tree validation | manifest, closed file set, SHA-256 content, read-only modes | `SeededArchiveQueryLease` or clone integrity FD |
| `tests.infra.workload_artifacts.build_immutable_tree` | `_publish_sealed_staging` + pinned tree validation | protocol/key manifest, closed file set, SHA-256 content, read-only modes | `clone_immutable_tree` reauthenticates after fast/fallback copy |
| `tests.infra.archive_templates.clone_archive_template` | fixture-owned SQLite quiescence and clone validation | SQLite snapshot plus detached clone file set | private writable clone; bootstrap identity is rebound |
| `tests.infra.whale_fixtures.clone_blob_tree` | `clone_archive_template` | same as archive-template clone | private writable clone |
| `devtools.clone_support.reflink_clone` | authenticated single-file fast/fallback helper | regular single-linked file, sidecar-free, size and SHA-256 | caller receives only after post-copy identity check |

The no-promote canary route intentionally has different semantics. It does
not clone a reusable package: `IndexGenerationStore.create` creates an
inactive generation whose durable tiers are identity-checked links, and
`IndexGenerationStore.promote` is the only active-pointer transition. Its
consumer is the lifecycle owner and generation metadata, not a fixture clone
lease. The canary therefore retains its generation-specific ownership and
promotion receipts while sharing the same fail-closed rules for symlinked
ancestors, replacement, and authenticated file access.

Every fast path is an optimization only. A failed reflink is discarded before
copy fallback, and the fallback is checked for the same authenticated file set
and content identity before the caller can use it.
