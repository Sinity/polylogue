# Daemon

## Runtime ownership

The daemon holds writer/rebuild exclusion for its lifetime. `DaemonWriteCoordinator` serializes publication and retains ownership until a cancelled operation actually terminates. HTTP, UDS and derivation owners share the process compute adapter and writer bridge (`polylogue/daemon/cli.py:1890`; `polylogue/daemon/write_coordinator.py:280`; `polylogue/daemon/convergence.py:127`).

`run_daemon_services` composes FTS, embedding and session-profile callbacks once. FTS runs at startup and periodically; session profiles run after admitted ingest and during the periodic sweep; embeddings use watcher scopes and the periodic backlog owner (`polylogue/daemon/cli.py:2443`; `polylogue/daemon/cli.py:2503`; `polylogue/daemon/cli.py:2540`; `polylogue/daemon/cli.py:2623`). These are source-route facts, not live deployment evidence.

## Domain derivations

The typed kernel validates prerequisite names against the supplied ordered domain list. It pages required and excess keys, inspects authoritative output, computes outside the writer lease, and admits each replacement through the writer bridge. Process-local continuation state is disposable. Reports distinguish pending policy work from failed attempts (`polylogue/daemon/derivation.py:377`; `polylogue/daemon/derivation.py:449`; `polylogue/daemon/convergence.py:127`).

Raw observations use the same owner for admitted raw-to-logical membership. FTS retains canonical triggers, identity membership and the FTS refresh guard; per-session replacement joins exact canonical session membership. Its selected global orphan partition runs at low cadence and streams its binding, but still requires archive-wide scan and transaction work (`polylogue/daemon/raw_observation_owner.py:1`; `polylogue/storage/fts/derivation.py:1`; `polylogue/operations/fts_derivation.py:1`).

Embeddings replace one message reference atomically. Validity requires vector presence, full recipe identity and the exact message semantic hash. Provider work occurs outside publication admission. Attempt and cost receipts remain operation evidence (`polylogue/storage/embeddings/derivation.py:1`; `polylogue/daemon/embedding_owner.py:1`).

Session counters share one thirteen-measure declaration. Canonical writes recompute from stored messages, and the session-summary adapter inspects and replaces the same partition (`polylogue/storage/derived/session/summary.py:1`). Session-profile publication uses its existing domain adapter and shared owner (`polylogue/daemon/session_profile_composition.py:1`; `polylogue/storage/derived/session/derivation.py:1`).

## Remaining stage execution

The generic stage engine remains for optional Sinex publication, raw-authority cache warming, attachment acquisition, Claude workflow, delegation evidence and standing queries. It still has path/session callbacks, barriers and stage state. Removing these requires moving each surviving product responsibility to its owner; the domain adoption does not establish complete stage retirement (`polylogue/daemon/convergence_stages.py:433`; `polylogue/daemon/convergence.py:732`).

`convergence_debt` remains disposable retry state for those surviving stage callers. FTS, embeddings, raw parsing and session profiles no longer use its stage rows as publication authority (`polylogue/daemon/cli.py:1476`; `polylogue/sources/live/convergence_outcome.py:1`).

## Readiness and intake

Readiness derives from domain inspection and is reported separately from operation health. FTS does not consult a freshness ledger, and debt cannot certify insight readiness (`polylogue/daemon/fts_status.py:196`; `polylogue/readiness/claim_guard.py:1`; `polylogue/storage/sqlite/archive_tiers/archive.py:1`).

Fair intake applies a process-local cooldown to repeated retryable failures. A stale cursor refusal remains retryable even when the same batch reports successful files. Terminal refusal isolates only the affected item (`polylogue/daemon/intake.py:182`; `polylogue/operations/intake_adapters.py:1`).

verified: cfcf58476011e7522a47a6df542e9839af8ab7a3 2026-09-13
