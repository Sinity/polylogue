# Polylogue upgraded Beads setup

Generated: 2026-07-07T00:05:00Z

This package turns the Beads snapshot into a delivery-grade setup. It keeps the original dependency-aware order, but adds release gates, lanes, readiness grades, proof requirements, acceptance-criteria patches, hidden dependency edges, and Beads memories that preserve the operating rules.

## What changed

- Active beads covered: 397.
- Proposed acceptance-criteria patches for beads that had none: 72.
- Proposed hard `blocks` dependency edges: 191.
- Proposed delivery labels: 1263.
- Proposed delivery note append operations: 397.
- Proposed durable Beads memories: 5.

## Release counts

- `A-trust-floor`: 59
- `L-external-legibility`: 29
- `E-variants-preferences`: 12
- `B-storage-rebuild-bytes`: 23
- `H-web-cockpit`: 18
- `D-agent-context-coordination`: 48
- `G-live-performance`: 25
- `F-lineage-compaction`: 12
- `M-substrate-consolidation`: 33
- `J-embeddings-retrieval`: 11
- `I-analytics-experiments`: 30
- `C-read-evidence-contract`: 60
- `K-interop-origin-export`: 31
- `N-horizon`: 6

## Upgraded readiness counts

- `A-implementation-ready`: 170
- `B-local-inspection-needed`: 43
- `D-horizon-ready`: 184

## Lane counts

- `read-contracts`: 60
- `substrate-consolidation`: 33
- `origin-interop-export`: 31
- `analytics-experiments`: 30
- `docs-demos-launch`: 29
- `agent-coordination`: 29
- `evidence-honesty`: 19
- `web-evidence-cockpit`: 18
- `blob-integrity`: 16
- `interactive-performance`: 16
- `context-memory`: 16
- `security-privacy`: 12
- `variants-preferences`: 12
- `lineage-compaction`: 12
- `usage-cost-honesty`: 11
- `storage-rebuild-scale`: 11
- `embeddings-retrieval`: 11
- `capture-reliability`: 6
- `horizon-spec`: 6
- `temporal-provenance`: 5
- `agent-write-safety`: 4
- `operational-resilience`: 3
- `live-substrate`: 3
- `agent-substrate`: 3
- `verification-readiness`: 1

## Files in this package

- `delivery_manifest.csv` and `delivery_manifest.json`: every active bead with release, lane, readiness, proof artifact, and impact flags.
- `patch_manifest.json`: the complete structured upgrade patch.
- `patches/beads_delta_ops.jsonl`: append-label, set-acceptance-if-empty, add-dependency, append-note, and add-memory operations.
- `patches/acceptance_criteria_patches.json`: proposed acceptance criteria for all active beads that lacked them.
- `patches/dependency_edges_to_add.csv`: hard dependency edges to add to Beads.
- `patched/polylogue-beads-upgraded-export.jsonl`: a fully patched export-shaped JSONL snapshot for review or import workflows.
- `scripts/apply_upgrade_to_export.py`: safe patcher that applies this package to a Beads export JSONL.
- `scripts/validate_upgrade_manifest.py`: validator for dangling IDs, missing delivery coverage, and blocker cycles.

## Use model

Use the patched export for review and diffing. For a live Beads database, treat `patch_manifest.json` as the source of truth and apply changes through whatever `bd` mutation commands your installed Beads version supports. The patcher intentionally only sets acceptance criteria when the field is empty and only appends notes/labels/dependencies, so it preserves existing human-authored content.
