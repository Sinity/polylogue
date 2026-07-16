# Upgrade summary

The upgraded Beads setup converts the active backlog into a gated delivery system. It does not discard the 397-step execution order; it overlays release gates and implementation rules onto it.

- Active beads classified: 397
- Acceptance criteria filled where empty: 72
- Hard blocker edges proposed: 191
- Delivery labels proposed: 1263
- Delivery notes proposed: 397
- Durable memories proposed: 5

The most important upgrade is the explicit dependency spine: public claims wait for evidence honesty and citations; agent context waits for safe write policy and the scheduler; browser posting waits for daemon security and blob integrity; analytics wait for measure registry and evidence semantics; interop waits for OriginSpec; federation waits for idempotent export/import.

The patched export lives at `patched/polylogue-beads-upgraded-export.jsonl`. The authoritative delta lives at `patch_manifest.json`.
