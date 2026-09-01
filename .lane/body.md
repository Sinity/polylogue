Summary

Reconstructs legacy append frontiers from retained source bytes after ops.db loss and uses transactionally consistent logical SQLite revisions.

Problem

Pre-offset append rows retained durable payloads without usable byte windows, so deleting disposable cursor state forced full capture. SQLite filesystem and page fingerprints changed for logically unchanged databases, and the logical digest could observe mixed WAL commits.

Solution

The planner admits legacy append windows only when the current source bytes and retained blobs prove a contiguous chain from a unique full predecessor. It recomputes offsets and revisions, including multi-append chains, and rejects replacement, truncation, and ambiguity. SQLite logical revisions run in one read transaction, and the watcher uses that identity.

Verification

- `nix develop --accept-flake-config --command devtools test tests/unit/sources/test_live_append_cursor_resynthesis.py tests/unit/sources/test_source_snapshot.py tests/unit/sources/parsers/test_codex_state.py tests/unit/storage/test_revision_replay.py`: 94 passed.
- `nix develop --accept-flake-config --command devtools test tests/unit/sources/test_live_watcher.py -k hermes_wal_revision_triggers_resnapshot`: 1 passed.
- `agentctl job start polylogue verify_quick --workspace worktree-5d257574066ef583`: passed all 18 quick gates at `cec1b37537b3c749fdabc1adfc8a15b8c96e7f5b`.
- `nix develop --accept-flake-config --command devtools verify`: refused because the compatible native testmon graph is absent.

Residual risk

Plain affected verification remains unmeasured until a compatible native testmon graph exists. The full corpus and live archive were not run.
