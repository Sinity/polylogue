Summary

- Reopen tier prototypes read-only and publish them with same-directory atomic replacement.
- Add a six-tier regression proving OPS and embeddings receive the same prototype reuse as the other tiers.

Problem

Prototype reuse existed for OPS and embeddings but the cached SQLite files were opened read-write and publication was not atomic, leaving the shared cache vulnerable to accidental mutation or partial publication under concurrent workers.

Solution

Prototype snapshots are finalized without write permissions, reopened through SQLite read-only URIs, and published from a staging file with `os.replace`. The regression initializes two independent databases for every tier and requires one fresh DDL and one prototype hit per tier.

Verification

- `nix develop --command devtools test tests/unit/storage/test_archive_tier_init.py -k 'prototype or embeddings or ops or tier_init'` — 13 passed.
- `git diff --check` — passed.
- `nix develop --command devtools test tests/unit/infra -k 'fixture or archive or workspace'` — interrupted after a long run with failures; not green.

Residual risk

The complete-corpus before/after receipts and the 16-worker pressure reproduction were not available in this lane. The change is limited to prototype publication and focused coverage.
