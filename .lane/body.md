Summary

Add deterministic configuration-evidence records and exact historical context resolution. Live bytes are SHA-256 addressed; committed git revisions can be enumerated; gaps and overlaps remain explicit. Structural invocations join only unique active declarations, and efficacy comparisons require cohort limits and judgment authority.

Problem

Actor and execution-context refs existed, but configuration artifacts had no reusable evidence model. Consumers could not distinguish an exact historical setup from missing or overlapping revisions without consulting current files.

Solution

Add `polylogue.context.configuration_evidence` with immutable artifact revisions, live-file capture, committed git-history capture, interval resolution, invocation joins, and an honest comparison record. Export the API through `polylogue.context`. Add tests covering content identity, changed contexts, gap/overlap handling, structural joins, required comparison metadata, and committed-versus-uncommitted git bytes.

Verification

- `nix develop --command devtools test tests/unit/context/test_configuration_evidence.py` — 6 passed.
- `nix develop --command devtools verify --quick` — all quick checks passed, including ruff, mypy, render, layering, pattern, CI/doc command, schema, oracle-integrity, reachability, definition-closure, timestamp, insight-honesty, schema-promotion, and schema-privacy checks.

Residual risk

This is the reusable storage-independent core only. It does not yet persist records in source.db, wire the live watcher or archive session rows, ingest the full setup inventory, or materialize Claude Workflow coverage. Those integrations require a separately scoped durable migration and production-route slice.
