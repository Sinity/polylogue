# In-Place Schema Upgrade Test Surface

This directory is the **dedicated verification lane** for in-place
schema upgrade helpers. Per `docs/internals.md` § "Schema Versioning
Model" and `CONTRIBUTING.md` § "Schema-Touching Changes", Polylogue
has **no schema migration chain**: a database opened at any
`user_version` other than the current `SCHEMA_VERSION` is rejected,
and the operator re-ingests from source.

That policy is the steady state. This directory exists so the
policy boundary is enforceable rather than aspirational.

## Policy

A PR that adds an in-place upgrade helper under
`polylogue/storage/sqlite/` (any function whose name matches the
historical migration shape — `build_vN_to_vM`,
`_apply_version_upgrade_plan`, `upgrade_vN_to_vM`, `migrate_vN_*`,
`ensure_schema_upgrades_vN`) must, in the same PR:

1. Add a fixture under `tests/data/schema_samples/` recording the
   exact source-version DB shape the helper claims to upgrade.
2. Add a driving test in this directory that:
   - opens the recorded fixture DB at the source version,
   - applies the helper,
   - asserts the post-upgrade row counts and schema shape match the
     target `SCHEMA_VERSION`,
   - asserts equivalence with a fresh-init DB seeded with the same
     logical content (round-trip).
3. Reference the helper symbol by its declared name in the test source
   text (the verification lane uses literal name matching for
   discovery).
4. Document the upgrade exception in the PR body, including why a
   fresh re-ingest is unacceptable for that specific transition.

## Lint

`devtools verify-schema-upgrade-lane` enforces the discovery contract:

- it scans `polylogue/storage/sqlite/` for migration-shaped helpers,
- if it finds one without a paired test under this directory that
  references it by name, the lint fails.

The lint runs as part of `devtools verify --lab`. It is intentionally
not in the fast default path: the policy boundary is an architectural
concern, not a per-edit gate, and the lint cost is low enough that the
`--lab` cadence is the right venue.

## Steady state

When no in-place upgrade helpers exist (current and intended state),
this directory contains only this README and the verification lane
passes cleanly. The directory must remain in the tree so the lane
surface is discoverable **before**, not after, someone tries to add an
upgrade path.
