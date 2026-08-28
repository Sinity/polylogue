Summary

Remove the pre-adoption Beads issue session route and its public origin/provider declarations, parser, detector bindings, watcher/config plumbing, UI/generated projections, fixtures, and tests. Preserve Beads interaction shape classification as a non-session artifact and retain the work-evidence adapters and `beads-issue` evidence reference kind.

Problem

Beads interaction ledgers were admitted as synthetic sessions even though they are repository work evidence. This created protocol-only user messages and exposed a public origin with no live session population.

Solution

Delete the session admission route without changing durable migrations or live archive data. Shape-matching interaction records now classify as `Provider.UNKNOWN` and `ArtifactKind.UNKNOWN` with `parse_as_session=False`. Durable schema narrowing remains owned by the census and migration successors.

Verification

- `bash nix/devtools-wrapper.sh test tests/unit/sources/test_origin_specs.py tests/unit/sources/test_artifact_taxonomy.py::test_beads_interaction_artifact_is_refused_as_session tests/unit/insights/test_work_effects.py tests/unit/core/test_sources.py tests/unit/cli/test_command_aux_runtime.py`: 88 passed.
- `python -m compileall -q polylogue devtools tests/unit/sources tests/unit/core tests/unit/insights/test_work_effects.py`: passed.
- `bash nix/devtools-wrapper.sh render all --check`: passed; all generated surfaces synchronized.
- `PATH="$PWD/.venv/bin:$PATH" bash nix/devtools-wrapper.sh verify --quick`: passed all checks.
- The requested broad selector was attempted and interrupted after expanding to 8,003 tests; its partial summary was 220 passed, 14 failed, and 82 setup errors in environment-sensitive archive/CLI paths. It is not represented as a green suite claim.

Residual risk

Existing durable archives may still carry historical Beads-origin schema vocabulary. No durable mutation was performed. The named census and migration successors must verify archive state before any schema narrowing or data transformation.
