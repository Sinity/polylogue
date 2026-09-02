Summary

Keep NULL action outcomes unknown when lowering `is_error` and `exit_code` predicates.

Problem

Action predicates used `COALESCE(..., 0)`, causing unknown provider outcomes to match known false or zero-value predicates. The regression fixture reproduced this by returning unknown and no-result actions for `is_error:false` and `exit_code:>=0`.

Solution

Lower action `is_error` predicates with direct comparisons and lower numeric `exit_code` predicates against the nullable column. Added three-state regression assertions covering unknown, zero, and one values.

Verification

`uv run devtools test tests/unit/storage/test_archive_tiers_archive.py::test_archive_facade_exposes_distinct_action_result_states` passed after the fix: 1 passed, 1 warning in 42.67s.

`uv run devtools verify --quick` passed after rebasing onto `origin/master`; every reported gate was ok.

The containing storage test file reported 49 passed and 1 inherited failure in stale-schema exception typing, where `SchemaSkewError` is raised while the existing test expects `SchemaVersionMismatchError`.

Residual risk

This change preserves existing query syntax and does not add an explicit unknown-selection predicate. Unknown outcomes remain excluded from known-value comparisons.
