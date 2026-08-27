Summary

Reject catalog pricing for paid provider-usage rows when cached-token rates are absent, and apply the same rule to the provider usage projection. Genuine free models remain valid zero-cost rows.

Problem

The provider rollup used `estimate_cost` even when a paid model had cached tokens and a zero cache rate, making the result look fully priced while silently assigning that lane a zero price. The projection already detected the condition, but the write path did not, and it treated free zero-rate models as incomplete.

Solution

The provider rollup now checks the resolved catalog entry before persisting `priced`/`cost_usd`; paid models with missing cache-read or cache-write rates return no pricing evidence. The event projection applies the paid-model guard, preserving free-model zero pricing. Regression tests cover the writer and projection paths.

Verification

`nix develop --command devtools test tests/unit/storage/test_usage_projection.py tests/unit/core/test_cost_compute.py tests/unit/storage/test_cost_queries.py tests/unit/insights/test_cost_basis_split.py tests/unit/cost/test_contract_suite.py` — 49 passed.

`nix develop --command devtools test tests/unit/storage/test_session_usage_reconciliation.py tests/unit/insights/test_cost_basis_split.py tests/unit/cost/test_contract_suite.py tests/unit/storage/test_reindex_derived_model_differential.py` — 31 passed, 1 inherited failure: `candidate_source_membership` lacks differential-harness classification.

`nix develop --command devtools verify --quick` — ruff format, ruff check, and mypy passed; docs-surface failed on the pre-existing duplicate `docs/artifact-publication.md` entry.

Residual risk

The full differential test and quick gate remain red for unrelated repository-baseline issues. No live archive or service state was modified.
