Summary

Add an immutable inventory for the six named workload measurement paths and record whether each emits the shared WorkloadReceipt or delegates through an explicit adapter. Fix pipeline-probe process-tree RSS receipt construction so aggregate bytes equal the separately rounded self and child measurements. Make verifier checkout detection tolerant of minimal subprocess test doubles and align its worktree mypy test with the current ephemeral policy.

Problem

The workload receipt contract had no executable inventory of its named consumers. Pipeline-probe could produce a receipt whose aggregate RSS differed from self plus child RSS because it rounded the aggregate MiB value before converting to bytes. The focused verifier tests also failed under the current worktree checkout policy.

Solution

Add `WorkloadAdapterDeclaration` and the six-entry `WORKLOAD_ADAPTER_DECLARATIONS` catalog, exported from `polylogue.scenarios`. Add an anti-vacuity test requiring every named path and both dispositions. Derive pipeline aggregate RSS from component bytes, preserving process scope and unit conversion.

Verification

`uv run devtools test tests/unit/scenarios/test_workload_receipts.py tests/unit/devtools/test_query_memory_budget.py tests/unit/devtools/test_pipeline_probe.py tests/unit/devtools/test_ingest_throughput_probe.py tests/unit/devtools/test_slo_catalog.py tests/unit/devtools/test_verify.py tests/integration/test_append_cohort_memory.py` passed: 81 passed, 1 warning.

`uv run devtools verify --quick` passed: all 17 quick steps reported success, including formatting, ruff, mypy, generated surfaces, layering, command checks, schema checks, oracle integrity, reachability, closure, honesty, and privacy gates.

Residual risk

The catalog records scenario-execution as an explicit delegation because execution specs do not themselves collect physical observations. Host-dependent live incident receipts and the non-perturbation proof remain runtime evidence, not established by this focused lane.
