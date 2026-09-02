Summary

Make CorpusArtifactManifest the sole manifest type for deterministic workload artifacts and remove the retired SeededArchiveManifest compatibility alias. The existing shared WorkloadEnvelopeSpec and WorkloadReceipt contract remains the common carrier for workload identity, phase observations, physical budgets, canaries, and explicit measurement-path dispositions.

Problem

The artifact substrate still exposed a second manifest name after CorpusArtifactManifest became canonical. That preserved a duplicate route and allowed callers to continue depending on the retired name. The packet also requires honest measurement-unavailable handling and anti-vacuity coverage across the shared receipt paths.

Solution

Update manifest readers, validators, builders, clone authentication, garbage-collection helpers, exports, and documentation strings to use CorpusArtifactManifest directly. No semantic expected result is carried by the manifest. The shared receipt implementation and named MCP exact-session and watcher append/cohort canaries are retained and exercised by the focused tests.

Verification

`uv run devtools test tests/unit/scenarios/test_workload_receipts.py tests/unit/infra/test_workload_artifacts.py` passed: 99 passed in 1109.24s.

`uv run devtools test tests/unit/devtools -k workload` passed: 3 passed, 960 deselected, 1 warning in 39.70s.

`uv run devtools test tests/unit -k envelope` ran 391 tests: 385 passed, 5 failed, and 1 error. The failures are inherited fixture/schema or generated-web-content issues outside this diff: query execution fixture tool-outcome evidence, deployment smoke schema version, coordination archive fixtures, and selection-shell asset text.

`uv run devtools verify --quick` passed after rebasing onto origin/master. All 17 gates reported ok, including oracle-integrity, consumer-reachability, and schema promotion/privacy checks.

Residual risk

The broad envelope selector retains six unrelated inherited failures. This lane does not claim live MCP or daemon incident measurements; those remain host-dependent observations. The implementation still has a hand-maintained measurement-path declaration inventory rather than kernel-derived declarations.
