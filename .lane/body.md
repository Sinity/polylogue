Summary

Bound durable excision policy to production acquisition admission. The canonical async acquisition service now builds one immutable snapshot before source traversal and carries it through the pending raw-admission request. The synchronous ArchiveStore governance routes use the same snapshot builder before raw payload, blob-reference, artifact, parsed-session, and hook writes. Schema identity is derived from the current tier versions.

Problem

The prior implementation exposed policy parameters only at low-level writers. Every production caller left them at `None`, so the guard was a no-op and `build_excision_policy_snapshot` was reachable only from a unit test.

Solution

The binding belongs at the acquisition orchestration and governance layers. `AcquisitionService.acquire_sources` captures policy before candidate traversal, `persist_raw_record` carries it into the admission request, and the async executor rejects the content hash before source-row side effects. Shared synchronous governance entrypoints resolve and forward the same policy to the raw-admission chokepoint. The regression invokes `AcquisitionService.acquire_sources` against a real source file and confirms the excised payload creates no `raw_sessions` row.

Verification

- `nix develop --accept-flake-config --command devtools test tests/unit/pipeline/test_acquisition_excision_policy.py tests/unit/security/test_excision_policy.py`: 3 passed.
- `nix develop --accept-flake-config --command devtools test tests/unit/pipeline/test_acquisition_blob_gc_age_gate.py tests/unit/pipeline/test_stage_independence.py tests/unit/storage/test_raw_admission_async_parity.py`: 24 passed; the selected acquisition blob test also hit the inherited fresh-bootstrap `index schema semantic manifest mismatch` for extra table `schema_identity` when run in the combined selection.
- `nix develop --accept-flake-config --command devtools test tests/unit/storage/test_source_items.py tests/unit/storage/test_raw_admission_async_parity.py tests/unit/security/test_excision_policy.py tests/unit/pipeline/test_acquisition_excision_policy.py`: 13 passed; two hook-route tests hit the same inherited schema-manifest mismatch.
- `nix develop --accept-flake-config --command devtools verify --quick`: passed after the route binding was added.
- Rebased onto `origin/master` at `6de9e5d4746b3ea9cb0c02b20b8e21b0850bd4b0`.

Residual risk

The source-generation projection remains lazily created by `publish_source_generation`; its numbered source migration and the broader durable excision transition are outside this route-binding repair. The legacy `source.excised_content` ledger remains unchanged.
