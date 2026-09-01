Summary

Triaged all 45 findings owned by the storage and devtools lane. Fixed 20 confirmed-small findings in six commits, classified one as already fixed, and filed 24 follow-up beads for confirmed-large findings. The live index census was read-only.

Problem

The deferred review packet covered generated-client contracts, verification state, cost provenance, message identity, tool outcomes, and schema compatibility. Fail-closed tool-outcome findings were checked against the live archive before changing admission behavior.

Solution

The branch preserves generated operation IDs, complete verification batches and failure tracing, embedding compatibility, provider cost provenance, message identity, repository tool outcomes, parser-declared unknown outcomes, topology evidence, and append-write cost/outcome consistency. The remaining schema, bootstrap, readiness, migration, query-plan, and topology-identity work is represented by beads below.

Verification

- `nix develop --accept-flake-config --command devtools test tests/unit/devtools/test_verify.py tests/unit/devtools/test_mypy_daemon_lifetime.py tests/unit/devtools/test_render_openapi.py tests/unit/storage/test_embedding_request_identity.py tests/unit/storage/test_source_items.py tests/unit/storage/test_cost_queries.py tests/unit/storage/test_origin_usage_report.py tests/unit/storage/test_delegations_view.py tests/unit/storage/test_tool_outcome.py tests/unit/storage/test_topology_hook_authority_conflict.py tests/unit/storage/test_unread_wire_batch_v46.py` passed: 140 passed.
- `nix develop --accept-flake-config --command devtools test tests/unit/storage/test_index_fast_forward_lifecycle.py tests/unit/storage/test_schema_policy_contracts.py tests/unit/storage/test_index_fast_forward_executor.py` returned 41 passed and 5 failed. The same command in a detached clean `origin/master` checkout returned 41 passed and 5 failed, establishing those failures as inherited schema-identity and historical-shape fixtures covered by the large findings.
- `nix develop --accept-flake-config --command devtools verify --quick` passed: ruff, mypy, generated surfaces, layering, patterns, CI commands, JavaScript tests, documentation commands, schema round-trip/versioning, oracle integrity, consumer reachability, definition closure, timestamp doctrine, insight honesty, and schema promotion audits all passed.
- Read-only live archive queries found 1,858,390 tool-result blocks: ChatGPT 23,050 and Hermes 17,337. Among those, `is_error` was NULL for 11 ChatGPT and 15,107 Hermes rows. These counts confirm that the parser-shaped unknown path is production data.
- Read-only source census found 12,146 supported Claude Code artifacts, 5,287 recognized-unparsed Claude Code artifacts, 435 supported AI Studio artifacts, and 1 supported Codex artifact.

Disposition table

| Finding | Class | Evidence |
| --- | --- | --- |
| 3892956871 | CONFIRMED, small | Fixed in 2f2195234: generated OpenAPI rendering preserves an explicit operationId with `setdefault`, with a regression test. |
| 3892956887 | CONFIRMED, large | Filed as polylogue-30xdb. Generated transport and 304 result modeling require cross-surface design. |
| 3892048306 | CONFIRMED, small | Fixed in 2f2195234: dmypy contract tests assert the timeout-bearing command. |
| 3893813737 | CONFIRMED, small | Fixed in 2f2195234: full verification uses bootstrap tracing to heal resumable testmon state. |
| 3893813747 | CONFIRMED, small | Fixed in 2f2195234: full-corpus batching is independent of testmon selection mode. |
| 3893813753 | CONFIRMED, small | Fixed in 2f2195234: complete runs retain testmon failure-state updates. |
| 3895663234 | CONFIRMED, small | Fixed in e6ab07362: embedding compatibility remains `archive-index-v79`, with an identity regression test. |
| 3902807506 | CONFIRMED, small | Fixed in e6ab07362: repository block projection selects `tool_outcome`. |
| 3897719980 | CONFIRMED, small | Fixed in e6ab07362: session-level reported cost supplies `origin_reported` provenance when model columns are absent. |
| 3897719967 | CONFIRMED, small | Fixed in e6ab07362: current-schema fixture writers use split cost columns. |
| 3893368128 | CONFIRMED, large | Filed as polylogue-3af3o. Runtime-index repair must precede manifest validation in bootstrap. |
| 3893368147 | CONFIRMED, large | Filed as polylogue-vdedm. Fast-forward bootstrap needs post-plan semantic validation. |
| 3894403065 | CONFIRMED, large | Filed as polylogue-175sz. Unstamped and older tiers need lifecycle adoption before identity enforcement. |
| 3894127160 | CONFIRMED, large | Filed as polylogue-6qpng. The production three-key ordering needs a derived-index lifecycle change. |
| 3897719973 | CONFIRMED, small | Fixed in e6ab07362: provider cost makes child cost non-estimated even without catalog pricing. |
| 3899029284 | CONFIRMED, large | Filed as polylogue-10yj3. Identity stamping must be atomic with the final declaration transaction. |
| 3893445740 | CONFIRMED, small | Fixed in e6ab07362: origin totals prefer provider-reported session cost. |
| 3894403051 | CONFIRMED, large | Filed as polylogue-7q7po. Canonical DDL and schema identity metadata need one bootstrap contract. |
| 3894403083 | CONFIRMED, large | Filed as polylogue-yijb2. Identity needs semantic rather than raw SQL hashing. |
| 3898037586 | CONFIRMED, large | Filed as polylogue-ugpay. Existing durable source archives need a numbered migration. |
| 3898037597 | CONFIRMED, small | Fixed in e6ab07362: source generation validates nullable origins before durable insertion. |
| 3893413406 | ALREADY-FIXED | Current origin/master selects `identity_source` in `_fetch_message_window` at `polylogue/storage/sqlite/archive_tiers/write.py:1717`, introduced by 7440d0881. |
| 3893413418 | CONFIRMED, small | Fixed in e6ab07362: full archive projections pass selected `identity_source` through the envelope. |
| 3896976359 | CONFIRMED, small | Fixed in e6ab07362: hook-only authoritative parent evidence classifies the session as SUBAGENT. |
| 3896976376 | CONFIRMED, large | Filed as polylogue-giuu2. Structural topology fields are omitted from content identity and need admission design. |
| 3897719957 | CONFIRMED, small | Fixed in e6ab07362: append writes clear stale provider totals before assigning the incoming session total. |
| 3902807478 | CONFIRMED, small | Fixed in e6ab07362: established parser-shaped unknown results are normalized to `not_reported`. |
| 3902807487 | CONFIRMED, small | Fixed in e6ab07362: the same ChatGPT and Hermes unknown path is admitted before refusal. |
| 3902807494 | CONFIRMED, small | Fixed in e6ab07362: append outcome derivation reconciles stored and incoming tool evidence. |
| 3902807497 | CONFIRMED, small | Fixed in e6ab07362: canonical and legacy outcome fields are synchronized atomically. |
| 3897038968 | CONFIRMED, large | Filed as polylogue-7kaw4. SchemaSkew handling crosses connection profiles, readiness, and public diagnostics. |
| 3897038982 | CONFIRMED, large | Filed as polylogue-y51wz. Tier ownership needs an explicit archive-root or tier contract. |
| 3897038986 | CONFIRMED, large | Filed as polylogue-oazf3. Attached sibling tiers need coordinated version validation. |
| 3897038994 | CONFIRMED, large | Filed as polylogue-bcgzx. Durable and derived tiers require distinct recovery remedies. |
| 3891721442 | CONFIRMED, large | Filed as polylogue-6bpip. Version-zero handling must distinguish writable bootstrap from read-only opens. |
| 3891721444 | CONFIRMED, large | Filed as polylogue-t1x4j. Schema refusal actions must derive from lifecycle plans. |
| 3893368136 | CONFIRMED, large | Filed as polylogue-o0j93. Readable opens need non-mutating semantic manifest validation. |
| 3894403074 | CONFIRMED, large | Filed as polylogue-6dxfd. Read-only opens need typed schema-identity enforcement. |
| 3898827738 | CONFIRMED, large | Filed as polylogue-3a0wi. Read validation needs the complete required schema contract. |
| 3898827741 | CONFIRMED, large | Filed as polylogue-gnaqw. Transient SQLite failures need separate typed handling from shape drift. |
| 3893368118 | CONFIRMED, large | Filed as polylogue-1qkf6. Same-version trigger variants need lifecycle compatibility rules. |
| 3893368144 | CONFIRMED, large | Filed as polylogue-19m8s. SQL normalization must preserve quoted literal case. |
| 3893368155 | CONFIRMED, large | Filed as polylogue-qkd19. Manifest caching needs explicit identity and invalidation semantics. |
| 3893368161 | CONFIRMED, large | Filed as polylogue-r9ttp. Manifest drift needs a typed schema-layout refusal. |
| 3897719963 | CONFIRMED, small | Fixed in e6ab07362: origin-reported pricing lanes read `provider_cost_usd`. |

Residual risk

The full corpus and daemon convergence were not run. Large findings are not implemented; their beads carry the packet claims and confirmation evidence.
