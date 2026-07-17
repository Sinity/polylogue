# Source anchor index
These anchors are from the unpacked current Polylogue snapshot: `/mnt/data/sysiter/tree78/polylogue` (`master @ 8a975a40`, Beads export generated 2026-07-06). Re-open files in the real working tree before editing.

## analytics
- `polylogue/archive/actions/followup.py` — Action/followup classification is a real structural analytics input.
- `polylogue/archive/actions/fields.py` — Action fields determine what can be measured without prose heuristics.
- `polylogue/insights/registry.py:294` — Insight registry should become measure/product registry input.
- `scripts/agent_forensics.py` — Existing forensics script is a proof artifact and candidate product surface.

## assertion_chokepoint
- `polylogue/storage/sqlite/archive_tiers/user_write.py:31` — ASSERTION_DEFAULT_STATUS is ACTIVE, so missing status currently means trusted active.
- `polylogue/storage/sqlite/archive_tiers/user_write.py:641` — upsert_blackboard_note passes author_kind and no explicit status into upsert_assertion.
- `polylogue/storage/sqlite/archive_tiers/user_write.py:901` — upsert_assertion is the single write chokepoint to patch.
- `polylogue/api/contracts/assertions.py` — Check public assertion request/response contract after changing default status behavior.

## attachments
- `polylogue/browser_capture/models.py:22` — BrowserCaptureAttachment has metadata and possible data fields; acquisition policy must be explicit.
- `polylogue/browser_capture/server.py:259` — Capture POST writes payloads to spool after admission checks.
- `polylogue/archive/attachment/models.py` — Normalized attachment model should carry acquired/missing/recoverable state.
- `polylogue/storage/blob_store.py` — Byte storage API and hash validation live here.

## blob_gc
- `polylogue/storage/blob_store.py:352` — detect_orphans only compares disk hashes to caller-supplied referenced IDs.
- `polylogue/storage/blob_store.py:387` — cleanup_orphans deletes caller-supplied hashes and lacks lease/ref/generation checks.
- `polylogue/storage/blob_gc.py:163` — _has_active_lease exists in the safer GC path.
- `polylogue/storage/blob_gc.py:307` — run_blob_gc is the safer planner/executor path to route destructive cleanup through.
- `polylogue/storage/blob_gc.py:393` — GC generation/age gate exists in run_blob_gc.

## bluegreen
- `CONTRIBUTING.md:102` — Derived-tier schema changes require rebuild/blue-green planning.
- `AGENTS.md:168` — Agent guidance says schema mismatch should rebuild or blue-green-replace derived tiers.
- `polylogue/cli/commands/reset.py` — Current reset/rebuild commands are the operator path to replace derived tiers.
- `polylogue/daemon/convergence_stages.py` — Daemon convergence/readiness state should represent generation progress honestly.

## coordination
- `polylogue/coordination/envelope.py` — Coordination envelope model exists; harden it as the shared payload.
- `polylogue/coordination/payloads.py` — Coordination payload types should stay small and evidence-ref oriented.
- `polylogue/coordination/rendering.py` — Rendered advisories should be scheduler-mediated, not chat spam.
- `tests/unit/coordination/test_envelope.py` — Existing envelope tests are the starting verification lane.
- `polylogue/mcp/server_prompts.py:219` — MCP prompt registration exists and can surface cookbook/roles.
- `polylogue/cli/commands/agents.py` — CLI agent commands are the operator-facing entry point.

## daemon_security
- `polylogue/daemon/http.py:983` — _check_auth_logic uses direct equality and allows all when token is unset.
- `polylogue/daemon/http.py:1037` — _check_auth currently accepts query-string access_token broadly.
- `polylogue/daemon/http.py:1294` — do_GET dispatches without central Host/Origin admission.
- `polylogue/daemon/http.py:1301` — _check_cross_origin applies only to POST and allows absent Origin.
- `polylogue/browser_capture/receiver.py:45` — BrowserCaptureReceiverConfig defaults auth_token to None.
- `polylogue/browser_capture/server.py:54` — _origin_allowed accepts absent Origin.
- `polylogue/browser_capture/server.py:68` — _check_token accepts every request when auth_token is None and uses direct equality otherwise.
- `polylogue/browser_capture/server.py:47` — Only per-request max body exists; add spool file/count/bytes governor.

## docs
- `README.md` — Public claims should be grounded through the claims ledger.
- `docs/agent-forensics.md` — Existing forensics docs are a pattern for proof artifacts.
- `docs/demo.md` — Demo docs should depend on evidence/citation machinery.

## embeddings
- `polylogue/api/embeddings.py` — Public embedding API seam.
- `polylogue/cli/commands/embed.py` — Operator embedding command and dry-run planner hooks.
- `polylogue/archive/query/retrieval.py` — Retrieval composition seam for FTS/vector/hybrid.
- `polylogue/archive/query/retrieval_search.py` — Search/retrieval runtime implementation.

## fts
- `polylogue/storage/sqlite/archive_tiers/index.py:296` — messages_fts table DDL lives in archive_tiers/index.py.
- `polylogue/storage/fts/sql.py:11` — messages_fts DDL copy also exists in storage/fts/sql.py.
- `polylogue/storage/fts/fts_lifecycle.py:292` — ensure_fts_triggers_sync owns runtime repair/recreation.
- `polylogue/storage/fts/fts_lifecycle.py:512` — Thread FTS rebuild logic is duplicated in lifecycle repair path.
- `polylogue/daemon/convergence_stages.py:988` — daemon convergence has another repair/readiness path for archive messages_fts.

## lineage
- `polylogue/archive/session/threads.py` — Session/thread lineage read and composition model.
- `polylogue/insights/topology.py` — Topology/lineage derived insight code.
- `polylogue/daemon/lineage_startup.py` — Daemon lineage startup/convergence path.
- `polylogue/archive/coverage.py` — Completeness/truncation cues live here.
- `polylogue/insights/postmortem.py` — Compaction/continuation postmortem evidence is mined here.

## object_refs
- `polylogue/core/refs.py` — Existing ref model should be extended, not bypassed.
- `polylogue/storage/sqlite/archive_tiers/user_write.py:901` — Findings should reuse assertion/candidate lifecycle.
- `polylogue/surfaces/payloads.py:747` — Surface payloads currently expose branch/message refs; expand consistently.
- `polylogue/mcp/payloads.py:377` — MCP message payloads carry variant/branch metadata today.

## read_contract
- `polylogue/archive/query/spec.py:440` — SessionQuerySpec is the core read selection object.
- `polylogue/archive/query/spec.py:498` — from_params centralizes parameter lowering.
- `polylogue/cli/query_contracts.py:70` — QueryOutputSpec is current output/render contract seam.
- `polylogue/cli/archive_query.py:164` — CLI archive query execution remains a dense orchestration point.
- `polylogue/daemon/http.py:1734` — Daemon split-archive route already has from_params path for some cases.
- `polylogue/daemon/http.py:3296` — Another daemon path builds SessionQuerySpec from raw query params.
- `polylogue/mcp/query_contracts.py:54` — MCP query normalization has a distinct lowering path.

## rigor
- `polylogue/insights/audit.py:173` — build_insight_rigor_audit_report is the audit entry point.
- `polylogue/insights/audit.py:194` — Current code iterates list_rigor_contracts, not the product registry.
- `polylogue/insights/audit.py:216` — Registry lookup is secondary and skipped for products without contracts.
- `polylogue/insights/rigor.py:85` — _RIGOR_MATRIX declares only a subset of registered products.
- `polylogue/insights/registry.py:294` — INSIGHT_REGISTRY is the universe the audit should iterate.

## sources
- `polylogue/sources/dispatch.py` — Current origin/source dispatch logic; target for OriginSpec consolidation.
- `polylogue/sources/import_preflight.py` — Preflight/readiness should report origin strictness and ambiguity.
- `polylogue/sources/provider_completeness.py` — Provider completeness is adjacent to OriginSpec readiness.
- `polylogue/sources/parsers/base.py` — Parser base contracts should be folded into OriginSpec.

## temporal
- `polylogue/core/dates.py:10` — parse_date has no injected clock parameter.
- `polylogue/core/dates.py:37` — RELATIVE_BASE uses ambient datetime.now.
- `polylogue/archive/query/expression.py:2440` — Query grammar recognizes relative-date literals.
- `polylogue/archive/query/spec.py:498` — SessionQuerySpec.from_params is the central query-spec constructor.
- `polylogue/insights/temporal_source.py:66` — classify_profile_hwm_source promotes any updated_at to provider_ts.
- `polylogue/insights/temporal_source.py:97` — classify_aggregate_hwm_source currently collapses all non-empty source updates to provider_ts.

## timestamp_copies
- `polylogue/context/selection.py:285` — One of several _parse_archive_datetime copies.
- `polylogue/mcp/archive_support.py:492` — MCP copy of _parse_archive_datetime.
- `polylogue/api/archive.py:514` — API copy of _parse_archive_datetime.
- `polylogue/cli/read_views/standard.py:232` — CLI read-view copy of _parse_archive_datetime.
- `polylogue/archive/query/archive_execution.py:113` — Archive query execution copy of _parse_archive_datetime.
- `polylogue/storage/insights/session/rebuild.py:763` — Divergent storage/insight copy that forces naive timestamps to UTC.
- `polylogue/daemon/provenance.py:84` — One of several _iso_from_epoch_ms copies.
- `polylogue/storage/embeddings/status_payload.py:338` — Lenient _iso_from_epoch_ms copy in embeddings status payload.

## usage
- `polylogue/storage/usage.py:478` — stale provider rollup stats are materialized/read here.
- `polylogue/storage/usage.py:797` — full stale diagnostics path can become expensive.
- `polylogue/storage/usage.py:891` — comments indicate corrected rows with cached/reasoning partitions.
- `polylogue/storage/usage.py:1033` — model rollup stats aggregate input/output/cache lanes.
- `polylogue/archive/semantic/cost_compute.py` — Inspect pricing/provenance computation before changing cost views.
- `polylogue/archive/semantic/subscription_pricing.py` — Subscription-credit view belongs here, distinct from API-list-equivalent cost.
- `scripts/cost_accounting_demo.py` — Existing demo captures usage/cost accounting expectations.

## variants
- `polylogue/core/identity_law.py:42` — Current identity includes variant_index only for provider sibling messages, not transformed content variants.
- `polylogue/storage/sqlite/queries/message_query_reads.py:34` — Read model projects message variant_index as branch_index.
- `polylogue/surfaces/payloads.py:747` — Reader payload maps message variant_index to branch_index.
- `polylogue/daemon/compare.py:220` — Compare/alignment semantics exist for message diffing, not content-transform alignment.

## web
- `polylogue/daemon/http.py` — Current web/daemon API surface.
- `polylogue/daemon/web_shell_workspace.py` — Current HTML/JS workbench shell; v2 should reduce inline state sprawl.
- `polylogue/daemon/web_assets.py` — Static asset ownership and generated-client integration point.
