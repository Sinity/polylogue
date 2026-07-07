---
created: "2026-06-28T00:00:00+00:00"
purpose: "Read-only construct-validity audit: what each core construct CLAIMS to represent vs what the code ACTUALLY computes"
status: complete
project: polylogue
---

# Construct-Validity Audit (read-only)

Method: for each core construct, compare name/docstring/usage CLAIM against what the code
actually computes/stores. Flag proxy-as-truth, conflation, silent-lie/misnamed,
ill-defined category, dead/fake field. Rank by severity. Validity-sound constructs listed
separately. No code changed.

---

## SUSPECT / BROKEN (ranked by severity)

### HIGH

| Construct | Claims to represent | Actually is | Failure kind | Evidence + concrete gap |
|---|---|---|---|---|
| ~~`RecoveryEvent` work-event miner (`_events_from_text`)~~ | ~~"what the agent/session DID" — durable facts: PR merged, tests passed, issue closed~~ | ~~Previously regex-mined arbitrary message text~~ | **Resolved in current source** | 2026-06-30 audit: `_events_from_text` no longer exists. `insights/transforms.py:_extract_events` derives command/test outcome events from paired `tool_result` structure (`tool_result_exit_code` / `tool_result_is_error`) and skips `unknown` outcomes. Pathology now consumes typed `ObservedEvent` metadata, with regression coverage in `tests/unit/insights/test_pathology.py`. |
| ~~`material_origin = HUMAN_AUTHORED` / `authored_user_message_count`~~ | ~~This row is text a human actually typed~~ | ~~Previously: `Role.USER` + `MessageType.MESSAGE`, where `MESSAGE` was the fall-through default after a fixed marker allowlist failed to match~~ | **Resolved in current source** | 2026-06-30 audit: `archive/message/artifacts.py:classify_material_origin` now explicitly has no `Role.USER + MESSAGE -> HUMAN_AUTHORED` fall-through and returns `UNKNOWN` unless positive evidence exists. Regression guard: `tests/unit/core/test_message_types.py::test_plain_user_message_does_not_imply_human_authorship`. |

### MEDIUM

| Construct | Claims to represent | Actually is | Failure kind | Evidence + concrete gap |
|---|---|---|---|---|
| `TopologyEdgeType` / `LinkType` `FORK` vs `RESUME` | 7 distinct, correctly-assigned relationship types incl. distinct `fork` (branch off) and `resume` (continue) | Codex parser collapses fork and resume into `FORK`; `RESUME` is never emitted by the parser; `BranchType` enum doesn't even contain RESUME | **Conflation** | `sources/parsers/codex.py:653` comment admits "`forked_from_id` (no subagent block) → user fork / **resume** of the parent" but `l.658-660` assigns `BranchType.FORK` for both. `core/enums.py:293-300` `BranchType` has only {continuation, sidechain, fork, subagent} — no resume — while `LinkType` (l.302-312) advertises `RESUME` and `REPAIRED` as if assignable. `branch_type_to_edge_type` (`identity_law`→`topology/edge.py:31-37`) maps the 4 BranchType values identity-wise; RESUME/REPAIRED edge types have no production assignment path. |
| `branch_type = CONTINUATION` (Codex multi-meta) | Parser observed a real continuation relationship | Heuristic guess: "there is a 2nd embedded session_meta id" | **Proxy-as-truth** | `sources/parsers/codex.py:661-663`: `elif len(session_metas_seen) > 1: parent_id = session_metas_seen[1]; branch_type = CONTINUATION`. The relationship type is inferred from the *count of embedded meta records*, not from an asserted parent/continuation marker. |
| `detect_paste` / `has_paste` (=1) | Message is dominated by pasted/inserted content | OR of 5 shape heuristics, two of which are pure length/format proxies | **Proxy-as-truth** | `archive/message/paste_detection.py:111` any text > 4000 chars → paste=1; `l.115` >70% inside code fences → paste=1. A genuinely long typed prose message or a heavily-code-quoting human answer is flagged "has_paste". The `[Pasted text #N]` marker (l.107) IS ground-truth, but it is unioned with the weak proxies, so the boolean cannot be trusted as "a paste occurred." `resolve_paste_boundary_state` is more honest (it labels `whole_message_fallback` vs `exact`), but the scalar `detect_paste` flattens that distinction. |
| `Provider` enum | (legacy) provider-wire identity | Single token mixing lab identity, product/runtime, and source-family | **Conflation (documented, partially cleaned)** | `core/enums.py:67-80`: `CHATGPT`/`CLAUDE_AI`/`CLAUDE_CODE`/`CODEX`/`GEMINI`/`GEMINI_CLI` mix product (chatgpt), source-family (claude-ai export vs claude-code session), and lab (gemini). Public read surfaces have largely moved to `Origin` (sound); `Provider` survives at parse/schema/provider-usage boundaries where it is legitimately provider-wire (`mcp/server_tools.py:656` provider_usage is a valid use). Severity medium because the leak is mostly closed and the residue is documented in `docs/architecture.md`. |
| Codex token lanes — two divergent normalizers | Disjoint input/output/cache lanes (post-7.69× double-count fix) | Two functions disagree: one keeps `uncached_input_tokens` (honest), the other maps provider `input_tokens` straight through | **Silent lie risk / conflation** | `sources/parsers/codex.py:150-167` `_codex_token_usage_payload` exposes `uncached_input_tokens` + `cached_input_tokens` separately (honest, supports disjoint billing). But `_token_usage` (l.178-190) maps provider `input_tokens`→`input_tokens` and `cache_read_tokens` separately **without subtracting cached from input**. Codex's reported `input_tokens` INCLUDES cached (~96% per MEMORY); `pricing.py:_cost_components` prices `input` and `cache_read` as additive lanes (l.~404+), so any path feeding `_token_usage` output into the cost components re-double-counts the cached portion. Which path wins at materialization needs a runtime trace to confirm; flag as suspect, not proven. |

### LOW

| Construct | Claims to represent | Actually is | Failure kind | Evidence |
|---|---|---|---|---|
| `_session_transform_timestamp` fallback | Session occurred-at time | Returns `1970-01-01T00:00:00+00:00` when no timestamp exists | **Silent bucketing** | `insights/transforms.py:112-114`. Missing timestamps are silently coerced to epoch-zero rather than carried as null; downstream ordering/temporal grouping buckets timestamp-less sessions at 1970. (Low: it is a documented sentinel, not random.) `core/timestamps.py:44` also silently drops epochs < 86400 as "not an epoch," so legitimate tiny epochs parse to None. |
| Decision/run-state extraction (`_DECISION_RE`, `_STATUS_HEADING_RE`) | Durable decisions made in the session | Any line matching `decision|decided|choose|chosen:` or `goal|done|next:` headings, regardless of speaker or whether a decision was reached | **Proxy-as-truth (mitigated)** | `insights/transforms.py:1896-1929`. Mitigated by the `candidate`/review-status machinery (these are surfaced as *candidates* needing acceptance, not asserted facts), so lower severity than the work-event miner which asserts directly. |

---

## GENUINELY VALID / SOUND constructs

| Construct | Why it is sound |
|---|---|
| `session_content_hash` (messages) | `pipeline/ids.py:125-168` hashes id/role/text/timestamp/content-blocks with NFC normalization and None/empty sentinels. Honestly named and complete for the message tree. |
| `content_hash` attachment subcomponent | `pipeline/ids.py:137-146` hashes only id/message_id/name/mime/size — and this is now *honest*: attachments are metadata-only by construction (#2468), and the hash documents exactly that. It does NOT claim to hash bytes; the docstring (l.130) lists "attachments" as a field, and the attachment payload makes the metadata-only scope explicit. Sound given the (separately disclosed) attachment-bytes limitation. |
| `attachments.acquisition_status` / nullable `blob_hash` (#2468 fix) | `storage/sqlite/archive_tiers/write.py:3436-3450` + `index.py:476-479`: `blob_hash` is now true SHA-256 of stored bytes or NULL; `acquisition_status ∈ {acquired, unavailable, unfetched}` honestly records fetch state. The prior fabricated-hash lie (`bytes.fromhex(attachment_id)`) is gone. The `CASE WHEN excluded.acquisition_status='acquired'` upsert (l.1769) is monotonic — never downgrades acquired→unfetched. Now a sound, honest construct. |
| `identity_law` IDs (session/message/block) | `core/identity_law.py` — deterministic, provider-native-id-wins with `position.variant_index` fallback; collision-safe for sibling regenerations. Does exactly what it claims. |
| `Origin` enum + public read filters | `core/enums.py:42-64` is a clean source-origin vocabulary; public CLI/MCP/API surfaces key on it. The intended split from `Provider` is real on the public surface. |
| `logical_session_count` vs `session_count` | `insights/archive_rollups.py:77,102` counts distinct `logical_session_id` (resolved root) vs physical rows. The count arithmetic is honest; it correctly distinguishes physical from logical. (Caveat: its *correctness* inherits the upstream FORK/RESUME conflation, but the count construct itself does not lie about what it measures.) |
| `MaterialOrigin` non-human lanes (`TOOL_RESULT`, `RUNTIME_CONTEXT`, `RUNTIME_PROTOCOL`, `OPERATOR_COMMAND`, generated packs) | When a marker *does* match, the classification is principled and useful — the failure is only the `HUMAN_AUTHORED` *default*, not the positively-matched non-human categories. |
| `CostBasis` / `CostUnavailableReason` taxonomy | `archive/semantic/pricing.py:24-46` explicitly separates `provider_reported` / `api_equivalent` / `subscription_equivalent` / `catalog_priced` and gives discrete unavailable reasons. `cost_enrichment.py` never downgrades a confident stored estimate. The construct is honest about its basis rather than presenting one number as "the cost" — the known `cost_usd`-is-API-list caveat is surfaced *through* this taxonomy, not hidden. |
| `TopologyEdgeRecord` durability/identity | `archive/topology/edge.py:44-69` — typed unresolved-assertion identity, out-of-order resolve, confidence bounded [0,1], hash-boundary-excluded. Sound. The only issue is upstream *assignment* of FORK/RESUME, not the edge record itself. |

---

## Notes
- The two HIGH findings share a root cause: **text-shape heuristics promoted to asserted facts without speaker/material_origin gating**. The work-event miner is the worst because it asserts directly; `material_origin=HUMAN_AUTHORED` is an allowlist-default masquerading as positive evidence.
- `session_phases` (prompt item 8) was not found as a live table/module in the current tree (`rg phase_type` returns no insight-layer hits); the historical "phase_type='phase' const / confidence=0.0" dead-field appears to have been removed in the #1743 split-file rewrite. Could not confirm a live fake-field instance — treat the old note as likely stale.

## ADDENDUM (2026-06-28 session 2): pathology_report / postmortem inherit the regex invalidity
Operator flagged that `pathology report` (and less likely `postmortem`) may themselves be
construct-invalid. CONFIRMED for pathology: `polylogue/insights/pathology.py` detects
`wasted_loop`/`missed_review`/`stale_context` over a `RunProjection`, whose `test_failed`/
`check_failed` events are emitted by `_events_from_text` — the SAME regex prose-miner flagged HIGH
above. So pathology findings inherit construct-invalidity through their INPUT (a "wasted loop" =
"repeated regex-guessed test_failed events", not repeated real failures).
- NOT inherently invalid as a CONCEPT (loop/stuck/abandonment detection is legit) — the INPUT is bad.
- SALVAGEABLE via the keystone (schema v16, commit 19490eadd): real per-tool is_error/exit_code now
  exist. Re-base run-projection (and thus pathology) on structured outcomes = redesign step 4. Then
  wasted_loop = repeated real is_error=1 in a run.
- DECISION (operator): leave pathology/postmortem in place for now (not fabricating a NEW bad class;
  downstream of one already targeted). Re-base when run-projection is rebuilt from structure.
- postmortem_bundle: needs a separate look (likely composes session reads + recovery digest; the
  digest part shares the run-projection dependency). Lower priority than pathology.
