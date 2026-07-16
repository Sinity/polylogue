---
created: "2026-06-28"
purpose: "Investigate capturing server_tool_use (web_search/web_fetch) billing counts from Claude JSONL"
status: complete
project: polylogue
---

# server_tool_use capture gap (Claude web_search / web_fetch billing)

Read-only investigation. No code changed. Goal: confirm the JSON shape, find
where the Claude parser drops it, and sketch the minimal capture plan.

## 1. Real JSON shape (confirmed against `~/.claude/projects/`)

There are TWO distinct `server_tool_use` surfaces in Claude records — do not
conflate them:

### (a) The billing counts — `message.usage.server_tool_use` (the target)

Every assistant record carries a usage block. The relevant sub-object:

```json
{
  "input_tokens": 0,
  "output_tokens": 0,
  "cache_creation_input_tokens": 0,
  "cache_read_input_tokens": 0,
  "server_tool_use": {
    "web_search_requests": 0,
    "web_fetch_requests": 0
  },
  "service_tier": null,
  "cache_creation": {
    "ephemeral_1h_input_tokens": 0,
    "ephemeral_5m_input_tokens": 0
  },
  "inference_geo": null,
  "iterations": null,
  "speed": null
}
```

The two integer counters `web_search_requests` and `web_fetch_requests` are the
separately-billed quantities (Anthropic bills web search/fetch per request, on
top of tokens). This is the data being dropped.

Corpus note: across this operator's Claude **Code** corpus the counts are
**always 0** (~397K usage occurrences scanned, all `web_search_requests:0` /
`web_fetch_requests:0`) — CLI sessions don't invoke web tools here. The shape is
nonetheless confirmed and stable; non-zero values appear in Claude.ai web
sessions. Capture should be implemented even though local Code data is all-zero,
because (i) the daemon also ingests Claude.ai exports through the AI parser path,
and (ii) Code may gain web tools.

### (b) A content block of type `server_tool_use` (already partially handled, NOT the billing data)

A separate thing also exists — an assistant content block representing a
server-side tool invocation (e.g. the `advisor` / `web_search` server tool):

```json
{ "type": "server_tool_use", "id": "srvtoolu_019gywpX3bigCDW3c7dCWzKz", "name": "advisor", "input": {} }
```

This is the invocation, not the count. It flows through
`content_blocks_from_segments` like any other block. It is NOT where the billed
request counts live. The task target is (a), the `usage.server_tool_use` object.
(There would also be matching `web_search_tool_result` / `web_fetch_tool_result`
result blocks when web tools actually run — none present in this corpus.)

## 2. Where the parser drops it

`polylogue/sources/parsers/claude/code_parser.py`:

- `_parse_code_records()` reads `msg_usage = message.get("usage")` (line ~273)
  and pulls exactly four token fields onto the `ParsedMessage`:
  `input_tokens`, `output_tokens`, `cache_read_input_tokens`,
  `cache_creation_input_tokens` (lines 304-307). `usage["server_tool_use"]`
  is never read.
- It also emits a `ParsedSessionEvent(event_type="message_usage", ...)` whose
  payload is built by `_message_usage_event_payload()` (lines 116-140). That
  helper copies only the same four token lanes into `last_token_usage`. The web
  tool counts are dropped here too.

So both the per-message model and the usage-event payload omit the counts. The
Claude.ai web parser (`ai_parser.py`) does not extract usage at all, so it is a
no-op there today regardless.

`ParsedMessage` model (`polylogue/sources/parsers/base_models.py`, lines
111-117) has the four token int fields and `model_name`/`duration_ms`; no web
tool fields.

## 3. Where it should be captured (storage)

This is **usage/billing** data, so it belongs with provider usage — NOT in
`web_content_constructs` (index schema v8). `web_content_constructs` captures
provider-native *content* constructs (citations, source refs, canvas/artifact
revisions, asset pointers) — i.e. surface (b)-style result artifacts, not
billing counters. Putting per-request billing counts there would split usage
accounting across two tables. Keep it in the usage lane.

Target table: `session_provider_usage_events`
(`polylogue/storage/sqlite/archive_tiers/index.py`, lines 575-605). Today it has
typed columns for the token lanes (`last_input_tokens`, `last_output_tokens`,
`last_cached_input_tokens`, `last_cache_write_tokens`,
`last_reasoning_output_tokens`, `last_total_tokens`, plus `total_*` mirrors) and
a `payload_json TEXT` column.

Write path: `_write_provider_usage_event()`
(`polylogue/storage/sqlite/archive_tiers/write.py`, lines 2235-2277). It maps
`payload["last_token_usage"]` keys onto the typed columns and stores the **entire
event payload verbatim** via `_json_dumps(event.payload)` into `payload_json`.

Key consequence: **anything the parser adds to the message_usage event payload
is already persisted into `payload_json` with no schema change.** A schema bump
is only needed if we want first-class, queryable typed columns.

## 4. Minimal capture plan

Two tiers, pick based on whether cost queries need to join on the counts.

### Tier A — payload-only (no schema bump, smallest change)

1. `code_parser.py:_message_usage_event_payload()` — read
   `usage.get("server_tool_use")` and, when present/non-empty, add to the
   payload, e.g.:
   ```python
   stu = usage.get("server_tool_use")
   if isinstance(stu, dict):
       payload["server_tool_use"] = {
           "web_search_requests": _safe_int(stu.get("web_search_requests")),
           "web_fetch_requests": _safe_int(stu.get("web_fetch_requests")),
       }
   ```
   It rides into `payload_json` automatically via the existing writer.
2. (Optional) mirror the same extraction in the Claude.ai `ai_parser.py` usage
   path if/when that path emits usage events.
3. No `ParsedMessage` field, no DDL change, no version bump. Downstream cost code
   (`archive/semantic/pricing.py`) can read the counts out of `payload_json`.

Tradeoff: not directly indexable/aggregatable in SQL without `json_extract`.

### Tier B — first-class typed columns (schema version bump v14 → v15)

If web-tool requests should be summable/joinable like token lanes:

1. Add `last_web_search_requests` / `last_web_fetch_requests` (+ optional
   `total_*`) INTEGER columns to `session_provider_usage_events` DDL in
   `index.py`.
2. Bump `INDEX_SCHEMA_VERSION` (`index.py:36`, currently `14`) to `15`.
3. Extend `_write_provider_usage_event()` INSERT (write.py:2244) to populate the
   new columns from the payload's `server_tool_use` sub-dict.
4. Parser change as in Tier A (the payload is still the carrier into the writer).
5. Per `CONTRIBUTING.md` "Schema-Touching Changes" + `docs/internals.md`
   §"Schema Versioning Model": this is a **deletes-then-defines** edit — no
   in-place upgrade. The PR body must carry a re-ingest plan
   (`polylogue ops reset --database && polylogued run`). `index.db` is derived,
   so it's a rebuild-from-source with no user-data loss; `source.db` untouched.
6. Update `self_verify.py` / any usage read models (`storage/usage.py`,
   `cli/commands/diagnostics.py`) and the provider-usage coverage matrix notes to
   mention the new lanes.

### Recommendation

Tier A first (idempotent, zero-risk, fully closes the data-loss gap — the bytes
are now durable in `payload_json`). Promote to Tier B only when a concrete cost
or analytics surface needs to aggregate web-tool requests in SQL. Both tiers
share the identical parser edit; Tier B is purely additive storage on top.

## Files touched in a fix
- `polylogue/sources/parsers/claude/code_parser.py` (`_message_usage_event_payload`, ~L116-140)
- (Tier B) `polylogue/storage/sqlite/archive_tiers/index.py` (DDL L575-598; version L36)
- (Tier B) `polylogue/storage/sqlite/archive_tiers/write.py` (`_write_provider_usage_event`, L2235-2277)
- (optional) `polylogue/sources/parsers/claude/ai_parser.py` for Claude.ai exports
