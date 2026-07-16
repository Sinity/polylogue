---
created: "2026-07-05"
purpose: "Design the hot-daemon local wire protocol (UDS) for the Polylogue thin client / live-preview composer"
status: "complete"
project: "polylogue"
agent: "A2 (swarm2)"
---

# A2 — Hot-daemon local wire protocol (UDS)

Design for the local protocol the **thin client** (CLI/TUI/composer) speaks to
the **warm resident daemon**. The client does no substrate work: it frames a
request, the warm daemon answers against resident state. Because the client only
speaks a protocol, it can later be rewritten in Go/Rust for a sub-10 ms floor
while the substrate stays Python.

Recommendation in one line: **length-prefixed msgpack frames over an
`AF_UNIX SOCK_STREAM` socket, multiplexed on one persistent connection with
request-id correlation, a first-class `cancel` control frame, and server-push
`chunk`/`end` frames for streaming preview** — fronting the *same* handler
functions and pydantic payload models the HTTP surface already uses.

---

## 1. What exists today (evidence), and why HTTP is not enough

The daemon HTTP API is `ThreadingHTTPServer` over TCP loopback with
`BaseHTTPRequestHandler` (`polylogue/daemon/http.py:18`, `:3871`). Every request
runs in its own thread and calls `asyncio.run(self._run_archive_query(...))`
which opens a **fresh** `Polylogue()` context per request
(`http.py:1243-1250`). Responses are orjson (`http.py:380`).

The methods the composer needs already exist as routes, but each pays the
cold-open tax and none of them is built for keystroke cadence or cancellation:

| Composer need | Nearest HTTP route | File | Gap |
| --- | --- | --- | --- |
| `query(spec)` | `GET /api/sessions` → `_do_list` / `_do_search_list` (SearchEnvelope) | `http.py:1758,1785,1873` | Cold-opens `Polylogue()`; computes full `count()` every call (`http.py:1823`). |
| `query(spec)` terminal units | `GET /api/query-units` | `http.py:2773` | Opens `ArchiveStore` per call (`http.py:2825`). |
| `complete(partial)` | `GET /api/query-completions` → `query_completion_payload` | `http.py:2908`; `archive/query/completions.py` | **Metadata only** — field/operator/unit/stage vocab. No *value* completion (tag/origin/repo/tool values from live data). |
| `preview(spec, top_n)` | (none) — approximated by `?limit=N` + `total` | `http.py:1862` | No cheap top-N + approximate-count method; full count each keystroke. |
| `status` | `GET /api/status` → `get_status_snapshot_payload` | `http.py:1721` | Fine as-is; reuse. |
| streaming | `GET /api/events` SSE | `daemon/events_http.py` | Event-**ledger** stream, not query-**result** stream; one socket per stream. |

Cancellation primitives already exist and are good: cooperative disconnect
probe via `MSG_PEEK` (`http.py:159 _socket_peer_disconnected`),
`_ClientDisconnectedDuringComputeError` (`http.py:155`), and the facet
budget's `_raise_if_client_disconnected()` (`http.py:1135`). But they are
driven only by *socket close*, and HTTP/1.1 cannot multiplex or cancel an
in-flight request while keeping the connection.

Degraded/error vocabulary already exists and should be reused verbatim:
`QueryErrorPayload {ok, error, detail, field}` (`http.py:1197`,
`daemon_safe_handler` at `:921`), `archive_busy` → 503 + `Retry-After`
(`http.py:947`), typed embedding-not-ready → 409 (`http.py:1816`),
`RouteReadinessPayload {state, route, reason, component, stale_available}`
(`http.py:427`), and the tier-version gate that returns `None` when the archive
is mid-rebuild (`_web_reader_archive_root`, `http.py:386`).

**Conclusion.** The transport swap (TCP loopback → UDS) is a *minor* latency
win by itself — loopback TCP is already fast. The load-bearing changes are
(a) a **warm resident event loop + `Polylogue` + read-connection pool** that
kills the `asyncio.run(Polylogue())` cold-open, and (b) a **multiplexed,
cancellable** frame protocol for keystroke-cadence `complete`/`preview`. UDS is
the right vehicle for both, plus it gives filesystem-permission auth for free.

### What the UDS path adds over HTTP (crisp)

1. **Auth = filesystem, not tokens.** Socket at mode `0600`, owner-only, plus
   `SO_PEERCRED` uid check. Replaces the bearer-token / loopback-bootstrap dance
   (`http.py:1105,1383`) for the local composer.
2. **Multiplexing + first-class `cancel` on one persistent connection.**
   Keystrokes fire overlapping `complete`/`preview`; the client cancels the
   superseded request over the *same* socket. HTTP/1.1 can't; HTTP/2 would be
   heavier than a bespoke frame.
3. **Length-prefixed msgpack.** No per-request HTTP header parse, no base64 for
   bytes, 2–5× smaller/faster to decode at these payload sizes.
4. **Forces the warm-daemon refactor.** A persistent connection naturally pairs
   with a resident loop + connection pool; the cold-open pattern cannot survive
   a long-lived multiplexed socket.
5. **Sub-ms transport floor for a future Go/Rust client.**

HTTP stays as the **browser/remote** surface (web shell, `/metrics`, `/healthz`,
SSE for the browser reader) and the `--no-daemon` break-glass. UDS is the
**local thin-client fast path**. Do not fork logic — see §8.

---

## 2. Transport

- **Socket:** `AF_UNIX`, `SOCK_STREAM`, at
  `${XDG_RUNTIME_DIR}/polylogue/daemon.sock` (fallback
  `${POLYLOGUE_ARCHIVE_ROOT}/run/daemon.sock`). Created with `umask 0077`
  → mode `0600`; directory `0700`.
- **Auth:** owner-only file perms + `SO_PEERCRED` (`getsockopt(SOL_SOCKET,
  SO_PEERCRED)`) asserting the peer uid == daemon uid. No token. Non-local /
  remote composer is out of scope for UDS and continues to use authenticated
  HTTP.
- **Lifecycle:** the composer opens the connection once and holds it for the
  session. One-shot CLI verbs open, `hello`, one `req`, read `resp`, close.
- **Concurrency:** one connection carries many concurrent logical requests. A
  slow `query` must not head-of-line-block a keystroke `complete` — the daemon
  runs each `req` as its own task and pulls read connections from a small pool
  (see §6).

### Framing

Stream sockets have no message boundaries, so every frame is length-prefixed:

```
┌────────────┬───────────────────────────────┐
│ u32 len BE │ msgpack map (len bytes)        │
└────────────┴───────────────────────────────┘
```

The msgpack map is the frame envelope:

```
{ "t": <frame-type str>, "id": <u32 request id>, ... type-specific fields }
```

- Length prefix bounds a frame at, say, 64 MiB (reject larger → `err
  frame_too_large`, close). `id` correlates responses to requests and scopes
  cancellation. `id` is client-assigned, monotonic per connection.
- Encoding is negotiated in `hello` (§4): `msgpack` (default) or `json`
  (human/debug). Same payload bytes, different codec.

### Frame types

Client → server: `hello`, `req`, `cancel`.
Server → client: `welcome`, `resp`, `chunk`, `end`, `err`, `event`.

```
req     { t:"req", id, method:"query|complete|preview|status|subscribe", spec:{…},
          stream:bool, top_n?, limit?, offset?, cursor? }
cancel  { t:"cancel", id }                       # cancel in-flight request `id`
resp    { t:"resp", id, meta:{…}, payload:{…} }  # terminal, non-streamed
chunk   { t:"chunk", id, seq, payload:{…} }      # streamed partial
end     { t:"end", id, meta:{…}, cancelled?:bool }
err     { t:"err", id, error, detail?, field?, retryable:bool, degraded_to?, meta:{…} }
event   { t:"event", id, kind, payload:{…} }     # server-push for subscribe
```

Every `resp`/`end`/`err` carries a `meta` block (§7) so the client always knows
the archive state and precision without a separate `status` round-trip.

---

## 3. Encoding — recommendation: msgpack, JSON opt-in

- **msgpack** default: the daemon already `model_dump(mode="python")`s pydantic
  models everywhere (e.g. `http.py:1900`); msgpack serializes those dicts
  directly, no base64 for blob bytes, ~2–5× smaller and faster than JSON to
  decode at keystroke cadence.
- **JSON** (via the existing orjson path, `http.py:380`) selectable in `hello`
  for curl-equivalent debugging and a pure-Python client.
- Payloads are the **same models** as HTTP: `SearchEnvelope`,
  `query_completion_payload`, `QueryErrorPayload`, `RouteReadinessPayload`, the
  status snapshot. No new schema; the composer already knows these shapes from
  the HTTP contract.

Open question: adding a `msgpack` wheel. It is a tiny, pure-optional dep; if
undesired, ship JSON-only first and add msgpack behind the `hello` negotiation
later. The framing does not change.

---

## 4. Versioning

Negotiated **once per connection**, not per route (contrast the per-route
`RouteContract`, `http.py:33`):

```
hello    { t:"hello", id:0, protocol:1, client:"polylogue-cli/…",
           codecs:["msgpack","json"], features:["value-complete","stream"] }
welcome  { t:"welcome", id:0, protocol:1, server_version:"…",
           codec:"msgpack", capabilities:["value-complete","stream","subscribe"],
           archive_state:"ready|converging|rebuilding|stale|unavailable" }
```

- `protocol` is a single integer, bumped only on breaking frame/method changes.
- Additive method/feature changes are advertised as `capabilities` strings, so
  an old client + new daemon (or vice versa) negotiate down gracefully.
- Mismatch (`hello.protocol` unsupported) → `err protocol_unsupported`, close.
- The `welcome` also front-loads `archive_state` so the composer can show a
  "converging" banner before its first query.

---

## 5. Core methods

All methods take a `spec`: either the flat param map that
`_build_query_spec_params` builds today (`http.py:980`) **or** a raw DSL
`expression` string the daemon compiles via `compile_expression_into`
(`http.py:1792`, `archive/query/expression.py`). Same lowering as every other
surface — no protocol-specific query logic.

### `query(spec, limit, offset, cursor) -> SearchEnvelope | list-envelope`

Full result set. Reuses `_do_list` / `_do_search_list` (`http.py:1785,1873`) →
`SearchEnvelope` for ranked queries, list-envelope for pure-structured. Honors
`clamp_query_limit` (`spec.py`, `MAX_QUERY_LIMIT=1000`). Optional `stream:true`
emits `chunk` per page then `end`.

### `complete(partial_query, cursor_pos) -> candidates`

Two layers, both cancellable but effectively instant:

1. **Structural** (exists): partial-parse `partial_query` up to `cursor_pos`,
   classify the token under the cursor (field name? operator? unit? pipeline
   stage? action? value slot for field X?), then call
   `query_completion_payload(kind, incomplete, unit, field)`
   (`completions.py`) for the vocab. Pure, no DB.
2. **Value completion** (NEW — the composer's headline need): when the cursor
   is in a value slot (`origin:`, `tag:`, `repo:`, `tool:`, `model:`), return
   ranked live values with a prefix filter. Backed by **resident in-memory
   facet indexes** the warm daemon keeps (built from the same families as
   `_FACET_COMPLETE_ARCHIVE_FAMILIES`, `http.py:133`), refreshed on ingest.
   "Memory-hungry OK" (brief) makes this a sorted/trie lookup, not a per-
   keystroke SQL scan.

Returns the existing `QueryCompletionCandidate` payload
(`completions.py:QueryCompletionCandidate.to_payload`) with `replace_start/
replace_end` so the client can splice the completion at the cursor.

### `preview(spec, top_n) -> {rows: top_n, count, count_precision, meta}`

Distinct from `query` because it runs at **keystroke cadence**:

- Bounded sample: `top_n` small (default 8). No pagination, no cursor.
- **Cheap count**, not full `count()`:
  - `exact` when the filter is structured and the cheap count path applies;
  - `estimate` from SQLite planner stats for large derived tables (mirror
    `boundary_table_count_precision`, see `docs/internals.md` workload probe);
  - `capped` via a `LIMIT top_n+1` probe → "≥N" when exactness is expensive.
  - Return `count_precision ∈ {exact, estimate, capped}` so the composer can
    render "≥500" vs "512".
- **Always cancellable** and typically `stream:true`: emit the `top_n` rows as
  `chunk`s the instant they materialize, then a final `chunk`/`end` carrying
  `count` + `count_precision`, so rows paint before the count resolves.

`preview` is a new thin handler but shares the spec-build and filter path with
`query`; it just swaps full-count for the cheap-count policy and caps rows.

### `status() -> status_snapshot`

Reuse `get_status_snapshot_payload()` (`http.py:1730`) plus `component_readiness`
and `daemon_liveness`. Served from a **cached** snapshot invalidated on the
event-ledger id (the same signal the HTTP ETag uses, `http.py:1722`), so it is a
memory read, not a DB scan. Adds the derived `archive_state` (§7).

---

## 6. Streaming & cancellation (the live-preview core)

**Daemon concurrency.** Resident asyncio loop. Each connection is one reader
task. Each `req` spawns a task keyed `(conn_id, request_id)` and tracked in a
map. `complete`/`preview`/`status` pull from a dedicated small **read
connection pool** (`mode=ro` / `query_only=ON`, already the read profile per
`docs/internals.md` WAL section) so a slow `query` cannot block a keystroke
`complete`. The daemon's own ingest writer is separate and serialized.

**Cancellation is first-class:**

- Client sends `cancel {id}` when a newer keystroke supersedes an in-flight
  `preview`/`complete`. The daemon cancels that task; the task checks a cancel
  token at every await point (between streamed rows, before the count leg),
  generalizing today's cooperative `_raise_if_client_disconnected()` pattern
  (`http.py:1135`) from *socket-close-only* to *explicit-cancel*.
- A cancelled request returns `end {cancelled:true}` (or `err cancelled` if it
  had emitted nothing) — never a partial-looking success.
- **Socket close cancels all in-flight requests** on that connection (reuse the
  `MSG_PEEK` disconnect probe, `http.py:159`).
- **Client-side debounce** (~30–50 ms) + supersede is the first line of
  defense; the daemon adds optional **server-side coalescing** — a `preview`
  whose `id` is already superseded by a newer `preview` on the same connection
  is dropped, mirroring the events coalesce threshold
  (`events_http.py:_resolve_coalesce_threshold`).

**`subscribe`** folds the existing `/api/events` SSE stream
(`events_http.py`) into this connection: `req {method:"subscribe",
spec:{kinds:[…], since:id}}` → server-push `event` frames until `cancel`. This
lets the composer live-refresh preview results when ingest lands new sessions,
over the same multiplexed socket instead of a second SSE connection.

---

## 7. Error & degraded-state envelopes

Every terminal frame (`resp`/`end`/`err`) carries a `meta` block:

```
meta { archive_state:"ready|converging|rebuilding|stale|unavailable",
       last_event_id:int, stale_available:bool, degraded?:{lane:"semantic→lexical", reason} }
```

`archive_state` is derived exactly like `_web_reader_archive_root` (`http.py:386`):
tier files present and `PRAGMA user_version` matching → `ready`; mismatch/rebuild
in progress → `converging`/`rebuilding`; else `unavailable`. This means the
composer sees convergence state on *every* answer, no separate poll.

Errors reuse `QueryErrorPayload` fields plus two protocol additions:

| `error` | Meaning | `retryable` | Composer action |
| --- | --- | --- | --- |
| `invalid_query` | DSL parse/compile error; `field` + position | false | Underline the offending token at `field`. |
| `archive_busy` | Transient write-lock (today's sqlite-busy, `http.py:947`) | true | Retry after backoff; keep last good preview. |
| `embedding_not_ready` | Vector lane unavailable (`http.py:1816`) | false | Auto-degrade to lexical; `meta.degraded` set. |
| `archive_converging` / `archive_rebuilding` | Derived tier mid-rebuild | true | Show banner; serve `stale_available` best-effort. |
| `cancelled` | Superseded/cancelled request | n/a | Drop silently. |
| `protocol_unsupported` | `hello` version mismatch | false | Fail connect. |
| `frame_too_large` / `bad_frame` | Framing violation | false | Close + reconnect. |
| `not_found` / `internal_error` | as HTTP | false / true | Surface. |

`degraded_to` (e.g. semantic→lexical) lets `query`/`preview` return *useful*
results with a warning instead of a hard 409, matching the "results may be
partial/stale" posture of `RouteReadinessPayload.stale_available`.

---

## 8. Reuse & layering — do not fork the logic

The UDS server is a **thin framing layer** in front of the *same* handlers, not
a parallel implementation. Concretely:

1. Extract a transport-agnostic core `daemon/rpc/` from the current HTTP
   handlers: each core function takes a spec, returns a pydantic model, is
   `async`, and honors a `CancelToken`. The HTTP handlers become thin adapters
   (params→spec→core→`_send_json`); the UDS handlers become thin adapters
   (frame→spec→core→frame). One place for query/complete/preview/status logic.
2. Reuse the DSL compile path (`compile_expression_into`), the payload models,
   the readiness/precision vocabulary, `clamp_query_limit`, and the cooperative
   cancel/disconnect pattern.
3. The warm daemon owns the resident `Polylogue` + read-connection pool + the
   in-memory facet/value-completion indexes; **both** HTTP and UDS handlers read
   through it, replacing per-request `asyncio.run(Polylogue())` (`http.py:1249`).

HTTP is not deleted: it remains the browser reader, `/metrics`, `/healthz`, and
the remote/authenticated surface. UDS is added as the local fast path.

---

## 9. Latency budgets (warm daemon, single-user local)

| Method | Path | Budget |
| --- | --- | --- |
| connect + `hello`/`welcome` | once per client session | < 2 ms |
| `complete` — structural | pure vocab, no DB | < 1 ms |
| `complete` — value | resident in-memory index prefix lookup | < 2 ms |
| `preview` — structured, `top_n=8` | pool read, capped count | < 5 ms |
| `preview` — FTS bm25 | ranked, capped count | < 15 ms |
| `preview` count leg (capped `LIMIT n+1`) | probe | < 3 ms |
| `query` — structured list | warm cache | < 10 ms |
| `query` — FTS bm25 | ranked | 20–60 ms |
| `query` — vector / hybrid RRF | embeddings.db | 40–120 ms |
| `status` | cached snapshot (event-id invalidated) | < 2 ms |
| `cancel` round-trip | task-map lookup + cancel | < 1 ms |

These assume the warm-daemon refactor; today's cold-open (`asyncio.run(
Polylogue())` opening 5 tiers per request) blows every budget, which is the
whole point of moving off it. Composer target: `complete` and `preview` stay in
single-digit ms so the preview repaints within one animation frame of a
keystroke.

---

## 10. Open questions for the operator

1. **msgpack dependency** acceptable, or JSON-only first with msgpack behind
   `hello` negotiation later? (Framing is identical either way.)
2. **Value-completion indexes resident in RAM** (my recommendation, given
   "memory-hungry OK") vs. on-demand SQL against `index.db`? Resident wins on
   latency and matches the hot-daemon frame; cost is a rebuild hook on ingest.
3. **Fold event-push into the RPC connection** (`subscribe`) vs. keep the
   browser's `/api/events` SSE separate? Recommend fold for the local composer;
   keep SSE for the browser reader.
4. **Break-glass `--no-daemon`**: confirm it may be read-only and slow (direct
   `ArchiveStore.open_existing`), i.e. no composer/preview support without the
   daemon. The brief already sanctions "require the daemon."
