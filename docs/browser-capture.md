# Browser Capture

Polylogue can receive browser-observed LLM sessions through a local-only
Manifest V3 extension.

Start the receiver:

```bash
polylogued browser-capture serve
```

For normal long-running local service use, `polylogued run` starts the browser
capture receiver together with live source watching.

CaptureJob update receipts validate the full current request digest. A request ID does not certify an obsolete digest shape. Browser capture files and source-bearing checkpoint carriers are original inputs; bookkeeping retirement preserves their normal readers and does not exclude them because they were previously acknowledged.

Protocol 2 CaptureJobs use the receiver store at `capture-jobs/v2/` under the browser-capture spool. The earlier `capture-jobs/registry.sqlite3` and its artifacts remain intact as retained evidence. The receiver does not upgrade or read that earlier registry in place.

CaptureJob control requests are staged to the receiver spool and decoded through
a disk-backed JSON view. Registry responses are encoded to a staged file and
sent with their exact `Content-Length` in chunks. Intent payloads, event refs
and payloads, native preparation metadata, and asset outcomes use lazy JSON
views through registry publication and reads. Their collection cardinality does
not require a decoded Python tree; individual strings, parser records, and
SQLite cells still occupy memory while being processed. Event pages stage each
event under its own two cell-view owners and release them before reading the
next event. One lazy page document then serves the events and their timelines;
open spill connections do not scale with the selected event count.

The receiver's existing `serve_forever` lifecycle owns CaptureJob retirement.
Each `service_actions` turn selects one eligible job through the existing
receiver-local indexes, persists an internal `retiring` marker, drains at most
64 rows from one leaf relation, and advances 64 artifact-directory entries.
The marker permanently fences scoped reads, adoption and mutations and excludes
the job from discovery. An expired lease is cleared when marking; a later clock
change cannot restore authority. Native assets and other leaves drain before
plans, acquisitions and the job parent, so a parent deletion cannot cascade an
unbounded membership. Each page commits independently and restart resumes from
the durable marker and physical artifact roots. The artifact frontier retains a
bounded pending page across registry-open, SQLite busy, query and I/O failures.
Only a successful root and physical check settles an entry; unchecked names
remain for the next lifecycle turn. Restart discards that disposable position
and safely rescans the physical namespace.

Create and discovery do no retirement deletion or artifact traversal. Discovery
excludes jobs already eligible for retirement before calculating its total and
page cursor, even before the lifecycle marks them. Adoption checks that same
eligibility inside its transaction and returns retryable 503 rather than
renewing an expired terminal job. Held jobs, live leases and unfinished native
custody remain inspectable. Creating
the same intent while its old eligible or retiring job owns the unique key
returns retryable HTTP 503 `capture_job_retirement_pending`; no new job is
partially created. Cleanup continues without client requests. Existing extension
transport retries retain the capture and retry this 503. Explicit registry GC
repeats the same row pages and retains a complete streamed artifact sweep.
Artifact root checks and active-reader locks still precede removal. Shutdown
stops between lifecycle turns and closes its disposable directory frontier; a failed turn logs the named registry failure
and remains retryable. No archive tier, receiver DDL or public field changes.

The
ordinary browser-action, pairing, health, and assertion control routes retain
their separate bounded in-memory request contract.

The receiver listens on `127.0.0.1:8765` by default and accepts the route contracts in `polylogue/browser_capture/route_contracts.py`:

- `GET /v1/status` -> `BrowserCaptureReceiverStatusPayload`
- `POST /v1/receiver/status-attest` with `receiver_id`, fresh `challenge` and request `proof` -> status payload authenticated by the `X-Polylogue-Status-Proof` response header; the bearer remains local
- `GET /v1/archive-state?provider=chatgpt&provider_session_id=...` -> `BrowserCaptureArchiveStatePayload`
- `POST /v1/browser-captures` with `BrowserCaptureEnvelope` -> `BrowserCaptureAcceptedPayload` or `BrowserCaptureErrorPayload`
- `PUT /v1/browser-action-attachments` -> streamed immutable attachment input, returning its SHA-256 `attachment_ref` and byte count
- `GET/POST /v1/browser-actions...` -> provider-neutral draft/submit intents, leases, exact receipts, explicit uncertain-submit reconciliation, and explicit operator approval for destructive submits

`/v1/archive-state` reports archive visibility, not just receiver spool
presence. Its `state`/`lifecycle` field is one of:

| State | Meaning |
| --- | --- |
| `missing` | No receiver artifact, raw acquisition row, or indexed session was found. |
| `spooled_only` | The receiver has a local artifact, but live ingest has not acquired it into `source.db`. |
| `ingest_pending` | `source.db.raw_sessions` has the capture, but `index.db.sessions` is missing it or has no messages yet. |
| `archived` | The capture has raw evidence, an indexed session, and at least one indexed message. Only this state sets `captured: true`. |
| `stale` | The receiver spool artifact is newer than the indexed archive row for the same provider session. Keep the daemon running; convergence should advance this without a manual repair command. |
| `failed` | The receiver artifact is unreadable or raw validation/parsing recorded a failure. |

For a ChatGPT temporary chat, archive checks and mission control use the
ephemeral identity in the current document's observed native payload. The URL
sentinel only admits capture; it is never an archive query key. An unavailable
or mismatched document identity stays unknown rather than borrowing a prior
conversation's stored status.

The payload includes bounded archive evidence (`raw_row_exists`, `raw_id`,
`indexed_session_exists`, `indexed_session_id`, `indexed_message_count`) and a
relative `artifact_ref`. It must not expose absolute paths. Deployment smoke
uses this endpoint as an invariant check: a receiver that says `captured: true`
without raw/index/message evidence is considered broken, not merely stale.

Inspect the running receiver's observed policy through the daemon with
`polylogued status` or `polylogue ops doctor --daemon`.
`polylogued browser-capture status` reads the configured receiver directly, including
standalone `browser-capture serve`, using existing credentials without minting or rotation.
For a standalone listener override, pass the matching `status --host HOST --port PORT`; omitted values use resolved settings. Credentialed observations use `POST /v1/receiver/status-attest`. A fresh challenge and request HMAC authenticate the caller before status disclosure; a response HMAC binds the challenge, persisted receiver identity and exact staged JSON bytes. The persisted bearer never crosses this observation socket, including through a relay. The returned identity is checked. For an explicitly unauthenticated listener, use `status --allow-no-auth` (or the matching configured/environment opt-out); this sends no credential and checks identity and disabled authentication. Use `--require-auth` to override a configured no-auth setting for a credentialed listener. Status waits for completion or operator cancellation, stages and validates responses incrementally, and retains the lazy origin roster only through output. Invalid JSON/schema is a named refusal; local response-spill failures report `receiver_observation_storage_failed`. These routes report the bound server's resolved authentication,
allowed origins and remote policy. Before bind or after shutdown, policy remains
unknown rather than being inferred from defaults. Status never includes bearer
token values. Its nonsecret identity file may be readable by other users, but must be an owner-controlled regular file without group/other write permission. The token remains owner-only. Descriptor reads refuse symlinks, and local identity/token failures are named before any receiver connection is opened.

## Control-plane browser boundary

Browser capture has a local receiver and an unpacked extension, but it does not
require Polylogue to borrow the operator's authenticated browser. For web-shell
or extension debugging, keep these paths distinct:

- agent work uses the existing Chrome only through Sinnix's `agent-window` control boundary, which parks proof windows on the named `agentbrowser` workspace;
- no declared Polylogue browser proof creates or copies a browser profile, launches Chrome or Chromium, or allocates a private CDP port.

The deployment fallback is useful for proving that the deployed daemon can
serve the web root to a real browser engine:

```bash
agentctl job start polylogue deployment_browser_smoke --workspace <workspace-id>
agentctl job result <job-id>
```

That smoke does not certify MCP browser launch, extension ids,
authenticated ChatGPT/Claude.ai pages, or shared-browser cookies.

Accepted captures are typed browser-capture envelopes and are written
atomically under the configured capture spool at `<provider>/...json`.
The filename is deterministic from provider and provider session id, so repeated
observation of the same web session replaces the same source artifact. Receiver
responses expose this artifact as an `artifact_ref` relative to the spool root;
absolute filesystem paths stay inside the receiver process.
A response carries a typed `outcome`: `accepted` publishes the submission, `noop` retains semantically identical content, and `superseded` retains a different resident revision. `submitted_content_hash` binds the receipt to the exact posted bytes; `content_hash`, `capture_id` and `accepted_identities` identify the retained artifact. Backfill completes accepted/noop captures using the retained hash. It retires superseded items explicitly, releases their retained envelopes and does not advance the incoming provider revision. A superseded response is not proof that its incoming turns were captured.

Snapshot convergence compares attachment occurrences within their observed
owner and provider attachment ID. Stable provider descriptors and compatible
known byte evidence guard replacement: an absent size or supported encoded
byte carrier (`content_base64`, `inline_base64`, or `data`) may be enriched,
while unequal known sizes, malformed carriers, or different known bytes for
the same occurrence remain a conflict. Native-plan acquisition outcomes, content
hashes, and raw-revision ordinal coordinates remain in the accepted envelope
but do not redefine the provider attachment across snapshots when a stable
native message owner is available. Attachments without that owner retain
their declared ordinal constraints. Repeated attachment IDs on different owner messages remain separate occurrences.

Every receiver response carries `X-Request-ID`. If the extension or a local
debug probe sends a safe `X-Request-ID` header, the receiver echoes its
sanitized value; otherwise it generates one. Receiver logs use the same id for
origin rejection, token rejection, malformed payloads, write failures, accepted
captures, and request timing. During branch-local extension work, copy that id
from the browser network panel or curl output into the run-local daemon log to
connect UI action, receiver decision, artifact ref, and duration.

The extension popup records a redacted debug log for the same lifecycle. Each
entry is timestamped and carries stage, method/path, request id, receiver
request id, provider/session identity, archive state, and error metadata where
available. It does not retain transcript text, raw provider payloads, message
bodies, or turn arrays. Operators should export this packet when debugging
capture behavior; it is the intended bridge between visible popup state,
service-worker requests, receiver responses, and daemon convergence evidence.

The popup refreshes status and receiver health automatically on open and while
it remains open. Receiver configuration and pairing writes share the worker's
existing storage mutation owner. A delayed health response cannot update pairing
after configuration changes. Isolated proof cleanup restores its exact owned
snapshot through that same configuration owner and waits for earlier pairing
writes; concurrent independent settings remain untouched. Capture, retry, focused-tab observation, open-conversation
convergence, and provider-inventory freshness are background invariants, not
operator controls. The primary surface exposes only genuine user intents such
as copying a stable conversation reference or opening its archive view.
Ordinary webpages and provider landing/project pages without a selected
conversation are neutral non-conversation states: they never inherit stale
fidelity metadata or appear as failed/unsupported conversations.

The extension lives in `browser-extension/` and can be loaded unpacked in
Chrome. It includes ChatGPT and Claude.ai provider adapters, a popup control
panel, receiver configuration, automatic current-page capture, badge state, and
archive-state feedback. Provider adapters should prefer provider-native
structured page/app payloads where available and use DOM text extraction only
as a compatibility fallback. The shared envelope carries session, turn,
attachment, provenance, and provider metadata semantics.

### Background inventory delta

An explicitly started background job uses a ChatGPT, Claude.ai, or Grok provider
adapter with four operations: inventory enumeration, native fetch, response
classification, and normalized capture. IndexedDB stores the inventory cursor,
queue state, attempts, eligibility deadline, lease owner/expiry, fidelity,
submitted envelope, and receiver receipt. MV3 alarms resume eligible work after
service-worker termination; expired leases return to their prior state.

Provider HTTP runs through a strict first-party page broker. The service worker
retains durable scheduling, budgets, and receiver ACK ownership, while an
existing or inactive provider tab supplies ephemeral authenticated context.
Page transport frames carry the originating installation's actual runtime ID
in their scoped protocol type. MAIN admission, stream chunks, ACKs and cancellation
remain bound to that owner; an older or foreign installation cannot answer a
candidate request. Explicit native captures and backfill supply that owner even
when automatic capture is paused. Passive ChatGPT observations retain a separate
borrowed clone for each registered enabled, paired owner. Offline pairing keeps
its original identity and queued delivery; it is not treated as unpaired. A
stale ownerless fetch wrapper cannot admit work through the current helper.

ChatGPT access-token and selected-account values never leave MAIN world;
Claude requests must match the organization selected by the UI. Only fixed
inventory and native-conversation operations are allowed. Responses stream into
immutable OPFS files; the scripting result carries a file reference and HTTP
metadata. Auth/challenge classification and response-shape drift remain explicit;
HTTP 200 without proven page context is not accepted as an empty inventory.
Claude jobs pin their initial UI-selected organization so a later account
switch cannot mix one organization's inventory with another organization's
native conversations. A `selected_organization_stale` pause requires cancelling
the old job and starting a new one; Resume is appropriate only after restoring
the original selection.

The default provider concurrency is one. Token cadence is conservative and
learned upward after throttling. `Retry-After` is authoritative, retry delays
use exponential full jitter, and every representable provider Retry-After
remains authoritative, including deadlines longer than a day. Auth/challenge
requires operator action; transport failures and 429s retain eligible work for
retry. Per-wake and daily request budgets govern scheduling. Receiver downtime
never marks an item complete or forces a second provider fetch: a durable body
reference waits for an idempotent receiver ACK.

An IndexedDB execution lease serializes inventory and native-fetch requests as
well as queue ownership. The coordinator atomically reserves provider budget
before network I/O. Active execution renews the job and item leases; expiration
recovers work after worker loss. Pause/cancel aborts and drains the active
provider read, increments the job generation, and updates queue entries, so a
late response cannot restore stale running state. Successful ACK finalization
atomically updates job, queue, and the provider-native revision ledger. A later
job skips only an exact trusted `(provider, native id, updated_at)` revision;
records without a revision are fetched. Retries do not discard acquired evidence
or stop after an attempt count. Actual storage quota failure is visible and
preserves existing delivery references.

The one-time disposable queue conversion makes old, unacquired byte-limit
refusals eligible again while preserving paused jobs. Acquired evidence on
such a row remains explicitly held for recovery. Database publication is atomic;
checkpoint conversion applies the same rule and never resumes provider traffic.
Foreground publication assigns a durable sequence in the queue transaction, so
equal timestamps preserve FIFO across worker restart; due scheduling uses a
separate index.

Backfill scheduling and progress fold indexed queue cursors without materializing
the ledger. The popup pages active job metadata and displays the complete active
count; paging does not pause, resume, or discard jobs.

The existing IndexedDB acquisition and delivery owners retain native reply files
and the receiver's immutable preparation reference. Native prose is normalized
by the ordinary canonical provider parsers at the receiver, using their existing
scratch-backed routes. The receiver CaptureJob registry retains exact raw
members, an unfinished envelope prefix, a canonical attachment plan, occurrence
receipts, and the completed envelope. The browser pages the attachment plan and
uses the original document's page credentials to acquire bytes. Each receipt
binds the plan digest, descriptor digest, and attachment ordinal. Duplicate
provider IDs retain separate occurrences; provider IDs alone do not grant an
attachment owner. Bytes already embedded in the native reply have an explicit
`retained_native_bytes` receipt naming their canonical digest and size.

The final suffix is sealed only after every planned attachment has a terminal
receipt. Publication rechecks the same lease and explicit hold/cancel fence in
the registry writer transaction, invokes the shared capture admission route,
and retains its acknowledgement before responding. Foreground completion and
backfill queue retirement use this acknowledgement. A retry after response loss
publishes the same retained artifact and obtains the same durable receipt.
Committed raw, prefix and asset custody survive cancellation and lease takeover.
A cancelled native acquisition is terminal even without a publication receipt.
Its artifacts retire only when the owning job is explicitly eligible and
non-authoritative, its retry state is completed or abandoned, its checkpoint
is acknowledged, and its lease has expired. Unpublished live acquisitions
continue to protect their custody from job collection.
Only an already admitted physical upload can renew its unchanged expired lease
when actual byte progress resumes; a later request must prove a current lease.
No timer turns a slow producer pause into failure.

Account backfill uses its positively derived `h1` scope. Foreground preparation
uses a separate invocation capability in the same registry. Preparation instance
and token identify the current operation; missing original instance, acquisition
sequence, or invocation stay absent. An old acquisition is not attributed to the
current browser profile or current tab. Grok retains the named conversation,
responses, and optional response_nodes replies; Claude retains its original
attachments, files, and extracted content. The standalone receiver retains a
complete browser capture artifact for ordinary daemon ingestion.

The browser still materializes selected header scalars, and canonical parsing,
scratch JSON, SQLite values and FTS retain whole-value consumers. This route does
not establish memory independent of the largest scalar or record. File transfer
measurements and canonical preparation/lowering qualification are separate.

Claude native captures retain attachments, files, and extracted content in the
original reply. The ordinary Claude parser owns their lineage order and message
identity; the extension does not create a second attachment projection when it
has acquired no additional bytes. Native envelope spool serialization preserves
provider mapping insertion order. Structural admission fingerprints continue to
ignore mapping-key order.

Retained payloads bearing the manufactured `chatgpt-native-compact-v1` bridge
projection marker are refused as an invalid capture format. Their lossy synthetic
nodes cannot establish provider-native provenance; retained evidence is not
rewritten or deleted. Original provider mappings and opaque provider metadata
remain supported.

After durable receiver ACK, cleanup retires delivery and record references. A current document's raw
revision cache remains pinned until supersession or document loss; acquired assets
follow that pin and independent unacknowledged record references. Cleanup checks
these indexed roots before deleting files. Native reply recovery uses the same durable owner,
not a page-local replay list. Grok reserves one operation identity before its
named endpoint reads, publishes each immutable reply into that operation, and
transfers all reply ownership into the normalized queue entry atomically. A
restart reuses only that same conversation and operation; incomplete acquired
evidence remains visibly pending. Cache publication uses a durable acquisition
sequence so equal-clock, delayed header parsing cannot replace a newer revision.

Foreground callers sharing one durable delivery reference share one physical
upload. Cancelling one caller leaves other callers' upload intact; the last
caller drains physical cancellation. Admission during that drain waits its
settlement before using the ACK or starting a new retained attempt. A valid ACK
returns before cleanup queued behind another delivery, and a failed archive-state
refresh remains explicit without replacing the accepted capture result.
The same sequence travels with the background-owned extension instance in capture
provenance. Receiver admission compares counters only for that same instance and
provider session, after provider revision and fidelity precedence. Observation
fields do not change structural fingerprints. Native replies retain their actual
acquisition timestamp through delayed normalization. Gemini reserves the durable
observation before reading its DOM snapshot; cancellation releases an unpublished
claim. Historical envelopes without a sequence retain their declared timestamp
evidence rather than receiving an invented acquisition order.
Provider-page refusal responses retain a bounded recent diagnostic window and
report its discarded count in `native_attempts_dropped`. This diagnostic retention
does not refuse capture or stop retries.
Sealed single-response acquisitions and ready Grok bundles retain independent
roots until normalization publishes its own custody. Losing the original document
does not discard those bytes. The popup reports pending acquisitions separately
from ready deliveries. Startup serializes already-normalized captures; it does not
start provider or attachment acquisition. An authorized capture in the same owned
conversation resumes earlier ready bundles before publishing its newer snapshot.
Foreground delivery drains use cursors,
and popup pages report the actual total. Conversion of acquired inline retry state
publishes its complete staged body and durable replacement reference before
retiring the original entry; an interruption resumes the same conversion.
Already-materialized DOM turns and inline attachments publish their records,
preparing body root, and delivery reference in one IndexedDB transaction.
Cancellation or a record-write failure before that commit rolls back the whole
publication. After commit, restart resumes serialization from those exact records,
including a failure before the OPFS writer opens. This does not bound the memory
used to acquire the original DOM snapshot.

The incremental native tokenizer retains its largest token or selected provider
record. A title, message string, tool argument, attachment descriptor, or individual
native record can be arbitrarily large. Those allocations and browser-native File
upload buffering remain memory limitations; the implementation does not impose a
size refusal or claim a bounded end-to-end memory footprint.

This workflow repairs gaps relative to an immutable export/GDPR baseline and a
user-selected cutoff. It honors provider authentication and controls and cannot
prove completeness beyond the authenticated inventory returned by the provider.
It never bypasses anti-bot checks, discovers deleted/ephemeral records absent
from inventory, or activates an operator foreground tab.
ChatGPT enumeration traverses all active/starred/archived flag partitions and
deduplicates repeated native ids in the durable queue.

### Automatic freshness convergence

An archived identity is not treated as proof that a growing provider
conversation is current. The extension maintains a durable freshness
queue keyed by `(provider, native conversation id)`. Open-page transcript
mutations, observed provider-native detail responses, ChatGPT `new-message`
service-worker events, newly submitted provider turns, and recurring inventory
head scans are coalesced as wake hints for that one queue. Hints never mark a
conversation complete.

The queue acquires the latest native conversation through the same owned,
inactive first-party transport used by historical backfill, then submits the
ordinary `BrowserCaptureEnvelope` to the canonical receiver. A user head or a
non-terminal assistant head remains queued with adaptive polling; a terminal
assistant head removes the item only if no newer hint arrived while the fetch
was in flight. Leases recover MV3 worker termination, provider warnings and
rate/auth failures use typed backoff. Pending identities remain queued until
they converge or receive an explicit cancellation.

An exact ChatGPT recapture may reuse a signed-in home tab. Before provider
traffic, the background owner reserves the native identity and acquisition
sequence against that tab's document. The invocation token accompanies raw
staging, header inspection and normalization; each boundary verifies the
requested URL and returned native identity. Cancellation closes admission
while the owned provider work drains. Worker restart settles the previous
invocation before admitting another. Retiring the invocation leaves acquired
raw evidence, cache pins and unacknowledged delivery references intact.

Every 15 minutes the extension checks one overlapping ChatGPT inventory
partition head, rotating across active/starred/archived partitions. Exact
provider `update_time` receipts already acknowledged by the backfill revision
ledger are skipped; changed or unknown revisions enter the same freshness
queue at a staggered cadence. An explicit historical backfill remains
authoritative while it is running, so the standing sweep yields instead of
competing for provider requests. The popup projects queued identities, leases,
due times, failures, last sweep state, and per-conversation timeline decisions.
This safety net is what allows action-owned or user-owned conversation tabs to
close after submission without making capture depend on those tabs.

## Dataflow and boundary

```text
ChatGPT/Claude.ai page/app state
  -> extension provider adapter (`browser-extension/src/content/*.js`)
  -> provider-neutral `BrowserCaptureEnvelope` (`browser-extension/src/common.js`)
  -> extension service worker POST `BrowserCaptureEnvelope`
  -> local receiver `POST /v1/browser-captures`
  -> typed Pydantic validation (`polylogue/browser_capture/models.py`)
  -> atomic source artifact write under the browser-capture spool
  -> live source watcher/parser (`polylogue/sources/parsers/browser_capture.py`)
  -> canonical archive session/message/attachment rows
```

The receiver is a capture ingress, not the web workbench API. The daemon status
API only reports component readiness and spool location; it does not accept raw
browser-capture payloads. Route metadata lives in
`polylogue/browser_capture/route_contracts.py` so tests and future OpenAPI/web
surfaces do not infer receiver DTO or auth semantics from handler branches.

## Provider payloads and coalescing

Browser capture is an acquisition path for the same provider sessions that
GDPR/Takeout-style imports already store. It must not create a duplicate
session merely because the evidence arrived through an extension.

| Provider/path | Adapter or source | Stored payload | Parser path | Coalescing key |
| --- | --- | --- | --- | --- |
| ChatGPT authenticated page | `chatgpt-native-v1` | Full `/backend-api/conversation/<id>` JSON under `raw_provider_payload` | delegated to the normal ChatGPT parser | `chatgpt-export:<conversation_id or id>` |
| ChatGPT authenticated page fallback | `chatgpt-dom-v1` | Visible turns in the browser-capture envelope | browser-capture DOM parser | `chatgpt-export:<url /c/<id>>` |
| ChatGPT GDPR/export file | provider import | Export JSON `mapping` payload | normal ChatGPT parser | `chatgpt-export:<conversation_id or id>` |
| ChatGPT shared-link helper | standalone public-share script | React Router share stream reduced to export-shaped messages | separate conversion input, not the extension payload | provider-native share conversation id when converted |
| Claude.ai authenticated page | `claude-ai-native-v1` when the conversation API response is observed; `claude-ai-dom-v1` fallback | Full `/api/organizations/.../chat_conversations/<id>` JSON under `raw_provider_payload`; otherwise visible turns in the browser-capture envelope | delegated to the normal Claude.ai parser for native payloads; browser-capture DOM parser for fallback | `claude-ai-export:<uuid or url /chat/<id>>` |
| Claude.ai GDPR/export file | provider import | `chat_messages` payload | normal Claude.ai parser | `claude-ai-export:<uuid>` |

The native bridges stage the original authenticated response bytes in OPFS.
The existing CaptureJob receiver retains each exact member and prepares canonical
messages, blocks, and an attachment plan through the ordinary provider parsers
and their scratch spill. Exact occurrence receipts complete the attachment
suffix before the immutable full envelope is sealed. The extension publishes
that artifact by reference through the same registry; its durable ACK precedes
retirement of the browser's byte custody. Cache reuse asks this same preparation
owner for the canonical prefix summary before assets or final publication. An
unchanged terminal reply can be reused; a running reply keeps the existing
provider fetch and follow-up path. Provider throttle admission precedes both.
Native preparation uses its separately
retained instance/token. Missing original observation instance, sequence, and
invocation remain null, and the original document and source URL come from the
staged admission rather than the current page. Full titles and other provider
fields remain in literal raw replay and the canonical artifact; preparation
control summaries do not copy full titles.

Extension-only checks use synthetic receiver contracts in the existing npm
environment. Actual JavaScript background-to-Python receiver and canonical
lowering parity is separately selected with
`devtools test tests/unit/browser_capture/test_native_extension_integration.py`
in the Python development environment with extension npm dependencies. These
controls do not qualify whole-scalar parser allocation, session-wide lowering,
or SQLite TEXT/JSON/FTS resource behaviour.

The retained capture envelope also accepts a Codex native record array under
`raw_provider_payload` when `session.provider` is `codex`. Acquisition adapters
can use this existing envelope to associate attachment evidence with an original
Codex transcript. The parser delegates the records to the ordinary Codex parser
and merges the envelope attachments, preserving message fields and fork lineage
on ingestion and retained replay. Invalid Codex record streams are refused. This
is an ingestion contract; the browser extension has no Codex page adapter.
For Codex, ChatGPT and Claude.ai native captures, the parsed native session ID
must match the envelope's declared session ID before attachment or lifecycle
evidence is merged. A disagreement produces a typed refusal.
An attachment turn without a provider message ID must retain an explicit native
turn ordinal and matching role and text. The parser resolves that witnessed
turn through its private message owner coordinate, preserving native IDs and
attachment bytes on replay. Missing or conflicting evidence is refused; a
default ordinal is not ownership evidence.

DOM extraction is a compatibility fallback for pages where no provider-native
payload has been observed yet. It still uses the provider-native conversation id
from the URL, so a later native capture or GDPR import for the same conversation
updates the same archive session instead of creating a second visible session.
Fallback DOM captures are therefore acceptable as temporary live evidence, but
not as a reason to prefer DOM over a clean provider payload.

Temporary chats are a typed browser-capture session property, not only a
provider metadata convention. The extension writes
`session.session_kind = "temporary"` when the page URL or provider-native
payload identifies a temporary conversation, and the parser persists the
existing `capture:temporary-chat` ingest flag from that typed field. Legacy
captures that only carry `provider_meta.session_kind = "temporary"` remain
accepted for already-spooled artifacts.

The Claude.ai content script also hooks same-origin conversation fetches and
uses native payloads when the response is the current
`/chat_conversations/<id>` JSON shape with `chat_messages`. DOM extraction
remains a fallback for pages where the provider response is not available to
the content script.

## Not a schema-inference subject

The browser-capture envelope is authored here: `browser-extension/` writes it
and `polylogue-browser-capture-native-host` reads it. Its structural contract
is therefore a decision, recorded in
`polylogue/browser_capture/models.py` and changed together with its writer --
not evidence to be discovered from a corpus the way a provider's wire format
is. Inferring a schema for it would publish a snapshot of our own model as if
it were an observation.

`polylogue/core/schema_subjects.py` records that as
`inference_excluded_reason` on the `browser-capture` subject, and the
declaration is the authority everywhere it matters:

- `parse_schema_source_input`, `inventory_schema_sources` and `infer_sources`
  refuse the token, so no capture artifact is ever inventoried, counted,
  sampled or reported as eligible-then-unsupported;
- the schema-source frontier refuses a declaration that gives the subject a
  root or a recorded baseline, so the recorded denominator is zero;
- `devtools schema commit --provider browser-capture` refuses instead of
  writing a package;
- `devtools schema frontier --list` prints the subject as declared
  non-applicable with this reason.

The denominator is therefore zero *by declaration*, which is a different and
stronger claim than "every candidate was refused at preflight". There is no
adapter to write; the earlier `browser_capture_adapter_unavailable` refusal
read as missing work and has been retired along with the inferred
`browser-capture` package it justified.

## Local auth and origin policy

Default CORS is extension-only: `chrome-extension://*`. Remote web origins such
as `https://chatgpt.com` and `https://claude.ai` are not accepted by default; the
extension service worker sends requests from its own extension origin after the
content adapter builds the envelope. Extra web origins require an explicit
receiver auth token. This keeps a normal web page from writing local capture
artifacts or reading receiver state merely because it is open in the browser.

A fresh extension profile obtains the receiver bearer from the native-messaging
host (`polylogue-browser-capture-native-host`). Loopback is not identity: any
local process can hold the receiver port while the daemon is stopped. The host
therefore sends a fresh 32-byte challenge to the endpoint's
`POST /v1/receiver/attest` and releases the bearer only when the answer is the
HMAC-SHA256, keyed by that bearer, over the receiver identity and the challenge.
The bearer itself never crosses the socket during this check. This possession proof alone does not establish endpoint ownership against a forwarding relay; native bootstrap transport remains a separate follow-up (`polylogue-xgj34`). An endpoint that
does not answer yields `receiver_unreachable`; one that answers with anything
else yields `receiver_authentication_failed`. Local response staging or spill failures yield `receiver_observation_storage_failed` in the native-messaging error envelope; they are distinct from peer reachability or authentication. When a status probe is refused
with `401`, the extension asks the host for the current bearer once per health
check; a second refusal is reported as `unauthorized` rather than retried.

If the receiver is unavailable, the extension surfaces an offline state instead
of dropping content silently.

## Branch-local extension proof modes

Use the declared `dev_loop_proof` AgentCTL operation when changing receiver, extension, or provider adapters from a branch. It binds the proof to a managed checkout; the Polylogue child selects its own loopback API and receiver ports and isolates XDG configuration. The proof checks the shared-Chrome control boundary with one owned `agentbrowser` target, proves receiver authentication and deterministic provider capture, then reports archive and API convergence through the canonical job result. See [`docs/dev-loop.md`](dev-loop.md) for the start, wait, and result commands.

Live shared-Chrome proof runs only through the declared `live_provider_proof`
AgentCTL operation. It must not create an alternative Polylogue daemon
lifecycle, ad hoc receiver lease, free CDP port, or direct Chrome launcher. It
uses Sinnix's `agent-window` control boundary in the running authenticated
browser, verifies each proof window hidden on `agentbrowser`, and closes
only proof-created targets. Select exact conversations through a private
`--conversations-file`; see [the live proof contract](dev-loop.md#shared-chrome-live-provider-proof).
Its standalone receiver verifies admitted native bytes without opening the archive,
and automatic capture remains paused after cleanup.

## Current residual map for #1824 / #1847

Fixed in this slice: default web origins no longer cross the local receiver
boundary unauthenticated; receiver routes have a small executable contract;
valid and malformed receiver payloads are tested at the HTTP boundary; accepted
and error DTOs carry receiver/schema/source identity; accepted and archive-state
DTOs use bounded artifact refs instead of local filesystem paths.

Still outside this slice: browser-capture artifacts still enter the archive as
provider sessions (`chatgpt`, `claude-ai`) rather than a distinct acquisition
source family; the daemon web API exposes status/read surfaces but not a full
web workbench flow; extension-id pinning is still operator policy rather than a
default because unpacked extension ids are local-install specific.

### Retained checkpoint inspection

CaptureJobs own current account-scoped recovery. The retired per-instance
checkpoint mirror is no longer written or read as a recovery fallback.
Existing files remain intact under the receiver's retained artifact root.
`GET /v1/capture-jobs/orphans?client_protocol=2` identifies their exact byte
digests. An authenticated
`GET /v1/capture-jobs/orphans/{source_digest}/payload?client_protocol=2`
streams the original bytes without assigning an account or consuming custody.
Historical account scope remains unresolved unless positive ownership proof
exists; the current browser account alone is insufficient. Inspection preserves
paused jobs and acquired references, and does not start provider traffic.

Current checkpoints use `canonical-artifact-v1`, declared by the CaptureJob
capabilities response. `PUT /v1/capture-jobs/{job_id}/checkpoint` sends a
file-backed body containing CAPTURE canonical JSON and an
`X-Polylogue-Checkpoint` descriptor. The descriptor contains the provider token,
43-character account pseudonym and lease proof, UUID request and lease identities,
safe-integer protocol/revision/generation/sequence values, and the exact SHA-256
digest. It contains no ledger records. The receiver compares the semantic
canonical digest, the exact received-byte digest, and the declared digest.
Whitespace, alternate escaping, numeric spelling and noncanonical key order
are refused instead of being acknowledged under a different byte identity.
Generic CAPTURE JSON shapes remain valid checkpoint payloads; the extension's
ledger recovery requires its declared jobs, queue and revisions arrays.

The browser snapshots those records and their borrowed custody in one
IndexedDB transaction, commits, then serializes an immutable OPFS file. Receiver
summaries contain only its digest, sequence and size. Authenticated artifact
reads use `/v1/capture-jobs/{job_id}/checkpoint-artifacts/{digest}` with the
same descriptor and a current account-scoped lease. Checkpoint replacement
retains older receipt roots. A read holds its exact inode while terminal-job
collection checks roots and active reads before deletion. Interrupted conversion
of the shipped SQLite checkpoint column preserves the original row until the
verified artifact and its reference commit. Restore stages records before
atomic publication and preserves existing local acquired custody and operator
pause state. An unchanged restored record can advance only with verified
receiver checkpoint evidence; same-job sequences are compared within that job,
and independent job snapshots use receiver receipt time. A changed local record
or acquired body keeps its existing authority. Ambiguous source evidence is a
visible refusal, with the received artifact retained. A published checkpoint
keeps its exact account, job, sequence and digest witness after artifact cleanup,
so discovering it again cannot resurrect locally retired records.

The old Chrome-storage ledger is read only for one-time conversion. Each inline
acquired envelope is staged and verified before its replacement queue reference
and conversion marker commit; the original ledger remains until every record is
published. Restart checks those markers before staging again. Accountless
historical jobs remain paused with their custody preserved. Automatic recovery
after a provider tab appears applies to positively account-scoped receiver
checkpoints. Conversion and restore do not start provider acquisition.

The vetted tokenizers still allocate one complete scalar or selected record.
Accepted prose, tool arguments and extracted attachment text can be unbounded;
these allocations and Chromium's file-upload buffering remain unmet
memory criteria. Streaming the checkpoint tree does not establish a bound for
those values.

Backfill export freezes queue rows in its IndexedDB snapshot transaction. Each
acquiring file freezes its committed prefix in a durably owned evidence copy;
these are separate publication points rather than one atomic snapshot time for
all files. The descriptor is returned only after every evidence copy and the
complete export artifact seal. Interrupted copying retains its original metadata
and source custody, and a published copy is reused after source append or seal.
The external JSON format is `polylogue.backfill-export.v2`: `ledger` preserves
the snapshotted queue rows, and `acquired` carries each queue item's native capture
metadata and exact staged envelope, raw response, and acquired asset bytes.
Each file includes byte length, SHA-256, and base64 parts. An interrupted
acquisition exports its committed prefix with its original acquisition metadata;
it does not invent a complete provider response. Append and seal share the file
owner while those immutable parts are copied. Every complete file's bytes are
verified against its declared digest. Export does not request provider data.
The popup opens the browser file picker directly from the export click, streams
and hashes the artifact into the selected file, and acknowledges the exact
snapshot only after the writable file closes. Cancellation, write failure or a
missing ACK leaves a visible pending export which reuses the same snapshot on
retry. Export ACK releases its snapshot records and artifact; borrowed queue,
native and asset custody remains owned by its original delivery/cache roots.
The popup requests only a bounded descriptor, never a complete ledger string.

CaptureJob discovery returns `jobs`, `total`, `cursor`, and `has_more` from one
page snapshot. Its cursor contains immutable `created_at` and `job_id`; lease
adoption and checkpoint replacement cannot move an existing job across the
cursor. Recovery consumes pages sequentially and retains a held lease as a
visible pending checkpoint. Orphan census uses the same response metadata with
`orphans` and a `source_digest` cursor, including unreadable retained custody.
Each row preserves its diagnostic message and reports `errno_class` from the
current failed read, or `null` when no exception observation survives.
There is no maximum number of recoverable jobs or orphan records.

Browser action attachments are uploaded as binary bodies with `Content-Length`
before enqueue. The action request carries `attachment_ref`, `name`, and
`mime_type`; inline base64 input is refused. The CLI streams each open input
file into the same receiver-owned input store. Acknowledged upload references
retain their original bytes independently of action delivery.

The extension streams the verified attachment download into the original owned
provider document in 64 KiB chunks, hashing incrementally before making a File
available to the composer. It removes transient page parts after success or
failure. The advertised `attachment_chunk_bytes` describes a transfer unit,
not a maximum attachment size. Provider upload constraints remain observable
provider failures; Polylogue does not impose the former 16 MiB attachment cap.

Receiver configuration, pairing-code exchange, reset, and health-result persistence
share the worker's storage mutation owner. A response from an earlier
configuration cannot recreate its pairing or credential after configuration
changes. The provider proof snapshots the endpoint, token, and pairing together
and restores them through `polylogue.configureReceiver`; it waits for an
in-flight configuration mutation before restoration. Its configure response
carries the admitted configuration revision, so restoration refuses independent
reset or configure even when endpoint and token values match. Restoration does not resume automatic capture.

Configured receivers pin the current owner-only persisted token for each HTTP operation, so `browser-capture token show --rotate` immediately invalidates the previous bearer without restarting the receiver. Direct library servers retain their explicit in-memory credential authority. An unavailable persisted credential produces `receiver_credential_unavailable` (503), without a frozen-token fallback.

Authenticated status first obtains `/v1/receiver/status-challenge`, a receiver-issued nonce owned by that kept-alive connection. `/v1/receiver/status-attest` consumes it once before checking the signed request. The client prohibits reconnecting before attestation; recorded requests cannot authorize another connection or a restarted receiver. Local native SQL settlement failures produce `receiver_observation_storage_failed` and retain the original cleanup owner.

Pairing-code redemption validates and consumes the code, then returns the receiver’s pinned token; it does not consult or mint an unrelated default credential. A receiver with authentication disabled refuses redemption with `receiver_auth_disabled`. The local `browser-capture action` command enqueues through its spool owner and has no receiver authentication options or credential publication side effect. Lazy JSON read failures are classified at the SQLite view producer; renderer and output exceptions retain their original identity.
