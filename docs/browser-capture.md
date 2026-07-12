# Browser Capture

Polylogue can receive browser-observed LLM sessions through a local-only
Manifest V3 extension.

Start the receiver:

```bash
polylogued browser-capture serve
```

For normal long-running local service use, `polylogued run` starts the browser
capture receiver together with live source watching.

The receiver listens on `127.0.0.1:8765` by default and accepts the route contracts in `polylogue/browser_capture/route_contracts.py`:

- `GET /v1/status` -> `BrowserCaptureReceiverStatusPayload`
- `GET /v1/archive-state?provider=chatgpt&provider_session_id=...` -> `BrowserCaptureArchiveStatePayload`
- `POST /v1/browser-captures` with `BrowserCaptureEnvelope` -> `BrowserCaptureAcceptedPayload` or `BrowserCaptureErrorPayload`

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

The payload includes bounded archive evidence (`raw_row_exists`, `raw_id`,
`indexed_session_exists`, `indexed_session_id`, `indexed_message_count`) and a
relative `artifact_ref`. It must not expose absolute paths. Deployment smoke
uses this endpoint as an invariant check: a receiver that says `captured: true`
without raw/index/message evidence is considered broken, not merely stale.

Inspect the receiver target directly with `polylogued browser-capture status`,
include it in the daemon component summary with `polylogued status`, or include
the same component status in archive health output with `polylogue ops doctor
--daemon`.

## Control-plane browser boundary

Browser capture has a local receiver and an unpacked extension, but it does not
require Polylogue to borrow the operator's authenticated browser. For web-shell
or extension debugging, keep these paths distinct:

- an agent-private Chrome/MCP browser can inspect the local workbench when the
  local control plane provides one;
- the operator's live browser/cookies are used only after explicit approval and
  should be copied into an ignored local profile, never into CI or cloud agents;
- `devtools workspace deployment-smoke --browser` launches a fresh headless
  Chrome/Chromium profile and reports the executable it resolved.

The deployment fallback is useful for proving that the deployed daemon can
serve the web root to a real browser engine:

```bash
devtools workspace deployment-smoke --browser --browser-executable "$(command -v google-chrome)"
```

That smoke does not certify private MCP browser launch, extension ids,
authenticated ChatGPT/Claude.ai pages, or copied-profile cookies.

Accepted captures are typed browser-capture envelopes and are written
atomically under the configured capture spool at `<provider>/...json`.
The filename is deterministic from provider and provider session id, so repeated
observation of the same web session replaces the same source artifact. Receiver
responses expose this artifact as an `artifact_ref` relative to the spool root;
absolute filesystem paths stay inside the receiver process.
An accepted response also carries `content_hash`, the SHA-256 of the exact
UTF-8 JSON request bytes. The receiver returns it only after the atomic spool
write succeeds. Background backfill therefore treats a response as a durable
ACK only when both `X-Request-ID` and the expected content hash match.

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

The popup refreshes status automatically on open and while it remains open.
Manual **Check status** is only an explicit refresh. Button controls expose
busy/success/failure states so a click has visible feedback even when archive
convergence takes several seconds.

The extension lives in `browser-extension/` and can be loaded unpacked in
Chrome. It includes ChatGPT and Claude.ai provider adapters, a popup control
panel, receiver configuration, current-page capture controls, badge state, and
archive-state feedback. Provider adapters should prefer provider-native
structured page/app payloads where available and use DOM text extraction only
as a compatibility fallback. The shared envelope carries session, turn,
attachment, provenance, and provider metadata semantics.

### Background inventory delta

An explicitly started background job uses a ChatGPT or Claude.ai provider
adapter with four operations: inventory enumeration, native fetch, response
classification, and normalized capture. IndexedDB stores the inventory cursor,
queue state, attempts, eligibility deadline, lease owner/expiry, fidelity,
submitted envelope, and receiver receipt. MV3 alarms resume eligible work after
service-worker termination; expired leases return to their prior state.

The default provider concurrency is one. Token cadence is conservative and
learned upward after throttling. `Retry-After` is authoritative, retry delays
use exponential full jitter, and repeated 429/403/challenge or transport
failures open a fail-paused circuit. Queue size, captures per wake, and daily
request count are bounded. Receiver downtime never marks an item complete and
does not force a second provider fetch: the native-full envelope waits durably
for an idempotent receiver ACK.

This workflow repairs gaps relative to an immutable export/GDPR baseline and a
user-selected cutoff. It honors provider authentication and controls and cannot
prove completeness beyond the authenticated inventory returned by the provider.
It never bypasses anti-bot checks, discovers deleted/ephemeral records absent
from inventory, or activates an operator foreground tab.

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

The ChatGPT content script hooks same-origin fetches for authenticated
conversation JSON and keeps only a bounded recent window in page memory. On
capture it sends that provider-native JSON as `raw_provider_payload`; the
browser-capture parser then delegates to the same ChatGPT parser used by direct
imports. This is the preferred path because it preserves branches, current
node, message ids, timestamps, model metadata, attachments, and other fields the
visible DOM cannot reliably expose.

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

## Local auth and origin policy

Default CORS is extension-only: `chrome-extension://*`. Remote web origins such
as `https://chatgpt.com` and `https://claude.ai` are not accepted by default; the
extension service worker sends requests from its own extension origin after the
content adapter builds the envelope. Extra web origins require an explicit
receiver auth token. This keeps a normal web page from writing local capture
artifacts or reading receiver state merely because it is open in the browser.

If the receiver is unavailable, the extension surfaces an offline state instead
of dropping content silently.

## Branch-local extension proof modes

Use `devtools workspace dev-loop` when changing the receiver, extension, or
provider adapters from a branch. The branch-local loop owns distinct proof
levels, from cloud-safe synthetic checks to local workstation evidence:

- `--extension-smoke` imports the real background worker with a Chrome API mock
  and proves receiver auth rejection, receiver status, and accepted capture
  writes without a GUI browser.
- `--browser-provider-smoke` loads the unpacked extension into headless
  Chrome/Chromium, maps deterministic ChatGPT and Claude fixture pages onto
  their real supported origins, and proves content-script capture plus
  receiver request-id/artifact evidence without cookies.
- `--browser-live-proof` is explicit operator-local evidence for authenticated
  copied-profile work. It opens a visible Chrome/Chromium with an
  operator-approved copied user-data-dir, live ChatGPT/Claude conversation
  URLs, and the unpacked extension; it writes a redacted proof summary and keeps
  any raw captured content inside ignored local receiver spool artifacts.

For a non-authenticated Chrome-family binary check, `--browser-smoke` sits
between those two layers: it loads the unpacked extension into a fresh headless
profile and proves that the MV3 service worker can talk to the branch-local
receiver. It still does not exercise provider cookies or live pages.

Use [`docs/visual-evidence.md`](visual-evidence.md) for the repo-owned reader
DOM evidence lane and [`docs/dev-loop.md`](dev-loop.md) for the full
branch-local browser proof ladder. Design screenshots or store media are not
accepted as browser-capture verification unless they are tied to one of those
commands and a run-local artifact.

Generate the copied-profile checklist first:

```bash
devtools workspace dev-loop --browser-plan
```

Then run the live proof only from a local workstation with a copied profile:

```bash
devtools workspace dev-loop --browser-live-proof \
  --browser-live-profile-dir .local/browser-profiles/<run-id>-chrome-user-data \
  --browser-live-chatgpt-url https://chatgpt.com/c/<conversation-id> \
  --browser-live-claude-url https://claude.ai/chat/<conversation-id>
```

The live proof refuses CI by default and rejects common live profile roots or
Chrome singleton lock files unless a local operator explicitly overrides the
guardrail. Summaries redact source URLs and provider session ids, omit raw turn
text, and record provider/adapter identity, role coverage, receiver request ids,
and spool artifact refs.

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
