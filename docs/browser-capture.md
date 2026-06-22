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

Inspect the receiver target directly with `polylogued browser-capture status`,
include it in the daemon component summary with `polylogued status`, or include
the same component status in archive health output with `polylogue ops doctor
--daemon`.

Accepted captures are typed browser-capture envelopes and are written
atomically under the configured capture spool at `<provider>/...json`.
The filename is deterministic from provider and provider session id, so repeated
observation of the same web session replaces the same source artifact. Receiver
responses expose this artifact as an `artifact_ref` relative to the spool root;
absolute filesystem paths stay inside the receiver process.

Every receiver response carries `X-Request-ID`. If the extension or a local
debug probe sends a safe `X-Request-ID` header, the receiver echoes its
sanitized value; otherwise it generates one. Receiver logs use the same id for
origin rejection, token rejection, malformed payloads, write failures, accepted
captures, and request timing. During branch-local extension work, copy that id
from the browser network panel or curl output into the run-local daemon log to
connect UI action, receiver decision, artifact ref, and duration.

The extension lives in `browser-extension/` and can be loaded unpacked in
Chrome. It includes ChatGPT and Claude.ai provider adapters, a popup control
panel, receiver configuration, current-page capture controls, badge state, and
archive-state feedback. Provider adapters should prefer provider-native
structured page/app payloads where available and use DOM text extraction only
as a compatibility fallback. The shared envelope carries session, turn,
attachment, provenance, and provider metadata semantics.

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

## Local auth and origin policy

Default CORS is extension-only: `chrome-extension://*`. Remote web origins such
as `https://chatgpt.com` and `https://claude.ai` are not accepted by default; the
extension service worker sends requests from its own extension origin after the
content adapter builds the envelope. Extra web origins require an explicit
receiver auth token. This keeps a normal web page from writing local capture
artifacts or reading receiver state merely because it is open in the browser.

If the receiver is unavailable, the extension surfaces an offline state instead
of dropping content silently.

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
