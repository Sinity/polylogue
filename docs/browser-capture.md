# Browser Capture

Polylogue can receive browser-observed LLM sessions through a local-only
Manifest V3 extension.

Start the receiver:

```bash
polylogue browser-capture serve
```

The receiver listens on `127.0.0.1:8765` by default and accepts:

- `GET /v1/status`
- `GET /v1/archive-state?provider=chatgpt&provider_session_id=...`
- `POST /v1/browser-captures`

Accepted captures are typed browser-capture envelopes and are written
atomically under the configured inbox at `browser-capture/<provider>/...json`.
The filename is deterministic from provider and provider session id, so repeated
observation of the same web session replaces the same source artifact.

The extension lives in `browser-extension/` and can be loaded unpacked in
Chrome. It includes ChatGPT and Claude.ai DOM adapters, a popup control panel,
receiver configuration, current-page capture controls, badge state, and
archive-state feedback. Provider adapters are intentionally thin; the shared
envelope carries session, turn, attachment, provenance, and provider metadata
semantics.

The transport is local-only. Browser origins are allowlisted to ChatGPT,
Claude.ai, and extension pages. If the receiver is unavailable, the extension
surfaces an offline state instead of dropping content silently.
