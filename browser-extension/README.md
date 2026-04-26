# Polylogue Browser Capture

This is a local-first Manifest V3 extension for capturing supported web LLM
sessions into Polylogue.

Run the receiver:

```bash
polylogue browser-capture serve
```

Then load this directory as an unpacked Chrome extension. The popup shows
receiver state, current-page support, archive capture state, the last accepted
capture, and the configured local receiver URL. Captures are posted to
`http://127.0.0.1:8765/v1/browser-captures`, written into the normal Polylogue
inbox, and parsed by the normal source dispatch path on the next ingest run.

The extension does not send session content to any remote service. It only
talks to the local receiver. When the receiver is unavailable, the extension
badge and popup report the offline state. ChatGPT and Claude.ai are the minimum
supported provider adapters in this slice.
