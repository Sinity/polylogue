# MK2 corrections before implementation

The MK2 artifact is useful, but do not copy it literally.

## Keep

- Four-surface boundary: `polylogue`, `polylogued`, `polylogue-mcp`, `devtools`.
- Shared query/read/status/derived kernel.
- Query-first CLI examples.
- Dynamic completion/fuzzy selector concept.
- Local web reader layout.
- TUI/dashboard as operator cockpit.
- Cold cockpit/dense local-first visual style.

## Change

- Remove every `--json` mention; use `--format json` only.
- Do not add `POLYLOGUE_DAEMON=1` feature flag.
- Do not remove `polylogue watch` until `polylogued` owns equivalent live ingestion.
- Do not remove `browser-capture serve` until receiver endpoints move into `polylogued`.
- Do not introduce `polylogue ui`.
- Do not make `insights` a detached top-level replacement for `products` without resolving derived-data placement.
- Rename the web inspector's `messages` tab to `outline`; messages belong in the main reader.
- Replace `telemetry` wording with `daemon metrics` or `status counters`; nothing leaves the machine.
- Use current open issue refs, not stale MK2 refs.
- Production web bundle must not use CDN React/Babel.

## Current implementation-sensitive facts

- `messages` and `raw` are deliberate read surfaces; keep them.
- `raw` can remain the CLI verb, but API/internal naming should be truthful: raw archive artifacts unless true provider records exist.
- `dashboard` already exists as a TUI concept; do not add another TUI command casually.
- Browser capture default in the human decision memo is automatic with prominent visible state. If open issue text still says manual default, add an explicit issue comment before implementing capture-mode behavior.
- The public alias policy is no aliases, no compatibility windows.

## Suggested issue-comment patch

Add this to the new surfaces issue or #635:

```text
Surface decision update from MK2 integration:

Preserve the query-first CLI grammar. Do not replace it with a generic task tree.
Move service/protocol modes out of `polylogue`: `polylogued` for live/capture/local API/web, and `polylogue-mcp` for MCP stdio.
`polylogue dashboard` remains the terminal operator cockpit unless deliberately clean-break renamed.
Do not add `polylogue ui`.
Do not expose `polylogue mcp` once `polylogue-mcp` lands.
Do not expose `polylogue watch` once `polylogued` owns live ingestion.
Do not expose `polylogue browser-capture serve` once the receiver is daemon-owned.
Do not replace `products` with a detached `insights` command. Derived data should appear inline in conversation/session views and through concrete aggregate nouns, with only a compact `derived status/export` admin surface if required.
No aliases and no `--json`; use `--format json` only.
```
