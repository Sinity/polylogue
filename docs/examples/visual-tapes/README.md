# Public Visual Tape Examples

These are current example renders from `devtools render visual-tapes --capture`.
They use only deterministic synthetic/demo data.

Regenerate from the repository root:

```bash
devtools render visual-tapes --output-dir docs/examples/visual-tapes --capture
```

The full default inventory also renders `demo-tour.gif`; the canonical copy for
that larger demo packet lives at `docs/examples/demo-tour/demo-tour.gif`.

## Files

- `demo-tour.tape` — tape spec for the one-command demo tour; its canonical GIF
  copy lives in `docs/examples/demo-tour/demo-tour.gif`.
- `query-tour.tape` / `query-tour.gif` — query/read drilldown against a seeded
  demo archive.
- `reader-evidence-tour.tape` / `reader-evidence-tour.gif` — browserless reader
  evidence lane against synthetic fixtures.
- `browser-capture-tour.tape` / `browser-capture-tour.gif` — headless Chrome
  proof that deterministic ChatGPT and Claude fixture pages are captured through
  the unpacked browser extension, receiver, archive/API convergence, popup/debug
  UI, and the daemon web reader DOM.
