# Verification and visual evidence plan

## Why this matters

The web reader and CLI polish must not become screenshots that rot. They should become evidence-bearing surfaces: generated from fixtures, checked automatically, and consumed by README/docs when curated.

## Visual evidence artifacts

Create scenario/exercise outputs for:

- `polylogue.cli.query_table`: terminal screenshot/cast of `polylogue "fts trigger" --repo polylogue list`.
- `polylogue.cli.select`: terminal screenshot/cast of `polylogue select conversation`.
- `polylogue.local_reader.search`: browser screenshot of `/search` with facet/result/reader layout.
- `polylogue.local_reader.conversation`: browser screenshot of `/c/{id}`.
- `polylogue.dashboard.status`: terminal/TUI screenshot of daemon/live/capture/doctor status.
- `polylogue.capture.status`: web/extension/capture state screenshot, deferred until browser receiver migration.

## Evidence envelope

Each generated visual artifact should record:

```json
{
  "artifact_id": "polylogue.local_reader.search",
  "kind": "screenshot",
  "surface": "web",
  "scenario": "local-reader.search.synthetic",
  "fixture": "synthetic-polylogue-demo-v1",
  "path": "docs/assets/polylogue/local-reader-search.png",
  "produced_at": "...",
  "command": "...",
  "privacy_class": "synthetic",
  "checks": {
    "exists": true,
    "not_blank": true,
    "dimensions": "pass",
    "no_private_paths": true,
    "expected_text": ["polylogued", "local-only", "fts trigger"],
    "readability": "pass"
  },
  "review": {
    "kind": "agent_judgment",
    "reviewed_at": "...",
    "fresh_until": "...",
    "verdict": "pass"
  }
}
```

## Automated checks

At minimum:

- file exists;
- size/dimensions in expected range;
- image not blank;
- no obvious clipping of key text;
- expected text visible if OCR/screenshot text extraction exists; otherwise DOM assertions for web and terminal output assertions for CLI;
- no absolute private paths in rendered screenshots;
- no non-synthetic fixture content;
- current-generation timestamp/hash exists;
- docs/README references only curated generated artifacts.

## Manual / agent aesthetic review

Treat aesthetic judgment as an agent/VLLM review cell:

- owner: coding agent or named reviewer;
- artifact id;
- last reviewed timestamp;
- freshness policy;
- notes;
- fail/warn/pass.

This does not replace automated checks. It catches “technically present but ugly/unreadable.”

## README storyboard

README should show the product story, not every feature:

1. Query-first CLI table.
2. Local web reader: search + conversation reader + provenance.
3. Live/capture/doctor status strip.
4. Fuzzy selector or command palette.
5. One short proof/confidence/verification visual only for contributors.

Do not lead with proof jargon. The verification story appears as “generated from real scenario evidence,” not “look at our obligation graph.”

## Verification commands

Suggested first-slice commands:

```bash
polylogued run --help
polylogue --help
polylogue "fts trigger" --repo polylogue list --format json
polylogue messages <fixture-id> --limit 3 --format json
pytest -q tests/daemon tests/cli/test_select.py tests/visual
devtools verify --quick
```

If visual tooling is not present yet, the first PR should add a clear pending visual evidence issue/manifest rather than pretending screenshots are verified.
