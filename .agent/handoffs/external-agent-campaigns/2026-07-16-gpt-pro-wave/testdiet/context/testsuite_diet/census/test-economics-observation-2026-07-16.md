---
created: 2026-07-16
purpose: Record what the existing test-economics substrate can and cannot currently support
status: current-artifact observation
project: polylogue
---

# Current test-economics observation

Regenerate:

```bash
devtools lab test-economics --json
```

On 2026-07-16 the command reports no
`.cache/coverage/coverage.json`, so coverage percentages and every coverage-
dependent quadrant are unavailable. The testmon data is present, but hub
imports make package-level exposure unusably coarse for diet decisions:

- `storage`, `daemon`, `mcp`, and `schemas` each report all 13,486 recorded
  tests and the same 17,223 seconds of wall-time exposure;
- `_root`, `core`, `paths`, `archive`, and `sources` also have very large
  transitive fan-out;
- therefore package exposure cannot tell which test owns or uniquely proves a
  behavior inside those packages.

Historical `fix:` commit counts still provide a rough risk prior: storage 261,
CLI 171, daemon 138, sources 124, archive 84, and pipeline 74 in the current
git history. These counts are biased by package size, age, and commit-message
discipline, so they prioritize investigation; they do not establish
under/over-testing.

Conclusion: reuse the report's extraction machinery, but a credible deletion
estimate requires a fresh coverage run with per-test contexts and behavior-
cluster grouping. Do not use the current package quadrants or fan-out totals to
authorize test additions/deletions.
