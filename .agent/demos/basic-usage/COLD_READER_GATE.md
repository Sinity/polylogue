# Cold-Reader Gate

Give a fresh reader only this directory and ask:

```text
Using only the files in this directory, list the eight Polylogue features
this demo claims to prove work end-to-end, state which surface (CLI, MCP)
each was exercised through, and identify the one walkthrough whose output
was deliberately captured against a non-default daemon URL — and why.
```

## Expected Passing Answer

- Names all eight walkthroughs: find (fielded query + pipeline aggregate),
  read (exact-ref transcript), search (FTS with provenance), resume
  (continuation command generation), cost (disjoint token/cost lanes),
  lineage (composed fork read), MCP (search → get_session_summary round-trip),
  status/health (archive readiness without a daemon).
- States that seven walkthroughs use the `polylogue` CLI directly and one
  (MCP) uses a real stdio JSON-RPC client/server exchange, not the CLI.
- Identifies walkthrough 8 (status/health) as the one run with
  `--daemon-url http://127.0.0.1:1`, and explains this forces the
  direct-archive fallback path so the captured output reproduces
  identically for a reader with no daemon running at all.
- Notices the demo runs against the deterministic seeded archive
  (`polylogue demo seed`), not the operator's live archive, and states why
  (public repository; the README's own "Why the demo archive" section).
- Does not treat any of the eight numeric results (token counts, hit counts,
  session counts) as a claim about a real corpus — they are properties of
  the fixed deterministic demo archive.

## Current Gate Evidence

- All eight command outputs are captured verbatim in this directory's
  numbered files; none are hand-edited except line-wrapping.
- `devtools workspace basic-usage-demo-check --archive-root <fresh-seed>`
  re-runs every command against a freshly seeded archive and asserts each
  output's shape (non-empty hit lists, expected JSON keys, expected session
  refs) — see `devtools/basic_usage_demo_check.py`.
- Status: ready for an external cold read; no private transcript content is
  present (deterministic demo archive only).
