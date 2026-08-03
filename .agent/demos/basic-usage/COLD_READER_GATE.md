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
- States that seven walkthroughs use the `polylogue` CLI directly through the
  real subprocess test route and one (MCP) uses a real stdio JSON-RPC
  client/server exchange, not the CLI.
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

## Replacement Verification Mapping

| Former check | Replacement | Production route exercised |
| --- | --- | --- |
| find query | `test_find_query_covers_fielded_filter_and_pipeline_aggregate` | Query-first CLI parser, SQL query plan, and select renderer |
| read | `test_read_renders_the_seeded_transcript` | Query-first CLI read/transcript renderer |
| search | `test_search_spans_multiple_origins` | CLI lexical search and result serialization |
| resume | `test_continue_generates_a_resume_command` | CLI continuation action |
| cost | `test_usage_reports_disjoint_token_lanes` | CLI usage insight and JSON renderer |
| lineage | `test_lineage_read_composes_parent_prefix_and_child_tail` | CLI read path and lineage recomposition |
| MCP round-trip | `test_mcp_query_and_get_round_trip` | Production stdio MCP query/get dispatcher |
| status/health | `test_status_reports_direct_archive_fallback_when_daemon_is_unreachable` | CLI direct-archive status fallback |

The replacement test seeds through `polylogue.demo.seed_demo_archive`, runs the
CLI through `tests.infra.cli_subprocess.run_cli`, and invokes the production
MCP stdio route. `polylogue demo verify --require-overlays` remains the
structural demo contract check. The captured outputs remain private-data-free
fixtures for cold reading; they are not a second QA command surface.
