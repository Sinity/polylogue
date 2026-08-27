Summary

No Polylogue source change is applicable. The incident is owned by the separate `bd` repository; its sibling worktree contains the complete fix and CLI regressions in commits `6bec1d071` and `66d10d11a`.

Problem

The reported runaway `bd list --all` tree renderer is not implemented or vendored in Polylogue. The bounded live route probe in `/realm/project/polylogue` completed successfully: `bd list --all --no-pager` exited 0, producing 260,523 bytes across 2,278 lines. The probe was file-size limited to avoid reproducing the historical runaway write.

Solution

The external `bd` patch carries branch/path cycle state, suppresses repeated nodes, collapses duplicate direct edges, and exercises both renderer and embedded-Dolt CLI routes. It passes the focused renderer tests and the embedded-Dolt production-route tests in `/realm/worktrees/beads-2bc2-list-tree`.

Verification

- `nix develop --command env TEST_RUN='^TestDisplayPrettyListWithDepsMode_SuppressesLongCycle$|^TestPrintPrettyTree_SuppressesHundredThousandDuplicateEdges$' ./scripts/test.sh ./cmd/bd/...` — PASS; `ok github.com/steveyegge/beads/cmd/bd (cached)`.
- `nix develop --command env BEADS_TEST_EMBEDDED_DOLT=1 TEST_RUN='^TestEmbeddedListAllTreeSuppressesImportedParentChildCycle$|^TestEmbeddedListTreeSuppressesHundredThousandDuplicateEdges$' ./scripts/test.sh ./cmd/bd/...` — PASS; `ok github.com/steveyegge/beads/cmd/bd 33.734s`.
- `bd list --all --no-pager` from `/realm/project/polylogue`, with a 1 MiB output-file limit — exit 0; 260,523 bytes, 2,278 lines.
- `nix develop --command devtools verify --quick` — PASS; all reported checks completed `ok`.
- `nix develop --command devtools verify` — REFUSED before affected tests; `devtools why`: `native_testmon_graph_unavailable`, required native environment `polylogue-960ae5728522c613238028f4a93948c5db75f51146fd392bc3649acda8937285` absent.

Residual risk

This Polylogue lane does not publish the external `bd` commits or repair a live Beads database. The external repository owner must land `6bec1d071` and `66d10d11a`; the full Polylogue affected verification must be rerun once its native testmon environment is available. The bead remains open and requires a named successor/dependency if closed as partial.
