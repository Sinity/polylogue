# D1 "The Receipts": Claim-vs-Evidence on a Real Merged PR

This file is a Demo Finding Packet artifact (`devtools/demo_packet.py`
`PACKET_FILENAMES` contract), not an agent session summary. It is consumed
by `devtools lab policy demo-packet-registry` and read by future operators
reproducing this demo -- it is checked-in repo content, not a report to the
orchestrating agent.

## Claim

`session_refs` typed `pull_request` evidence resolves a real merged PR to
its authoring/dispatch session, and specific sentences from that PR's body
can be checked against the session's own recorded `blocks` -- with claims
that have no matching evidence marked as such, not silently trusted.

## Corpus

The live archive (`/realm/db/polylogue`, read-only), scoped to one session:
`claude-code-session:5ecdb160-495a-4d9b-b80a-3a24886af8cc`, resolved via
`session_refs WHERE kind='pull_request' AND repo='Sinity/polylogue' AND
ref_number=3282`. This is the real, merged PR
[Sinity/polylogue#3282](https://github.com/Sinity/polylogue/pull/3282)
("perf(storage): defer FTS repair off the live-ingest write path").

## Method

1. Resolved PR #3282 to a session structurally through `session_refs` (the
   table PR #3425 populated and PR #3431 wired into
   `insights/session_commit.py:build_correlation_result` /
   `insights/correlation_view.py`'s `read --view correlation` surface --
   not a regex/time-window guess).
2. Fetched the live PR body via `gh pr view 3282 --json body`.
3. For each individually falsifiable claim in that body, searched the
   resolved session's `blocks` table (`tool_use`/`tool_result` joined by
   `tool_id`) for matching structural evidence.
4. Recorded each claim's status: `supported` (matching block evidence
   found) or `not_supported` (no matching block, regardless of what the PR
   prose says).

## Findings

Claim-vs-evidence table (full block citations in `evidence.ndjson`):

| # | PR #3282 claim | Evidence found in the session | Status |
|---|---|---|---|
| 1 | This session authored/opened the PR | `tool_use` block runs `gh pr create --title "perf(storage): defer FTS repair off the live-ingest write path" --body "..."` with body text byte-identical to the live-fetched PR body; `tool_result` returns `https://github.com/Sinity/polylogue/pull/3282` | **supported** |
| 2 | "`devtools verify --quick` -- pass (ruff format/check, mypy, render all, topology/layering/...)" | `tool_use` runs `timeout 180 devtools verify --quick`; `tool_result` is a structured run-JSON with every step's `exit` field `0` (17 steps enumerated, `total_duration_s: 32.99`, top-level `exit_code: 0`) | **supported** (structural -- the exit codes, not a trusted "pass" word) |
| 3 | "`rebuild_index` bulk FTS materialization checkpoints progress (base for #3281, rebased here after that merge)" | `tool_use` runs `devtools test tests/unit/maintenance/test_rebuild_index_bulk_build.py` (+4 more files) in `/realm/worktrees/polylogue-membership-head-provenance`; `tool_result` pytest summary: `123 passed in 8.05s`; a following `git commit` in the same worktree stages exactly `polylogue/maintenance/rebuild_index.py` -- the one file this line's claim is about, matching the PR's own file diff (`polylogue/maintenance/rebuild_index.py 1 1`) | **supported** |
| 4 | "`devtools test tests/unit/sources/test_live_batch_support.py tests/unit/sources/test_live_catchup_planning.py tests/unit/storage/test_revision_replay.py tests/unit/storage/test_fts_identity_ledger.py tests/unit/storage/test_fts_repair_sql.py tests/unit/storage/test_bulk_fts_prefix_reextract.py tests/unit/daemon/test_daemon_cli.py -- all passing (see individual commit messages for per-commit pass counts)" | That exact 7-file string appears **only** inside the `gh pr create --body` tool_input (i.e. inside the PR body text itself) -- 0 rows when searching this session's `tool_use` blocks for the string with the `gh pr create` block excluded | **not independently verified in this session** |

## Specimens

See `evidence.ndjson` for the full block-id citations behind each row
above, including the exact `tool_result` text for rows 2 and 3.

## Counterexamples

**Finding 4 is a real, structurally-confirmed gap, not an artifact of
sloppy search.** The PR body's own parenthetical for that claim --
"see individual commit messages for per-commit pass counts" -- already
admits the aggregate 7-file invocation was never run as one shot; this
session's block evidence confirms it structurally: the string is prose
inside the PR body draft, never an executed command. This is the intended
behavior of a claim-vs-evidence packet: a claim the PR body asserts in
prose, with no matching structural evidence in the resolved session, must
render as unsupported -- not silently upgraded because the surrounding
claims (1-3) checked out.

**This session is a merge-conductor, not the file-editing session.**
`SELECT tool_name, count(*) ... GROUP BY tool_name` over this session's 56
`tool_use` blocks returns `Bash=53, Read=3` -- zero `Edit`/`Write` blocks.
The PR's actual code changes were authored in separately dispatched worker
sessions across several git worktrees
(`/realm/worktrees/polylogue-membership-head*`); this session orchestrates
`git`/`gh`/`devtools` across them and opens the PR. `session_refs` correctly
resolves PR #3282 to *this* session (the one that ran `gh pr create`), which
is the right target for "which session can I ask about this PR's own
claims" -- but it is not the right target for "which session edited file
X", a different (currently unresolved by this packet) question.

## Limits

- This packet checks 4 claims from one PR's body, not every sentence. It is
  a method demonstration (structural claim-vs-evidence resolution through
  `session_refs`), not an audit of PR #3282's full body.
- This is the **live-archive operator variant** only. The epic's own design
  (`polylogue-212`) calls for two variants per demo: a public seeded-corpus
  reproduction (seed 1843) and a live-archive operator variant. `session_refs`
  `pull_request` rows are a real, provider-native Claude Code capability
  (pr-link sidecar records) that the deterministic seed fixture does not
  currently populate, so the public variant is not built by this packet --
  named as remaining scope in the owning bead (`polylogue-xyel`) rather than
  claimed done here.
- The multi-worktree merge-conductor pattern found in Finding 4/Counterexamples
  means `session_refs`'s PR-to-session resolution answers "which session
  opened this PR", not "which session wrote this specific line of this
  specific file" -- a real, useful distinction this packet surfaces but does
  not resolve further (that would need session-to-commit-to-worktree
  lineage, which `polylogue-cijx.1`'s notes document as a separate, still-
  open problem for the durable `session_commits` table, unrelated to the
  `session_refs` mechanism this packet exercises).

## Non-claims

- This packet does not prove every sentence in PR #3282's body is
  independently verified -- only the four claims explicitly checked above
  are scored; claim 4 is explicitly scored `not_supported`.
- This packet does not establish that `session_refs` correctly resolves
  every PR reference archive-wide -- only that it resolves this one case
  with structural evidence.
- This packet does not reproduce on the public seed corpus (seed 1843); it
  requires read-only access to the live archive and the `Sinity/polylogue`
  GitHub history.

## Reproduce

```bash
# 1. resolve PR -> session
sqlite3 "file:/realm/db/polylogue/index.db?mode=ro" \
  "SELECT session_id, repo, ref_number, url FROM session_refs \
   WHERE kind='pull_request' AND repo='Sinity/polylogue' AND ref_number=3282"

# 2. cross-check through the CLI's own correlation surface
POLYLOGUE_ARCHIVE_ROOT=/realm/db/polylogue POLYLOGUE_FORCE_PLAIN=1 \
  polylogue find "id:claude-code-session:5ecdb160-495a-4d9b-b80a-3a24886af8cc" \
  then read --view correlation --format json

# 3. fetch the live PR body
gh pr view 3282 --repo Sinity/polylogue --json body

# 4. tool_name distribution (merge-conductor finding)
sqlite3 "file:/realm/db/polylogue/index.db?mode=ro" \
  "SELECT tool_name, count(*) FROM blocks \
   WHERE session_id='claude-code-session:5ecdb160-495a-4d9b-b80a-3a24886af8cc' \
   AND block_type='tool_use' GROUP BY tool_name ORDER BY 2 DESC"

# 5. the devtools verify --quick evidence (claim 2)
sqlite3 "file:/realm/db/polylogue/index.db?mode=ro" \
  "SELECT tr.text FROM blocks tu JOIN blocks tr \
   ON tr.tool_id=tu.tool_id AND tr.block_type='tool_result' AND tr.session_id=tu.session_id \
   WHERE tu.session_id='claude-code-session:5ecdb160-495a-4d9b-b80a-3a24886af8cc' \
   AND tu.tool_input LIKE '%devtools verify --quick%'"

# 6. the negative-control count (claim 4 -- must return 0)
sqlite3 "file:/realm/db/polylogue/index.db?mode=ro" \
  "SELECT count(*) FROM blocks \
   WHERE session_id='claude-code-session:5ecdb160-495a-4d9b-b80a-3a24886af8cc' \
   AND block_type='tool_use' AND tool_input LIKE '%test_live_batch_support.py%' \
   AND tool_input NOT LIKE '%gh pr create%'"
```

See `evidence.ndjson` for every cited ref and `checks.json` for the
pass/fail summary.
