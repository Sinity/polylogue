# D1 "The Receipts": Claim-vs-Evidence on a Real Merged PR

Predeclaration receipt: `artifact:d1-receipts-predeclaration`.

Pick a real merged, agent-authored PR from this repository. Resolve it to
its authoring/dispatch session **structurally** — via `session_refs`
(kind=`pull_request`), not by regex-scanning message prose or a time-window
heuristic. Then check specific sentences from the PR body against that
session's own recorded tool_use/tool_result blocks: does the evidence
actually support the claim, or is the claim resting on the PR body's own
prose with nothing underneath it?

Product primitives only: `session_refs` (the typed evidence table wired by
PR #3425/#3431), `polylogue read --view correlation`, and structural SQL
reads over `blocks`/`session_refs` for citation (mirroring the exact
read-only query style PR #3392 and PR #3282 themselves used in their own
Verification sections — this demo does not invent a new access pattern).

## Steps

1. Resolve PR → session structurally:
   ```sql
   SELECT session_id, repo, ref_number, url
   FROM session_refs
   WHERE kind = 'pull_request' AND repo = 'Sinity/polylogue' AND ref_number = 3282;
   ```
   Cross-check the same resolution through the CLI's own read surface:
   `polylogue find "id:<session_id>" then read --view correlation --format json`
   (this is the surface PR #3425/#3431 wired `session_refs` into —
   `insights/session_commit.py:build_correlation_result` and
   `insights/correlation_view.py`).

2. Fetch the PR body from GitHub (`gh pr view 3282 --json body`) and pull
   out individually falsifiable sentences — not the whole prose block, each
   claim on its own.

3. For each claim, search the resolved session's own `blocks` rows
   (`tool_use`/`tool_result`, joined by `tool_id`) for structural evidence:
   an exact command, an exact exit code, an exact pytest summary line. A
   claim with no matching block is marked **not independently verified in
   this session** — never silently upgraded to "supported" because the PR
   body asserts it.

4. Render the two columns: claimed sentence | observed block evidence
   (drillable via the cited `block:` ref), with an explicit status per row.

## Note on this run

This demo's session turned out to be a **merge-conductor** session: its own
`blocks` are almost entirely `Bash` (53 of 56 tool_use blocks) plus 3 `Read`
calls — zero `Edit`/`Write` tool_use. The actual file edits for PR #3282
happened in separately dispatched worker sessions across multiple git
worktrees (`/realm/worktrees/polylogue-membership-head*`); this session
orchestrates `git`, `gh pr create`, and `devtools test`/`devtools verify`
invocations across those worktrees and stitches the result into one PR.

This is itself a real, useful finding, not a inconvenience to hide: the PR
body's own Verification section names a 7-file `devtools test` invocation
("`devtools test tests/unit/sources/test_live_batch_support.py ...` — all
passing, **see individual commit messages for per-commit pass counts**") —
its own parenthetical admits the aggregate command was never run as one
shot. Searching this session's blocks confirms it: the 7-file string only
appears inside the `gh pr create --body` tool_input (i.e. inside the PR body
text itself), never as an actual invoked command. That specific claim is
marked **not independently verified in this session** in `report.md` and
`checks.json` — precisely the honesty discipline this packet exists to
enforce, applied to itself.
