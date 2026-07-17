## [user 307]
▎ *You have the forked context: the polylogue bundle, the swarm2 daemon/composer/thin-client designs, and our discussion of queries-as-objects, delegation units, and annotation recipes. Ground every claim in the bundle (cite file:line / bead id); tag [evidence] vs [proposal]. Be opinionated — one defended recommendation plus the runner-up you rejected and why. Answer in layers: (1) with today's substrate, (2) with a near-term substrate change, (3) the full direction. End with Open questions for the operator and What's missing (a claim unverified, a source unread, a modality not run). The snapshot is dated; live code may have moved — flag assumptions to verify locally. Don't re-derive swarm2.*

Analytical [A] over the bundle:
 [A] Session replay/reproduction. Feasibility + design: re-execute a session's tool-calls against the repo to reproduce/verify work. The tracing + git-correlation + replay model; what the archive must store.

## [assistant 396]
Using the 2026-07-05 bundle snapshot. Assumption to verify locally: the code/beads may have moved since this tarball and since the Claude Code session that later mentioned more bead changes. I am treating the uploaded bundle as authoritative for this memo, not live master.

## Recommendation

[proposal] Build **reproduction-first verification**, not “exact session replay,” as the next substrate slice.

The defended recommendation is: **make Polylogue produce a replayable work-trace bundle whose default operation is to verify the final claimed work from a clean repo state, not blindly re-run every historical tool call.** The object should answer: “Given this session, which repo state did it start from, what files/actions/commands/results did it touch, which commits/artifacts did it produce, and can we independently re-run the relevant verification commands or apply the resulting patch?”

The runner-up I reject is: **raw command-stream replay**, i.e. “take every Bash/tool call from the session and re-execute it in order.” It is tempting because the archive already has tool commands and exit codes, but it is too brittle and unsafe as the main product claim: the bundle currently does not guarantee action-level cwd/env/sandbox/network state, pre/post file hashes, secrets policy, package-cache state, or per-tool side effects. Raw replay should exist later as a best-effort mode for carefully classified commands, not as the headline.

The better framing is a replay ladder:

`L0 visual playback → L1 evidence reconstruction → L2 final-state verification → L3 patch/commit reproduction → L4 classified command replay → L5 deterministic full replay`

Polylogue is already credible at L1 today, can reach L2/L3 with a small substrate addition, and should only claim L4/L5 for sessions captured under stricter instrumentation.

## Layer 1 — With today’s substrate

[evidence] Polylogue already stores the raw source evidence in `source.db`: `raw_sessions` records origin, native id, source path, blob hash/size, acquisition and parse/validation state (`polylogue/storage/sqlite/archive_tiers/source.py:15-33`), and `blob_refs` records durable blob references (`source.py:51-59`). Raw artifacts are also modeled with `artifact_kind`, `support_status`, source path/index, parse eligibility, cohort/link keys, sidecar agent type, and first/last observed timestamps (`source.py:84-102`). This means a reproduction report can point back to original captured session payloads and artifact observations, not only derived rows.

[evidence] The parsed index already has session-level git hints: `sessions` includes `git_branch`, `git_repository_url`, and `commit_hash` (`polylogue/storage/sqlite/archive_tiers/index.py:51-55`). It also has `session_working_dirs`, `repos`, `session_repos`, and `session_commits` tables; `session_commits` stores `commit_sha`, `repo_id`, `detection_type`, `method`, `confidence`, and `evidence_json` (`index.py:470-512`). Codex parsing promotes provider git metadata into typed `git_branch`, `git_repository_url`, and `git_commit_hash` fields (`polylogue/sources/parsers/codex.py:847-888`), and the writer persists working dirs, repo edges, and explicit commit refs as `session_commits` with `detection_type='explicit_ref'`, `method='parser-git-meta'`, and confidence `1.0` (`polylogue/storage/sqlite/archive_tiers/write.py:3045-3053`, `3212-3276`).

[evidence] The action substrate is strong enough for **claim-vs-evidence verification**. Blocks store `tool_name`, `tool_id`, `tool_input`, `semantic_type`, `tool_result_is_error`, `tool_result_exit_code`, generated `tool_command`, generated `tool_path`, and generated `search_text` (`polylogue/storage/sqlite/archive_tiers/index.py:182-208`). The public `actions` view pairs `tool_use` blocks with matching `tool_result` blocks and exposes `tool_command`, `tool_path`, `tool_input`, `output_text`, `is_error`, and `exit_code` (`index.py:324-343`). There is also an index specifically for structured tool-result failures (`idx_blocks_tool_result_outcome`, `index.py:217-224`). The proof-artifacts page explicitly says the failure predicate is normalized structure, `is_error=1` or non-zero `exit_code`, and assistant prose is only follow-up acknowledgment (`docs/proof-artifacts.md:11-14`). fileciteturn1file18

[evidence] The query surface already treats replay-relevant things as units. The swarm brief says the query grammar supports structural `exists <unit>(...)` over `observed-event`, `context-snapshot`, `message`, `action`, `block`, `assertion`, `file`, and `run`, with terminal actions such as `read`, `analyze`, `select`, `mark`, `delete`, and `continue` (`SWARM_BRIEF.md:9-18`). Query metadata gives common fields such as `command`, `exit_code`, `is_error`, `output`, `path`, `tool`, and `duration_ms` (`polylogue/archive/query/metadata.py:474-505`), and terminal query descriptors exist for `assertions`, `files`, `runs`, `observed-events`, and `context-snapshots` (`metadata.py:792-867`). fileciteturn1file8

[evidence] There is already a materialized run/event/context layer: `session_runs` stores run refs, parent run refs, agent refs, context snapshot refs, harness, role `main|subagent`, status, cwd, git branch, lineage/evidence refs, transcript ref, payload JSON, and search text (`polylogue/storage/sqlite/archive_tiers/index.py:913-938`). `session_observed_events` stores event kind, subject/object refs, evidence refs, payload, and search text (`index.py:952-968`). `session_context_snapshots` stores context snapshot refs, boundary, inheritance mode, segment refs, evidence refs, metadata, payload, and search text (`index.py:1000-1015`).

[evidence] There is already a work-event read model: `session_work_events` stores type, summary, confidence, start/end indexes, times/duration, `file_paths_json`, `tools_used_json`, evidence, inference, and search text (`polylogue/storage/sqlite/archive_tiers/index.py:689-709`). This is not a replay log, but it is a useful mid-level reconstruction layer.

[evidence] The codebase already has an embryonic session↔commit detector. `polylogue/insights/session_commit.py` states that it detects likely commits through time windows, file-overlap scoring, and explicit references (`session_commit.py:1-10`). It extracts referenced files from tool calls/text (`session_commit.py:148-186`), derives scan windows (`session_commit.py:218-238`), runs `git log --since/--until --name-only` (`session_commit.py:290-305`), scores commits by explicit refs/file overlap/time-window fallback (`session_commit.py:319-363`), and builds a correlation result with commits, issue refs, PR refs, and file paths (`session_commit.py:387-449`). But the persistence hook is still a placeholder (`persist_session_commits`, `session_commit.py:455-468`), so this is not yet a complete product path.

[evidence] Bead `polylogue-7xv` already names the exact missing capability: “Native git/repo awareness: session-to-commit/branch/repo correlation in Polylogue.” Its design says to start from evidence already in the archive, infer session→repo from cwd/git tool calls, session→commit from SHAs in tool results plus Claude-session trailers in commit messages, materialize as a derived read model with evidence refs, and add DSL predicates/projections. Bead `polylogue-bby.12` already names “Session replay: play a session back the way it happened,” but it is explicitly a **derived visual view** over occurred timestamps, cost, change rails, and commit markers; it is not command re-execution. Bead `polylogue-6mv` draws the boundary: Polylogue owns raw AI-session evidence; Sinex owns redacted machine-timeline events and causal windows; Polylogue should emit derived privacy-preserving agent-trace events rather than pushing raw transcripts into Sinex. Bead `polylogue-x4s` and `polylogue-4c0` cover devloop/assertion and session↔bead work-history joins.

[proposal] So with today’s substrate, the honest demo is not “replay the session.” It is:

`reconstruct session work trace → correlate likely commits/files → compare claimed success against structured tool outcomes → generate a reproduction checklist`

A today-level output could be:

```text
session:<id>
  actions: ordered tool calls, outputs, error flags, exit codes
  files: paths referenced by tool calls / file units
  repo: inferred from cwd/repo metadata
  commits: explicit or candidate commits with confidence
  final claims: assistant/user summary messages
  verdict: verified / contradicted / unverified
  reproduction instructions: checkout commit X, inspect diff Y, run commands Z
```

[proposal] The correct today-level command surface is something like:

```text
polylogue read session:<id> --projection work-trace
polylogue query 'actions where session:<id> | group by tool | count'
polylogue query 'actions where session:<id> AND (is_error:true OR exit_code > 0)'
polylogue query 'files where session:<id>'
polylogue query 'runs where session:<id>'
```

The exact syntax may differ, but the units/fields exist in the snapshot.

[evidence] Current limitation: exact re-execution is not supported as a substrate. The action view exposes command/path/input/output/error/exit, but the schema does not guarantee normalized action-level cwd, environment, shell, timeout, stdin, sandbox/network policy, tool version, before/after file hashes, or pre/post git status (`index.py:324-343`; `session_runs.cwd` exists only at run level in `index.py:931`). This is enough for forensic reconstruction and many final verification checks; it is not enough for safe deterministic command replay.

## Layer 2 — Near-term substrate change

[proposal] The near-term change should be a **Work Trace + Reproduction Attempt** substrate, not a full replay VM.

Create a derived relation:

```text
session_work_trace_events
```

It should normalize existing evidence into an ordered trace:

```text
trace_event_ref
session_id
run_ref nullable
message_id
block_id / tool_use_block_id / tool_result_block_id
position
occurred_at_ms / started_at_ms / ended_at_ms
kind: message | tool_call | tool_result | file_ref | repo_snapshot | commit_ref | assertion | artifact
tool_name
semantic_type
command_text
argv_json nullable
input_json
output_blob_ref / output_text_excerpt
is_error
exit_code
cwd
repo_id
git_head_before
git_head_after
touched_paths_json
evidence_refs_json
replayability_class
redaction_json
payload_json
```

[evidence] This is not a new silo. It composes the existing `actions` view (`index.py:324-343`), `session_runs` (`index.py:913-938`), `session_work_events` (`index.py:689-709`), `session_repos/session_commits` (`index.py:491-512`), `raw_artifacts` (`source.py:84-102`), and `assertions` (`user.py:7-31`).

[proposal] Add one durable or semi-durable object:

```text
reproduction_attempt
```

This records an actual attempt to reproduce or verify work:

```text
attempt_ref
session_id
analysis_id / query_run_ref nullable
base_repo_ref
base_commit
target_commit / target_patch_ref nullable
selected_trace_event_refs
runner_version
mode: final-verify | patch-apply | command-replay | visual-only
sandbox_profile
started_at
finished_at
status: passed | failed | partial | unsafe | unverified
verifier_commands_json
result_events_json
produced_artifact_refs_json
assertion_refs_json
```

Store the attempt result as assertions/judgments over the session, commit, and trace events. The user-tier assertions table already supports `target_ref`, `scope_ref`, `kind`, `value_json`, `body_text`, `author_ref`, `author_kind`, `evidence_refs_json`, `status`, `visibility`, `confidence`, `staleness_json`, and `context_policy_json` (`polylogue/storage/sqlite/archive_tiers/user.py:7-31`); assertion kinds include `annotation`, `correction`, `saved_query`, `recall_pack`, `judgment`, `run_state`, `prompt_eval`, and `pathology` (`polylogue/core/enums.py:399-428`). The missing part is a first-class target kind and import/write affordance for reproduction attempts.

[proposal] Add target kinds beyond the current user-state registry. Today the canonical registry supports `session`, `message`, `work_event`, `thread`, `block`, `attachment`, and `paste_span`, and explicitly says some stable identities are not listed until external work lands (`polylogue/core/user_state_targets.py:1-21`, `29-95`). For replay/reproduction, add:

```text
query_run
result_relation
analysis_run
trace_event
reproduction_attempt
artifact
repo
commit
bead
```

[evidence] This fits the B8 contract rather than fighting it. B8 recommends a single `PolylogueService` with `query`, `read`, `preview`, `complete`, `act`, `status`, plus possibly `facets` (`B8_contract.md:134-168`). `read` should execute `QueryProjectionSpec` rather than named views (`B8_contract.md:188-205`), and `act` is the typed mutation/effect verb (`B8_contract.md:221-241`). fileciteturn1file8

[proposal] Therefore do **not** add a top-level `replay` RPC. Use:

```text
read({ selection: session/ref, projection: work-trace | reproduction-plan })
act(kind="record-reproduction-attempt", ...)
act(kind="run-safe-verifiers", ...)
```

The daemon/CLI/MCP surfaces remain thin and shared. This is important because B7 identifies `cli/archive_query.py` as a second engine and says it should shrink to formatting while substrate execution moves behind `Polylogue` (`B7_thinning.md:29-65`). fileciteturn1file17

[proposal] The near-term replayability classifier is the safety hinge:

```text
pure_read          e.g. ls, cat, rg, git status, git diff
safe_verify        e.g. pytest, mypy, devtools verify, cargo test
mutating_patch     e.g. file edits, sed -i, apply_patch, git checkout
networked          e.g. curl, gh, npm install, provider API calls
secret_sensitive   e.g. env, credentials, cloud deploys
interactive        e.g. editor, TUI, prompts
unknown
```

Only `pure_read` and `safe_verify` should be auto-runnable by default. Mutating commands should be replayed only inside a disposable worktree/sandbox, and network/secret-sensitive commands should be plan-only unless explicitly allowed.

[proposal] The near-term runner should operate this way:

1. Resolve session → trace events.
2. Resolve repo root and candidate base commit.
3. Build a disposable worktree at base commit.
4. Optionally apply the session-produced patch or checkout the target commit.
5. Run verifier commands, not every historical command.
6. Compare verifier results with the original session’s structured tool outcomes.
7. Write a `reproduction_attempt` plus `judgment` assertions.
8. Render a proof card.

[evidence] This preserves the existing daemon direction. A3 says the daemon request handler should become the single execution core, with `--no-daemon`/library/MCP/tests invoking the same core in-process, and that warm daemon is required for speed/composer while break-glass is a transport swap, not a parallel codebase (`A3_daemonless.md:7-31`). It also notes the Python substrate must remain directly constructable for CI/cloud/tests/recovery (`A3_daemonless.md:120-145`). fileciteturn1file17

[proposal] For session replay specifically, near-term UI should split two products that bead `bby.12` currently risks conflating:

`session playback`: timeline/scrubber/cost/file/commit rails, no side effects.

`session reproduction`: creates a sandbox, applies target state, runs verifiers, writes a verdict.

Keep `bby.12` for playback. Add a new bead for reproduction.

Suggested bead:

```text
polylogue-REPRO.1 — Work-trace reproduction harness: verify a session’s repo work from clean checkout

Parent: polylogue-7xv or polylogue-1xc/3tl depending operator priority
Priority: P1 if tied to external proof demo; otherwise P2
Type: feature

Description:
Polylogue can already reconstruct action evidence and correlate sessions to repos/commits,
but cannot safely reproduce a coding session’s work. Build a work-trace/reproduction
harness that turns one session into a replayable proof bundle: repo base, trace events,
touched files, produced commits/patches, verifier commands, and a recorded reproduction verdict.

Design:
Derived work-trace materializer over actions/session_runs/session_work_events/session_repos/
session_commits/raw_artifacts/assertions. Add replayability classification and a reproduction
attempt record. Default mode is final-state verification in a disposable git worktree, not
blind command replay. `read --projection work-trace` renders the trace; `act(kind=run-safe-verifiers)`
records the attempt as assertions/judgments.

Acceptance:
1. Given a fixture coding session with Bash/tool calls, file edits, and a final test command,
   Polylogue renders a work-trace with ordered actions, cwd/repo/commit, touched paths,
   error/exit-code evidence, and source refs.
2. Given a repo fixture, reproduction creates a disposable worktree at the selected base,
   applies the target patch/commit or checks out target commit, runs verifier commands,
   and records pass/fail as a reproduction_attempt plus assertions.
3. Unsafe/network/secret/interactively mutating commands are classified and not auto-run.
4. A report card cites original action refs and reproduction attempt refs.
5. `devtools test <fixture tests>` green; no CLI direct storage reads added.
```

## Layer 3 — Full direction

[proposal] The full design is **replay as a provenance DAG**, not a script log.

The graph should connect:

```text
session → run → message → block/action
action → tool result
action → file observation/artifact
action → repo snapshot
repo snapshot → commit/patch/status
query/run/analysis → trace selection
reproduction attempt → verifier commands/results
assertions/judgments → evidence refs
report → query/reproduction/cohort refs
```

[evidence] The existing architecture is already pointed this way. The swarm brief’s read algebra is `Query × Projection × Render`, with query units for actions/files/runs/observed-events/context-snapshots/assertions and projection families including sessions/messages/blocks/actions/raw/context/chronicle/neighbors/correlation/temporal/assertions (`SWARM_BRIEF.md:9-28`). B8’s `ReadRequest` recommendation explicitly wires selection + projection + render as the read input (`B8_contract.md:188-205`). fileciteturn1file8

[proposal] Full replay requires storing more than Polylogue currently guarantees. The archive must store, or link to a system that stores:

```text
Per action:
- normalized tool invocation
- raw tool input JSON
- shell/script/argv split
- cwd
- environment allowlist + redacted secret markers
- stdin if any
- timeout
- sandbox/network policy
- started_at/ended_at
- stdout/stderr/output refs with truncation markers
- exit code and error flag
- tool/runtime version

Per repo:
- repo root and origin
- branch
- HEAD before/after each mutating cluster
- index/worktree status before/after
- patch/diff after mutating cluster
- changed path list
- file hashes before/after
- file modes, renames, deletions
- submodule/LFS/annex references
- lockfile/devenv/Nix flake state

Per artifact:
- path/URI
- content hash/blob ref
- producer action/session
- consumer action/session
- media/type/support status
- redaction/privacy tier

Per reproduction attempt:
- base commit
- target commit/patch
- runner version
- selected event range
- sandbox profile
- verifier commands
- outputs
- verdict
- evidence refs
```

[evidence] The raw-log already points to the broader machine-observability layer: terminal sessions, web provenance, neovim logs, ActivityWatch-like data, and explicit desire to reconstruct what was happening across the desktop are recurring themes; one raw-log entry specifically says Polylogue could correlate/connect logs with git commits, issues, and PRs (`logs.raw-log.md`, 2026-04-22 13:45:04), and another notes the need to analyze wall-clock time in agent sessions such as edit/build/test cycles (`logs.raw-log.md`, 2026-04-21 23:24:03). fileciteturn1file4

[evidence] Bead `polylogue-6mv` is the important boundary: Polylogue should own raw AI-session evidence while Sinex owns redacted machine-timeline events and causal windows. Bead `polylogue-fs1.9` already points to a future Polylogue→Sinex derived agent-trace emitter. So full L5 replay belongs to the combined Polylogue+Sinex system, not Polylogue alone.

[proposal] Full deterministic replay should be opt-in and capture-mode-dependent. Sessions captured before the work-trace substrate can often be reconstructed and verified, but not deterministically replayed. Sessions captured after the substrate lands can be classed by evidence completeness:

```text
replay_class A: exact verifier reproduction available
replay_class B: patch/commit reproduction available
replay_class C: action-level command replay possible for safe commands
replay_class D: visual/evidence replay only
replay_class E: insufficient evidence
```

[proposal] The strongest external demo would not say “we replayed every command.” It would say:

“Polylogue reconstructed this agent session, checked out the repo at the recorded base, applied the produced diff/commit, reran the exact verifier class, and recorded a pass/fail verdict with links to the original tool outputs and reproduction outputs.”

That is defensible, useful, and far less brittle than pretending every historical command can be replayed.

## Feasibility verdict

[evidence] **Forensic reconstruction is high-feasibility today.** The archive already stores raw payloads, structured tool use/results, exit codes, file paths, sessions/runs/events/context snapshots, repo metadata, and some commit links (`source.py:15-102`; `index.py:182-224`, `324-343`, `470-512`, `689-709`, `913-1015`).

[evidence] **Final-state verification is medium-feasibility today, high with a near-term work-trace substrate.** Today a human/agent can inspect actions and run likely verification commands manually. Near-term, Polylogue can generate the worktree/verifier plan and record the verdict.

[evidence] **Patch/commit reproduction is medium-feasibility near-term.** The open `polylogue-7xv` bead already scopes native git/repo awareness; the current code has typed repo/commit tables and explicit parser-git commit persistence, but the older `session_commit.py` detector still has a placeholder persistence function (`session_commit.py:455-468`).

[evidence] **Blind command re-execution is low-feasibility today.** The action view does not guarantee normalized action-level cwd/env/stdin/sandbox/tool version/pre/post file hashes (`index.py:324-343`), and some commands are unsafe, nondeterministic, networked, or secret-bearing.

[proposal] **Deterministic full replay is a long-term capture-mode claim, not an archive-backfill claim.** It needs Sinex-like machine traces plus stricter repo/file snapshots and sandbox/environment capture.

## Concrete architecture

[proposal] Add these derived/read objects:

```text
work_trace_event
repo_snapshot
artifact_observation_edge
reproduction_plan
reproduction_attempt
reproduction_verdict
```

[proposal] Add these projection presets, not hardcoded silos:

```text
work-trace
reproduction-plan
reproduction-card
claim-vs-reproduction
session-playback
```

[proposal] Add these query examples:

```text
actions where session:<id> AND (is_error:true OR exit_code > 0)
files where session:<id> | group by path | count
runs where session:<id>
observed-events where session:<id> AND kind:tool_finished
assertions where target_ref:reproduction_attempt:<id>
```

[proposal] Add these commands as thin client calls over the shared service:

```text
polylogue read session:<id> --projection work-trace
polylogue reproduce plan session:<id> --json
polylogue reproduce verify session:<id> --mode final-state --worktree /tmp/...
polylogue reproduce card reproduction_attempt:<id>
```

Under B8’s contract, `plan` and `card` are `read` projections; `verify` is an `act` because it creates a reproduction attempt and writes assertions.

## What the archive must store

[evidence] Already stored or partially stored:

```text
raw session payload and blob refs                 source.py:15-59
raw artifact observations                         source.py:84-102
session git branch/repo/commit hints              index.py:51-55
messages and blocks with structured tool fields   index.py:92-125, 182-208
tool result error/exit fields                     index.py:195-196
actions paired tool_use/tool_result view          index.py:324-343
working dirs, repos, session_repos, commits       index.py:470-512
session work events with file/tool evidence       index.py:689-709
runs/observed-events/context snapshots            index.py:913-1015
assertions/judgments over evidence refs           user.py:7-31; enums.py:399-447
```

[proposal] Missing or insufficiently normalized:

```text
action-level cwd/env/sandbox/network/timeout
argv/script normalization separate from raw command text
stdin and input-file refs
stdout/stderr blob refs with truncation markers
pre/post git HEAD and dirty status
pre/post file hashes for touched paths
patch bundles linked to action ranges
artifact producer/consumer edges
dependency lockfile/devenv/Nix state
safe/unsafe replayability classification
reproduction attempt object and verdict assertions
query_run/result_relation/analysis_run targets
```

## Bead implications

[evidence] This should not become a Fable/demo silo. The existing bead set already implies the capability across multiple towers:

`polylogue-bby.12`: visual session replay.

`polylogue-7xv`: native git/repo/session↔commit awareness.

`polylogue-6mv`: Polylogue/Sinex boundary for agent traces.

`polylogue-x4s`: devloop state as assertions/substrate.

`polylogue-4c0`: session↔bead work-history joins.

`polylogue-s7ae.5`: two agents, one repo, overlap/message/context/handoff proof.

`polylogue-fs1.9`: Polylogue→Sinex derived agent-trace emitter.

`polylogue-9l5.7`: statistics/measure registry with construct-validity metadata.

`polylogue-3tl`: external legibility / citable proof artifacts.

[proposal] Add one new explicit bead because none of those owns “verify/reproduce a session’s repo work” end-to-end. Parent choice:

Best parent if proof-driven: `polylogue-3tl` or a child under `polylogue-sru`/`tf2` demo lineage.

Best parent if substrate-driven: `polylogue-7xv`.

I would parent it under `polylogue-7xv` and add a `relates-to` edge to `polylogue-3tl`, `polylogue-bby.12`, `polylogue-4c0`, and `polylogue-6mv`.

## Open questions for the operator

1. [proposal] Do you want the first shipped claim to be “verifies final work” or “replays command history”? I strongly recommend the first.

2. [proposal] Should the reproduction attempt be allowed to run networked commands at all, or should networked steps always be plan-only unless explicitly whitelisted?

3. [proposal] What is the canonical repo environment target: plain git worktree, Nix/devenv shell, container, or “use whatever the session used”? I recommend disposable git worktree + optional Nix/devenv profile when detected.

4. [proposal] Should Polylogue store patch bundles itself, or only link commits/artifacts and let git remain the patch store? I recommend content-addressed patch artifacts for session-produced uncommitted changes, but ordinary git commits for committed work.

5. [proposal] How much belongs in Polylogue vs Sinex? My answer: Polylogue owns session semantics, tool evidence, reproduction plans, and verdicts; Sinex supplies machine/process/filesystem timeline for higher replay classes.

6. [proposal] Should reproduction attempts be durable by default? I recommend yes if user/agent explicitly ran `reproduce verify`; no for ordinary preview/planning.

## What’s missing

[evidence] I did not run the daemon, tests, or live `polylogue` commands. The line refs are from the uploaded tarball and markdown reports, not live checkout.

[evidence] I did not verify whether `polylogue-7xv`, `polylogue-bby.12`, or the later bead-set changes have moved after the bundle snapshot. The pasted Claude Code session indicates later bead churn and even mentions the snapshot hazard; live `bd show` should confirm status/parents before filing new work. fileciteturn1file9

[evidence] I did not inspect every provider parser for whether action-level cwd/env is sometimes present inside raw `tool_input`; I only verified that the normalized schema does not guarantee it.

[evidence] I did not inspect the full implementation of `query_files`/file unit lowering; I relied on the query metadata and table schemas.

[evidence] I did not read or run Sinex sources; the Polylogue↔Sinex boundary is grounded in bead `polylogue-6mv` and the bundle descriptions, not current Sinex code.

[evidence] I did not check whether `session_commit.py`’s placeholder persistence has been replaced in live code. In this snapshot it is still placeholder (`polylogue/insights/session_commit.py:455-468`).

[evidence] I did not test whether old archived sessions have enough raw data to reconstruct cwd/env from provider-specific payloads. That is a necessary sampling task before claiming backfill coverage.
