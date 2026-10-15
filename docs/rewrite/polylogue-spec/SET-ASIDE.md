# Set aside

Features and scope that the rewrite does not build now, kept here so nothing
is forgotten. Nothing on this list is rejected. Each entry says what it was,
why it is set aside, and what would bring it back. When an entry comes back,
it moves into the spec and its row here is deleted.

Mechanisms the rewrite replaces rather than defers (retry ledgers, repair
commands, periodic loops, the operation registry) are not listed here. They
are in `evidence/python-not-carried.md` with the rule that replaces each.

## Python surfaces not rebuilt

Measured use is from `evidence/deep-surface-use.md` (the operator's shell
history, archived agent tool calls, Lynchpin's imports).

| Python surface | What it did | Why set aside | What brings it back |
|---|---|---|---|
| `compare` | blind pairwise comparison of two sessions or answers | partly planned: `comparison` notes with blind judging are in slice 12 (`spec/user-data.md`, Judgment); the side-by-side reading view for two references is not designed | a person ranking variants side by side; the view reads two references and writes a `comparison` note |
| `judge` (assertion candidates) | triage of candidate claims extracted from sessions | no recorded use | claims (`plan/slices.md`, slice 7): build judging when claims exist to judge |
| `annotations` | batch import of typed annotations with provenance | no recorded use | an outside tool producing annotations in bulk; notes already carry kind, author and revision, so this is an import format for notes |
| `insights`, `facets` | registry-driven insight reports; counts of sessions by field, optionally within a search | three uses each | the query language's grouping covers both (`spec/query.md`); a dedicated view only if grouping proves awkward in use |
| `agents`, MCP `agent_coordination`, `blackboard_list`, `work_threads` | which agents work where; a shared blackboard; sessions grouped into work threads | coordination calls were real (61 in four months) | live state replaces the first (`spec/model.md`, live state). Work threads (sessions grouped by repository and lineage, shown as one line of work) and a shared blackboard are not designed yet: the `context` view and peers cover part; design them with the workbench (slice 12) |
| `materialize-incident-evidence`, `reconcile-work-effects` | built a graph of work evidence from sessions, then checked its claims against observed git and Beads effects | no recorded use | commit links, PR links and run-task linkage exist (`spec/analysis.md`); reconciling what an agent claimed against what changed fits claims (slice 7) |
| `debt` | listed pending convergence work by kind | replaced: staleness is a query and `status` shows waiting work | not coming back as a command |
| `scan-secrets` | a separate pass that flagged secrets in stored text | replaced: the writer computes `looks_like_secret` | not coming back as a command |
| `backup` | copied the durability tiers | the archive's backup is the SQLite backup API plus the blob files (`spec/storage/README.md`), run by Sinnix's backup | a person wanting Polylogue to run its own backup schedule |
| `diagnostics`, `paths`, `maintenance` | archive and daemon checks, canonical paths, maintenance tasks | `status` and `doctor`-style output cover checks; maintenance commands are not built (invariants instead of repair) | not coming back as commands |
| `tutorial`, `manual`, `agent` (cold-start manual) | first-run checklist; CLI documentation; agent integration text | help, completions and `explain` cover them | a new user other than the operator |
| `demo` | seeded a deterministic demo archive | the Morrow example archive is the demo (`examples/archive/`) | already back, in that form |
| Python API, 133 of 148 methods | library calls with no outside caller | Lynchpin uses about 15; the rewrite's client is 10 calls | an outside caller that needs one |
| MCP `write`, `run`, `maintenance`, `record_work_event`, `emit_decision` | writes and work records from agents | no calls under these names | `note` covers agent writes; work events return with runs if agents need to record them explicitly |
| HTTP API for other programs, with bearer tokens | 51 routes, usable by any local program | other programs use the socket client; the web reader is the only HTTP user (`decisions/process.md`) | a program that cannot speak the socket protocol |
| Analysis MCP tools (`find_abandoned_sessions`, `workflow_shape_distribution`, `cost_rollups`) | one tool per report | rarely called and mostly failing | `search` with grouping covers them; a per-report tool only with a caller |
| "Phases" (`get_session_phase_insights`) | split a session into heuristic phases | the phases were heuristic and unverified | a phase definition checked against sessions the operator labels |
| `devtools verify` evidence lane (`$XDG_STATE_HOME/polylogue/verification/evidence.jsonl`) | one `polylogue.verification-receipt` row per verification run: status, revision, per-step outcomes and times (RFC 3339 with offset); 12,225 rows on this machine; Sinex reads it as `work.polylogue-verification` | it records the Python repository's own custom verification tool, which the rewrite replaces with plain `cargo test` and flake checks (the operator's "tests over custom verification", `evidence/intent-web-journal.md`). The existing file is left in place and stays readable by Sinex | a need to analyse Rust test history across checkouts: then a test-run lane from the test runner's machine-readable output, declared to Sinex as a new source |

## Sources not built yet

Listed with their parsers' prerequisites in `spec/sources/others.md`,
"Planned sources": OpenTelemetry GenAI span files, generic OpenAI chat
arrays, the Codex AppServer event stream, Meta Muse Code, Claude Code Web
runs, A2A task streams.

Claude Code workflow run records (`<session>/workflows/wf_*.json`) and
workflow journals (`subagents/workflows/<run>/journal.jsonl`) are stored but
not parsed (`spec/sources/claude-code.md`). The Python version parses both
into facts (`polylogue/sources/parsers/claude/orchestration.py`): run ID,
agent ID, attempt ID, the spawning tool call's ID, agent type, description,
and whether an entry is the run's result. Without them a workflow's
subagent transcripts still belong to the session that ran the workflow
(their records carry its session ID), but not to a particular Workflow tool
call, a run, or an attempt, so a retried attempt is not told apart from the
one whose result was used. 20 runs on this machine on 2026-10-12. What
brings it back: a parser mapping journal entries and run records to
subagent links with run and attempt attributes, and the result flag to a
`used` mark on the attempt; the bytes are already stored.

## Later features already in the plan

Research features and the workbench are placed in `plan/slices.md`
(slices 7–12) and are built on first real use, not dropped.

## Designs set aside

| Design | What it would do | Why set aside | What brings it back |
|---|---|---|---|
| Learned defaults | the archive notices how settings and flags are actually used (a flag passed on most runs) and proposes a default as a judged candidate; accepting writes the setting | no recorded need since the rewrite; it would be a watched query over usage plus a judgment, nothing new (`polylogue-1jc`, from its inherited learned-defaults acceptance) | the operator repeating the same flags often enough to want them proposed; build it from the session records already stored (the commands agents ran, the flags in them) |
| Sealed sessions in Polylogue's own blob store | encrypt each session's blobs with its own data key, wrapped by a key the operator holds (as Sinex's sealed class, `sinex-spec/decisions/privacy.md`), so excision becomes deleting a key and reaches old backups once the key table's backup retention passes | without Sinex, filler rewrites already remove the content from the archive itself; sealing adds key management for the one gain of bounding backups | the operator wanting excision to reach old backups without running Sinex; the `hashed-blobs` crate's sealing step (`put_sealed`, in the format of joint #23: a fresh per-encoding frame key derived from the data key, an OS-random salt and the keyed blob name, with authenticated version, salt, layout and frame metadata; sealed blobs named by a keyed hash, joint #24) would be shared with Sinex. Sealing a session would also need keyed hashes wherever Polylogue stores a hash of its content (`text.hash`, an imported input's ID, excision refused lists), or those would confirm the content as the blob name would |
| A firehose view | every chat from every source scrolling past at full detail (journal, 2026-05-09) | not designed; the live view and the change feed cover what is happening now | a design on the canvas, built on the change feed |
| Editing a harness's current state | treat the archive as the record and trim or reshape a harness's live session file (journal, 2026-05-03) | it writes into provider files, which Polylogue only reads; needs a safety design | a harness API for replacing context, or the operator asking for it with a recovery plan |
| Agent sessions behind an MCP | start, address and steer agent sessions of any harness through one MCP (journal, 2026-05-12) | pings (slice 12) cover addressing one session; a general session manager is a larger product | the workbench in use and a need beyond pings |
| Local-model analytics | analysis by a local model (journal, 2026-05-08: something fitting 10 GB of VRAM) | embeddings plan a local model later (`spec/analysis.md`); other model-based analysis goes through notes and passes | a concrete analysis worth running locally |

## Gaps found by the vision catalog

From `vision/uses.toml` (`python3 vision/check_uses.py --gaps` lists the uses
that need each). Gaps fixed in the spec, and those that need Sinex, are in
`vision/README.md`.

| Gap | What brings it back |
|---|---|
| Stated plans and todo lists are not units (Claude Code `todos/` and `plans/` stored but not parsed; plans stated in chats unstructured) | A `plans` unit: TodoWrite calls and todo files parsed into items with state and time, plan files as documents, and chat-stated plans as claims written by an annotation pass; then "stated but never done" is a set operation (query.md, planned extensions) |
| No dataset export for training or evaluation | An export mode that writes (context, prompt, accepted outcome) records with filters, de-duplication and the privacy redaction rules, from sessions with an `outcome` note |
| No entity identity across sessions and sources (a person, project or concept mentioned in several sessions, or in Polylogue and Sinex, is not one entity) | Set aside with Sinex's knowledge-graph layer (the operator, 2026-05-25: "the knowledge graph likely goes away"; `sinex-spec/SET-ASIDE.md`). The way back is the narrow form: identity only from shared identifiers (a repository path, an address, a handle) or an operator confirmation, never from names in text; repositories already join through `repo` and `cwd`. A joint gap (`polylogue-sinex-joint/vision/uses.toml`) |
| No index of code facts across sessions | A derived table of files, symbols and the sessions that read or changed them, from tool calls' paths and edits; directory briefings would read it |
