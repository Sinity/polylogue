# Polylogue Agent Conventions

Repo-local conventions for agents working in this checkout. Always-loaded
operating rules live in `CLAUDE.md` (= `AGENTS.md`); this file holds the
repo-agent conventions that do not need to be in every context window.
Kept deliberately parallel to `sinex/.agent/CONVENTIONS.md` — divergences are
intentional and marked.

**Task authority is external.** Do not recreate task workspaces, task-state
exports, or branch/PR carriers in this repository. A configured append-only
interaction ledger is archive evidence only; it is never a worktree control
surface.

## Dispatch Agent Definitions

`.claude/agents/lane.md` and `.claude/agents/triage.md` are the canonical
standing contract for dispatched worktree agents in this repo — worktree
discipline, red-first bug fixes, verification commands, PR shape, and
never-merge. Dispatch prompts that use `subagent_type: lane` or
`subagent_type: triage` only need to carry task-specific content (which
bead/files/scope); the operating rules are already loaded from the agent
definition, not re-typed per dispatch. If the standing contract changes,
edit those two files, not the per-dispatch prompt template.

## Directory Shape

```text
.agent/
  README.md          # orientation (tracked)
  CONVENTIONS.md     # this file (tracked)
  demos/             # curated demo shelf (tracked)
  handoffs/          # external-agent handoff corpora (tracked)
  reports/           # report artifacts — gitignored, local-only
  task-history/      # historical task notes — gitignored, local-only
  scratch/           # gitignored thinking space
  archive/           # retired scaffolds kept as evidence — gitignored, local-only
```

## Execution Tactics

- **Async is coordinator-only.** An interactive/coordinator session may
  background long test runs and imports and do light work while they run.
  Dispatched lane agents (worktree subagents) must run every command
  synchronously in their own foreground turn — never launch a background job
  and idle-wait on it across turns (2026-08-01: three lanes each stalled for
  multiple turns "waiting for the background job", burning wall-clock and
  coordinator attention until manually interrupted).
- **No poll loops on background agents (coordinator).** Completion
  notifications for background subagents are automatic — never
  `ScheduleWakeup`/`Monitor` as a "done yet?" poll loop. Full rule + the two
  legitimate exceptions (genuine `Monitor` until-conditions, genuine
  wall-clock `ScheduleWakeup` deadlines) live in `CLAUDE.md`'s "Coordinator
  dispatch: no poll loops" section (polylogue-kzse6) — this is the same rule
  the `lane`/`triage` agent definitions carry for a dispatched agent that
  itself spawns background work.
- **Serial heavy, parallel light.** Serialize anything sharing the pytest
  temp DBs (`/realm/tmp/polylogue-pytest`) or the archive DB; parallel tool
  calls are for reads/searches only.
- **Proof ladder.** Narrowest proof while iterating (`devtools test` with
  testmon affected-selection or a targeted `-k` filter); the broad suite once
  per publishable phase. Never blanket `pytest tests/unit` as an iteration
  step (see CLAUDE.md).

## Greedy Batch / PR Cadence

Batch related implementation into the largest coherent phase. A green
substep is a checkpoint, not a publishing trigger; split only at a genuine
boundary (risk, reviewability, dependency, ownership, deployment, or failure
isolation).

## Scratch, Demos, Git Boundary

`.agent/scratch/` is gitignored thinking space (README + research notes);
promote durable repository guidance to CLAUDE.md/CONVENTIONS.md. `.agent/demos/`
is a curated shelf, not a dump. The tracked `.agent` surface stays small;
everything else is ignored live state.

## Pathology Zoo Growth Rule

Every production incident adds its smallest production-ingested reproduction to `tests/infra/pathology_zoo.py` in the fix PR. Add the pathology label and motivating bead id to the queryable manifest with the fixture, rather than leaving the incident represented only by a comment or a parser-local example.

## Fixture Identifier Hygiene

This repo is **public**. Never commit a real session/conversation identifier
(Codex/Claude Code/ChatGPT/etc. UUID or native id) from the operator's live
archive into `tests/`, `docs/`, or `.agent/` — even as a bare id with no
transcript text attached, and even inside handoff/demo evidence bundles.
Test fixtures, doc examples, and demo receipts must use freshly generated
synthetic UUIDs (`python3 -c "import uuid; print(uuid.uuid4())"` or
equivalent) that do not resolve against the live archive. Before adding a new
fixture id, or copying one out of a real ingest/debug session, generate a
fresh one rather than reusing what you saw on screen. If you discover a real
id already committed at tip, replace it in place (keeping any paired
assertions consistent) rather than leaving it — polylogue-b629 is the
precedent (17+ live-archive identifiers found across tests/docs/.agent
2026-07-31, fixed by wholesale synthetic-id substitution across the affected
files; see that bead's history before deciding whether an occurrence is real
or already-synthetic).
