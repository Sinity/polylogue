---
created: 2026-07-10
purpose: Architecture-fit design for consuming Sinnix attested agent-job evidence
status: conditional-proposal
project: polylogue
---

# Agent-job evidence integration

## Decision

The only architecture-fitting Polylogue integration is a **thin, read-only
external adapter**, but do not implement it or create its Bead against Sinnix
commit `a4fdef4c` exactly as it stands. First finish a small consumer contract
in Sinnix, then run a paired workflow using Sinnix CLI refs directly. Build the
Polylogue projection only if that comparison demonstrates recurring integration
cost or a coordination decision that the adapter would improve.

Do **not** add an agent-job table, a new ingest provider, a scheduler, a queue,
or an interrupt surface to Polylogue. Do **not** materialize these manifests as
`session_runs` or `session_observed_events`. A launcher receipt is evidence
about an external execution; it is not a chat session, an archive-derived run,
or proof that the job's claimed work is correct.

If that gate passes, the highest-leverage product slice is:

> `agent_coordination(view="conflicts")` and `polylogue agents conflicts
> --json` show bounded, repo-scoped, attested external executions with stable
> job/work-item/artifact refs, while managed processes are de-duplicated from
> the noisy `ps` heuristic. Sinnix remains the sole owner of launch, liveness
> attestation, resource policy, wait, and interrupt.

This would repair a demonstrated weakness in `s7ae.1` rather than create an
orchestrator. Direct Sinnix CLI use is a legitimate terminal state if the
paired test shows that composing the same refs into Polylogue adds no behavioral
value.

## Evidence read

### Sinnix job contract at `a4fdef4c`

`run_agent_prompt.sh` atomically emits schema-v1 manifests containing:

- job ID and recorded lifecycle (`starting|running|completed|failed`);
- backend, model, effort, declared role, and declared work item;
- worktree, branch, prompt path/hash, and log/JSON/final artifact paths;
- launcher PID, systemd scope/cgroup, requested resource overrides, and exit
  status.

`agent_job_control.sh` lists/statuses manifests and permits destructive control
only by job ID after PID/cgroup/worktree checks. This is materially safer than
the PID/title inference used during the current coordination session.

The current contract has four gaps relevant to a Polylogue consumer:

1. `repo` is `git rev-parse --show-toplevel`. In a linked worktree it equals
   the worktree path and therefore duplicates `worktree`; it does not identify
   the shared repository. Verified locally: the canonical Polylogue checkout
   and `/realm/tmp/worktrees/polylogue-runtime` have different top-levels but
   the same absolute git common dir, `/realm/project/polylogue/.git`.
2. There is no exact provider/session ref. Polylogue cannot honestly join a job
   to `codex-session:*` or `claude-code-session:*`; model/worktree/time inference
   is not sufficient for identity.
3. Interrupting the scope can bypass the final `write_manifest` call and leave
   recorded lifecycle as `running`. Recorded lifecycle and observed liveness
   must remain separate.
4. The read surface exposes raw `ActiveState`, `SubState`, `ControlGroup`, and
   systemd resource-property names. Those are Sinnix implementation details,
   not Polylogue product vocabulary. The helper is also a skill script, not yet
   a stable installed consumer command.

The coordinator reports that follow-up Sinnix work has already repaired a
collected-scope list failure and abnormal-exit/scope-launch lifecycle
finalization. Those changes should be verified from their eventual commit and
tests; they reinforce that `a4fdef4c` is a baseline under correction, not a
consumer contract Polylogue should freeze prematurely.

### Existing Polylogue coordination shape

`polylogue/coordination/envelope.py` already composes a JSON-first envelope and
exposes it unchanged through CLI and MCP. `s7ae.4` added archive-derived session
trees, activity, proof refs, subagent exchanges, and context refs.

The live peer/resource half remains heuristic:

- `_process_payloads` scans `ps` and matches `codex|claude|gemini` or resource
  words anywhere in the full command line;
- a harmless argument or internal harness payload containing one of those
  words can become a false peer/resource;
- the resulting identity is a PID plus guessed kind, not a durable execution
  handle;
- handoff discovery still points at the retired
  `.agent/conductor-devloop/*.md` scaffold.

This explains the noisy `systemd-journald`/internal-plumbing observations from
dogfooding. The Sinnix manifest is useful primarily because it can replace
guesswork for **managed** jobs. It should not be stretched into archive
semantics it does not have.

## Architecture-fit gate

### Option 1: no Polylogue integration

**Fit:** cleanest ownership. Agents call Sinnix job controls directly.

**Value:** it is already sufficient for safe control. A coordinator can list
Sinnix jobs, retain exact refs, and pass them to status/wait/interrupt without
any Polylogue change. It avoids cross-repo coupling and may be the fastest
workflow once wrapped in the shared agent-orchestration skill.

**Cost:** Polylogue's advertised coordination envelope continues to misidentify
managed peers/resources and cannot hand an agent the stable job/work-item/result
refs that now exist. The coordinator must manually reconcile Sinnix's exact job
view with Polylogue's session/work-item/proof view, and each agent must know to
call both.

**Verdict:** required until the Sinnix consumer contract is complete and
possibly the best terminal state. Run a paired behavior test first. The adapter
pays for itself only if manual reconciliation causes a missed/incorrect
coordination decision, repeated prompting/tool calls, or false peer/resource
conclusions that a composed view removes.

### Option 2: thin external adapter

**Fit:** Sinnix publishes a normalized read snapshot; Polylogue converts it to
a bounded coordination projection. No persistence, no second lifecycle owner,
no actuator, and no new MCP tool.

**Value:** exact job/worktree/work-item identity for managed executions;
artifact discovery; truthful recorded-vs-observed state; de-duplication of
managed PIDs from `ps`; one compact view available to every MCP-capable agent.

**Verdict:** conditionally recommended after the Sinnix prerequisite and paired
behavior gate below. Do not build it merely because the DTO is easy to add.

### Option 3: native Polylogue object/model

Possible versions include a durable `execution_jobs` table, a new source
provider, or insertion into `session_runs`/`session_observed_events`.

**Why it is wrong now:**

- the job is not a session and has no exact session identity;
- `session_runs` are rebuildable projections derived from archived sessions,
  while XDG job manifests are external mutable receipts;
- an index rebuild would either lose the external facts or unexpectedly depend
  on live Sinnix state;
- provider parsing already derives main/subagent runs, so a launcher-derived run
  risks duplicate executions with incompatible identity/status semantics;
- native storage would force retention, deletion, privacy, and source-of-truth
  policies before any demonstrated historical query needs them;
- prompt/final/log locators are not verification receipts and must not enter
  `proof_refs` merely because the process exited zero.

**Verdict:** reject. Reconsider only after exact session linking exists and
there are repeated historical queries that the live adapter cannot answer. At
that point the general analysis/provenance substrate (`rxdo`) is a better place
to evaluate an external-execution object than the coordination envelope.

## Ownership boundaries

### Sinnix owns

- launch, wait, status, interrupt/cancel, and all destructive authorization;
- PID, cgroup, worktree, and service/scope attestation;
- resource policy and its systemd representation;
- atomic manifest evolution, compatibility, retention/pruning, permissions,
  and artifact-file lifecycle;
- normalized observed state and the distinction between recorded and live;
- a stable repository identity across worktrees;
- the stable installed read command and its JSON contract;
- eventually, adapters for cloud-task handles if Sinnix chooses to unify them.

### Polylogue owns

- a read-only adapter with a hard timeout and graceful absence/degradation;
- bounded repo/work-item filtering and projection into agent vocabulary;
- provenance, source/recorded timestamps, and explicit uncertainty;
- de-duplication of externally managed jobs from fallback process heuristics;
- the shared CLI/MCP/rendered projection;
- optional exact links to existing session/work-item/artifact refs when the
  producer supplies them. Polylogue must not infer a session link.

### Vocabulary that must not leak into Polylogue payloads

`systemd`, unit/scope names, cgroup paths, `ActiveState`, `SubState`,
`MemoryHigh`, `MemoryMax`, `CPUWeight`, and `IOWeight` remain behind the Sinnix
adapter. Polylogue exposes only generic terms such as `recorded_state`,
`observed_state`, `worktree`, `work_item_ref`, and artifact refs.

## Sinnix prerequisite

Amend `sinnix-056.1` before treating its read output as a cross-project
contract:

1. Add `repo_identity` based on the absolute git common dir (or an equivalent
   stable repository ID) while retaining worktree separately.
2. Make terminal interruption/cancellation persist a terminal recorded state,
   or expose it as incomplete while normalized observation reports stopped.
3. Install one stable read/control command, suggested shape:
   `sinnix-agent-job list --json` and `sinnix-agent-job status --job ID --json`.
4. Emit normalized consumer fields:
   `observed_state = running|stopped|unknown`, `attested = true|false`, and
   `observed_at`; keep raw systemd details in an optional Sinnix-only detail
   mode.
5. Add optional exact `session_ref` when a backend can attest it. Absence must
   remain `null`, never inferred.

This is not feature creep: `sinnix-056.1` already claims repo/worktree binding,
live status, and lifecycle. The additions make those existing claims true for
linked worktrees and external consumers.

Cloud tasks are out of this first contract. Commit `a4fdef4c` attests local
headless processes; presenting Codex Cloud tasks as if they had the same PID,
cgroup, or lifecycle proof would be false. A future Sinnix cloud adapter may
emit the same normalized consumer vocabulary with a different source.

## Minimum Polylogue slice

### Adapter

Add a small module under `polylogue/coordination/` that calls the stable Sinnix
read command through the existing bounded `CommandRunner`. Do not scan
`$XDG_STATE_HOME` directly; producer-owned discovery and schema evolution
belong in Sinnix.

The adapter:

- times out through the existing two-second runner;
- accepts only the supported consumer schema/version;
- filters by exact `repo_identity`, then newest-first by observation/update
  time;
- clamps rows to the envelope limit before rendering;
- reports unavailable, invalid, and partial states without raising;
- does no filesystem artifact reads and no control calls;
- correlates a PID only when Sinnix says the observation is attested.

### Payload

Add two small typed payloads:

```text
CoordinationExecutionSourcePayload
  source                    # "sinnix-agent-job"
  state                     # available|unavailable|invalid|partial
  observed_at
  valid_count / invalid_count
  provenance

CoordinationExternalExecutionPayload
  external_ref              # opaque producer ref, not a Polylogue ObjectRef
  backend / model / effort
  recorded_state / observed_state / attested
  recorded_at / observed_at
  repo_identity / worktree / branch
  work_item_ref / role
  session_ref               # optional, producer-attested only
  prompt_sha256
  artifact_refs             # bounded file refs/locators, never bodies
  exit_status
  provenance
```

Add `execution_source` and `external_executions` to
`AgentCoordinationPayload`. Keep the public representation free of launcher PID
and systemd/resource-policy fields unless a later demonstrated workflow needs a
generic resource measurement.

### Projection and surfaces

- Include external executions in `status` and `conflicts` views.
- Keep `self`, `work-item`, and `handoff` views unchanged unless an exact
  self/session link is present.
- Tighten `conflicts` to the live coordination subset: peers, external
  executions, resource episodes, overlaps, repo/work-item, and adapter health.
  Do not include session trees, full subagent exchanges, proof summaries, or
  context snapshots in this view. This makes it the compact agent poll.
- Reuse `polylogue agents conflicts --json` and the existing
  `agent_coordination(view="conflicts")` MCP tool/prompt. No new MCP tool,
  command family, queue, or write role.
- For managed jobs, de-duplicate the matching PID from `peers` and
  `resource_episodes`. Keep a narrowly classified `ps` fallback for unmanaged
  interactive agents, but match executable/`comm` identity rather than arbitrary
  argument substrings.
- Enforce an output budget for `conflicts`, for example <=8 KiB at `limit=5`,
  with artifact refs and summaries bounded. Evidence detail remains reachable
  through refs/Sinnix status.

### Semantics

- `recorded_state=completed` plus `exit_status=0` means the launcher completed;
  it does not mean tests passed or the patch is correct.
- `observed_state=running` is shown only when the Sinnix observation is
  attested. A stale manifest is not a running peer.
- Artifact paths are locators, not proof refs. Proof continues to come from
  structured archived tool/test/command outcomes.
- Same-repo/same-work-item execution is awareness, never automatically
  blocking.
- No context injection is part of this slice. `37t.11` may later admit a
  compact ref through the one scheduler if outcome data shows it is useful.

## Proof tests

Use consumer-contract fixtures and the existing fake `CommandRunner`; do not
start systemd or a real model in Polylogue unit tests.

1. **Producer contract fixture:** a supported normalized Sinnix response with
   two linked worktrees sharing one `repo_identity` projects both jobs into the
   same canonical-repo envelope.
2. **Repo isolation:** a different git common dir and a lookalike path prefix
   are excluded.
3. **Recorded versus observed:** a manifest recorded `running` but observed
   `stopped` is not counted as an active peer; an attested running job is.
4. **Unknown/malformed version:** the envelope degrades with explicit adapter
   state/counts and does not crash or reinterpret fields.
5. **Boundedness:** newest-first result clamp, bounded artifact refs, and a
   serialized `conflicts limit=5` size assertion.
6. **De-duplication:** an attested managed PID appears once under external
   executions, not again as a heuristic peer/resource.
7. **False-positive regression:** a process whose arbitrary argument mentions
   `codex`, `claude`, `python`, or `nix` is not classified solely from that
   substring.
8. **Surface parity:** CLI JSON and MCP return the same typed fields; no new MCP
   registry entry is introduced.
9. **No control:** a spy runner proves the adapter invokes only the read/list
   command and never `interrupt`, `stop`, or another mutating verb.
10. **Anti-vacuity mutation:** remove/change `repo_identity` or the attestation
    bit in the fixture and assert the same-repo/active-peer expectation fails.

One live dogfood proof after unit verification should launch two lightweight
jobs in separate worktrees, show both from the canonical checkout, stop one via
Sinnix, and show it cease to be active without editing Polylogue state. Capture
the before/after envelope and exact external refs. This proves composition, not
model output quality.

## Conditional Beads surgery

Do not perform these Beads edits yet. They are the exact surgery to apply only
after the Sinnix producer contract is stable and the paired behavior gate shows
that direct Sinnix CLI refs are insufficient. Canonical checkout safety remains
a separate prerequisite.

### Then create `polylogue-s7ae.7` (P1, size S/M)

**Title:** `Project attested external agent jobs into the coordination envelope`

**Description:** The shipped coordination envelope guesses peers/resources
from command-line keywords, producing false identities and no durable handle.
Sinnix now owns attested local job handles. Consume its normalized read contract
as bounded external evidence so agents can see exact managed jobs without
turning Polylogue into an executor.

**Design:** the Minimum Polylogue slice above. Explicit non-goals: persistence,
new provider/source, scheduler, queue, launch/wait/interrupt, systemd fields,
session inference, and treating exit zero/artifact existence as proof.

**AC:** the ten proof-test clauses above plus the two-worktree live dogfood
artifact. Require the `conflicts limit=5` compact budget and CLI/MCP parity.

**Relationships:**

- parent-child: `polylogue-s7ae`;
- `polylogue-s7ae.1` blocks `.7` (the closed envelope is its substrate);
- `.7` blocks `polylogue-s7ae.5` (the live proof should use attested jobs, not
  process-keyword guesses);
- relates-to `polylogue-s7ae.3` only; message/advisory delivery does not block
  this read adapter.

### Amend existing coordination beads

- **`polylogue-s7ae`**: append a boundary note: external runtimes own
  execution/control/resource policy; Polylogue composes observation, refs,
  outcomes, and context. “Operational” does not authorize a scheduler.
- **`polylogue-s7ae.1`**: do not reopen. Append a post-close correction naming
  the `ps` false-positive/unstable-identity gap and pointing to `.7`.
- **`polylogue-s7ae.4`**: no scope change and do not reopen. It composes
  archive-derived evidence; `.7` is an orthogonal external live-evidence
  adapter.
- **`polylogue-s7ae.3`**: append that future resource advisories consume the
  normalized generic execution projection, never systemd/cgroup details. Do
  not make `.3` and `.7` block one another.
- **`polylogue-s7ae.5`**: replace the process-table proof leg with `.7`'s
  attested external refs. Remove every reference to
  `.agent/conductor-devloop/*.md`; use a real scoped blackboard/HANDOFF
  assertion delivered through `.3`/`37t.11`. Preserve awareness-not-blocking.

### Context spine classification

- **`polylogue-37t.11`**: orthogonal to implementation. It remains the only
  future context-injection arbiter; `.7` must not wait for it and must not
  inject anything. `.5` already depends on it for the later live proof.
- **`polylogue-37t.12`**: no relationship. It is the candidate-assertion
  judgment queue, upstream of trusted context injection. Mapping an execution
  manifest onto it would confuse operator judgment with process observation.

### Sinnix bead

Before closing **`sinnix-056.1`**, amend its AC with the five Sinnix prerequisite
items above, especially shared git-common-dir identity, normalized attested
observation, interruption terminal state, and a stable installed JSON read
command. Link `sinnix-056.1` and `polylogue-s7ae.7` by external reference in
their notes; neither repo becomes the other's task authority.

## Sequencing and drop order

1. Finish and verify the Sinnix consumer contract in `sinnix-056.1`.
2. Run two lightweight worktree jobs using **only** `sinnix-agent-job` refs for
   execution control while also capturing the current Polylogue conflicts
   envelope. Record whether manual reconciliation changes or delays a concrete
   coordinator decision, and whether current Polylogue output creates a false
   conclusion.
3. If direct Sinnix CLI use is sufficient, stop: keep option 1 and improve the
   orchestration skill/docs instead of Polylogue.
4. If the paired evidence proves composition value, create `.7` and land one
   focused Polylogue PR: adapter + payload + compact projection +
   de-dup/classifier repair + tests.
5. Repeat the same paired scenario through Polylogue and capture before/after
   JSON plus tool-call/output-size changes.
6. Continue `.3`/`37t.11`/`.5` for messages, trusted context delivery, and the
   full paired-agent proof.

If step 1 does not land, do not create the Polylogue Bead or scan private state
or duplicate Sinnix attestation logic. If either paired proof shows the compact
view does not change a coordinator decision, reduce tool/reconciliation cost,
or eliminate false peers, retain the Sinnix controls and stop at option 1; do
not promote the adapter into a native model to justify the work.
