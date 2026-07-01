# Polylogue Conductor Devloop

This is the tracked contextless resume entrypoint for the active Polylogue
dogfood/demo loop.

The process docs and helper scripts are tracked. Current loop state is local and
ignored: `ACTIVE-LOOP.md`, `OPERATING-LOG.md`, `EVENTS.jsonl`, `DEMO-RADAR.md`,
generated manifests, demos, scratch notes, and task history are not meant to be
committed.

If the current chat history is cleared and the operator says only:

```text
continue the devloop setup in .agent
```

then do this:

1. From `/realm/project/polylogue`, read `.agent/DEVLOOP.md`.
2. Run `.agent/scripts/devloop-status`.
3. Run `.agent/scripts/devloop-review`.
4. Read `RUNBOOK.md`, `INTEGRATION.md`, and `INDEX.md`.
5. If local `ACTIVE-LOOP.md` exists, read it and continue that slice.
6. If local state is absent, start in `Direction`: choose a slice from the goal,
   current repo state, and any available archive/daemon evidence.
7. If review reports stale generated scaffold state, run
   `.agent/scripts/devloop-sync` and review again.
8. Use the focus modes in `RUNBOOK.md`: Direction, Evidence, Construction,
   Proof, Artifact, Velocity, and Meta.
9. Record material transitions with `.agent/scripts/devloop-focus`; start new
   slices with `.agent/scripts/devloop-start`.
10. Before ending, refresh/review and leave `ACTIVE-LOOP.md` plus
   `OPERATING-LOG.md` resumable.

The process goal is indefinite: continuously choose the highest-value
live-archive capability slice, produce inspectable artifacts proving Polylogue
improves agents with real history, collapse silos into shared
acquisition/query/projection/rendering substrate, verify on the active archive
or live capture, maintain logs and handoffs, and reprioritize by evidence.

Default state lives here, not in `/realm/inbox`. The default demo shelf is
`.agent/demos`, and it is current-curated rather than append-only.

## How The Devloop Works

The conductor loop is a lightweight operating system for long-running
Polylogue dogfood work. It is not a one-shot task plan. Its job is to keep the
agent oriented around live evidence, current demos, reusable substrate, and
process health across context loss and across many small slices.

The loop has seven focus modes:

- `Direction` chooses the next capability slice.
- `Evidence` inspects the current source tree, active archive, daemon state,
  logs, issues, docs, and demo artifacts.
- `Construction` edits code, docs, scripts, or artifacts.
- `Proof` verifies the exact claim with the narrowest sufficient command.
- `Artifact` makes the result inspectable outside chat, usually under
  `.agent/demos` or in the conductor packet.
- `Velocity` removes or records friction that slows the next loop.
- `Meta` improves the devloop itself when the process drifts or the operator
  corrects the agent.

Material focus changes should be logged with:

```bash
.agent/scripts/devloop-focus <from> <to> "<trigger>" "<decision>"
```

The normal slice flow is:

1. Run `.agent/scripts/devloop-status` and `.agent/scripts/devloop-review`.
2. Read `ACTIVE-LOOP.md` when present.
3. Pick one slice from current evidence, not from stale memory.
4. Record the slice with `devloop-start` or `devloop-log`.
5. Gather evidence before editing.
6. Make the smallest coherent shared-substrate or artifact change.
7. Prove the specific claim.
8. Update the demo shelf or conductor state when the result should survive.
9. Run `devloop-sync` and `devloop-review` before claiming a clean checkpoint.

The proof ladder is:

- source review proves shape;
- focused tests prove parser/storage semantics;
- CLI/API/MCP/daemon probes prove surface contracts;
- real archive artifacts prove operator value;
- broad `devtools verify` proves phase readiness, not every small edit.

## State Files And Their Roles

- `ACTIVE-LOOP.md` — current slice, focus transition, accepted warnings, and
  next action. It also carries `Meta Origin: yes/no` so process slices remain
  auditable after they move from `Meta` into evidence, proof, or velocity.
- `OPERATING-LOG.md` — timestamped log of decisions, evidence, actions, proof,
  and next decisions.
- `EVENTS.jsonl` — structured event sidecar generated from the operating log.
- `DEMO-RADAR.md` — demo candidates, selected artifact, proof/caveat, and next
  demo question.
- `RUNBOOK.md` — operational protocol and focus-mode rules.
- `INTEGRATION.md` — PR-shaped replay and publication protocol for the
  long-running workbench branch.
- `PROCESS.md`, `TACTICS.md`, `VELOCITY.md` — compact process, tactical, and
  speed rules.
- `SELF-PROMPTS.md` — durable goal, primary self-prompt, adversarial prompt,
  and tactical prompt.
- `ADVERSARIAL-REVIEW.md` — known failure modes and mitigations.
- `INDEX.md` — tracked routing guide for the active packet and archived-context
  boundary. This must survive checkout; current state belongs in ignored files.

Current local state is intentionally ignored by git. The tracked scaffold tells
a future agent how to resume; the ignored state tells it where this exact loop
currently is.

Durable cross-loop conventions live in `.agent/includes/`. Do not bury durable
architecture or process memory only in ignored conductor history.

`devloop-script-hashes.tsv` is generated local state, but it is load-bearing:
`devloop-sync` refreshes hashes for every `devloop-*` primitive and
`lib-devloop`, and `devloop-review` fails stale hashes so a future agent can
trust that the packet describes the scripts currently in the checkout.

## Backlog And Prioritization Model

The devloop does keep future workload, but it is deliberately evidence-shaped
rather than a static ticket queue. The current priority comes from:

1. `ACTIVE-LOOP.md` — immediate slice and next action.
2. `DEMO-RADAR.md` — current demo pressure and the next demo question.
3. `OPERATING-LOG.md` — recent findings and next decisions.
4. `.agent/includes/` — durable cross-loop conventions and architectural
   direction.
5. `.agent/archive/conductor-history/` — archaeology only, when older audit
   notes are specifically relevant.

Prioritize work in this order unless fresh evidence contradicts it:

1. Fix anything that makes live archive evidence false or ambiguous: wrong
   archive root, stale schema, duplicate daemon, stale generated state, or
   misleading counts.
2. Fix user-facing lies and construct-invalid wording: e.g. saying "pasted
   content" where the archive only proves paste evidence.
3. Repair query/projection/rendering substrate when a useful demo or CLI/API
   workflow would otherwise be bespoke or false.
4. Produce or refresh a current demo that proves a capability on the canonical
   archive.
5. Improve performance/velocity when command latency or host pressure is
   slowing repeated loops.
6. Do meta/scaffold work only when it leaves an executable consequence or a
   materially better next-slice decision.

## Meta And Self-Improvement

The devloop has an explicit `Meta` mode. Use it when the operator corrects the
agent, repeated friction appears, archive/process state becomes confusing, or a
loop feels vague. Meta work is only valuable if it changes future behavior.

Acceptable meta outcomes:

- a stronger `devloop-review` check;
- a sharper `devloop-status` or `devloop-velocity` signal;
- a corrected README/RUNBOOK/SELF-PROMPTS instruction;
- a cleaned scratch/current-state boundary;
- a new executable tripwire for stale archive roots, duplicate daemons, stale
  generated state, or empty log entries;
- a better prioritization rule grounded in evidence.

Non-outcomes:

- apology prose;
- broad process notes with no executable consequence;
- adding ceremony that does not speed up or harden the next slice;
- preserving compatibility trash under a new name.

When in doubt, make the next loop harder to derail and easier to resume.
