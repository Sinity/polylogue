# Polylogue presentability plan: a frontier-agent execution program

## Objective

Produce a public-facing Polylogue cut that a stranger can understand, run, inspect, and cite without requiring the entire roadmap to complete.

The presentable cut is not “all open Beads closed.” It is a coherent vertical slice:

```text
category statement
    → deterministic incident
        → semantic transcript rendering
            → evidence-resolved claim
                → explicit caveat/refusal
                    → public packet and recording
                        → reproducible install path
```

The plan deliberately favors a small number of forceful proofs over broad surface completion.

## Definition of presentable

The cut passes when a fresh reader can:

1. explain Polylogue in one sentence after viewing the repository page;
2. run one command from a supported install path without private data;
3. see a tool call and result rendered semantically rather than as generic chat;
4. watch “The Receipts” classify an outcome from structure and resolve it to evidence;
5. watch “Count It Once” distinguish copied lineage from unique work;
6. see one query correctly refuse an unsupported claim;
7. inspect a public claims ledger and proof packet;
8. distinguish shipped capability from roadmap;
9. find installation, demos, proof, security, architecture, and Sinex interop in two clicks;
10. reproduce the recordings from committed commands.

## The minimum presentable cut

### Required

- `polylogue-xyel`: real D1 receipts demo through the existing demo-packet contract;
- a new `polylogue-212` child for **Count It Once**;
- first useful vertical slice of `polylogue-ap7`: shell/test command and result cards in CLI and web;
- `polylogue-3tl.12`: README de-meta and claim tightening;
- `polylogue-3tl.15`: anti-grep proof card;
- a static first slice toward `polylogue-3tl.16`: claims YAML and linter, without falsely closing the full object-level claims program;
- `polylogue-3tl.4`: static public findings page for claim-vs-evidence, scoped to existing artifact evidence;
- `polylogue-3tl.5` maintenance/regeneration of visual recordings if needed;
- `polylogue-3tl.7`: prove at least source-checkout and Nix install lanes; do not wait for every packaging target;
- `polylogue-3tl.9`: public-doc and visual coverage check for the new surfaces;
- `polylogue-3tl.10`: launch copy and asset bundle;
- `polylogue-6bu`: docs-site build/link verification;
- `polylogue-ttu`: a narrow docs information-architecture pass;
- `polylogue-avg` or the smallest compatible readiness projection needed to label incomplete demo/query state;
- `polylogue-cpf.4` slice for the exact demo paths: timeout, degraded, truncated, unsupported, and incomplete signals must not look ordinary.

### Important but not launch-blocking

- full `polylogue-ap7` tool taxonomy;
- `polylogue-212.1` private/live multi-hour forensic variant;
- `polylogue-212.5` live self-capture;
- `polylogue-212.6` resumption experiment;
- outcome-conditioned cost (`polylogue-212.3`);
- full first-class finding objects and query-result provenance;
- multi-agent live proof (`polylogue-s7ae.5`);
- full Sinex-backed rebuild.

### Explicitly defer from first launch

- multi-model leaderboard (`polylogue-3tl.3`);
- productivity or generalized memory-uplift claims;
- “cost wasted on failures” headline without denominator coverage;
- full-day personal reconstruction;
- autonomous coordination or actuation;
- broad performance claims from one private archive.

---

# 1. Work decomposition

Use eight coding lanes plus one coordinator. More simultaneous writers increase merge entropy faster than throughput. Additional agents should review, design fixtures, audit claims, or test rather than touch shared files.

## Lane C0 — Integration coordinator

**Owns:** integration branch, Beads state, file-ownership ledger, merge order, final verification, public packet.

**Writes only:**

- `.agent/swarm/` coordination files;
- integration conflict resolutions;
- final release packet;
- Beads comments/status after evidence exists.

**Never implements a whole feature while coordinating.** The coordinator’s scarce resource is maintaining a coherent product cut.

## Lane C1 — Narrative and documentation

**Beads:** `polylogue-3tl.12`, `.15`, narrow `.16`, `polylogue-ttu`.

**Exclusive files:**

- `README.md`;
- `docs/demos.md`;
- `docs/public-claims.yaml`;
- `docs/sinex-interop.md`;
- `docs/proof-artifacts.md`;
- `devtools/docs_surface.py`;
- generated `docs/README.md` after registry changes.

**Deliverables:**

- one stable category statement;
- install/run path that matches current support;
- anti-grep card;
- explicit facts/capabilities/aspirations language;
- claims ledger;
- docs map.

**Verification:**

```bash
devtools render docs-surface
devtools render docs-surface --check
devtools verify doc-commands
```

## Lane C2 — Site and repository landing surface

**Beads:** `polylogue-3tl.4`, `.8`, `.9`, `polylogue-6bu`.

**Exclusive files:**

- `docs/site/pages.toml`;
- `devtools/pages_templates.py`;
- `devtools/pages_style.py`;
- `devtools/pages_builder.py`;
- site-specific tests;
- social-preview source asset and repository-settings checklist.

**Deliverables:**

- replace “Your AI memory” with the flight-recorder category;
- add Demos, Findings/Proof, and Sinex Integration to navigation;
- homepage cards for The Receipts, Count It Once, and Honest Refusal;
- clear link to the deterministic command;
- generated-site link and route check;
- platform settings checklist for description, topics, social card, and pinned artifacts.

**Verification:**

```bash
devtools render pages
devtools render pages --check
devtools verify docs-drift
pytest -q tests/unit/site
```

## Lane C3 — Shared Incident 14:32 fixture

**Beads:** supports `polylogue-212`, `polylogue-xyel`, new Count It Once child, and demo corpus audit.

**Exclusive files:**

- `polylogue/demo/` fixture/scenario implementation;
- dedicated demo fixture directories;
- `polylogue/demo/constructs.py` and independent oracle data;
- corresponding demo unit tests.

**Deliverables:**

- declarative scenario manifest;
- failed structural test and contradictory claim;
- successful control;
- copied-prefix lineage plus fresh subagent control;
- compaction boundary;
- accepted and stale assertion controls;
- context snapshot;
- attachment bytes;
- independent expected-result manifest.

**Design constraint:** fixture generation and oracle computation must not use the same reducer. The scenario declares ground truth; Polylogue independently derives it.

**Verification:**

```bash
pytest -q tests/unit/demo tests/unit/sources tests/unit/storage/test_lineage_normalization.py
polylogue demo seed --root /tmp/polylogue-incident-1432 --force --with-overlays --format json
polylogue demo verify --root /tmp/polylogue-incident-1432 --require-overlays --format json
```

## Lane C4 — The Receipts packet

**Bead:** `polylogue-xyel`, with closure evidence for the relevant slice of `polylogue-212.2` only when all ACs are met.

**Exclusive files:**

- `.agent/demos/receipts/` or canonical public packet directory;
- `devtools/demo_packet.py` extensions specific to packet v2;
- receipts runner and tests;
- generated packet/report/recording source.

**Inputs:** fixture contract from C3, rendering contract from C5.

**Deliverables:**

- `polylogue demo receipts` or equivalent canonical runner;
- claim/evidence two-column result;
- positive, negative, and unsupported controls;
- stable refs and exact outcome fields;
- packet validation and generated report;
- recording tape generated from the command.

**Verification:**

```bash
pytest -q tests/unit/devtools/test_demo_packet.py tests/unit/devtools/test_claim_vs_evidence.py
polylogue demo receipts --root /tmp/polylogue-incident-1432 --out-dir /tmp/receipts
python -m devtools.demo_packet validate /tmp/receipts/packet.yaml
```

## Lane C5 — Semantic transcript renderer

**Bead:** first vertical slice of `polylogue-ap7`.

**Exclusive files:**

- `polylogue/rendering/`;
- shared render descriptors;
- CLI transcript renderer adapters;
- web reader card renderer files agreed in the ownership ledger;
- rendering snapshots/tests.

**First slice only:**

1. shell/test command card;
2. structural result card with exit status, duration, and folded output;
3. file edit diff card;
4. generic fallback preserving unknown tool data;
5. evidence and caveat badges.

Do not attempt every provider tool spelling in the first branch. Normalize through semantic families and prove fallback behavior.

**Contract:** renderers consume typed, surface-neutral descriptors. CLI and web do not implement separate outcome logic.

**Verification:**

```bash
pytest -q tests/unit/rendering tests/unit/site/test_renderer_behavior.py tests/unit/cli/test_query_fmt.py
```

**Visual acceptance:** screenshot the same four records in CLI and web, compare semantic fields rather than pixel identity.

## Lane C6 — Count It Once

**Bead:** create a child under `polylogue-212`, related to `polylogue-4ts`.

**Exclusive files:**

- new demo runner;
- lineage demo packet/report;
- only demo-specific view adapter files;
- tests for expected physical/logical totals.

Avoid editing core lineage algorithms unless the independent oracle finds an actual defect. If a defect is found, stop and create a focused blocker for the owning lineage lane rather than hiding a repair inside demo code.

**Deliverables:**

- physical session count/usage;
- logical unique count/usage;
- replay/copy delta;
- shared-prefix topology evidence;
- fresh-subagent negative control;
- explicit disclaimer that logical accounting is not provider billing.

## Lane C7 — Readiness and loud degradation

**Beads:** compatible slices of `polylogue-avg` and `polylogue-cpf.4`; coordinate with active P1 accounting work rather than editing the same modules.

**Exclusive files:** chosen readiness/view-envelope modules and tests listed before work begins.

**Deliverables for launch paths:**

- demo/result envelope can state `complete`, `degraded`, `not_supported`, `timed_out`, or `truncated`;
- FTS/search freshness visible;
- missing modality/source visible;
- no successful-looking empty result on the demo routes;
- no broad redesign of all daemon readiness unless required.

**Verification:**

- targeted unit tests;
- honest anti-demo packet;
- inject stale FTS or missing data and ensure visible non-success state.

## Lane C8 — Install, launch, and visual artifacts

**Beads:** `polylogue-3tl.7`, `.9`, `.10`, existing visual-tape work.

**Exclusive files:**

- install proof scripts/receipts;
- `docs/examples/visual-tapes/` tapes and generated assets;
- launch-kit directory;
- distribution matrix docs;
- no changes to core demo logic.

**Deliverables:**

- source-checkout lane proof;
- Nix one-shot lane proof;
- optional package lanes clearly marked unproven until actually exercised;
- regenerated GIF/MP4 or terminal recording from canonical demo commands;
- announcement copy, short description, repository topics, social-card text;
- checksum manifest.

---

# 2. Dependency and merge order

The coordinator should merge in this order:

```text
1. C3 scenario/oracle substrate
2. C7 result-state vocabulary needed by demos
3. C5 shared semantic renderer slice
4. C4 The Receipts
5. C6 Count It Once
6. C1 narrative/docs against actual commands and outputs
7. C2 site routes and homepage
8. C8 install proofs, recordings, and launch kit
9. final integration fixes and generated surfaces
```

C1 and C2 can begin from design stubs but should rebase or reconcile after the actual command names and packet paths settle.

Do not merge polished copy before the corresponding command exists. Do not let an implementation agent casually rewrite public claims.

---

# 3. Worktree layout

From a clean integration checkout:

```bash
ROOT=/work/polylogue
WT=/work/polylogue-wt
BASE=master

mkdir -p "$WT"

git -C "$ROOT" switch -c presentable/integration

git -C "$ROOT" worktree add -b presentable/docs      "$WT/docs"      "$BASE"
git -C "$ROOT" worktree add -b presentable/site      "$WT/site"      "$BASE"
git -C "$ROOT" worktree add -b presentable/scenario  "$WT/scenario"  "$BASE"
git -C "$ROOT" worktree add -b presentable/receipts  "$WT/receipts"  "$BASE"
git -C "$ROOT" worktree add -b presentable/render    "$WT/render"    "$BASE"
git -C "$ROOT" worktree add -b presentable/count     "$WT/count"     "$BASE"
git -C "$ROOT" worktree add -b presentable/readiness "$WT/readiness" "$BASE"
git -C "$ROOT" worktree add -b presentable/launch    "$WT/launch"    "$BASE"
```

The actual working repository may use another base branch. Record the exact commit in `.agent/swarm/baseline.json` before launching agents.

## Branch-local runtime isolation

Polylogue already supports branch-local archives and ports. Assign each worktree a lane ID:

| Lane | API port | Browser port | Archive root |
|---|---:|---:|---|
| integration | 8780 | 8880 | `/tmp/polylogue-presentable/integration` |
| scenario | 8781 | 8881 | `/tmp/polylogue-presentable/scenario` |
| receipts | 8782 | 8882 | `/tmp/polylogue-presentable/receipts` |
| render | 8783 | 8883 | `/tmp/polylogue-presentable/render` |
| count | 8784 | 8884 | `/tmp/polylogue-presentable/count` |
| readiness | 8785 | 8885 | `/tmp/polylogue-presentable/readiness` |

Example:

```bash
export POLYLOGUE_ARCHIVE_ROOT=/tmp/polylogue-presentable/receipts
export POLYLOGUE_API_PORT=8782
export POLYLOGUE_BROWSER_CAPTURE_PORT=8882
```

Use the branch-local `devtools dev-loop` commands where applicable rather than inventing another launcher.

---

# 4. Coordination protocol

## 4.1 The file-ownership ledger

Create `.agent/swarm/ownership.yaml` on the integration branch:

```yaml
lanes:
  docs:
    owner: C1
    paths:
      - README.md
      - docs/demos.md
      - docs/public-claims.yaml
      - docs/sinex-interop.md
      - docs/proof-artifacts.md
      - devtools/docs_surface.py
  renderer:
    owner: C5
    paths:
      - polylogue/rendering/**
      - tests/unit/rendering/**
```

An agent must not edit outside its paths without writing a handoff request. This is more important than clean commit history.

## 4.2 Handoffs are files and Beads comments, not chat memory

Each lane maintains:

```text
.agent/swarm/<lane>/status.md
.agent/swarm/<lane>/handoff.md
.agent/swarm/<lane>/verification.json
```

`status.md` contains current state and blockers. `handoff.md` contains exact refs, commands, changed files, and assumptions. `verification.json` contains machine-readable commands and exit codes.

Use Beads to record task state and evidence refs. Do not use GitHub Issues.

## 4.3 Agent reporting format

Every coding agent returns:

```text
Bead(s):
Base commit:
Head commit:
Files changed:
Behavior added:
Claims now supported:
Claims still unsupported:
Verification run:
Generated artifacts:
Known conflicts:
Recommended merge order:
```

## 4.4 Commit policy for speed

Sacrifice aesthetic history, not traceability.

Allowed:

- large checkpoint commits;
- fixup commits;
- cherry-picking only the useful range;
- merging worktree branches with ordinary merge commits;
- copying a small non-overlapping file set when cherry-pick conflict cost is excessive;
- squashing only at the final public boundary.

Disallowed:

- uncommitted handoff;
- two agents editing the same generated file without coordination;
- force-pushing another agent’s branch;
- closing Beads before verification artifacts exist;
- hiding failed checks by deleting output.

A fast integration command sequence is:

```bash
git -C "$ROOT" merge --no-ff presentable/scenario
git -C "$ROOT" merge --no-ff presentable/readiness
git -C "$ROOT" merge --no-ff presentable/render
git -C "$ROOT" merge --no-ff presentable/receipts
git -C "$ROOT" merge --no-ff presentable/count
```

For a heavily conflicted docs branch, use a path-level import rather than spending an hour preserving commit topology:

```bash
git -C "$ROOT" checkout presentable/docs -- README.md docs/demos.md docs/public-claims.yaml
git -C "$ROOT" commit -m 'docs: import presentability surface'
```

Record the source branch/head in the commit body.

---

# 5. Single-machine resource scheduling

The bottleneck is not the number of agents. It is CPU, RAM, database/process isolation, and merge attention.

## Recommended concurrency

- up to four Python-heavy implementation/test lanes concurrently;
- one browser/daemon visual lane at a time;
- one full validation lane at a time;
- docs/review agents can run concurrently with tests;
- Sinex Rust builds should not compete with Polylogue full-suite runs unless the machine has substantial spare memory.

## Test tiers

### Tier A — per-edit

Run the narrowest owned tests only.

```bash
pytest -q path/to/owned/tests
ruff check changed/files.py
mypy changed/package
```

### Tier B — pre-handoff

Run the subsystem suite and generated-surface checks.

### Tier C — post-merge

The coordinator runs:

```bash
devtools render all
devtools render all --check
devtools verify --quick
pytest -q tests/unit/demo tests/unit/rendering tests/unit/site tests/unit/devtools
```

### Tier D — candidate cut

Run the repository’s prescribed full verification and release-readiness lanes. Do not ask every branch agent to run the full suite independently.

## Build and cache hygiene

- share immutable package/download caches where safe;
- keep branch-local archives, logs, ports, and generated outputs isolated;
- avoid running multiple mutation or scale suites simultaneously;
- preserve failed test output under `.agent/swarm/<lane>/artifacts/`;
- use `nice`/`ionice` for recording or full-suite jobs if they disturb interactive agents;
- reserve one CPU core and enough RAM for the coordinator and editor/agent runtime.

---

# 6. Beads handling

## Existing Beads to use directly

| Work | Bead | Closure posture |
|---|---|---|
| Real receipts packet | `polylogue-xyel` | Close when packet runner, controls, refs, and validator pass |
| Semantic renderer | `polylogue-ap7` | Do not close if only first tool-family slice lands; comment exact completed slice or create children |
| README de-meta | `polylogue-3tl.12` | Close after claims audit and generated/public links agree |
| Anti-grep card | `polylogue-3tl.15` | Close when grounded in public receipts/finding |
| Claims ledger | `polylogue-3tl.16` | Keep open if only static v1 lands; full Bead has broader blockers |
| Findings page | `polylogue-3tl.4` | Close only if stable public URL, method, caveats, and reproduction path exist |
| Install matrix | `polylogue-3tl.7` | Keep open unless every AC lane is actually proven; record partial Nix/source receipts |
| GitHub surface | `polylogue-3tl.8` | Requires repository-setting changes outside code; produce checklist and apply where access exists |
| Docs/visual ownership | `polylogue-3tl.9` | Close only after standing check exists |
| Launch kit | `polylogue-3tl.10` | Its current dependencies are broad; land artifacts without falsely closing it |
| Docs IA | `polylogue-ttu` | Narrow closure only if its full orphan/stale-doc AC is met |
| Site verification | `polylogue-6bu` | Close with rendered-site route/link proof |
| Honest refusal | `polylogue-212.8` | Already closed; reuse, do not reimplement |

## New child Beads to create

1. `Demo: Count It Once — copied-prefix physical/logical accounting receipt` under `polylogue-212`.
2. `Evidence Lab: Incident 14:32 shared deterministic scenario and independent oracle` under `polylogue-212`.
3. `Demo packet v2: oracle, controls, scope, material manifest, and recording parity` under `polylogue-212.7` or related.
4. `Semantic renderer slice: shell/test/edit descriptors shared across CLI/web` under `polylogue-ap7` if the Beads model supports children.
5. `Static public claims ledger bootstrap` related to `polylogue-3tl.16`, explicitly not the full first-class finding system.

The new Beads should declare file footprints so the swarm scheduler can avoid collisions.

---

# 7. Integration gates

## Gate G0 — baseline

- clean base commit recorded;
- active work such as `polylogue-f2qv.2` assigned and excluded from overlapping file scopes;
- worktrees created;
- ownership ledger committed;
- current full or quick verification result preserved.

## Gate G1 — fixture truth

- Incident 14:32 seeds and verifies;
- independent oracle exists;
- controls are nonempty;
- no private data;
- construct audit updated.

## Gate G2 — evidence-visible product

- semantic shell/test cards render in CLI and web;
- failure outcome comes from structure;
- refs resolve;
- unsupported and degraded states are visibly distinct.

## Gate G3 — demo packets

- The Receipts packet validates;
- Count It Once packet validates;
- Honest Refusal packet validates;
- recordings derive from the same commands;
- packet hashes and software revision recorded.

## Gate G4 — public surface

- README, site, docs map, claims ledger, demos, proof page, and interop page agree;
- no “Your AI memory” leftovers on public surfaces unless historical/quoted;
- no retired GitHub-issue roadmap links;
- supported install commands pass;
- links and generated surfaces pass.

## Gate G5 — cold reader

Give the repository and demo packet to a fresh agent with no prior chat context. Ask it to answer:

1. What is Polylogue?
2. What did each demo prove?
3. Which claims are field observations versus fixture capabilities?
4. What is not implemented?
5. How does Sinex fit?
6. Which command should a new user run?

Fail the gate if the reader invents uplift, assumes Sinex-backed storage already exists, treats logical cost as billing, or cannot locate evidence.

## Gate G6 — candidate cut

- final prescribed verification;
- package/install receipts;
- launch assets checksummed;
- Beads statuses and comments reflect actual evidence;
- known limitations copied into release/announcement artifacts.

---

# 8. The ambitious extension after the minimum cut

Once the minimum presentable cut is stable, reuse the same corpus and surfaces for:

1. Context Autopsy.
2. Sinex Missing Source.
3. Sinex Import It Twice.
4. Joint World Around the Claim.
5. Sinex-backed Polylogue rebuild proof.
6. Preregistered resumption duel.
7. Two-agent coordination proof in separate worktrees.

Do not create another demo corpus. Extending one evidence world is both more impressive and more rigorous because cross-demo consistency becomes testable.

---

# 9. Coordinator’s final checklist

```text
[ ] One stable one-liner everywhere
[ ] No unsupported benefit claim
[ ] Current install command works
[ ] The Receipts runs from clean state
[ ] Count It Once runs from clean state
[ ] Honest Refusal runs from clean state
[ ] Every packet validates
[ ] Every recording comes from packet commands
[ ] CLI and web show the same semantic outcome
[ ] Claims ledger covers every quantitative/comparative statement
[ ] Findings page states sample, method, date, and caveats
[ ] Docs routes and links pass
[ ] No private paths, hostnames, tokens, or transcript text in artifacts
[ ] Beads—not GitHub Issues—describe roadmap state
[ ] Sinex direction is clearly future unless backed-mode proof exists
[ ] Cold reader can explain the project without this chat
```
