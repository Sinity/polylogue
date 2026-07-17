# Demo portfolio redesign

## The core correction

Both repositories have sophisticated demo backlogs, but the portfolios currently mix four different jobs:

- helping a stranger understand the category;
- proving one bounded product claim;
- evaluating a scientific hypothesis;
- torturing the system under faults or scale.

One artifact cannot do all four well. A rapid-fire query montage can prove breadth while failing to explain value. A large private-archive packet can reveal a genuine phenomenon while being impossible for an outsider to reproduce. A deterministic fixture can prove semantics while saying nothing about field scale. A fault-injection test can prove durability while being a poor landing-page story.

The portfolio should therefore have four explicit tiers.

| Tier | Audience | Primary job | Runtime posture | Public result |
|---|---|---|---|---|
| Hero story | First-time reader | Make the category and value obvious | Deterministic, visually narrated, under a few interactions | A short recording plus an evidence packet |
| Proof card | Technical evaluator | Falsify one bounded claim | Deterministic or tightly controlled | Claim, oracle, negative control, artifacts, caveats |
| Experiment | Researcher/operator | Compare strategies or estimate an effect | Preregistered, paired/blinded where possible | Protocol, raw results, analysis, limitations |
| Torture proof | Maintainer/operator | Demonstrate failure behavior and recovery | Fault injection, scale, interruption | Timeline, invariants, loss accounting, recovery proof |

A fifth, separate class is a **field packet**: evidence from a real private deployment. It can demonstrate scale and discover failure modes, but it is not a public benchmark and must never be presented as one.

## Construct-validity contract

Every public demo should ship a manifest containing the following fields.

1. **Question.** The concrete user or operator question.
2. **Claim.** One bounded statement the demo is intended to support.
3. **Construct.** The real concept being measured: structural tool failure, logical lineage, source coverage, replay revision, context utility, and so on.
4. **Intervention or fixture.** What is planted, changed, or held constant.
5. **Observable.** The exact product output used to judge the claim.
6. **Oracle.** An independent structural source of truth. Assistant prose is not an oracle for tool success; the product's own derived label is not an independent oracle for itself.
7. **Negative control.** A nearby case that should not trigger the result.
8. **Confounds.** Plausible alternate explanations and how they are bounded.
9. **Evidence reachability.** Stable refs or material anchors from every headline result to inspectable evidence.
10. **Caveats.** What the result does not establish.
11. **Regeneration.** One command and declared dependencies.
12. **Artifacts.** Machine-readable report, human report, transcript, visual media, and fixture manifest.

A demo is rejected when it merely shows that a command returns data, when the expected answer is encoded in the presentation script rather than product state, when the oracle is circular, or when a comparison arm receives materially different information without declaring it.

## Scoring rubric

Score each candidate from zero to four on each axis. A hero story should score at least 24/32 and must score at least three on comprehension, oracle strength, and visual force.

| Axis | 0 | 4 |
|---|---|---|
| Immediate comprehension | Requires architecture knowledge | The question and result are obvious without narration |
| Category differentiation | Could be a grep/dashboard demo | Demonstrates a capability ordinary chat/event viewers fundamentally lack |
| Oracle strength | Self-reported or circular | Independent structural or byte-level truth |
| Evidence reachability | Screenshot only | Every result resolves to exact source evidence |
| Negative controls | None | Positive and multiple plausible negative cases |
| Determinism | Depends on private state | Rebuildable private-data-free fixture |
| Visual force | Dense terminal dump | One clear visual transition with meaningful semantics |
| Honest scope | Implies generality | Explicitly separates capability, field evidence, and experiment |

---

# Polylogue portfolio

## Public hero 1: The Receipts

**Question:** The assistant said the tests passed. What actually happened?

**Story:** A deterministic coding-agent session contains a failed test, an assistant continuation, a later repair, and a final successful run. Polylogue renders the assistant claim and the structural tool result as distinct objects. The viewer drills from the claim to the tool call, exit code, output, source block, and session ref.

**Bounded claim:** Polylogue can distinguish conversational claims from structural tool outcomes and resolve both to evidence.

**Oracle:** Provider-native tool-result metadata and process exit code embedded in the source fixture.

**Negative controls:**

- prose containing the word “error” with no failed tool result;
- a successful tool result whose output contains failure-like text;
- an unpaired or unavailable result, which must render unknown rather than failed.

**Visual sequence:**

1. The session card shows “assistant: tests pass” beside a red structural result.
2. Selecting the discrepancy opens a semantic shell card with command, exit code, output, and refs.
3. A later green result shows the repair.
4. The packet ends with “supported,” “contradicted,” and “unknown” examples.

**Beads:** `polylogue-212.2`, a thin vertical slice of `polylogue-ap7`, and the evidence-honesty work under `polylogue-cpf.4`.

This should replace archive facets as the first public result.

## Public hero 2: History Has Branches

**Question:** How much distinct work happened when agents forked and resumed from copied prefixes?

**Story:** A parent session, copied-prefix fork, fresh subagent, and compaction boundary are shown as a topology. The physical view counts every provider artifact; the logical view composes inherited context once and displays only the fork's unique tail as new work.

**Bounded claim:** Polylogue can preserve physical provider evidence while exposing a separate logical composition that does not double-count copied prefixes.

**Oracle:** Fixture-level message IDs, declared branch point, and expected unique-tail manifest.

**Negative controls:**

- a fresh subagent that must not be deduplicated as a copied prefix;
- two coincidentally identical messages without a supported lineage edge;
- a compaction summary, which remains a real generated message rather than being erased.

**Visual sequence:** topology map → physical count → logical count → exact composed transcript.

**Beads:** `polylogue-4ts`; current deterministic lineage fixtures already provide much of the substrate.

## Public hero 3: Resume With Reviewed Evidence

**Question:** What should the next agent receive, and why?

**Story:** A prior session contains source evidence, an agent-proposed lesson, an operator-accepted correction, a stale candidate, and an explicit omission caused by the token budget. Polylogue compiles a context image. The accepted assertion appears; the candidate does not; the stale item is flagged; omitted evidence is listed; a context-delivery record captures exactly what crossed the boundary.

**Bounded claim:** Polylogue can compile bounded context from evidence and reviewed assertions without silently promoting agent-generated candidates.

**Oracle:** Assertion lifecycle rows, context policy, compiled segment manifest, and delivery snapshot.

**Negative controls:** rejected assertion, non-injectable candidate, stale assertion, and over-budget evidence.

**Important non-claim:** This proves selection and provenance, not improved agent performance. Performance requires the blinded resumption experiment.

**Beads:** `polylogue-37t`, `polylogue-212.6`, and the context-safety surface.

## Public hero 4: The Honest No

**Question:** Which browser pages was I reading minute by minute while this session unfolded?

In standalone Polylogue, the answer should be `not_supported` unless browser or ambient-machine evidence was actually captured. The demo explains the missing evidence class instead of fabricating a reconstruction.

In Sinex-backed mode, the same question can become supported when Sinex supplies browser, focus, terminal, or filesystem evidence with coverage state.

**Bounded claim:** Polylogue distinguishes unsupported inference from absent results and exposes the missing capability or evidence.

**Correction to the current anti-demo:** Do not describe this as an eternally unsupported product question. It is unsupported by the standalone transcript archive, but a primary combined-system use case.

**Beads:** `polylogue-212.8`, `polylogue-avg`, `polylogue-cpf.4`, and `sinex-4j2.2`.

## Proof cards

### Why not grep?

Use one fixed corpus and answer the same six questions with grep and Polylogue. The card should not caricature grep; it should show the boundary between lexical retrieval and domain interpretation.

| Question | Grep | Polylogue oracle |
|---|---|---|
| Find `pytest` text | Yes | Lexical FTS |
| Pair a tool call with its result | Provider-specific manual parsing | Typed block relation |
| Determine structural failure | Parse every provider convention | `is_error`/exit metadata |
| Separate human text from injected protocol context | No general rule | `material_origin` |
| Avoid fork-prefix double counting | No | Logical session composition |
| Resolve a finding to source material | No common ref model | Stable refs and raw evidence |

Bead: `polylogue-3tl.15`.

### Token lanes are disjoint

Use a Codex fixture with high cache reuse and a Claude control. Compare naïve inclusive-input pricing with the normalized disjoint-lane result. The oracle is the provider field contract and hand-computed expected price, not Polylogue's own aggregate.

Beads: `polylogue-f2qv.2`, `.3`, `.4`, and `.5` as applicable.

### Physical is not logical

A compact version of History Has Branches focused only on counts and cost. This should be a proof card rather than another hero recording.

### Every finding resolves

Take one published finding and programmatically walk all refs to terminal source evidence. Inject one broken ref as a negative control. This should gate the findings publishing lane.

Beads: `polylogue-3tl.4`, `.16`, `polylogue-rxdo.4`.

## Experiments

### Blinded resumption duel

Compare raw live references with a production-generated, bounded, freshness-aware context image. Pair checkpoints from the same workstream. Blind graders to arm labels. Predeclare reconstruction questions, task continuation quality, evidence correctness, time/call budget, and stale-claim penalties. Include deliberately stale memory checkpoints as negative controls.

The existing small pilot is useful design evidence, not a public efficacy claim.

### Retrieval ablation

Compare lexical, vector, and hybrid lanes on adjudicated queries. Measure hit@k, evidence correctness, redundant context, and downstream context utility. Do not evaluate by whether the system retrieves the chunk it used to create the query.

## Existing Polylogue demos: keep, move, or replace

| Existing item | Decision | Reason |
|---|---|---|
| One-command deterministic tour | Keep, rewrite narrative | Strong substrate; poor first-result storytelling |
| D1 The Receipts | Promote to flagship | Clearest category differentiation |
| D2 cost by outcome | Keep as later proof/field packet | Outcome attribution is harder and confounded; not launch wedge |
| D4 behavioral archaeology | Move to technical proof shelf | Demonstrates DSL breadth, not immediate value |
| D5 session watched itself | Keep after daemon stability | Impressive, but operationally fragile and less fundamental |
| D8 resume triage | Rebuild around judgment/context provenance | Strong thesis, but must separate capability from uplift |
| Forensic Q&A | Keep as deep evaluator demo | Valuable after the ontology is understood |
| Honesty anti-demo | Promote, reframe for Sinex-backed future | Demonstrates epistemic discipline and product boundary |
| Fable-as-Foreman | Defer | Rhetorically attractive, weak launch construct validity |

---

# Sinex portfolio

## Public hero 1: Reconstruct Tuesday, Including the Hole

**Question:** What happened around the failed build, and where is the record incomplete?

A deterministic day fixture contains independent source families: terminal, Git, filesystem, browser, focus/application activity, Beads intent, and one Polylogue agent session. One source is deliberately unavailable for a bounded interval.

The resulting timeline shows evidence tracks, a highlighted incident window, exact material refs, and a visible gap band. The answer must distinguish:

- no event observed while the source was healthy;
- source unavailable;
- material captured but not yet interpreted;
- projection stale;
- result omitted by query budget.

**Bounded claim:** Sinex can join heterogeneous evidence around an interval while preserving source-specific provenance and coverage gaps.

**Oracle:** Fixture manifests for every source plus the induced outage interval.

**Negative controls:** healthy empty interval, captured-but-unparsed material, and a similar event outside the join window.

**Beads:** `sinex-cem.8`, `sinex-jdp`, `sinex-a4w.3.4`, `sinex-bm1`, and `sinex-vhm`.

## Public hero 2: The System Changes Its Mind Honestly

**Question:** What happens when a parser was wrong for months?

Parser semantics v1 intentionally misclassifies a retained source record. A fix introduces semantics v2. Replay emits a new interpretation, current projections change, the old interpretation remains inspectable, and affected descendants are invalidated or recomputed.

**Bounded claim:** Sinex can revise interpretation without rewriting source material or pretending the prior interpretation never existed.

**Oracle:** Immutable source bytes and a fixture-level expected reading for v1 and v2.

**Negative controls:** a source record unaffected by the parser change and a derived record whose semantics version should remain unchanged.

**Beads:** `sinex-cem.3`, `sinex-cem.14`, `sinex-0vx`, and replay/invalidation work.

## Public hero 3: It Diagnosed Its Own Blind Spot

Induce a source outage while other sources remain healthy. The product should show source freshness degradation, affected query coverage, remediation evidence, and a permanent scar in the incident packet. Restoring the source must not retroactively turn the gap into an observed-empty interval.

**Bounded claim:** Sinex reports capture unavailability as coverage error rather than silently returning an apparently complete empty result.

**Oracle:** Fault-injection controller timestamps and source heartbeat/material manifests.

**Beads:** `sinex-cem.2`, `sinex-jdp`, `sinex-60r`, and `sinex-u3n`.

## Proof cards

- **Import it twice.** Prove occurrence idempotence while retaining replay-specific interpretation identity. Beads `sinex-cem.13`, `sinex-908`.
- **Run twice, pay once.** Prove deterministic model-effect caching with prompt/input/recipe identity and a deliberately changed recipe negative control. Bead `sinex-cem.4`.
- **Crash mid-thought.** Kill the producer or event engine at every receipt boundary and prove no acknowledged item disappears. Bead `sinex-cem.15`.
- **Disclosure control is not forgetting.** Split the existing retroactive privacy idea into a redacted-view proof and a separate physical-excision proof. Do not claim deletion while original bytes remain. Bead `sinex-cem.1` should be renamed or its acceptance criteria narrowed.
- **Hybrid retrieval earns its cost.** Use adjudicated queries and report hit@k plus latency/cost. Kill the vector lane if it does not improve a declared metric. Bead `sinex-cem.5`.

## Existing Sinex demos: keep, move, or replace

| Existing item | Decision | Reason |
|---|---|---|
| `sinexctl ops verify --demo` | Keep as smoke gate | Useful operational check, not a thesis demo |
| Current recall shell/SQL packet | Keep as field/depth evidence | Bypasses normal product boundary and is not broadly multi-source |
| `.1` planted-secret privacy | Split and rename | Disclosure control and physical forgetting are different constructs |
| `.2` capture outage | Promote to flagship | Distinctive and visually legible |
| `.3` replay bugfix | Merge narrative with `.14` | One stronger interpretation-revision hero is better than two overlapping demos |
| `.4` run-twice-costs-once | Keep as proof card | Strong bounded claim, not first-screen value |
| `.5` hit@k | Keep as experiment/proof | Scientific infrastructure decision |
| `.6` semantic lane duel | Keep as adjudication experiment | Too much prerequisite context for launch |
| `.7` blinded resumption | Make the joint thesis experiment | Strongest eventual combined-system result |
| `.8` reconstruct Tuesday | Promote to flagship | Best explanation of Sinex's broad value |
| `.9` focus actuation | Defer | Visually impressive but creates safety/category distraction |
| `.10` devloop as source | Use as dogfood field packet | Compelling after the public ontology is understood |
| `.11` corpus audit | Make a standing gate | Infrastructure, not a public story |
| `.12` experiment substrate | Keep as shared infrastructure | Enables credible claims |
| `.13` import twice | Promote as proof card | Simple and strong |
| `.14` interval revision | Merge into “changes its mind honestly” | Strong temporal concept |
| `.15` crash no loss | Promote as torture proof | Core durability claim |

---

# Combined flagship: Resume This Bead

This should become the eventual joint launch demonstration.

## Question

> Resume this work item. What was intended, what did the agents try, what actually changed, what was verified, what remains uncertain, and what should the next agent receive?

## Evidence legs

- **Beads:** work-item intent, dependencies, state changes, and acceptance criteria.
- **Polylogue:** sessions, messages, tool calls/results, subagents, lineage, compaction, usage, candidate and reviewed assertions, and prior context deliveries.
- **Sinex:** terminal, Git, filesystem, browser, focus, service/CI/deployment evidence, source coverage, and material refs.

## Product output

An Agent Work Packet contains:

1. intent and current task state;
2. physical and logical agent-session topology;
3. attempted actions and structural outcomes;
4. repository and environment changes;
5. verification evidence;
6. accepted lessons, rejected candidates, and stale claims;
7. missing evidence and source-health caveats;
8. a bounded context image for the next agent;
9. a context-delivery snapshot after handoff.

## Construct-validity controls

- One agent claim contradicts machine evidence.
- One relevant source is unavailable for a known interval.
- One candidate lesson is deliberately wrong and remains non-injectable.
- One accepted assertion becomes stale after a later commit.
- One fork copies a prefix and must not inflate logical work.
- One task-state transition is missing, preventing a “completed” verdict.

## Claims

The deterministic demo may claim reconstruction and provenance. It may not claim improved agent performance.

The later blinded duel may estimate whether the combined packet improves resumption under a fixed budget.

## Beads

`sinex-a4w.3.9`, `sinex-4j2`, `sinex-cem.7`, `polylogue-37t`, `polylogue-4ts`, and the relevant Beads/task integration work.

---

# Artifact format

Every demo directory should have the same shape:

```text
demo-id/
  manifest.yaml          # claim, construct, oracle, controls, caveats
  fixture/               # private-data-free inputs or fixture generator
  run.sh                 # one command, no hidden state
  report.json            # machine-readable result and evidence refs
  report.md              # human explanation
  transcript.txt         # exact commands and bounded output
  media/                  # tape, GIF/MP4, screenshots
  evidence/               # resolved source excerpts or material manifests
  validation/             # checksums, negative-control results, test logs
```

The visual recording is an index into this packet, not the proof by itself.
