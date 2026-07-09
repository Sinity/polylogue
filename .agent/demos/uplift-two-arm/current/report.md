# Handoff-Pack Uplift Re-Run (n=5 pilot)

Successor to campaign `polylogue-jxe`, which closed diagnostic-negative (raw-ref 8/10 vs handoff-pack 5/10, n=1, attributed to packet staleness). This re-run tests whether a *fresh* pack (regenerated at continuation time) plus live query access beats raw-ref plus live query access, per the protocol decision recorded on `polylogue-cfk`.

## Claim

Directional evidence (4 of 5 pairs) that a fresh, bounded context summary given alongside live query access produces a better devloop-state reconstruction than live query access alone, at this sample size. **Not a publishable claim** — n=5 is a de-noising step per the original protocol's own staged design (n=1 pilot → n≥3 de-noise → n=12-20 publishable), not the final tier.

## Method

5 pairs, each at a distinct real checkpoint commit in this session's own devloop history. Each pair: two isolated Agent-tool subagents (worktree-isolated), one given only `bd`/`git`/repo-source access ("raw-ref"), one given the same access plus a hand-written bounded summary of recent shipped work and standing goals ("handoff-pack"). Both attempt the same task: reconstruct devloop state and recommend a specific next bead. A ground-truth file was written *before* dispatching each pair's arms, recording the checkpoint and (where knowable) what actually happened next. A separate, blinded judge subagent scored each pair against its ground truth without being told which arm was which.

## Corpus

This session's own live devloop (2026-07-09), five checkpoints: commits `6c12e9234`, `64c079d6e`, `01592e5e9`, `a2ee55ec4`, `697470661`.

## Findings

| Pair | Checkpoint | raw-ref | handoff-pack | Winner |
|---|---|---|---|---|
| 1 | 6c12e9234 | 29 (judge_1); 22 (judge_2) | 33 (judge_1); 35 (judge_2) | handoff-pack (both judges) |
| 2 | 64c079d6e | 25 | 31 | handoff-pack |
| 3 | 01592e5e9 | 31 | 19 | raw-ref |
| 4 | a2ee55ec4 | 17 | 32 | handoff-pack |
| 5 | 697470661 | 12 | 36 | handoff-pack |

Handoff-pack wins 4 of 5 pairs. Mean scores: raw-ref 22.8/40, handoff-pack 30.2/40.

**Where handoff-pack won**, the differentiator was consistently *open-thread coverage* and *avoiding a total miss on the checkpoint's own significance* — e.g. in pair 5, the raw-ref arm called the checkpoint commit itself "an unrelated shipped fix" when it was the load-bearing event; in pair 4, raw-ref never identified the literally-in-progress bead as in-progress. The pack arms consistently surfaced named epics/threads (pj8, 3tl, cfk, the 212.x chain, the lineage epic) that the raw-ref arms, working from `bd`/`git` alone, more often missed or under-weighted.

**Where raw-ref won (pair 3)**, the pack arm made a confident, uncaveated factual error — asserting two beads (`4ts.4`, `xnkf`) were "already closed" at a checkpoint where they had not yet shipped. This is a genuine, real failure mode: a pack can encode information that's subtly ahead of (or stale relative to) the actual checkpoint, and an arm that trusts it uncritically can assert something false with high confidence. The raw-ref arm, forced to derive everything from the live state at that exact commit, could not make this specific class of error.

## Specimens

All 10 arm transcripts and 6 judge verdicts are in `arms/*.md` and recorded inline above; raw scores in `pairs.json`. `pairs.json`'s pair-1 row uses judge_1's scores (29/33) as the aggregate table's value with judge_2's (22/35) not separately averaged in, since both judges agreed on the winning direction and picking either alone doesn't change the sign-test outcome; a future revision could instead average the two.

## Counterexamples

Pair 3 is the explicit counterexample: a well-resourced arm can still fail when its input encodes inaccurate-for-this-checkpoint information and it doesn't verify against live state before asserting. This is exactly the "packet staleness" failure mode the original `jxe` campaign identified — not fully eliminated by giving the pack arm live query access, since it can simply not use that access to catch a factual error the pack primed it to assume.

## Limits

- **n=5 is not a publishable sample.** The protocol's own staged design calls for n≥12-20 for a publishable uplift claim. This result should be read as "worth continuing to n=12-20," not as a conclusion.
- **Correlated subject and rater.** All 5 checkpoints come from the same session/devloop, authored, ground-truthed, and (for the initial pilot pair) partially scored within the same overall session that generated the subject matter. This is a real construct-validity limit shared with the original `jxe` pilot.
- **Weaker isolation than the original protocol.** The original `jxe`/`jxe.2` pilot used fully separate Claude Code CLI invocations with distinct tracked `agent_id`s per arm. This re-run uses `Agent`-tool subagents in isolated git worktrees within the same orchestrating session — real process/context isolation, but not as strong a guarantee as fully independent CLI sessions.
- **Pair 1's blind was compromised** (the pack arm's own text self-referenced "the pack"), discovered after dispatch. A second independent judge, explicitly told to check for and account for such a leak, still scored the same direction (35 vs 22), which is reassuring but doesn't retroactively fix the methodology for that pair — later pairs' arm prompts were corrected to forbid self-referential process commentary.
- **This pilot did not test the "packet staleness" root cause directly** — i.e., it didn't measure whether `qt3`'s fast regeneration + `yps` freshness metadata actually keep a pack fresh under load; it assumed the packs handed to the pack arms were fresh (hand-written summaries, not regenerated via the production pack-generation pipeline). A full re-run per the original protocol would generate the pack through the actual pipeline at continuation time and verify freshness metadata, which this pilot did not do.

## Reproduce

The five ground-truth files (`metrics/ground-truth*.json`), the `pairs.json` aggregate, and every arm/judge transcript are committed under this directory. Re-running with the production pack-generation pipeline (rather than hand-written summaries) and extending to n≥12 pairs, drawn from genuinely independent subjects rather than one session's own consecutive checkpoints, is the next step toward a publishable result — tracked as a follow-up on `polylogue-cfk` rather than assumed complete by this pilot.
