[← Back to Docs](../../README.md)

# Reader-Comprehension Test Harness (polylogue-3tl.19)

A structural coverage lint (`devtools verify docs-coverage`) or a public-claims
ledger entry can prove a README claim is *true*. Neither proves it *lands* —
that a stranger who sees the first fold for a fixed exposure can actually
state the category, the outcome, the differentiator, and the one thing it
explicitly does not claim. This harness measures that, as a bounded,
single-blind, N-arm comprehension test, distinct from and complementary to
the structural gates in `docs/demos.md`.

## What this is

[`reader-test-runner.html`](reader-test-runner.html) is a self-contained,
offline HTML tool (no build step, no server, no external requests — open it
directly in a browser). For one session it:

1. randomly assigns (or lets an operator fix) one of N configured arms;
2. shows that arm's screenshot for a timed exposure, then hides it;
3. collects **unaided** free-text answers (category, outcome, differentiator,
   proof, boundary/non-claim, confusion);
4. re-shows the image and collects first-action and false-belief answers;
5. records independent 0–2/0–3 scores against the rubric below;
6. persists everything to `localStorage` and exports JSON/CSV.

## Scoring doctrine — anchored to `docs/demos.md`, not invented fresh

`docs/demos.md` already has a claim/oracle/controls/falsifier/non-claims
vocabulary for judging whether a demo proves what it says. This harness reuses
those terms instead of introducing a parallel scoring scheme:

| Harness dimension | `docs/demos.md` doctrine term | What it checks |
| --- | --- | --- |
| Category, outcome, differentiator | **claim** recognition | Did the reader state the same claim the copy makes, unprompted? |
| Boundary/non-claim | **non-claims** | Did the reader notice the one thing the copy explicitly says it does *not* establish? |
| False beliefs | **falsifier** | Any nonzero false-belief count falsifies the arm regardless of its other scores — this is the harness's explicit falsifier, matching the demo-packet contract's requirement that every proof declare one. |
| Independent scorer, not the participant | **oracle** | The score comes from a scorer applying the rubric, not the participant's self-report of how well they think they did. |
| A comparative baseline (>=2 arms, one of them `current`) | **comparative baseline** | A single-arm "reads fine" result is not comparative; the harness always configures at least a current-vs-candidate pair. |

## Non-claims (read before running or citing a result)

- A handful of local sessions is not a population-prevalence claim about how
  README readers in general respond. State sample size and selection
  alongside any result (polylogue-3tl.19 AC4).
- This tool measures stated comprehension and recall after a **timed static
  exposure**, not the full live README experience (scrolling, following
  links, running the actual demo command). A candidate that reads well here
  can still fail in the wild for reasons this tool cannot see.
- It does not measure downstream agent performance or task success.
- A candidate advances only when it wins or ties comprehension **without**
  introducing a new false belief — a higher category/outcome score paired
  with a nonzero false-belief count is not a win.

## Status: harness built, arms not yet populated, no run yet

This is an honest status, not a placeholder to be quietly promoted:

- **Built and adapted** from the external design study
  (`.agent/scratch/readme-positioning-2026-07-14/`, gitignored — the
  prototype protocol, not its generated code, was reused): the exposure
  timer, unaided-then-revealed question flow, doctrine-aligned rubric,
  localStorage persistence, and JSON/CSV export all work as shipped.
- **Repointed, not fixed**: the original prototype referenced
  `../prototypes/screenshots/*.png` files that do not exist in this repo (they
  were generated for a joint Sinex+Polylogue study, gitignored, never
  committed here). This build ships with **zero arms pre-populated with a
  screenshot** — the "Arm configuration" panel shows an explicit
  "no screenshot configured" placeholder instead of a silently-broken image,
  so a missing screenshot cannot be scored as "read fine" by accident.
- **Not done: a real run.** Populating an arm needs a real, freshly generated
  full-fold screenshot of a candidate README (e.g. via a pinned-viewport
  browser capture of the rendered Markdown, following the same "no
  hand-authored visual passed off as generated" discipline as
  `docs/visual-evidence.md`), and a real independent reader per session. Both
  are out of scope for this pass. `polylogue-3tl.19` AC3 (a real 3-arm run
  producing a decision-grade result before promoting a candidate README) is
  explicitly **deferred**, not silently marked done.

## How to run a real session once arms are populated

1. Generate a full-fold screenshot for each candidate (`current` plus at
   least one alternative) and save it under this directory or
   `docs/examples/visual-tapes/`.
2. Open `reader-test-runner.html` directly in a browser (`file://` is fine —
   it makes no network requests).
3. Fill in each arm's name and screenshot path in the "Arm configuration"
   panel.
4. Run one session per participant, save it, and repeat across participants
   and arms.
5. Export JSON/CSV and compute per-arm comprehension rate and false-belief
   rate. Report sample size and selection method alongside the numbers — this
   is not a population study.
