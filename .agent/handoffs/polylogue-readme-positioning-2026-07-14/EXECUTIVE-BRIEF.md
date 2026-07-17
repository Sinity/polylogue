EXECUTIVE-BRIEF.md
# Executive brief

## Decision

Use **recall-first** as Sinex’s repository front door and **receipts-first** as Polylogue’s repository front door.

Keep the other ideas as selectable narratives rather than diluting those pages:

- Sinex replay-first for technical differentiation;
- Sinex rebuildable-first for concept transmission and launch material;
- Sinex event-spine for contributor/infrastructure audiences;
- Polylogue search-first for archive-discovery traffic;
- Polylogue continuity-first for agent-handoff material after outcome experiments.

The repositories do not primarily need more promises. They need a shorter path from promise to inspectable proof, plus machinery that prevents the proof from drifting away from the product.

## Sinex: the important correction

The product work should not begin by creating recall again. The supplied code already has the root `sinexctl recall` command and a typed `ContextSummaryView` containing source summaries, sessions, attention spans, intervals, evidence/lineage refs, and an interleaved timeline with explicit gap/caveat items.

The public gap is narrower:

1. publish a disclosure-safe scenario through the existing recall envelope;
2. add a compact renderer over `payload.timeline`, not a parallel query path;
3. make IDs and timestamps stable enough for reproducible media;
4. encode missing-evidence, duplicate, unrelated-event, derivation-withheld, privacy, and freshness falsifiers;
5. generate the visual from the typed owner;
6. promote it only after cold-reader and clean-route tests.

The committed work graph records a closed internal multi-source campaign, but its personal packet is absent from the supplied snapshot. That supports “this path has worked in dogfood,” not an unwatermarked public receipt. Default to a deterministic public analog unless disclosure authority explicitly clears a scrubbed field packet.

Start with [`agent-tasks/START-HERE-sinex.md`](agent-tasks/START-HERE-sinex.md), then execute [`agent-tasks/sinex-01-public-recall-proof.md`](agent-tasks/sinex-01-public-recall-proof.md) before treating the layout prototype as evidence.

## Polylogue: the important correction

Polylogue already owns a highly legible deterministic proof: an assistant claims all tests pass, the structural action at claim time failed, a later matching action succeeds, and a prose-only `error` control contributes no structural failure. That event should dominate the first screen.

The implementation work is discipline around an existing proof:

1. freeze the typed compact-output contract;
2. remove visible prompt-escape defects through the real renderer/recording route;
3. make recipe, normalized output, tape, and committed media freshness one failing gate;
4. benchmark `uvx`, installed CLI, pipx, Homebrew, Nix, and source routes on supported clean environments;
5. place the measured winner directly after a compact receipt;
6. test whether readers retain “evidence at the claim boundary,” not merely “search AI chats.”

Start with [`agent-tasks/START-HERE-polylogue.md`](agent-tasks/START-HERE-polylogue.md), then close [`agent-tasks/polylogue-03-visual-drift-gate.md`](agent-tasks/polylogue-03-visual-drift-gate.md) before replacing public media.

## What can ship in copy before final visuals

Sinex can safely name the current `sinexctl recall` capability now, provided the README labels the image as a sanitized layout prototype and distinguishes the generic deterministic system walkthrough from the unshipped public recall receipt.

Polylogue can safely lead with the deterministic claim-versus-structural-outcome story now. The replacement card must remain prototype-labeled until generated through the repository’s visual owner.

## The decision experiment

Run only three arms per project in the first round:

| Project | Control | Candidate A | Candidate B |
|---|---|---|---|
| Sinex | current README | recall-first | rebuildable-first |
| Polylogue | current README | receipts-first | search-first |

Measure four things separately: unaided category recall after 30–60 seconds; the concrete differentiator the reader can restate; the limitation/non-claim the reader noticed; and time from a clean supported environment to meaningful product output.

A candidate advances only when its claim is current, its route succeeds, its media is generated/fresh, and it wins or ties comprehension without creating a new false belief. The structural linter is only a regression signal.

## Non-negotiable stop conditions

Stop promotion when a visual is hand-authored rather than generated, a command is assumed rather than measured, missing evidence is rendered as inactivity, a later success rewrites an earlier failure boundary, a private packet enters public paths without authority, an analogy is presented as compatibility, or a capability is promoted as measured outcome.

The complete option system is in [`DECISION-MAP.md`](DECISION-MAP.md), the component inventory is in [`library/CATALOG.md`](library/CATALOG.md), and the rendered comparison is in [`prototypes/gallery.html`](prototypes/gallery.html).

