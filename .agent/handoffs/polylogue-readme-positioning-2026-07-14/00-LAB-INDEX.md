# First-Impressions Lab for Sinex and Polylogue

Start with the [`EXECUTIVE-BRIEF.md`](EXECUTIVE-BRIEF.md). This delivery is a composable working library rather than a single redesign prescription: seven complete README compositions, thirty reusable components, twenty rendered prototype/contact-sheet images, agent-ready implementation contracts, reader/timing experiments, patch previews, overlays, and validation tooling.

Source snapshots inspected on 2026-07-13:

- Sinex commit `1e47b9d1` — `docs: consolidate the current documentation surface (#2513)`;
- Polylogue commit `59bcbe28e` — `docs: tighten the public documentation surface (#2846)`.

## Main findings

**Sinex already has the product surface the first impression needs.** The root `sinexctl recall` command accepts `--at`/`--window`; its JSON/YAML payload contains raw events, sessions, attention spans, intervals, refs, and explicit gap/caveat timeline items. The implementation task is publication plus compact rendering: build a disclosure-safe deterministic scenario or clear a scrubbed field packet, then render the existing typed recall view. Do not invent a second recall subsystem.

**Polylogue already has a deterministic first-screen proof.** Its receipt demo structurally distinguishes a failed action at claim time from a later repair and includes an anti-grep control. The highest-leverage work is to simplify the front door, fix the visibly broken prompt recording through the real renderer, measure the clean-machine command winner, and make recipe/output/media drift fail automatically.

## Recommended starting points

For Sinex, start with [`generated/README.sinex.recall-first.md`](generated/README.sinex.recall-first.md). The deeper conceptual alternative is [`generated/README.sinex.rebuildable-first.md`](generated/README.sinex.rebuildable-first.md); replay-first and event-spine-first remain available for narrower audiences.

For Polylogue, start with [`generated/README.polylogue.receipts-first.md`](generated/README.polylogue.receipts-first.md). Search-first is the fast-action alternative; continuity-first is bounded to capability language until a paired handoff experiment exists.

The shared composition pattern is: category and outcome → one concrete proof → one action → one falsifier/non-claim → short capability shelf → honest status/trust boundary → architecture.

## Use the library

Open [`prototypes/playground.html`](prototypes/playground.html) to compare the options side by side, [`prototypes/gallery.html`](prototypes/gallery.html) for the visual index, or the [`first-fold contact sheet`](prototypes/screenshots-contact-sheet.png) for one static scan. [`library/CATALOG.md`](library/CATALOG.md) lists every component, status, and composition use. [`DECISION-MAP.md`](DECISION-MAP.md) explains which narratives combine cleanly and which create claim debt.

Give a coding agent [`agent-tasks/START-HERE-sinex.md`](agent-tasks/START-HERE-sinex.md) or [`agent-tasks/START-HERE-polylogue.md`](agent-tasks/START-HERE-polylogue.md). The individual packets contain code owners, routes, falsifiers, stop conditions, acceptance criteria, and focused verification commands.

Run human tests with [`experiments/reader-test-runner.html`](experiments/reader-test-runner.html). Use the route manifests and timing harness to replace installation assumptions with cold/warm measurements. Run `make check` to verify composition freshness, component metadata, prototype inventory, overlays, links, patches, checksums, and script compilation.

## Repository map

- `audit/` — grounded diagnosis and first-screen claim ledger;
- `library/` — small Markdown components and generated catalog;
- `manifests/` — composition recipes;
- `generated/` — complete README candidates;
- `prototypes/` — proof cards, first folds, social previews, and interactive comparisons;
- `agent-tasks/` — repository-native implementation contracts;
- `experiments/` — cold-reader, visual, claim-integrity, and time-to-proof protocols;
- `scripts/` — composer, linter, route benchmark, renderer, and validator;
- `reports/` — structural comparisons and execution limitations;
- `patches/` — unified README diffs against the supplied snapshots;
- `overlays/` — repository-shaped candidate files with prototype boundaries preserved.

## Compose another option

```bash
python scripts/compose_readme.py manifests/sinex-rebuildable-first.toml \
  --source snapshots/sinex.README.current.md \
  --output /tmp/README.sinex.md

python scripts/compose_readme.py manifests/polylogue-search-first.toml \
  --source snapshots/polylogue.README.current.md \
  --output /tmp/README.polylogue.md
```

A manifest selects ordered components and a heading in the current README from which the technical tail is retained. Hero, proof, quick-start, status, and capability experiments remain independent instead of requiring a full rewrite each time.

## Merge boundary

The prose and diffs are review candidates, not branch authority. Reconcile commands, status, Beads owners, generated artifacts, and current `master` before applying.

Polylogue’s receipt facts are grounded in its committed deterministic fixture. The replacement visual must still be generated from the real route and tied to a freshness check; do not commit the hand-built prototype as evidence.

Sinex’s `sinexctl recall` capability is current. The watermarked image is not evidence, and the generic deterministic walkthrough does not yet seed the shown recall story. An unwatermarked public visual requires either a disclosure-cleared scrubbed packet or a deterministic analog generated from the typed recall envelope.

Execution attempts and unavailable dependencies are recorded in [`reports/verification-notes.md`](reports/verification-notes.md).
