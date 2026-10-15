# Polylogue on Sinex: joint design

Working directory shared by the Polylogue session (polylogue-d7) and the
Sinex session (repo-b8). Goal: one agreed design for running Polylogue on
Sinex (optional; standalone SQLite Polylogue stays complete) and for the
infrastructure the two Rust codebases share.

Inputs: `/realm/inbox/spec/polylogue-spec/` (especially `decisions/sinex.md`,
`spec/storage.md`, `spec/model.md`, `evidence/blobstore-design.md`) and
`/realm/inbox/spec/sinex-spec/` (especially `spec/storage.md` §9,
`evidence/studies/polylogue-on-sinex-comparison.md`, `decisions/privacy.md`), plus
both projects' Beads (`polylogue-303r`, `sinex-4j2`, `sinex-zi6`).

## Current realization work

[realization/README.md](realization/README.md) owns the derived indexed requirement map, M0/M1 task contracts, disposable Beads trial and remaining prose review.

[BOOTSTRAP-INVENTORY.md](BOOTSTRAP-INVENTORY.md) records the composition audit, replacement categories and minimum executable acceptance.

[REALIZATION-PLAN.md](REALIZATION-PLAN.md) owns the long-term delivery strategy: fresh workspaces in both existing repositories after verified preservation, bootstrap and product milestones, requirement/task mapping, inspectable end-to-end demonstrations and role ownership. [PLAN-2026-10-13.md](PLAN-2026-10-13.md) records the narrower prototype work and its remaining contracts. Neither authorizes deployment, state reset or publication.

The topic positions/comments below preserve the original joint sessions' reasoning. Apply current outcomes through their owning decisions and plans; the 2026-10-11 in-place Sinex assumption is superseded by [the revised decision](../sinex-spec/decisions/fresh-or-fix.md).

## How we work

- One file per topic in `topics/`. Each has three sections: **Polylogue
  position** (edited only by polylogue-d7), **Sinex position** (edited only
  by repo-b8), and **Proposal** (drafted by the topic's owner, commented on
  by the other in a `Comments` list, never rewritten by the non-owner).
- `decisions.md` is the table of outcomes. A row moves to *agreed* only when
  both sessions have written "agree" in its topic file. Rows needing the
  operator are marked *operator* with both options and a default, and are
  copied into both specs' `questions.md`.
- `LOG.md`: timestamped, one line per change, prefixed with the session.
- Agreed outcomes are then written into each project's own spec by that
  project's session. This directory holds the reasoning, not a third spec.

## Topics

| File | Topic | Owner (drafts the proposal) |
|---|---|---|
| `topics/01-contract.md` | data split, identity mapping (Polylogue IDs and `sinex:occ`/`sinex:eq`/`sinex:ev`), revisions, lineage as typed references, where judgments, assertions and deliveries live, excision as purge | repo-b8 |
| `topics/02-projections.md` | who owns the Postgres projections of Polylogue data; what Polylogue's index rebuilds from; the decisive proof (Sinex slice 6 acceptance and Polylogue's rebuild law) | polylogue-d7 |
| `topics/03-transport.md` | producer protocol (`sinex-zi6`) over the external ingress, versus Sinex reading Polylogue's change feed; in-process admission (Sinex Q5b); failure and degraded states | repo-b8 |
| `topics/04-content-store.md` | shared content store: `polylogue-blobs` as the base, appended pieces, digest (SHA-256 vs BLAKE3), excision, privacy (Sinex `decisions/privacy.md`) | polylogue-d7 |
| `topics/05-shared-crates.md` | model types, ID and time rules, testkit (fixtures, laws, example-world checkers), xtask patterns, lints, toolchain (stable vs nightly), where shared crates live | polylogue-d7 |
| `topics/06-sequencing.md` | which slices of each project come first, so neither waits on the other; when Polylogue's slice 13 moves earlier | repo-b8 |

## Beyond the topics

| Path | What it holds | Check |
|---|---|---|
| `decisions.md` | the agreed outcomes (31 rows) | |
| `glossary.md` | the words both projects share, or deliberately use differently | |
| `query-laws.md` | the query laws both languages hold, and where each project checks them | |
| `vision/` | uses that need both projects | `joint-vision` |
| `outline/` | the boundary: which component on each side owns each joint interface | `joint-outline` |
| `visual-tokens.css` | the one visual language (decision 21), copied into Polylogue's render crate | `polylogue-visual-tokens` |
| `prototype/contract/` | Polylogue publishing to Sinex admission: revisions, framing, excision with purge validation (no live message left citing only deleted inputs), the replicated frontier (`frontier.py`), the route move (`route_move.py`, `route_e2e.py`) | `joint-contract` |
| `prototype/hashed-blobs/` | the shared content-store crate: plain, delta and sealed blobs | `joint-hashed-blobs` |
| `prototype/logical-day/` | the shared logical-day crate | `joint-logical-day` |
| `prototype/signals/` | one evaluator for both projects' signal rules | `joint-signals` |
| `prototype/namespaces/` | cross-system query namespaces, with the colliding words derived from both evaluators | `joint-namespaces` |
| `prototype/masking/` | credential masking (decision 31): masked by default, `--reveal` only on a terminal with no harness (the terminal test carries most of the weight, since an agent's shell captures stdout; the harness variables are the second guard), the reader's page holding no secret until a click | `joint-masking` |
| `prototype/ux-u6/` | Polylogue on Sinex on one afternoon of both status lines | `joint-ux-u6` |
