# Slice 1 starts here

One page for whoever begins slice 1 (Claude Code, end to end) in the
existing Polylogue repository after preservation and executable bootstrap (`decisions/repository.md`).
`plan/slice-01.md` is the contract; this page is the way in.

## Read, in order

1. `DIGEST.md`: what Polylogue is and how the package is layered.
2. `plan/slice-01.md`: scope, signatures, work split, acceptance.
3. `spec/architecture.md`: the system on one page, one session end to end.
4. `spec/model.md`: Identity (kinds of sameness, provider and provisional
   IDs), Message, Time (quality and precision).
5. `spec/sources/README.md` (the per-record framework, law 120) and
   `spec/sources/claude-code.md`.
6. `spec/storage/README.md`, `blobs.md` (observing a file, batched
   durability), `archive-db.md`, `index-db.md`.
7. `spec/convergence.md`: the loop and how it finds its work.
8. `spec/query.md`: Principles (2: errors and command words), Grammar, the `sessions` unit's slice-1 fields.
9. `spec/surfaces/cli.md` and `decisions/process.md`: the socket, the
   commands, exit codes.
10. `plan/testing/laws.md` (the index rows with slice 1) and
    `plan/rewrite-rules.md`.
11. `examples/archive/FACTS.md`: the Morrow world every example runs on.
12. `skeleton/crates/` and `outline/components.toml`: the signatures and
    the components with `slice = "1"`.

## Seed from the prototype, and what not to

Seed (copy, then bring to the skeleton's signatures; their tests come too):

- `prototype/crates/sources`: the framework (`frame.rs`) and the Claude
  Code parser with its carry; law 120's accounting and append equivalence
  at every split point pass there.
- `prototype/crates/model`: types, `choose_version` and merging. Not its
  identity rules (`ids.rs`: colon handling, the fallback hash input,
  occurrence counting): they follow an earlier draft and are rewritten from
  `spec/model.md`, Identity (the provider's raw record as canonical JSON
  without volatile fields, `n` counted per input revision), as
  `prototype/README.md` warns.
- `prototype/crates/store/src/inputs.rs`: the input layer, including
  `first_batch` (batched durability; without it a first build waits on
  syncs, `evidence/real-converge.md`, the evening section).
- `prototype/crates/engine`: the collector's change classes and the loop,
  with `tests/converge.rs` (idempotence, a stop after every durable step,
  law 81, law 121). The prototype has no `head_rewritten` class, which
  slice 1 needs (`spec/sources/claude-code.md`; law 21): add it from
  `spec/storage/blobs.md`. It also produces `replaced` and `renamed`, which
  are slice 2: slice 1 is not required to produce them, keeps the code, and
  does not test it until slice 2.
- `prototype/crates/store/src/write.rs`: the writer's core, rewritten behind
  the skeleton's `WriterHandle` channel.

Do not seed: the daemon prototype's briefings, reports and live state
(slice 2 and later); the Codex, ChatGPT, Gemini, Hermes and Grok parsers
(their slices; keep them as the starting point there); the prototype's
`query` crate (stubs; the grammar starts from `spec/query.md`, with
`prototypes/query-ref/query_ref.py` as the reference evaluator to compare
against, not to translate). The real-data examples (`real_converge.rs`,
`codex_*`) are tools for acceptance item 3, not product code.

Where the seeded code is behind the spec (the spec wins, and no check
covers these yet): `inputs.rs` and the engine use `INSERT OR IGNORE` (three
statements each), which rules R24 and R92 forbid: a key collision becomes
an error or goes through the merge rule; the prototype numbers Codex
blocks with one counter, while Codex reasoning items have their own
(`spec/model.md`, Identity; Claude Code keeps one counter); it normalizes
commands by the older rule (`spec/analysis.md`, normalized command, writing
forms included); it knows fewer title sources than `spec/model.md` lists
(`index`, which slice 1 parses from `sessions-index.json`, and `job`); it
reads repository identity from the working directory, while the spec joins
sessions to repo-lookup inputs; its briefing header does not name the time zone
(`spec/user-data.md`).

## The first three commits

1. **Preservation and executable bootstrap.** Follow the joint realization plan and repository decision:
   verified pre-rewrite refs/private bundle, recoverable WIP, deployment-consumer inventory; then a
   coherent fresh-tree PR with target types/edges and usable build/check routes. Copy owning spec/docs
   and neutral examples to their target home, wire schema/type comparisons, and test actual ownership
   failures. No supported route contains todo!(); later signatures stay unsupported. The current
   skeleton audit/replacement inventory must settle the exact contents before PR creation.
2. **Model and sources.** Seeded types, identity and the Claude Code parser,
   passing the parser's examples, law 120 and append equivalence.
3. **Store and engine.** `hashed-blobs` (the joint crate), the input layer
   with batched durability, `cksumvfs` built and registered
   (`spec/storage/README.md`), the writer, and the loop, passing the
   convergence tests above.

Then the query grammar, rendering, the CLI, MCP and the socket (workers D,
F, G), and the acceptance runs.

## Acceptance, in short

`plan/slice-01.md`, Acceptance: the slice-1 examples on the
Claude-Code-only home (known cost $7.87); every law whose slice is 1, with
the slice-1 lineage laws (4 and 11 for copies and late parents, 29–34 and
99; `plan/slice-01.md`, acceptance), with 11, 29–34 and 99, law 14 and laws
71, 123 and 125 written first, since they have no prototype test yet; real data (every file parsed or
with a reason, `xtask compare-old` explained, the first build under 10
minutes and 2 GB with IO pressure recorded and a tmpfs control for a miss,
searches under 200 ms); and the operator using it.

## Open decisions

- **Bootstrap contents.** The joint realization plan's replacement inventory/skeleton audit must
  establish actual types, dependencies, supported commands and package consumers before replacement.
  This packet does not certify readiness or authorize state moves/deployment.
- **Parse commits.** The writer commits once per session (a parse too large
  for one bounded transaction in steps with its parse_state), the remaining
  IO-sensitive part of a first build (14 s against 2.9 s on tmpfs for
  120 MB in the prototype). Recommended: keep it until acceptance item 3
  shows the target missed, then group several small sessions per
  transaction as collect batches its first observations.
