# How it gets built

The operator wants this done differently from the last year: planned
top-down first, then built one piece at a time, each piece verifiably right
before the next starts, with a working prototype in a day or two rather than
weeks of blind agent work.

The source-replacement route is `decisions/repository.md`: verified pre-rewrite preservation, a fresh
executable bootstrap PR on the existing default branch, then vertical slices. A composition skeleton
with todo!() bodies is not an executable bootstrap. The replacement inventory/skeleton audit must
settle actual types, dependencies, unsupported registries and build consumers before dispatch.

## The shape of the work

1. **Spec first (this directory).** Product behavior, data model, storage,
   query language, surfaces, with concrete examples of inputs and outputs. The
   operator reads the examples and corrects them; disagreements become edits
   to the spec before any code.
2. **Vertical slices.** Each slice delivers something the operator can run
   end to end on real data, built in its final form (no stand-in code meant
   to be replaced later). Slice 1 is Claude Code only: read the files, keep
   the bytes, parse, store, search, read, and the MCP `search` and `read` tools. Later slices widen
   (more providers) or deepen (lineage, analysis, live updates). The order is
   in `plan/slices.md`.
3. **Each slice, in order:**
   1. *Design*: the slice's section of the spec is made precise, including
      exact types, function signatures at crate boundaries, schema changes and
      the acceptance examples. Written by the orchestrating model, checked
      against the full spec so the slice does not paint the design into a
      corner.
   2. *Examples reviewed*: the product lead checks independent expected facts and offers the
      milestone demo for operator inspection. Resolve actual ambiguity; operator availability is not
      a per-patch gate.
   3. *Implement*: parallel workers write the code (see below).
   4. *Verify*: discriminating checks while iterating; affected broad regression and slice examples
      at coherent integration; scoped private-corpus runs when their consequence requires them.
   5. *Operator tries it*: offer the milestone demo; daily-use acceptance remains open without
      the operator's experience or the actual client workflow it names.
   6. *Merge*: one PR per coherent delivery under actual forge gates. Merging code does not close
      unmet slice/usefulness or deployment criteria.
4. **Preserve the old implementation independently of the new tree.** Packages remain pinned to
   the verified pre-rewrite implementation until separate cutover; refresh whether it is running or
   parked. Never let old/new daemons write the same archive. Source replacement is not archive reset.
   See `decisions/repository.md`, `plan/migration.md` and the joint realization plan.

## Speed

Thousands of lines of code take minutes to write when the design is exact.
What made the last year slow was not typing; it was vague designs, agents
inventing machinery to cover the vagueness, and verification loops. The
countermeasures:

- **Exact interfaces before parallel work.** The designer writes the crate
  boundary types and function signatures (with `todo!()` bodies) and the
  tests that pin behavior. Workers fill in bodies. The compiler then catches
  most integration mistakes immediately.
- **Product lead, integrator and named milestone evaluation.** The joint realization plan owns
  cross-project roles. Authors own ordinary correctness review and scoped checks; a runtime evaluator
  inspects a fixed milestone client scenario with independent expected facts, not every patch.
  Independent workers use isolated worktrees with coherent outcomes. A shared checkout is allowed
  only with disjoint ownership and enforced prohibition of worker Git/index mutations; this is an
  exception, not the default. AgentCTL owns queued work and its checkout route. Preserve other edits
  and coordinate shared files with the integrator; do not dispatch every layer before dependencies exist.
- **The named integrator owns shared-file integration** (`Cargo.toml`s, each `lib.rs`,
  module lists, the provider registry, SQL files, `deny.toml`) and writes
  them in the signature step; workers own directories. Parser cases are
  example directories picked up by one test, so workers never edit a shared
  test file. One concept per file, around 600 lines at most; a file past
  that is split along the concept. Each provider uses the same three files
  (wire structs, mapping to model types, grouping into sessions), so a new
  provider is a copy of a known shape.
- **Model choice follows verification and measured results.** Use strong judgment for unresolved
  interfaces, reading evidence, design and review. Cheaper models can implement settled functions when
  tests/compiler and author review discriminate wrong behavior. Measure cost per accepted change;
  do not encode volatile prices or assume a compiler validates the intended outcome.
- **Workers get interface lists, not whole files.** Each worker prompt lists
  the existing public functions and types it may reuse (generated from the
  code), because agents skip reuse and reviewers don't notice duplication
  (`evidence/research-ai-codebases.md`). After each slice, the Opus review
  looks specifically for duplicated logic across workers' files and merges
  it before the slice is merged.
- **Measure the model mix.** Slice 1 records cost per accepted change for
  Haiku and Opus workers; the share of work given to Haiku in later slices
  follows that number, since published evidence on mixing is weak.
- **Verification batched at the end of each step**, not after every edit.
  A test that is wrongly red for an hour costs nothing; an agent re-running
  the suite after every small change costs hours.
- **Tasks name their spec section and their example.** A task's
  acceptance is behavior on an example input with its expected output,
  never "evidence that X holds". A finding outside the task's section goes
  to `questions.md` (a product question) or a task of its own (a real-data
  bug with its input), not into the same change.
- **Review asks three questions**, in a fixed prompt: does this produce wrong
  output on real data, and on which input? Does it contradict the spec?
  Could the same behavior be had with fewer concepts (tables, states,
  commands, nouns)? No "converged only at zero gaps", no adversarial threat
  brief. `xtask surface` shows the change in public surface (tables,
  columns, commands, MCP tools, public items, dependencies) for the PR
  description, so growth in concepts is visible without counting lines.
- **No product-side certification machinery.** Reproducible synthetic demonstrations and readable
  expected/actual output help the operator inspect software. Raw run/private-corpus evidence stays
  with its owner. No closure engine, proof score or duplicate scheduler. Code acceptance, daily use
  and deployed-client acceptance are distinct claims; leave unmet criteria open.

## Historical rough estimates (not acceptance)

| Step | Expected time |
|---|---|
| Spec first draft | today |
| Operator review of spec and slice 1 examples | operator's pace |
| Slice 1 (Claude Code end to end) | a few hours of agent work after review |
| Slices 2–4 (Codex, ChatGPT/claude.ai exports, live daemon) | about a day each |
| Remaining providers, lineage, analysis basics | days |
| Daily use replaces the Python version | within the first week or two |

These are estimates from the size of the core (tens of thousands of lines of
Rust), not commitments; the plan does not depend on them.

## Mistakes not to repeat

`plan/lessons.md` lists, one line each, the mistakes the Python version made
that its Beads recorded, grouped by area. Each slice's design step reads the
section for its area.

## Keeping the result small

The September campaign showed that a directive to shrink does not hold on its
own: code grew 40% during work meant as cleanup. The structure has to make
the small shape the default:

- the principles in `spec/principles.md` (one mechanism per job, no repair,
  no compatibility, correct by construction) are applied at design time, in
  each slice's design step, not discovered later;
- each slice's design states what it adds and roughly how many lines that
  should take, so growth beyond that is visible in review;
- the test suite is built from laws and examples (`plan/testing.md`), so a
  redesign does not require rewriting hundreds of tests;
- AGENTS.md in the new repository is short and states these rules in plain
  words.
