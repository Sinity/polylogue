# Presentation: how Polylogue is explained to someone new

How the README, the demo and the first-run experience present Polylogue to
a person who has never seen it, from the recorded designs rewritten in
plain words.

## What Polylogue says it is

A local archive of your AI chat and agent sessions that keeps the exact
record: what was said, which commands ran and how they ended, which session
continued which, what it cost. It lets you check what an agent claimed
against what actually happened.

It is not presented as an observability platform, a memory system for
agents or an evaluation framework, though it feeds all three. The honest
difference from other tools: it keeps every input file byte for byte, works
across providers, and treats tool outcomes, lineage and cost as records you
can cite, not as summaries.

## The README

1. **One reproducible proof first.** A claim checked against the record,
   from the demo archive:

   ```console
   $ polylogue demo
   $ polylogue --archive ~/.local/share/polylogue-demo 'messages where written_by:model "all tests pass"' list
   ```

   finds a model message saying "all tests pass", and

   ```console
   $ polylogue --archive ~/.local/share/polylogue-demo 'tool_calls where command:"cargo test" outcome:error' list
   ```

   shows the `cargo test` run just before it, which exited with code 101.
   The later fix and passing run are also there and do not overwrite the
   failure. A plain text search over the files would find the claim but not
   the failed run's outcome; that contrast is the point.
2. **Three groups of what it does**, each with one command and its real
   output: find and read past sessions; lineage and cost; context for the
   next agent (briefings and notes).
3. **Install** (Nix package, static Linux binary) and the first run against
   your own Claude Code sessions.

The README says what Polylogue does and how to run it, nothing about the
project's own process: no statements about how its claims are phrased,
which measurements are pending or how it is developed (operator,
2026-07-04, on such a line: "why would it be in the readme? it's meta.").

Every command shown in the README exists and runs: the README's command
blocks are executed by the same test harness as the spec's examples
(`plan/testing.md`, layer 1), so public copy never describes something that
does not run. No screenshots stand in for commands. The README's
screencasts of the web reader are recorded by the same real-browser run
that verifies the reader over Morrow (`surfaces/web.md`, how the reader is
verified), so its media are a byproduct of a passing check and cannot show
what the reader does not do (operator, 2026-05-01: "pretty screencasts
generated as byproduct artifacts of verifiability").

## Two demonstrations

The **real archive** is the main demonstration: the operator's own
sessions, with the story of each project told from stored facts and checked
notes (`workflows.md` §17). Being specific to one person is the point: it
shows results from actual use (`evidence/intent-journal.md`). Anything
taken from it for public view goes through the share step
(`workflows.md` §19).

The **Morrow archive** is the reproducible one: anyone can build it and
run every command in the README against it.

## The demo archive

`polylogue demo` builds a small synthetic archive from the Morrow world
(`examples/archive/FACTS.md`), starts a daemon for it, and prints the
commands to run, each with `--archive ~/.local/share/polylogue-demo`; the
README's commands carry the same option. It marks the directory as a demo archive, refuses to build
into an existing directory it did not create, can never write into the
operator's real archive, and the real hook spool is never collected into it.
Slice 8 adds one case to that world for the README proof: a model message
claiming the tests pass, placed after a failed test run and before the
passing one. The main example archive's totals do not change; the case lives
in the demo archive.

The demo archive never calls a paid embedding provider; semantic search in
the demo uses vectors shipped with it or none.

Published demonstrations are documented query sequences with their outputs
against the demo archive or chosen public data, never one-off code.

## The first outside user

The first run for someone else: a clean install, the demo, one query against
their own data that is answered from the record, one honest "not available"
(a provider they don't use, or cost unknown for a subscription chat), and an
output they can cite (a session or message ID).

Acceptance for slice 8: a real person other than the operator, with no help,
installs from the release binary or the Nix package, runs the demo and
answers one real question about their own sessions using only the README. An
agent never stands in for that person.

Bead and audit IDs behind each section: `evidence/beads/spec-sources-g3.md`.
