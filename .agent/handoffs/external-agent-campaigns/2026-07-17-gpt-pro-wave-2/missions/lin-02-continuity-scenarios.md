Title: "Seven continuity replay scenarios with independent known-answer oracles (the t8t black-box program)"

Result ZIP: `lin-02-continuity-scenarios-r01.zip`

## Mission

Bead `polylogue-t8t` (P0, read its complete record in the snapshot's
`.beads/issues.jsonl`) specifies seven black-box continuity jobs that a cold
agent must be able to complete against the archive: resume, forensic lookup,
prior art, decisions, postmortem, cost audit, and self-inspection — plus the
parallel-agent incident replay. The design is execution-grade; the code does
not exist. An earlier external attempt was REJECTED for implementing only two
of seven jobs and introducing a competing scenario seam — do not repeat that:
find and reuse the repo's existing scenario/fixture machinery
(`polylogue/scenarios/`, `tests/infra/`, demo corpus seeding) and Beads notes
about where scenario declarations belong.

Implement the full declaration set + oracle machinery:

1. Seven scenario declarations, each: the operator-language question, the
   fixture archive state it runs against, the expected answer expressed as
   INDEPENDENT planted facts (from fixture construction/source evidence —
   never computed via the production query route under test), and the
   allowed query surfaces.
2. A known-answer oracle harness that executes a scenario through the real
   public route (CLI/API/MCP-shaped entry), compares against planted facts,
   and reports per-scenario pass/fail with evidence refs — designed so the
   same scenarios can later run against the live archive as the z9gh.7
   terminal gate (parameterize the corpus, don't hardcode fixtures).
3. Fixtures for all seven jobs, including the parallel-agent incident shape
   (coordinator + worker children where the corrected known-answer population
   is 129 coordinator children: 91 in the specific run, 38 not — mirror that
   structure synthetically).
4. Tests proving oracle independence: mutate the production query route
   (e.g. drop a filter) and show the scenario FAILS; mutate a planted fact
   and show the mismatch is reported with the right diagnosis.

## Constraints

- No new scenario framework if an existing seam serves; extend
  `scenarios`/test-infra idioms. State explicitly which seam you chose and
  why it is not a competing parallel abstraction.
- Scenario fixtures are synthetic/sanitized only.

## Deliverable emphasis

HANDOFF.md: scenario inventory table (job → question → oracle facts →
route), harness entry points, how the z9gh.7 live replay would invoke these
against the real archive, exactly which t8t acceptance criteria are
satisfied vs remaining, and proposed follow-up beads.
