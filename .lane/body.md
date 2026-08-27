Summary

Add a versioned `verify falsification` gate that composes the existing safety
and semantic-fidelity evidence with an independent cursor-pagination oracle and
the declaration-driven interaction matrix. The report records populations,
sampling, stable references, versions, blind spots, resource cost, and mutation
controls.

Problem

The repository had independent evidence slices but no single reproducible
result tying their first three truth/operability checks to interaction coverage.
That made reruns and the interaction gate difficult to compare.

Solution

Add `devtools/falsification_program.py`, its contract artifact, a seeded query
mutation control, the command-catalog entry, and anti-vacuity tests. Register
the existing `context` command owner so the interaction matrix reflects the
current CLI surface.

Verification

- `./.venv/bin/python -m devtools test tests/unit/devtools/test_falsification_program.py`
- `python -m devtools render devtools-reference --check`
- `git diff --check`
- `python -m compileall -q devtools/falsification_program.py tests/unit/devtools/test_falsification_program.py`

The focused test exercises the semantic, query, and interaction routes. Full
safety/rebuild execution is opt-in because the existing rebuild witness timed
out at 120 seconds in this environment; `--execute-safety` records that result
for a full campaign run. The system interpreter could not load `aiosqlite`, so
the checkout virtualenv is the valid test environment.

Residual risk

The aggregate command does not open live archives, does not replace the full
query-law corpus, and does not measure cold human task success. Those remain
explicit blind spots in the report and require their owning campaigns.
