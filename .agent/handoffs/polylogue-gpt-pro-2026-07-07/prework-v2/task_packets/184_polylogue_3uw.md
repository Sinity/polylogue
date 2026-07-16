# 184. polylogue-3uw — Capture-completeness: the instrument's coverage error as a standing measure

Priority/type/status: **P2 / task / open**. Lane: **10-analytics-experiments**. Release: **I-analytics-experiments**. Readiness: **blocked-hard**.

Hard blockers: polylogue-d1y

## What the bead says

Convergence legibility answers 'how converged is what we ingested'; nothing answers 'how much of what EXISTS did we ingest'. Sessions known to have happened (hook SessionStart fired, harness wrote a file, extension saw a chat) versus sessions fully archived = the coverage error, per origin, over time. An instrument that does not know its own coverage error cannot honestly caveat its findings — and silent capture regressions (the hibernation threat) currently have no number to trip on.

## Existing design note

Three evidence sources joined against the archive: hook events (SessionStart without a matching archived session after a grace window = a miss), watcher-root file inventories (files seen vs raw rows), extension capture-gap events (3v1). Materialize as a per-origin coverage measure (9l5.7 registry, tier=structural) with a trailing-window trend; surface in ops status + daemon health (alert on regression) + the day page's open-loops. The drift sentinel (da1) alerts on shape drift; this alerts on VOLUME drift — together they are the hibernation-mode floor instrumentation.

## Acceptance criteria

Coverage renders per origin on the live archive with the known-miss list drillable to refs; a seeded missed-session scenario trips the health alert; findings' sample-frame stanzas can cite the coverage number for their window.

## Static mechanism / likely defect

Issue description localizes the mechanism: Convergence legibility answers 'how converged is what we ingested'; nothing answers 'how much of what EXISTS did we ingest'. Sessions known to have happened (hook SessionStart fired, harness wrote a file, extension saw a chat) versus sessions fully archived = the coverage error, per origin, over time. An instrument that does not know its own coverage error cannot honestly caveat its findings — and silent capture regressions (the hibernation threat) currently have no number to trip on. Design direction: Three evidence sources joined against the archive: hook events (SessionStart without a matching archived session after a grace window = a miss), watcher-root file inventories (files seen vs raw rows), extension capture-gap events (3v1). Materialize as a per-origin coverage measure (9l5.7 registry, tier=structural) with a trailing-window trend; surface in ops status + daemon health (alert on regression) + the day pag…

## Source anchors to inspect first

- No precise source anchor was localized in this static pass. Start from the bead description and repository search.

## Implementation plan

1. Three evidence sources joined against the archive: hook events (SessionStart without a matching archived session after a grace window = a miss), watcher-root file inventories (files seen vs raw rows), extension capture-gap events (3v1).
2. Materialize as a per-origin coverage measure (9l5.7 registry, tier=structural) with a trailing-window trend
3. surface in ops status + daemon health (alert on regression) + the day page's open-loops.
4. The drift sentinel (da1) alerts on shape drift
5. this alerts on VOLUME drift — together they are the hibernation-mode floor instrumentation.

## Tests to add

- Acceptance proof: Coverage renders per origin on the live archive with the known-miss list drillable to refs
- Acceptance proof: a seeded missed-session scenario trips the health alert
- Acceptance proof: findings' sample-frame stanzas can cite the coverage number for their window.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit tests/integration -k "query or contract or citation or insight"`

## Pitfalls

- Do not publish a number or success claim until evidence tier, denominator, and caveat rendering exist.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
