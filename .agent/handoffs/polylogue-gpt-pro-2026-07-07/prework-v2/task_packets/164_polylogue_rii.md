# 164. polylogue-rii — Live substrate intake: agents write work-events; evidence materializes in-loop

Priority/type/status: **P2 / epic / open**. Lane: **06-agent-context-coordination**. Release: **D-agent-context-coordination**. Readiness: **epic-needs-child-closure**.

## What the bead says

Invert the relationship for live agents: work lands in Polylogue as it happens (push), and the agent reads context/evidence back in-loop. OPERATOR GATE: direction confirmed as worth phasing, full program needs explicit green-light before a large build. Hermes-specific ingestion lives in the Hermes bridge program; this program owns the generic write-leg and intake seams. GH issue thread (body + comments) is input, not authority; this bead's scope statement wins where they conflict.

## Existing design note

Invert the relationship for live agents: work lands in Polylogue as it happens (push) and the agent reads context/evidence back in-loop. OPERATOR GATE: the direction is confirmed worth phasing, but the full program needs an explicit green-light before a large build. This epic owns the GENERIC write-leg and intake seams (rii.1 is the first child); Hermes-specific ingestion lives in the Hermes bridge (fs1). Treat the GH issue thread as input, not authority; this bead's scope statement wins where they conflict.

## Acceptance criteria

- The generic write-leg + intake seam scope is defined and split into child beads (rii.1 = the agent work-event write-leg); Hermes-specific ingestion is explicitly excluded and pointed at fs1.
- The program stays gated: no large build starts until an explicit operator green-light is recorded as a bead comment.
- The epic advances when rii.1 lands and an agent's pushed work-event materializes into the run-projection read-models within one convergence cycle (see rii.1 acceptance).

## Static mechanism / likely defect

Issue description localizes the mechanism: Invert the relationship for live agents: work lands in Polylogue as it happens (push), and the agent reads context/evidence back in-loop. OPERATOR GATE: direction confirmed as worth phasing, full program needs explicit green-light before a large build. Hermes-specific ingestion lives in the Hermes bridge program; this program owns the generic write-leg and intake seams. GH issue thread (body + comments) is input, not authority; this bead's scope statement wins where they conflict. Design direction: Invert the relationship for live agents: work lands in Polylogue as it happens (push) and the agent reads context/evidence back in-loop. OPERATOR GATE: the direction is confirmed worth phasing, but the full program needs an explicit green-light before a large build. This epic owns the GENERIC write-leg and intake seams (rii.1 is the first child); Hermes-specific ingestion lives in the Hermes bridge (fs1). Treat the …

## Source anchors to inspect first

- No precise source anchor was localized in this static pass. Start from the bead description and repository search.

## Implementation plan

1. Invert the relationship for live agents: work lands in Polylogue as it happens (push) and the agent reads context/evidence back in-loop.
2. OPERATOR GATE: the direction is confirmed worth phasing, but the full program needs an explicit green-light before a large build.
3. This epic owns the GENERIC write-leg and intake seams (rii.1 is the first child)
4. Hermes-specific ingestion lives in the Hermes bridge (fs1).
5. Treat the GH issue thread as input, not authority
6. this bead's scope statement wins where they conflict.

## Tests to add

- Acceptance proof: The generic write-leg + intake seam scope is defined and split into child beads (rii.1 = the agent work-event write-leg)
- Acceptance proof: Hermes-specific ingestion is explicitly excluded and pointed at fs1.
- Acceptance proof: The program stays gated: no large build starts until an explicit operator green-light is recorded as a bead comment.
- Acceptance proof: The epic advances when rii.1 lands and an agent's pushed work-event materializes into the run-projection read-models within one convergence cycle (see rii.1 acceptance).

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit/coordination tests/unit/mcp -q`

## Pitfalls

- Do not implement the epic as a single broad PR; use it to close/split child work and verify the terminal invariant.
- Do not add direct context injection or trusted memory writes outside the scheduler/candidate policy.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
