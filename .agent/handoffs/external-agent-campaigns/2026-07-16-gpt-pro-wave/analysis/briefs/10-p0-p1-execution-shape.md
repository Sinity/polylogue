Title: "[analysis 10] P0/P1 program shape and dependency adjudication"

Job ID: `analysis-10`
Result ZIP: `analysis-10-p0-p1-execution-shape-r01.zip`

Audit the live P0/P1 portfolio as a dependency graph rather than a flat list.
Identify epics versus executable leaves, duplicated parent/child priority,
blocking edges that are genuine versus historical residue, and P1s that are
effectively required for P0 closure. Preserve project ambition: the goal is a
more tractable execution program, not demotion by convenience.

Use current Beads notes, merged history, and source anchors. Deliver a
decision-ready active frontier, safe parallel partitions, shared-file hotspots,
promotion/demotion recommendations with evidence, and a clear account of which
items must not be launched until a prerequisite receipt exists.
