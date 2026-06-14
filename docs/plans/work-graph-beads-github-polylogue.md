# Work graph boundary

Beads owns internal work planning.
GitHub owns public collaboration.
Polylogue owns evidence, traces, context, KV assertions, WorkPackets, reports, and recovery views.

Polylogue should import Beads items as stable refs, observe Beads CLI usage in agent sessions, link Beads items to sessions/runs/PRs/checks/KV through WorkPackets, and preserve mappings between Beads items and GitHub issues or PRs.

A WorkPacket is not a task tracker entry. It is the evidence envelope for an attempt or outcome.

Acceptance: one Beads item can show linked runs, PRs, checks, and KV; a successor agent can recover work state from Beads plus Polylogue; Polylogue does not reimplement Beads.