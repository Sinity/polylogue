# Coding-agent master prompt for Polylogue urgent/correctness packet work v2

You are implementing one packet from `polylogue_urgent_bead_static_prework_v2`.

Rules:

1. Start from the packet file, then open every source anchor in the real working tree. Do not edit from the line numbers blindly; the packet was generated from snapshot `master @ 8a975a40` on 2026-07-06.
2. Confirm the mechanism before coding. If the code has moved, update the packet note in your final report.
3. Write or update a focused failing test first. The test must encode the bead acceptance criteria, not just exercise the changed function.
4. Patch the smallest shared seam. Prefer central contracts and write/read chokepoints over caller-by-caller fixes.
5. Preserve evidence honesty: missing is not zero, text-derived is not observed, fallback time is not provider time, agent-written is not user-trusted.
6. Run the packet verification commands and report exact commands, exit codes, and residual failures.
7. Do not close an epic from a child packet. Epics close only after child closure or explicit split.

Deliverable format:

- Bead ID and packet file.
- Source anchors inspected.
- Confirmed mechanism or corrected mechanism.
- Tests added/changed.
- Code changed.
- Verification commands and results.
- Any new beads needed.
