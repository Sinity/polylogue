# GPT-Pro delivery-upgrade corpus (2026-07-07)

Captured 811-msg session + `upgrade-setup/` (delivery-gate release
definitions) + `prework-v2/` (194 per-bead execution accelerator packets,
`task_packets/<NNN>_<bead-id>.md` — source anchors, mechanism, implementation
plan, tests, verification commands, pitfalls, generated from master @
`8a975a40`) + `prework-v1-superseded/`. Backs ~188 open/closed beads'
`[Prework packet 2026-07-07]` notes.

**Path note**: the original `.agent/scratch/` location had a
`task_packets/task_packets/` doubled-segment path (worked around with a
self-symlink, `task_packets -> .`) — collapsed to a single segment during
the 2026-07-16 move to `.agent/handoffs/`; bead notes were rewritten to
match. Re-verify source anchors against current master before coding —
line numbers are snapshot-relative to the 8a975a40 generation commit.
