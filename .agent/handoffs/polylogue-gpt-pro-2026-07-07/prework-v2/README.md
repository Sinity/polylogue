# Polylogue urgent/correctness/critical-path static prework v2

Generated from the current available Polylogue snapshot, not from the older v1 source directory.

Snapshot:

- Beads export: `/mnt/data/sysiter/polylogue_pkg78/polylogue/polylogue-beads-export.jsonl`
- Source tree: `/mnt/data/sysiter/tree78/polylogue`
- Export generated: 2026-07-06T215821Z
- Source git: master @ 8a975a40

Contents:

- `systematic_triage.md` — overview of all active beads and what changed from v1.
- `triage_matrix.csv` / `triage_matrix.json` — all 397 active beads with lane, release, readiness, blockers, and packet availability.
- `task_packets/` — 194 individual coding-agent task packets.
- `urgent_correctness_task_packets_v2.md` — combined summary of every packet.
- `task_packets.json` / `task_packets.jsonl` — machine-readable packet data.
- `task_index.csv` — packet order and handoff index.
- `source_anchor_index.md` / `.csv` — inspected source anchors grouped by mechanism.
- `critical_path_execution_order.md` — implementation order for the packetized critical path.
- `verification_lanes.md` — reusable exit gates by failure class.
- `packet_quality_audit.md` — source-localized vs spec/checklist packet quality classes.
- `coding_agent_master_prompt_v2.md` — pasteable master prompt for an implementation agent.

Scope:

This is static prework, not a source patch and not a live Beads DB migration. It turns urgent/correctness/critical-path beads into task packets that coding agents can implement with minimal rediscovery.

The v2 pass is broader than v1: v1 had 29 packets; v2 has 194 packets and a full active-bead triage matrix.
