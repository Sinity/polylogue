# 087. polylogue-h6r — Agent identity: a stable who-did-this tuple for every session

Priority/type/status: **P2 / task / open**. Lane: **03-lineage-compaction-truth**. Release: **F-lineage-compaction**. Readiness: **blocked-hard**.

Hard blockers: polylogue-7aw

## What the bead says

Half the analytics tower assumes 'per agent' partitions the schema cannot express: a model name is not an agent — the same model under different CLAUDE.md versions, skills, or MCP profiles is behaviorally a different worker. Calibration (h10), setup A/B, advisory tuning, and the evaluation instrument all need a stable agent-identity key or they average across regime changes and mislead exactly when the setup changes (which is when you look).

## Existing design note

agent_identity = (model_name, harness+version, config_state_ref, role) materialized per session as a derived column set: model/harness from session metadata (exists), config_state_ref from the 7aw config-artifact joins (the hash of the config set in force at session start), role from subagent dispatch context where present. Store as columns on session_profiles (the hot-row home) + a registry of observed identities with first/last seen. Measures gain identity as a grouping unit ('measure X by agent'). Degrades honestly: sessions predating config capture get config_state_ref=unknown, and identity-partitioned measures state the unknown fraction. Small bead, foundational — blocks h10's per-agent calibration claims.

## Acceptance criteria

Identity tuple materialized for new sessions on the live machine (config ref resolving via 7aw); an identity-partitioned measure runs with the unknown fraction stated; h10's calibration curves key on identity, not bare model name.

## Static mechanism / likely defect

Issue description localizes the mechanism: Half the analytics tower assumes 'per agent' partitions the schema cannot express: a model name is not an agent — the same model under different CLAUDE.md versions, skills, or MCP profiles is behaviorally a different worker. Calibration (h10), setup A/B, advisory tuning, and the evaluation instrument all need a stable agent-identity key or they average across regime changes and mislead exactly when the setup changes (which is when you look). Design direction: agent_identity = (model_name, harness+version, config_state_ref, role) materialized per session as a derived column set: model/harness from session metadata (exists), config_state_ref from the 7aw config-artifact joins (the hash of the config set in force at session start), role from subagent dispatch context where present. Store as columns on session_profiles (the hot-row home) + a registry of observed identities w…

## Source anchors to inspect first

- `polylogue/archive/session/threads.py` — Session/thread lineage read and composition model.
- `polylogue/insights/topology.py` — Topology/lineage derived insight code.
- `polylogue/daemon/lineage_startup.py` — Daemon lineage startup/convergence path.
- `polylogue/archive/coverage.py` — Completeness/truncation cues live here.
- `polylogue/insights/postmortem.py` — Compaction/continuation postmortem evidence is mined here.
- `polylogue/core/refs.py` — Existing ref model should be extended, not bypassed.
- `polylogue/storage/sqlite/archive_tiers/user_write.py:901` — Findings should reuse assertion/candidate lifecycle.
- `polylogue/surfaces/payloads.py:747` — Surface payloads currently expose branch/message refs; expand consistently.
- `polylogue/mcp/payloads.py:377` — MCP message payloads carry variant/branch metadata today.

## Implementation plan

1. agent_identity = (model_name, harness+version, config_state_ref, role) materialized per session as a derived column set: model/harness from session metadata (exists), config_state_ref from the 7aw config-artifact joins (the hash of the config set in force at session start), role from subagent dispatch context where present.
2. Store as columns on session_profiles (the hot-row home) + a registry of observed identities with first/last seen.
3. Measures gain identity as a grouping unit ('measure X by agent').
4. Degrades honestly: sessions predating config capture get config_state_ref=unknown, and identity-partitioned measures state the unknown fraction.
5. Small bead, foundational — blocks h10's per-agent calibration claims.

## Tests to add

- Acceptance proof: Identity tuple materialized for new sessions on the live machine (config ref resolving via 7aw)
- Acceptance proof: an identity-partitioned measure runs with the unknown fraction stated
- Acceptance proof: h10's calibration curves key on identity, not bare model name.

## Verification commands

- `devtools test <focused tests added for this bead>`

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
