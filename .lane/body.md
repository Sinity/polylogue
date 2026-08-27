Summary

Record the route-B decision and its evidence for the reindex gate. No product code or archive state changed.

Problem

The operator requested a decision about recapturing still-reachable provider conversations before the reacquisition window. The durable bead record reports a ChatGPT reachability census of 15/15 random samples and 7/8 oldest samples, with one oldest conversation gone. Claude was not censused. This supports live recapture, but does not prove the full population or complete the recapture.

Solution

Keep route B selected for ChatGPT, and require a per-origin reachability census, agent-window batch recapture, and provider-native-identity deduplication against route A before the clean source generation. The declared `live_provider_proof` operation was run at job `5f9d7065-5fb7-4356-aa2b-46e57439711c`; it failed its `ok` contract and produced no provider capture evidence. No undeclared browser automation or operator-tab mutation was substituted.

Verification

- `agentctl --plain job start polylogue live_provider_proof --workspace 015fcee5-0066-406b-8b85-9a2cd3a002c7` — admitted job `5f9d7065-5fb7-4356-aa2b-46e57439711c`.
- `agentctl --plain job wait 5f9d7065-5fb7-4356-aa2b-46e57439711c` — terminal `phase=failed`, `ExecMainStatus=1`.
- `agentctl --plain job result 5f9d7065-5fb7-4356-aa2b-46e57439711c` — `ok=false`, `live provider proof reported an unsuccessful result`.
- `bd show polylogue-4t7y7` — operator decision and ChatGPT census evidence present in the task record.
- `devtools verify --quick` — managed job `43d511bd-add3-42ea-9dd2-0c23d528ba31`, `exit_code=0`, `semantic_status=non-test-passed`, 17 static gates passed.

Residual risk

The bead remains open. Full ChatGPT population reachability, all Claude reachability, live batch capture, and A/B dedup are not evidenced. The existing census is sampled and pre-reindex.
