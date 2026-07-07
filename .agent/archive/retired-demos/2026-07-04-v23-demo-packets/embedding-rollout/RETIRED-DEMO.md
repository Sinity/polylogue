# Embedding Rollout Demo

This demo records the current live-archive semantic-search rollout state.
It is a current artifact, not a historical append-only log. Regenerate it after
meaningful embedding progress or after changing embedding product behavior.

Archive root: `/home/sinity/.local/share/polylogue`
Index schema: v23

## Current Proof

The latest bounded proof ran against the active archive with:

```bash
polylogue ops embed backfill \
  --yes \
  --max-sessions 25 \
  --max-messages 5000 \
  --max-cost-usd 1 \
  --format json
```

Latest material catch-up run:

- processed sessions: 25
- embedded sessions: 25
- embedded messages: 498
- errors: 0
- estimated window cost: $0.0249

Current status:

- embedded sessions: 6,371
- embedded messages: 426,894
- failure rows: 0
- needs-reindex rows: 0
- retrieval ready: true
- next preflight window: 9,389 pending sessions, 1,068,925 pending messages,
  estimated at $53.44625

Current read-only progress (`current/full-run-progress.json`) shows:

- embedded message rows: 426,894
- embedding status rows: 6,371
- failure rows: 0
- needs-reindex rows: 0

The demo proves that Polylogue can run a bounded prose-only embedding backfill
on the active archive, inspect the before/after state, distinguish material
catch-up progress from newer zero-progress daemon scans, and keep semantic-search
coverage visible through status/preflight artifacts.

## Files

- `current/summary.json` — compact claim, non-claim, proof fields, caveats,
  and source-file list for the demo shelf index.
- `current/status-before.json` and `current/status-after.json` — normal
  `polylogue ops embed status --format json` before and after the bounded run.
- `current/preflight-before.json` and `current/preflight-next.json` — bounded
  window preflights for the completed run and the next comparable window.
- `current/backfill-result.json` — structured result from the bounded provider
  call.
- `current/status.json` — current detailed embedding status payload.
- `current/preflight-remaining.json` — current read-only pending/cost estimate
  for `--min-messages 2`.
- `current/full-run-progress.json` — service state plus direct `embeddings.db`
  row counts joined with the pending preflight.
- `regenerate.sh` — read-only refresh for current status/progress artifacts.

## Caveats

- This is not full embedding convergence. The archive still has pending
  prose-bearing sessions.
- Cost estimates are approximate and use the product's configured estimated
  tokens per message.
- The active prod daemon can still write older zero-count catch-up rows until
  deployed code catches up; `latest_material_catchup_run`, `backfill-result.json`,
  and before/after status are the proof for this packet.
