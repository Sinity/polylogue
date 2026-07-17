# External-agent campaign workspace

This directory is the repository-side, provider-independent home for external
agent campaign inputs and returned artifacts. It is local orchestration data,
not a Polylogue product protocol and not browser-extension state.

Each campaign run is a self-contained directory with one or more workload
subdirectories. A workload has:

- `campaign.json`: stable campaign, workload, job, dependency, prompt, and
  expected-result identities;
- `briefs/`: concise job-specific source material;
- `prompts/`: fully rendered prompts ready to paste or submit;
- `results/index.json`: a rebuildable query projection of immutable returned
  attempt/package receipts;
- `results/README.md`: the on-disk result convention.

The prompt renderer combines a workload's brief with the named shared contract:

```bash
python .agent/handoffs/external-agent-campaigns/render_prompts.py \
  .agent/handoffs/external-agent-campaigns/2026-07-16-gpt-pro-wave/testdiet
```

Use `--check` at a publication boundary. The rendered prompt is the dispatch
artifact; a launcher never needs to understand prompt fragments.

## Stable identities

- Campaign: a dated, readable execution portfolio, for example
  `gpt-pro-wave-2026-07-16`.
- Workload: a reusable problem family within the campaign, for example
  `testdiet` or `deep-research`.
- Job: `<workload>-NN`, with a fixed two-digit number and readable slug.
- Attempt: `aNN`; a retry or same-chat repair is a new attempt, not an
  overwrite.
- Package revision: `rNN` for workloads whose executor returns a package.

An orchestrator may add provider conversation IDs, Polylogue ObjectRefs,
content hashes, Beads, worktrees, agents, PRs, and merge outcomes to
`results/index.json`. It assigns stable local filenames to conversation-native
artifacts such as Deep Research responses and must not infer completion from a
filename alone.

## Result path

Returned material is stored as:

```text
results/<job-id>/<attempt-id>/
  result.json
  raw/                 # immutable downloaded/captured provider artifacts
  extracted/           # deterministic extraction of raw packages
  integration/         # local adjudication, worktree, test, PR receipts
```

`raw/` is hash-addressed evidence. `extracted/` can be rebuilt. `integration/`
records what Polylogue maintainers actually accepted and verified. Large or
private artifacts stay outside Git when policy requires it; `result.json`
then records the absolute custody path and checksum rather than pretending the
file is absent.

Machine-readable shapes are documented by `schemas/campaign.schema.json` and
`schemas/result.schema.json`. `campaign.json` is dispatch intent and should
remain stable after launch. Per-attempt `result.json` is immutable evidence.
`results/index.json` is a rebuildable reconciliation projection across those
receipts. Its sole outcome field is `state`; it must never carry a second
`status` field or independently adjudicate an artifact. Rebuild and verify it
with:

```bash
python .agent/handoffs/external-agent-campaigns/reconcile_results.py \
  .agent/handoffs/external-agent-campaigns/2026-07-16-gpt-pro-wave --check
python .agent/handoffs/external-agent-campaigns/reconcile_results.py \
  .agent/handoffs/external-agent-campaigns/2026-07-16-gpt-pro-wave --write
```

`result.json` is the immutable attempt form (`aNN`). `receipt.json` is the
immutable package-revision form (`rNN`) for a revision without a separate
provider attempt. The reconciler accepts both explicitly, rejects ambiguous
identity mappings, and never rewrites either receipt.

## Attachment policy

For ChatGPT Pro, attachment bytes are not assumed to consume the active prompt
context. Attach all relevant authorized evidence, including a complete Chisel
project-state archive, and provide navigation instructions. Do not remove
useful evidence merely because it is large. Reduce inputs only for a real
provider upload/retrieval limit, privacy boundary, or irrelevance.

Generated result packages must not copy the supplied repository/state archive
back into their output. They contain only new analysis, patches, tests,
evidence, and explanation.

For any externally linked artifact, acquire it before triage with the generic
URL custody command. It accepts HTTP(S) and file URLs, streams with a declared
byte ceiling, hashes while writing, atomically publishes the raw artifact, and
writes `acquisition.json`; it does not know a provider or browser workflow.

```bash
python .agent/handoffs/external-agent-campaigns/acquire_artifact.py WAVE_ROOT \
  --workload WORKLOAD --job JOB --attempt aNN --url URL
```
