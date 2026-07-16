# Results

Create `results/<job-id>/<attempt-id>/result.json` when a chat is launched or
imported. Store every downloaded/captured package under `raw/` by its declared
filename plus SHA-256; never overwrite a prior revision. `result.json` records
provider conversation/turn, prompt hash, snapshot identity, package hash,
validation, triage, iteration/supersedes lineage, and integration receipts.

The initial empty ledger is `index.json`. It is not execution state until an
orchestrator or operator writes evidence-backed entries.
