Title: "[testdiet 03] Incremental and rebuild equivalence"

Job ID: `testdiet-03`
Result ZIP: `testdiet-03-rebuild-equivalence-r01.zip`

## Mission

Implement an incremental-versus-rebuild equivalence survivor over Polylogue's
actual durable source evidence and rebuildable index route. Replay one realized
workload identity through incremental ingest/update/reprocess and through a
fresh index rebuild, then compare independently selected public facts:
session/message/block identity and active path, attachments, lineage, search,
materialized insight outputs, and absence/unknown semantics relevant to the
chosen canary.

Read `architecture/05-derived-freshness.md`. For every compared derived fact,
bind equivalence to the exact source identity, recipe identity, output
contract, and active generation—not row counts, timestamps, or a boolean stale
flag. Incremental, targeted, restarted, and rebuild routes must reach the same
canonical facts for the same derivation key while retaining distinct attempt
receipts.

Keep user-tier overlays outside content/workload identity and prove they are
preserved or deliberately rejoined rather than folded into raw content hashes.
The expected fact set must originate from planted input facts and declared
transform invariants, not from serializing one production result and comparing
it to another.

Name an omitted-rebuild-stage or stale-dependent-row mutation that the test
must kill. Respect durable-versus-derived schema rules and the sole-writer
architecture. Propose duplicated rebuild/example checks for later
certification, but do not remove them or invent another rebuild harness.
