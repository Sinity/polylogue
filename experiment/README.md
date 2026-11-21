# Compact final-file search screen

This unmerged experiment tests two immutable parsed layouts on synthetic data.
The full-versioned reference stores each body, observation and selection in
SQLite. The compact mode stores selected typed bodies in one sequential file
per partition, with explicit additional claims and immutable older-body bases.
Both keep the original bytes and publish `CURRENT` only after every final
file is committed, checkpointed and synced.

Search now has one global native FTS5 owner, shared by both final partitions.
It assigns one identity per distinct raw text and indexes each text once per
search channel. Both FTS tables use the production tokenizer, stroke folding,
64 MiB hash size, automerge 16 and contentless-delete declarations. Partition
writers send bounded batches while their selected models are still present;
the reference first reads its selected stored bodies. Search commit and sync
precede publication. This replaces the former per-block, per-partition FTS
tables; it does not copy their native trees into a production archive.

The hosted check runs the existing native version/citation and interrupted
publication controls, then compares complete vocabulary and exact native BM25
scores with the signed production runtime. Its small fixture repeats text
across partitions and channels, tests stroke folding and block-local phrases,
and retains an older observation. Production search membership can survive
in retained historical versions. A selected-body-only population is therefore
a hypothesis to test, not an equivalent interpretation assumed by the check.
Mismatches are retained and fail the screen. No timed comparison runs here.

The parser/model pin and native SQLite dependency versions are unchanged from
the retained experiment. The production oracle version is checked explicitly.
This does not qualify arbitrary provider inputs or a production format change.

## Ordinary reads are not implemented by this schema

The stored frames return exact typed messages and original citation ranges,
including explicitly addressed old versions. That is insufficient for the
product's ordinary page contract. Completing it needs these existing owners:

- selected time/source order, field-group routes and active-path/branch facts;
- session items, block/layout facts, tool results, events and query projections;
- prepared selection and sealed item/fragment continuation, including stale
  generation refusal;
- durable parse/checkpoint references for ordinary reopen and stored-input
  reconstruction, alongside original and historical-version custody.

Those projections contain queryable facts and cannot be replaced by a
message-count cursor, a source-position sort or a shadow complete index. Their
construction, indexes, reads and durable writes would belong in any later
end-to-end comparison. There is no production schema/version change, reader
adapter or deployment in this diagnostic branch.
