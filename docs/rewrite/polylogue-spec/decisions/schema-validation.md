# Decision: validating provider input

Status: **decided.** Decided by the spec authors from the measurements below and the operator's rulings over time; the operator may overrule.

## History

- Early 2026: schemas inferred from the real archive, strict validation
  always on, so parsers can stay simple ("validate once, inside parsing",
  2026-03-30).
- 2026-10-11: the Python implementation's validation turned out to be
  advisory (it blocked nothing) yet used 53% of ingest CPU, because it loaded
  each input into a SQLite tree before checking it. The operator asked
  whether validation should run at all, given that parsing already rejects
  malformed input.

## Decision

**The typed parse is the validation.** Each provider's record types are Rust
structs (`serde` deserialization). A record either fits its type or is
reported: as a parse error if required fields are missing or wrong, or as an
unrecognized field or record type if it has things the struct does not know.
That is strict, happens once, costs nothing extra, and catches format drift.

The inferred JSON schemas stay, for three offline jobs (`spec/analysis.md`):

1. checking that the Rust record types cover what the real data contains
   (fields seen in data but not read by any parser are listed);
2. generating synthetic test inputs with the real distribution
   (`plan/testing.md`);
3. a drift report when a provider's format changes.

Schema checking never runs inside ingest.

## Measured (2026-10-11)

All 19.9 GB of the operator's Claude Code data (5.83M records) was parsed
with typed wire structs (`evidence/parse-throughput.md`): about 600 MB/s per
core, 159 MB peak memory, 0 records failing the typed parse. Zero failures
shows the typed parse alone validates little, because a new field cannot make
it fail. Counting ignored fields with `serde_ignored` found 185 unknown paths;
that count, reported in `status` and as drift, is what makes the parse a
validation. Both are required.
