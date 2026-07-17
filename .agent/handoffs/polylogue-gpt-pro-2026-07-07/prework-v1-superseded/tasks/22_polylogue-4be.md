# 22. polylogue-4be — Create a real restore drill for backup proof

Priority: **P2**  
Lane: **backup-integrity**  
Readiness: **ready-now / devtools+ops artifact**

## Why this is urgent / critical-path

A backup that has not been restored is not evidence. Polylogue’s archive is irreplaceable enough that restore proof is critical path.

## Static diagnosis / likely mechanism

The bead states there are multiple backup layers but no restore test. The restore drill should prove latest backup set can produce an archive that passes integrity checks and basic user queries within expected lag.

## Implementation plan

Implementation shape:
1. Add `polylogue ops restore-drill` or `devtools restore-drill`.
2. Locate latest configured backup set; restore to a scratch root, never over live archive.
3. Run `PRAGMA integrity_check` for each SQLite tier.
4. Run a 10-query battery: session count, message count, blob-reference debt, one `find`, one `read`, one insight/profile read, one attachment lookup, one usage/cost summary, one tag/user-state read, one health/status command.
5. Compare counts to live archive with an expected-lag tolerance and record differences.
6. Emit JSON + Markdown ops artifact with timing, backup source, restored root, counts, query results, and failure reasons.
7. Add a corruption fixture that fails loudly.
8. Wire Sinnix quarterly timer or emit exact downstream patch instructions.

## Test plan

Tests:
- synthetic backup restored to scratch and query battery passes.
- corrupted DB/file fails with clear failure.
- drill refuses to write into live archive path.
- count-lag tolerance behaves as declared.

## Verification command / proof

`devtools test tests/unit/operations/test_restore_drill*.py -k 'restore or backup or corruption'` plus one real restore drill artifact.

## Pitfalls

Do not close on “backup command ran.” The acceptance condition is a restored scratch archive that answers queries.

## Files/functions to inspect or touch

- `polylogue/operations/backup*`
- `polylogue/daemon/backup*`
- `devtools/*`
- `archive tier path helpers`
- `Sinnix timer module`
