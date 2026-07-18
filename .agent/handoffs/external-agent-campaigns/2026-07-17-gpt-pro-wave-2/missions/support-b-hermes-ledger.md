# Lane B support — Hermes verification-ledger ingestion and fidelity harness

Work from the attached fresh Chisel archive. Produce an integration-ready
implementation package for the Hermes verification-evidence surface. A paired
local lane owns real `~/.hermes` inspection and can validate private producer
bytes; you must not invent claims that such bytes were supplied.

## Mission

Implement the vertical slice that makes structured Hermes verification evidence
first-class archive data: artifact identification/acquisition boundary,
normalization of command/canonical command/kind/scope/status/exit code/output
summary/changed paths, deterministic session correlation, and typed degraded
states for missing or unmatched identity. Reuse the repository's raw → parser →
normalized model route and structural tool-outcome conventions (`NULL` means
unknown), rather than adding a bespoke SQLite reader or prose-derived error
classifier.

Provide a synthetic producer-shaped fixture corpus plus end-to-end tests for
idempotent replay, partial/malformed rows, an unmatched session, status/exit
code semantics, and a query/read route exposing the evidence. Include a
fidelity declaration that names what must be validated by the local real-byte
lane before this can be called exact.

## Boundary

Do not modify a real Hermes installation, pretend the fixture is real evidence,
or expand into the complete state.db/ATOF watcher program. Design the code so
the later real-byte fixture can replace the synthetic fixture without a parser
rewrite.

## Required package

Return `HANDOFF.md`, `PATCH.diff`, `TESTS.md`, and `EVIDENCE.md`. Explain the
source anchors and why each test fails under the relevant production mutation.
No fictional execution results.
