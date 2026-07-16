# Verification lanes

Use these as packet exit gates. Focused tests are always required before broad verify.

## Safety/security

- Host/Origin/CSRF/token negative tests
- Receiver token and spool quota tests
- Agent-write candidate/non-injected tests
- Secret/excision tests

## Data integrity

- Blob lease/ref/generation race tests
- Missing blob classifier totals
- Restore drill with SHA-256 verification
- Attachment acquisition state fixtures

## Temporal/evidence honesty

- Frozen clock query tests
- Weakest temporal source aggregate tests
- Empty/uncovered numeric-field tests
- Text-derived provenance rendering tests

## Read/query contracts

- CLI/daemon/MCP/Python parity
- Projection/render snapshots
- FTS unavailable vs structural-only routing
- Citation resolver drift tests

## Lineage/context

- Composed-session transaction tests
- Transcript completeness/truncation tests
- Scheduler context-ledger determinism
- Compaction loss/reground fixtures

## Scale/live

- Blue-green generation swap/crash tests
- Bulk ingest resource envelope
- Daemon heartbeat/crash forensics
- SSE/push/cache invalidation tests

## Analytics/public

- Measure registry validation
- Claims ledger coverage
- Demo anti-demo not_supported path
- Cold-reader/public proof artifacts

