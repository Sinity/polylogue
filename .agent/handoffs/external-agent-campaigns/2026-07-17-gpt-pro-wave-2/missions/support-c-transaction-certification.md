# Lane C support — transaction continuation certification across MCP/API/HTTP

Work from the attached fresh Chisel archive. Produce a cohesive implementation
package that certifies the canonical bounded, resumable read transaction across
every public read surface. This supports a local lane that owns live archive
incident-scale proof and deployment/client configuration.

## Mission

Find the canonical query transaction, continuation cursor/result references,
budget/deadline enforcement, and the MCP/API/HTTP adapters. Implement a
cross-surface certification harness using a deliberately overflowing synthetic
corpus. It must prove all of the following through production routes:

1. each surface returns a bounded first page plus a lossless continuation;
2. continuation exhausts the same stable transaction without duplicate or
   missing members;
3. an expired/invalid cursor fails typed and never silently restarts at page
   one;
4. cancellation/deadline/budget outcomes retain enough receipt identity to
   resume or explain safely; and
5. examples emitted by MCP discovery parse under the real query grammar.

If adapters currently diverge, repair the smallest shared substrate or adapter
contract rather than adding compatibility aliases or a test-only façade. The
test must force an actual envelope overflow; normal tiny fixtures are not
evidence of continuation behavior.

## Boundary

Do not reintroduce retired MCP names, add aliases, redesign role algebra, or
claim real 4.85M-block performance. The local lane will conduct that live proof
and reconcile client rollout.

## Required package

Return `HANDOFF.md`, `PATCH.diff`, `TESTS.md`, and `EVIDENCE.md`, with a surface
matrix and exact production dependencies exercised. Do not claim commands ran.
