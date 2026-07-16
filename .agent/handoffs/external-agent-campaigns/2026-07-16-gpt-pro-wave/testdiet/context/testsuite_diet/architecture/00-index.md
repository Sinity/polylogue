---
created: 2026-07-16
purpose: Index Sol-adjudicated architecture decisions needed by the Testsuite Diet laws
status: recommended-decision-set
project: polylogue
---

# Testsuite Diet architecture decisions

## Decision status

These documents replace “architecture must decide” placeholders in the Diet
with a recommended product contract. They are implementation inputs, not a new
verification framework and not claims that the production mechanisms have
landed.

The recommendations are the default. An implementation packet should not ask
the operator to choose among the rejected alternatives again unless new source
evidence invalidates an invariant recorded here. The only remaining operator
actions are live authority grants or explicit trust replacement; they are not
open-ended architecture questions.

## Decision map

| Domain | Laws | Recommended contract | Operator decision |
| --- | --- | --- | --- |
| [Evidence authority and identity](01-evidence-authority-and-identity.md) | L01-L03 | One typed raw-authority reconciler; byte proof outranks metadata; conflict blocks; archive and acquisition identities stay distinct | Authorize a specific live repair plan only after its immutable preflight receipt |
| [Lineage composition](02-lineage-composition-and-snapshots.md) | L05-L06 | Store canonical divergent tails plus typed edges; compose under one read snapshot; unresolved lineage degrades explicitly without inventing content | None |
| [Concurrent writes and publication](03-concurrent-writes-publication-and-resume.md) | L07-L10 | Atomic SQL/CAS for single-row state, immediate transactions for multi-row invariants, durable reservations and saga receipts across files | None |
| [Destructive and authentication boundaries](04-destructive-and-authentication-boundaries.md) | L23-L24 | One executable operation gateway; preview-bound confirmation for irreversible work; stable authenticated receiver pairing | Explicit re-pair only when trusted receiver identity changes |
| [Derived freshness](05-derived-freshness.md) | L11-L13 | Currentness is exact source identity plus recipe identity plus generation; one protocol, domain-specific ledgers | None |
| [Query cancellation and bounds](06-query-cancellation-and-bounds.md) | L14-L17 | One query execution context, weighted admission, dedicated interruptible SQLite readers, lossless page/spool results | Numeric budgets are evidence-tuned defaults, not semantic choices |
| [Evidence and public algebra](07-evidence-provenance-and-public-algebra.md) | L18-L20, L25 | `EvidenceValue` preserves independent epistemic axes; domain facts remain canonical; surfaces only project | None |
| [Configuration and paths](08-configuration-and-path-coherence.md) | L26-L28 | Load the existing five layers once; inject one resolved archive/path identity; retire ambient parallel resolution | None |
| [Capture and deployed status](09-capture-delivery-and-deployed-status.md) | L29-L30 | Production service graph, durable delivery state, per-component snapshots, runtime/build/archive identity and evidence-bound termination receipts | Permission for optional local host evidence; unknown remains valid |

## Shared architectural vocabulary

The designs deliberately reuse a small set of shapes without forcing unrelated
domains into one table or lifecycle:

- **identity key** — immutable fields that say what logical thing or recipe a
  record represents;
- **generation** — a monotonic attempt/publication discriminator, never a
  substitute for identity;
- **receipt** — an immutable account of a planned or completed effect, including
  inputs, authority, result, resources, and postflight;
- **typed blocked state** — evidence is insufficient or contradictory; no
  arbitrary winner and no fabricated default;
- **completeness/freshness projection** — public observation of what is known,
  missing, stale, degraded, or awaiting work;
- **domain owner** — the subsystem that computes and persists facts. Shared
  protocols do not create a universal evidence, lifecycle, or scheduler table.

## Cross-decision rules

1. Durable truth is never inferred from a rebuildable projection.
2. Timestamps, filenames, counts, and row presence may diagnose; they do not
   establish content authority or currentness where exact identity is available.
3. Multi-file SQLite/filesystem work is a receipt-backed saga, not fictitious
   cross-database ACID.
4. Every retry is idempotent under the same identity and generation.
5. A public surface cannot strengthen authority, completeness, freshness, or
   certainty relative to its weakest required evidence.
6. Expensive logical results may be paged or spooled, but never silently
   truncated by a transport convenience cap.
7. Security and destructive policy are executable at the operation boundary;
   declaration-only scans and per-surface booleans are supporting checks only.
8. Age is an observability and reconciliation signal. It is not, by itself,
   proof that a writer, lease, publication, or capture is dead.

## Implementation posture

Each design distinguishes current mechanisms from missing seams. Implement the
smallest production contract that has at least two real consumers; do not build
a generic framework in anticipation. Storage, security, and query-execution
changes use isolated or serialized hotspot branches. Survivor laws follow the
production seam; independent mutation certification follows in a disposable
worktree.
