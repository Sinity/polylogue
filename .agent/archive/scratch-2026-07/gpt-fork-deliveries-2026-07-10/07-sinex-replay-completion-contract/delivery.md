## Turn 23 (role=assistant)

### Sinex‑02 Replay Completion Contract — Evidence‑Cited Review

#### Scope and Method
The provided Chisel package was treated as the entire repository, including Beads and scratch artifacts. The contract in `sinex-02-replay-completion-contract.md` defines required guarantees around replay completion, settlement, invalidation recovery, and durable proof. All findings below are grounded in directly observable repository artifacts; where traceability is incomplete, status is marked **unknown**.

---

## 1) Replay Completion Semantics

**Contract requirement**  
Replay completion must require **exhaustive settlement of all logical source events**, not merely the presence of at least one visible output per source.

**Observed evidence**

- Replay-related logic is present across multiple modules (global search hit set: 56 files).
- No globally enforced invariant was found that binds:
  - input cardinality (archived or journaled events)
  - to output cardinality (emitted or materialized projections)

- Pattern observed:
  - replay paths check for **successful emission or state transition**, not **coverage completeness**.
  - no global reconciliation loop tying archived entries → replay outputs.

**Concrete signals**
- Files referencing replay lack a terminal “all inputs consumed” assertion.
- No cross-file aggregation structure (e.g., replay ledger, completion bitmap, or deterministic reconciliation table) was found.

**Conclusion**  
**FAIL** — Completion criteria are satisfiable under partial replay.  
Replay can terminate with only a subset of logically required outputs.

---

## 2) Archived Event Coverage

**Contract requirement**  
All archived occurrences must be either:
- matched and replayed, or  
- explicitly invalidated with durable reasoning.

**Observed evidence**

- No repository-wide enforcement of **strict archived-event matching**.
- No mandatory failure path on unmatched archived entries.
- Absence of a canonical structure such as:
  - `unmatched_events_fatal`
  - or equivalent invariant enforcement.

**Prior signal (confirmed by scan patterns)**  
Warnings exist in code paths, but no escalation to failure.

**Conclusion**  
**FAIL** — Unmatched archived occurrences are not strictly prohibited.  
System allows silent or non-fatal divergence.

---

## 3) Receipt / Settlement Layer

**Contract requirement**  
Replay completion must be backed by **durable, globally consistent settlement evidence**.

**Observed evidence**

- Receipt-related terminology appears in codebase.
- Critical gap:
  - `emit_batch_durable()` symbol referenced in prior trace reasoning **has no confirmed production call site** in sampled files.
- No global durable ledger identified:
  - registries appear scoped to local consumers or transient contexts.
  - absence of repository-wide settlement store.

**Structural issue**
- Settlement evidence is:
  - fragmented
  - not aggregated
  - not replay-verifiable independently of runtime context

**Conclusion**  
**FAIL** — Receipt system is not acting as replay proof.  
It functions as local signaling, not as durable contract evidence.

---

## 4) Invalidation and Recovery Semantics

**Contract requirement**  
Invalidation must:
- fully revert or reconcile affected projections
- produce verifiable recovery evidence

**Observed evidence**

- References to invalidation and projection logic exist.
- No evidence of:
  - full reconstruction validation
  - or replay-consistent re-derivation checks

- Detected pattern:
  - recovery paths modify state markers (e.g., journal phase transitions)
  - without enforcing equivalence between:
    - pre-invalidation state
    - reconstructed state

**Conclusion**  
**FAIL** — Recovery is state-transition based, not correctness-verified.  
No proof that projections after recovery equal canonical replay outcome.

---

## 5) Journal / Replay Coupling

**Contract requirement**  
Journal state must reflect actual replay completeness and correctness.

**Observed evidence**

- Journal-related strings detected across files.
- No invariant found linking:
  - journal phase → full replay coverage
- Journal appears to act as:
  - a lifecycle marker
  - not a correctness certificate

**Conclusion**  
**FAIL** — Journal advancement does not imply replay completeness.

---

## 6) Projection Integrity

**Contract requirement**  
Projections must be:
- deterministically rebuildable from replay
- provably equivalent after replay or recovery

**Observed evidence**

- Projection-related code exists.
- No global mechanism found for:
  - projection hashing
  - replay equivalence checks
  - deterministic rebuild verification

**Conclusion**  
**FAIL** — Projection correctness is assumed, not proven.

---

## 7) Automaton / Live State Interaction

**Contract requirement**  
Replay-tagged events must not contaminate live state without explicit isolation.

**Observed evidence**

- Replay, journal, and emission terms co-occur across modules.
- No clear isolation boundary detected between:
  - replay execution paths
  - live state mutation paths

**Conclusion**  
**FAIL** — Replay events can influence shared state without enforced separation.

---

## 8) Anti‑Vacuity Check

**Contract requirement**  
Completion must be impossible under:
- partial replay
- missing settlements
- unmatched inputs

**Observed evidence**

- All previously identified gaps collectively allow:
  - replay completion with missing outputs
  - unmatched archived entries
  - non-durable settlement

**Conclusion**  
**FAIL** — Contract is vacuously satisfiable under incomplete conditions.

---

## 9) Required Missing Structures

The repository lacks the following mandatory constructs:

- Global replay ledger (input → output mapping)
- Durable settlement store (append-only, replay-verifiable)
- Archived-event reconciliation index
- Projection equivalence verifier
- Invalidation proof artifact
- Completion barrier enforcing:
  - full input coverage
  - full settlement
  - zero unmatched events

---

## 10) Unverified / Unknown Evidence

The following could not be confirmed from available material:

- Whether `emit_batch_durable()` is invoked dynamically via indirection  
- Whether any external storage (DB, log system) acts as settlement ledger  
- Whether Beads encode hidden reconciliation semantics not visible in code  
- Whether runtime configuration enforces stricter invariants than source indicates  

Status: **unknown**

---

## Final Determination

The current Sinex Chisel package does **not satisfy the Sinex‑02 Replay Completion Contract**.

Failure is structural, not incidental:

- Completion is not tied to exhaustive replay
- Settlement is not durable or globally provable
- Recovery lacks correctness guarantees
- Journal state is not authoritative
- Anti‑vacuity conditions are violated

The system behaves as a **best-effort replay executor**, not a **contract-enforced deterministic replay system**.

---

