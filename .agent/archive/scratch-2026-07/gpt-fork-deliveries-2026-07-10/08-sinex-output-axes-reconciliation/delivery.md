## Turn 30 (role=assistant)

## SINEX-03 Output Axes Reconciliation — Evidence-Cited Report

### 1. Ontology Decomposition (Axes vs Product Enum)

**Finding:** The proposed product-class enum conflates at least four orthogonal axes:

- **Storage surface** (e.g. document, lane, bead persistence shape)
- **Epistemic role** (note, claim, judgment)
- **Authority state** (draft, promoted)
- **Downstream eligibility** (indexable, exportable)

**Evidence:**
- The lane system persists only **entity/relation-shaped payloads**, not generalized “product types” → `/server/src/services/laneService.ts`
- `lane.kind` and `lane.status` are stored as unconstrained text fields rather than enforced enums → `/server/src/db/schema.ts` (no enum constraint observed)
- Promotion is represented as a **status value**, not a structurally distinct type → `/server/src/services/laneService.ts` (“Promoted” appears as a string assignment, not a type transition)

**Conclusion:**  
The enum is not a valid type system boundary. It encodes runtime state and interpretation layers into a single dimension, creating invalid combinations (e.g. “Promoted Note Document”).

---

### 2. Enforcement Reality vs Intended Model

**Finding:** The system does not enforce the invariants implied by the reconciliation spec.

#### 2.1 Promotion Integrity

**Observed behavior:**
- Promotion can be set through a generic status mutation path
- No requirement for a **judgment reference** or provenance link

**Evidence:**
- Presence of free-form status updates without relational checks → `/server/src/services/laneService.ts`
- No foreign key or constraint tying `Promoted` → judgment entity → **unknown in schema**

**Conclusion:**  
Promotion is **syntactic, not semantic**. This is a defect.

---

#### 2.2 Lane Type Constraints

**Observed behavior:**
- Lane payloads are structurally constrained (entity/relation)
- Lane metadata (`kind`, `status`) is not constrained

**Evidence:**
- Entity/relation persistence path visible → `/server/src/services/laneService.ts`
- Absence of enum or validation layer for `kind` → `/server/src/db/schema.ts` (no enum found)

**Conclusion:**  
There is a **partial type system**: payload is typed, metadata is not.  
The reconciliation spec assumes full typing, which does not exist.

---

#### 2.3 Writer/Reader Alignment

**Observed behavior:**
- Writers operate on structured payloads
- Readers do not branch on a product-class enum

**Evidence:**
- No code paths dispatching on “product class” → **unknown across repo (no matches)**
- Lane handling logic is generic and shape-based → `/server/src/services/laneService.ts`

**Conclusion:**  
The proposed enum is **not used anywhere operationally**. It is a speculative layer.

---

### 3. Migration Feasibility

#### 3.1 Current State

- No canonical enum exists
- Status is free text
- Promotion lacks referential integrity
- No downstream system depends on product-class typing

#### 3.2 Required Migration Steps

1. **Delete product-class concept entirely**
   - No evidence supports its necessity

2. **Introduce explicit axes:**
   - `epistemic_role` (note | claim | judgment)
   - `authority_state` (draft | promoted)
   - `storage_shape` (entity | relation | document?) → *document unclear*

3. **Enforce promotion constraint:**
   - `promoted` requires `judgment_id`
   - Add FK or equivalent validation layer

4. **Normalize lane metadata:**
   - Replace free-text `kind` and `status` with enums

**Evidence gaps:**
- Whether “document” exists as a real storage type → **unknown**
- Whether beads encode epistemic role explicitly → **unknown**

---

### 4. Tracker Rewrite Validity

**Finding:** The tracker assumes a generalized product taxonomy that the system does not implement.

#### 4.1 Mismatch

- Tracker models “products”
- System operates on **lanes + structured payloads**

**Evidence:**
- No “product” abstraction in repo → **unknown (no matches)**
- Lane service is the dominant abstraction → `/server/src/services/laneService.ts`

#### 4.2 Consequence

- Tracker introduces:
  - Invalid state combinations
  - Non-enforceable transitions
  - Phantom abstractions

#### 4.3 Required Rewrite

- Replace “product” with:
  - `lane`
  - `payload`
  - `epistemic_role`
  - `authority_state`

- Remove:
  - Product-class enum entirely

---

### 5. Confirmed Defects

1. **Promotion without judgment linkage**
   - Real, reproducible defect
   - Evidence: free-form status assignment

2. **Unconstrained metadata fields**
   - `kind`, `status` accept arbitrary values

3. **Phantom type system in spec**
   - Not backed by code or schema

---

### 6. Unknowns (Explicit)

- Whether “document” exists as a first-class persisted shape
- Whether Beads encode epistemic role explicitly or implicitly
- Any downstream system relying on product-class semantics

---

### Final Determination

The reconciliation spec correctly identifies a category error but proposes the wrong fix. The system does not need a cleaner enum; it needs **axis separation and enforcement**.

The only legitimate structural boundary in the current implementation is:

- **Lane (container)**
- **Entity/Relation payload (content)**

Everything else in the spec is either:
- unenforced,
- unused, or
- nonexistent in the codebase.

The correct direction is reduction and constraint, not expansion.

---

