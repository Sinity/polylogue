## Turn 17 (role=assistant)

# Semantic fingerprint and comparison protocol for changed-only convergence

This report executes the attached review brief against the supplied Sinex Chisel package. fileciteturn0file0

## Decision

Changed-only convergence must not be implemented as one content hash. It requires a versioned comparison bundle with four distinct functions:

1. A **logical occurrence key** that pairs an existing interpretation with its shadow candidate.
2. A **semantic envelope** that determines whether the paired interpretations mean the same thing.
3. An **evaluator manifest** that records the current semantics, policy, configuration, and code under which the shadow candidate was produced.
4. **Nondeterminism receipts** that make order-, clock-, model-, privacy-, and external-effect-dependent evaluation reproducible.

Only the semantic envelope participates directly in equality. The evaluator manifest and receipts determine whether the equality result is admissible as proof.

This separation is required because full replay deliberately gives every interpretation a new random UUIDv7, while material coordinates represent stable occurrence identity (`crate/sinex-primitives/src/events/occurrence.rs:11-38`, symbol `MaterialOccurrenceKey`; `crate/sinexd/src/runtime/automaton/adapter/output.rs:311-338`, symbol `AutomatonRuntime::build_output_event`). It also preserves the settled distinction between always-reminting replay and retain-green changed-only convergence (`sinex-scratchpad.xml:35556-35582`, section `The load-bearing event model`; `sinex-scratchpad.xml:35789-35803`, section `Required semantic distinction`).

The current package is not ready for registry-wide green retention. The canonicalizer can be the first pilot after semantic payload projection is available. Most windowed, reconciliation, clock-reading, and effect-producing automata must remain fail-closed until stable occurrence identities and reproducibility receipts exist.

The normative decision predicate is:

```text
green(old, candidate) =
    unique_logical_match(old, candidate)
    AND exact_semantic_envelope_bytes(old) == exact_semantic_envelope_bytes(candidate)
    AND current_schema_valid(old)
    AND current_schema_valid(candidate)
    AND operational_keys_retention_compatible(old, candidate)
    AND nondeterminism_contract_satisfied(candidate)
    AND every_parent_is_green_or_certified_retained_boundary(candidate)
```

Digest equality is only an index and comparison prefilter. A node is never colored green solely because two hashes match.

A third internal result, `unknown`, is necessary. Missing projectors, ambiguous occurrence identity, missing effects, or unverifiable outside-scope parents are not evidence of red semantic difference. They are absence of proof. An `unknown` scope must stop or fall back to full replay; it must never be silently promoted to green.

---

## 1. Fingerprint composition

### 1.1 Versioned comparison bundle

Use BLAKE3-256, already used for source-byte and blob integrity in the repository, with explicit domain separation and length-prefixed fields:

```text
logical_occurrence_digest =
    BLAKE3("sinex/fingerprint/v1/logical-occurrence" || framed(logical_occurrence_bytes))

canonical_payload_digest =
    BLAKE3("sinex/fingerprint/v1/canonical-payload" || framed(canonical_payload_bytes))

parent_commitment_digest =
    BLAKE3("sinex/fingerprint/v1/parent-commitment" || framed(parent_commitment_bytes))

semantic_envelope_digest =
    BLAKE3("sinex/fingerprint/v1/semantic-envelope" || framed(semantic_envelope_bytes))

evaluator_manifest_digest =
    BLAKE3("sinex/fingerprint/v1/evaluator-manifest" || framed(evaluator_manifest_bytes))

effect_receipt_digest =
    BLAKE3("sinex/fingerprint/v1/effect-receipts" || framed(effect_receipt_bytes))
```

Every field must be tagged and length-prefixed. Concatenating unframed strings is forbidden. `fingerprint_version` identifies the entire canonicalization, projection, logical-identity, and parent-commitment protocol, not merely the hash algorithm.

The existing `SourceRecordFingerprint` is not this protocol. It fingerprints structural source-record shape—format, keys, and type map—not an interpreted event’s semantic value (`crate/sinex-primitives/src/parser/fingerprint.rs:1-16`, symbol `SourceRecordFingerprint`).

### 1.2 Material-provenance events

`Event<T>` currently contains source, event type, payload, timestamp and quality, host, schema ID, provenance, source-byte hash, blobs, and synthetic metadata (`crate/sinex-primitives/src/events/mod.rs:36-122`, symbol `Event<T>`).

The exact material comparison record is:

| Component | Exact v1 fields |
|---|---|
| Event-kind discriminator | Literal `material` |
| Logical occurrence | Resolved EventContract identity; `source`; `event_type`; `source_material_id`; start-anchor kind; `anchor_byte`; contract-declared parser `OccurrenceKey` fields when required; contract-declared material output slot |
| Semantic envelope | The complete logical-occurrence bytes; `source`; `event_type`; canonical projected payload; exact `ts_orig`; exact `ts_quality`; `offset_start`; `offset_end`; `offset_kind`; associated blob content identities |
| Validity gates | Material provenance must be present; derived parents must be absent; current payload schema must validate; the logical occurrence must be unique |
| Evaluator manifest | Current parser ID and version; parser code/build identity; effective parser configuration and checkpoint state; source/runtime/event/admission contracts; timestamp resolver; schema/projector; privacy snapshot; effect receipts where applicable |

The start anchor, not the full interpreted content, establishes occurrence. `MaterialAnchor` is explicitly required to remain stable across re-reads (`crate/sinex-primitives/src/parser/mod.rs:354-394`, symbol `MaterialAnchor`). The persisted `MaterialOccurrenceKey` is `(source_material_id, anchor_byte)` and is explicitly distinct from random interpretation identity (`crate/sinex-primitives/src/events/occurrence.rs:11-38`, symbol `MaterialOccurrenceKey`).

The physical key is not always sufficient. Directory, database-row, stream, and other source contracts may need their declared `OccurrenceKey` fields. If two material events remain indistinguishable under the contract’s occurrence fields, convergence must return `unknown` under regime 3 rather than inventing an ordinal or using payload content as identity. This follows the three-regime doctrine in `sinex-908` (`sinex-beads.xml:178-183`, issue `sinex-908`).

`offset_end` is semantic content, not occurrence identity. A source row or interval whose end extends remains the same start-anchored occurrence but has changed interpretation content. The settled interval rule makes the same distinction: start evidence stays stable while the end can move (`sinex-scratchpad.xml:35806-35822`, section `Derived occurrence, revision, and idempotence`; `sinex-beads.xml:194-205`, issues `sinex-y8v` and `sinex-ecy`).

### 1.3 Derived events

The exact derived comparison record is:

| Component | Exact v1 fields |
|---|---|
| Event-kind discriminator | Literal `derived` |
| Logical occurrence | Resolved output EventContract; producer identity; output `source`; output `event_type`; `automaton_model`; declared reconciliation kind; stable scope identity; stable output-slot identity; designated start/input occurrences; output role for legitimate multi-output derivations |
| Semantic envelope | The complete logical-occurrence bytes; `source`; `event_type`; canonical projected payload; exact `ts_orig`; exact `ts_quality` presence, normally null; `temporal_policy`; `automaton_model`; transitive logical-parent commitment; associated blob content identities |
| Operational compatibility | Existing and candidate scope/equivalence keys must identify the same declared slot, and retaining the old raw keys must remain compatible with current replacement behavior |
| Validity gates | Non-empty parent set; current payload schema validation; unique occurrence/slot match; class-specific nondeterminism proof |
| Evaluator manifest | Producer code/build identity; current `semantics_version`; effective automaton configuration; contracts; schema/projector; privacy snapshot; operation logical clock; input-order receipt; model/external-effect receipts |

`DerivedOutput` contains payload, selected timestamp, parent interpretation IDs, temporal policy, semantics version, scope/equivalence keys, aggregation metadata, and optional event-type override (`crate/sinexd/src/runtime/automaton/output.rs:30-81`, symbol `DerivedOutput`). The adapter then mints a fresh random UUIDv7 and persists the automaton model (`crate/sinexd/src/runtime/automaton/adapter/output.rs:258-339`, symbol `AutomatonRuntime::build_output_event`).

#### Logical parent identity

Raw parent UUIDs must not participate in semantic equality. A parent UUID identifies one interpretation and changes during replay. The parent commitment instead contains:

```text
parent_commitment_v1 =
    reconciliation semantics
    + parent count
    + for each immediate edge:
        parent role, where declared
        parent logical-occurrence digest
        parent transitive material-frontier commitment
```

For unordered lineage, entries are sorted by `(parent_role, logical_occurrence_digest)` while retaining multiplicity. For genuinely ordered input semantics, the sequence lives in a separate input-order receipt. Current provenance cannot prove ordered or repeated parent input because `canonicalize_source_event_ids` sorts and deduplicates the UUID list (`crate/sinex-primitives/src/events/builder.rs:625-647`, symbol `canonicalize_source_event_ids`).

This is the correct reading of “transitive occurrence.” Derived rows do not have material provenance columns, but their occurrence can be rooted transitively in designated material evidence (`sinex-scratchpad.xml:35817-35822`, section `Derived occurrence, revision, and idempotence`).

The digest remains a comparison coordinate only. It must never be assigned to `Event.id` or used to derive a UUID. Event identity remains random UUIDv7 interpretation identity.

### 1.4 Interpretation UUIDs inside payloads

Canonical payload equality cannot be raw JSON equality. Current derived payloads serialize interpretation IDs:

- `TerminalCommandCanonicalizer::process` writes the trigger UUID into `CanonicalCommandPayload.source_events` (`crate/sinexd/src/automata/canonicalizer.rs:58-80`).
- `TagApplier::process` writes its triggering event UUID as `entity_id` (`crate/sinexd/src/automata/tag_applier.rs:59-84`).
- `evaluate_pending_workspace_instruction` writes observation UUIDs into `matched_event_ids` (`crate/sinexd/src/automata/instruction_reconciler.rs:192-218`).
- Terminal document parsing derives `natural_key` and `document_id` from a parent interpretation UUID (`crate/sinexd/src/automata/document_parser.rs:241-275`, symbols `process_terminal_command` and its document-ID construction).
- Interval identifiers are built from parent event IDs (`crate/sinexd/src/automata/interval_lift.rs:371-440`, symbols `close_with` and same-event interval construction).

These fields must be transformed by a schema-owned semantic projector. The existing JSON Schema registry is the appropriate declaration surface because it already stores versioned schema content and content hashes (`crate/sinex-schema/src/defs/sinex_schemas.rs:22-75`, symbols `EventPayloadSchemas` and `EventPayloadSchemaRecord`).

The v1 schema extensions should include:

```json
{
  "x-sinex-semantic-role": "parent_interpretation_reference",
  "x-sinex-parent-role": "observation",
  "x-sinex-collection-semantics": "sequence | set | multiset",
  "x-sinex-derived-identity-projector": "projector-name/version"
}
```

A parent interpretation reference is resolved to its logical occurrence before payload canonicalization. Derived identifiers computed from a parent UUID must be recomputed in the comparison projection from the parent logical occurrence.

This cannot be implemented through field-name stripping. `entity_id`, for example, is sometimes a stable deterministic domain identity: the entity resolver deliberately produces UUIDv5 entity IDs and distinguishes them from UUIDv7 provenance IDs (`crate/sinexd/src/automata/entity_resolver.rs:25-47`, symbol `ResolverState`). Projectors therefore belong to the specific source/event schema, not a global list of suspicious names.

An unresolved reference or unknown projector produces `unknown`, not green.

### 1.5 Candidate inputs adjudicated

| Candidate input | Decision |
|---|---|
| Canonical payload | Included in semantic equality after current schema projection and the database boundary |
| Event source and type | Included in logical matching and semantic equality |
| `ts_orig` | Included exactly, through nanosecond resolution |
| `ts_quality` | Included exactly, including null versus a named quality tier |
| `semantics_version` | Recorded in the evaluator manifest and operation record; deliberately excluded from direct semantic equality |
| `automaton_model` | Included in logical identity and semantic equality |
| Parser identity and version | Evaluator-manifest evidence, not event-content equality |
| Effective policy/config versions | Evaluator-manifest evidence; output-changing effects are caught by the envelope; nondeterministic effects additionally require receipts |
| Logical parents | Included as transitive occurrence commitments; interpretation UUIDs excluded |
| `scope_key` and `equivalence_key` | Used only when the producer contract declares them stable occurrence/scope coordinates; retention also requires operational replacement compatibility |
| Payload schema | Current schema validity and projector identity are gates; the surrogate `payload_schema_id` is not semantic equality |
| Associated blobs | Compared by content identity and declared sequence/set semantics, not blob UUID |
| `anchor_payload_hash` | Excluded from semantic equality; retained only as an independent source-integrity check |

A `semantics_version` change alone must not force red. The changed-only operation exists to execute the current semantics and retain an old interpretation when its current result is unchanged. Directly hashing the stored version would reduce convergence to full reminting after every semantics bump. The current source describes `semantics_version` as the version of module logic for replay (`crate/sinex-primitives/src/events/mod.rs:103-105`, field `Event::semantics_version`), and the operation log is the proper place to record that an old interpretation was revalidated under a newer evaluator.

`automaton_model` is different. It declares whether an output is a transduction, window result, or reconciliation product and therefore changes replacement and lineage semantics. A model change is semantically material even when one JSON payload happens to match.

### 1.6 Effective evaluator manifest

The evaluator manifest is canonical JSON containing the exact behavior-affecting values actually used by the shadow evaluation:

- Parser ID, parser version, code artifact/build digest, manifest, required input keys, parser-local checkpoint state, baseline adapter configuration, and source-binding configuration. `MaterialParser` processes records in order and exposes both configuration and replay-relevant checkpoint state (`crate/sinex-primitives/src/parser/mod.rs:700-798`, symbol `MaterialParser`). `ParserManifest` owns parser identity, version, input shapes, output sources/types, and privacy/sensitivity declarations (`crate/sinex-primitives/src/parser/mod.rs:982-1017`, symbol `ParserManifest`).
- Source contract fields: ID, namespace, event pairs, privacy tier, horizons, retention, occurrence identity, and access scope (`crate/sinex-primitives/src/source_contracts/contract.rs:7-26`, symbol `SourceContract`).
- Behavior-affecting runtime binding fields: implementation, adapter, output event type, privacy context, source ID, checkpoint/runtime/material lifecycle, and transport (`crate/sinex-primitives/src/source_contracts/runtime.rs:11-52`, symbol `SourceRuntimeBinding`). Resource limits belong in the manifest only when they alter admitted output.
- EventContract payload, occurrence, temporal, provenance, disclosure, admission, and output-kind contracts (`crate/sinex-primitives/src/event_contracts.rs:18-96`, symbols `PayloadContract`, `OccurrenceContract`, `TemporalContract`, `ProvenanceContract`, and `EventContract`).
- Admission policy ID and exact behaviors for schema validation, occurrence admission, disclosure, malformed material, resource pressure, and proposal routing (`crate/sinex-primitives/src/admission_policy.rs:14-86`, symbols `AdmissionPolicyId` and `AdmissionPolicy`).
- Active payload schema version, schema-content hash, projector annotations, and active/inactive state (`crate/sinex-schema/src/defs/sinex_schemas.rs:61-75`, symbol `EventPayloadSchemaRecord`).
- Exact automaton configuration. Current examples include analytics gap, maximum duration, and event-count environment values (`crate/sinexd/src/automata/analytics.rs:28-104`, functions `window_gap_threshold`, `window_max_duration`, and `window_max_events`), health configuration (`crate/sinexd/src/automata/health.rs:27-135`), and entity-enricher configuration (`crate/sinexd/src/automata/entity_enricher.rs:24-82`).
- Exact privacy snapshot: enabled rules, matcher value/config, action, label, field scopes, priority, dictionaries and terms, recognizer backend kind/endpoint/config, key namespace and non-secret key fingerprint, external-recognizer behavior, and the actual compiled snapshot generation. The policy engine caches a compiled snapshot and can continue with stale policy on refresh failure (`crate/sinexd/src/event_engine/policy.rs:431-541`, symbol `compile_rules`; `crate/sinexd/src/event_engine/policy.rs:658-765`, symbol `PolicyEngine`).
- Timestamp resolver version and configuration, plus the operation’s frozen logical clock where clock-sensitive code is involved.
- Model, external I/O, and privacy effect receipts. `ModelEffectRequest` already identifies provider, model, prompt version, schema version, and input hash, while `ModelEffectRecord` contains request, output, replay policy, and output hash (`crate/sinex-primitives/src/llm.rs:169-241`, symbols `ModelEffectRequest` and `ModelEffectRecord`).

Evaluator-manifest equality is not required for green. A config or code change may legitimately produce the same semantic envelope. The manifest records exactly what was evaluated and allows later audit.

### 1.7 Deliberately excluded fields

The following fields are excluded from semantic equality:

- `Event.id`: random interpretation identity.
- Raw `source_event_ids`: replaced by the logical-parent commitment.
- Interpretation UUIDs inside payloads: replaced only through schema-owned projectors.
- `ts_coided`: generated from the interpretation UUID and therefore necessarily changes on remint.
- `ts_persisted`: storage timing.
- `module_run_id`, `host`, and process/deployment identity. A domain-significant machine identity belongs in the payload or EventContract; replay host identity does not.
- `created_by_operation_id`, provenance operation ID, convergence operation ID, and other operation-control metadata.
- `payload_schema_id`: a surrogate UUID. Current schema content and validity are the meaningful gates.
- Blob surrogate UUIDs: use content hashes and relevant intrinsic metadata.
- Aggregation telemetry that is not part of the persisted payload.
- `semantics_version`: evaluator evidence, as discussed above.
- `anchor_payload_hash`: source-integrity evidence, not interpretation equality.

The event schema confirms that `ts_coided` is generated from the UUID and `ts_persisted` defaults to database time (`crate/sinex-schema/src/defs/events.rs:169-236`, symbol `Events::create_table_statement`). Updating either would rewrite provenance rather than record revalidation.

### 1.8 Anti-vacuity matrix

| Equality dimension | Change that must make the old node non-green | Change that must not make it red |
|---|---|---|
| Material occurrence | Change `source_material_id`, start anchor, or declared occurrence fields: old becomes absent/red and candidate is new | Change interpretation UUID while keeping occurrence coordinates |
| Derived occurrence | Change stable scope/output slot or designated start occurrence | Change only parent interpretation UUIDs while retaining the same parent occurrences |
| Source/type | `command.canonical` → another event type, or source namespace change | Process host or module-run change |
| Object payload | Change a semantic key or value | Reorder object keys or alter insignificant JSON whitespace |
| Numbers | `1` → `1.01` | `1`, `1.0`, and `1e0` after JSONB numeric normalization |
| Strings/Unicode | Change Unicode scalar values, including NFC text to distinct NFD scalars | Literal UTF-8 versus equivalent JSON escape spelling |
| Null/absence | Present `null` → absent field, or the reverse | Reordering unrelated keys |
| NUL handling | Change a character that survives the JSONB boundary | Add/remove a NUL that admission strips before both values reach JSONB |
| Timestamp | Any change to seconds, nanoseconds, offset-equivalent instant, or presence | `ts_coided` or `ts_persisted` change |
| Quality/policy | Change `ts_quality` or derived `temporal_policy` | Operation timestamp change |
| Provenance span | Same start occurrence but changed `offset_end` or `offset_kind` is a changed interpretation | Re-minted parent UUID with the same parent logical occurrence |
| Automaton contract | Change `automaton_model` | Bump `semantics_version` while producing the same envelope |
| Scope/equivalence | Change the declared logical output slot | Change a display ordinal that is not used as identity after the occurrence fixes |
| Policy/config | Redaction, timestamp, threshold, or contract change that changes the candidate envelope | Policy/config version change whose effective candidate is byte-identical and deterministic |
| Schema | Current schema rejects the retained payload, or the projector now gives different semantic bytes | New schema UUID/version with equivalent constraints and projection |
| Blobs | Associated content hash, meaningful MIME metadata, sequence, or multiplicity changes | Blob surrogate UUID changes for the same content |
| Operational metadata | None of these alone may cause red | Operation ID, module run, host, persistence time |
| Anchor hash | It cannot establish green; a source-integrity mismatch blocks the decision | A representation-only change to independently stored hash metadata cannot constitute semantic difference |

---

## 2. Canonical payload encoding specification

### 2.1 Boundary decision

Fingerprint **after** the database boundary.

For an existing event, read the stored JSONB value. For a shadow candidate:

1. Execute the same admission and privacy path.
2. Run the same recursive NUL stripping.
3. Insert it into an isolated shadow or temporary table with the same JSONB column semantics and validation behavior.
4. Read the JSONB value back.
5. Apply the schema-owned semantic projector.
6. Canonicalize and fingerprint that projected value.

Admission currently strips NUL characters recursively from keys and values immediately before constructing the persisted row (`crate/sinexd/src/event_engine/admission.rs:1160-1217`, symbol `admitted_to_stream_rows`; `crate/sinex-primitives/src/validation/core.rs:305-341`, symbol `strip_postgres_jsonb_nul_chars`). The event payload is stored as JSONB (`crate/sinex-schema/src/defs/events.rs:169-221`, symbol `Events::create_table_statement`).

Pre-database fingerprinting would distinguish values that Sinex cannot persist distinctly. It would also fail to model JSONB’s object-key, duplicate-key, and numeric behavior.

Do not hash PostgreSQL’s JSONB binary representation or `jsonb::text` output directly. Those are storage/version representations, not the protocol. Parse the post-JSONB value into an arbitrary-precision canonical AST and emit the protocol bytes below.

### 2.2 `fingerprint_version`

The first version should be named unambiguously, for example:

```text
fingerprint_version = "sinex-semantic-v1"
hash_algorithm       = "blake3-256"
payload_encoding     = "sinex-canonical-json-v1"
```

Changing key ordering, numeric normalization, projection annotations, encrypted-token treatment, parent commitment, or logical occurrence construction requires a new `fingerprint_version`.

### 2.3 Canonical JSON rules

**Objects**

- Sort object keys recursively by the raw UTF-8 bytes of their decoded Unicode scalar sequence.
- Do not apply NFC, NFD, case folding, locale collation, or any other Unicode normalization.
- Emit no insignificant whitespace.
- Duplicate keys are already lost or resolved by JSONB. V1 fingerprints the value that survived the database boundary. Future admission may reject duplicates, but convergence cannot recover a discarded legacy duplicate.
- A missing key and a present key with `null` remain distinct.

**Arrays**

- Preserve order by default.
- A schema annotation may declare an array a `set` or `multiset`.
- A set is canonicalized by member bytes, rejects or collapses duplicates according to its EventContract, and sorts members.
- A multiset sorts members by canonical bytes but retains duplicate counts.
- Lineage arrays containing interpretation UUIDs must first be projected to logical parent references.
- A sequence must never be sorted merely to obtain a matching fingerprint.

**Strings**

- Decode JSON escapes first.
- Emit Unicode scalar values directly as UTF-8 except characters that JSON requires to be escaped.
- Escape quotation mark, reverse solidus, and control characters deterministically.
- No Unicode normalization.
- U+0000 must not exist after the persistence boundary. Encountering it in a supposedly persisted value is corruption and produces `unknown`.
- Escaped and unescaped spellings of the same scalar produce identical bytes.

**Numbers**

- Parse as arbitrary-precision decimal, never binary `f64`.
- Reject NaN, positive infinity, and negative infinity.
- Canonicalize negative zero to `0`.
- Remove leading integer zeroes.
- Remove insignificant trailing fractional zeroes.
- Remove an empty fractional component.
- Expand exponent notation to plain decimal.
- Therefore `1`, `1.0`, and `1e0` encode as `1`; `0.0010` and `1e-3` encode as `0.001`.
- Automata may still use floating-point arithmetic internally. The canonicalizer only stabilizes the serialized decimal value; it does not make a floating-point reduction deterministic.

`Timestamp` preserves nanoseconds through the separate PostgreSQL sub-nanosecond field (`crate/sinex-primitives/src/primitives/timestamp.rs:80-117`, symbols `Timestamp::subnano`, `from_postgres_timestamp`, and `to_postgres_parts`). Timestamp envelope encoding should therefore be a signed UTC epoch-second plus unsigned nanosecond-of-second, not a display-formatted RFC3339 string.

**Booleans and null**

- Encode exactly as `true`, `false`, and `null`.

### 2.4 Semantic projection order

Projection occurs after the JSONB round trip but before canonical encoding:

```text
persisted JSONB
→ resolve schema and projector
→ convert interpretation references to logical occurrence references
→ normalize encrypted semantic tokens where authorized
→ apply declared set/multiset semantics
→ canonical JSON
→ canonical payload digest
```

For randomly encrypted privacy values, the trusted projector should decrypt and replace the value with a keyed plaintext digest or compare through an exact effect receipt. It must never store plaintext in the fingerprint table. Encryption uses random XChaCha nonces, so equal plaintext does not imply equal ciphertext (`crate/sinex-primitives/src/privacy/envelope.rs:16-34`, symbol `encrypt_token`). Deterministic keyed BLAKE3 tokens already have stable comparison behavior (`crate/sinex-primitives/src/privacy/envelope.rs:58-64`).

No key or receipt means `unknown`, not a guess based on ciphertext.

---

## 3. Nondeterminism contract

The registry contains exactly 16 hosted automata (`crate/sinexd/src/automata/registry.rs:56-146`, constant `AUTOMATA`).

Class meanings:

- **D — deterministic:** same canonical input envelope and declared config produce the same output.
- **O — order-sensitive:** batching or input order can change state, cardinality, payload, timestamp, or key.
- **C — clock-sensitive:** wall-clock reads or an unspecified flush cutoff can change output.
- **E — effect-dependent:** model calls, mutable external I/O, random privacy effects, or another recorded effect can change output.

Config-sensitive behavior is recorded separately in the evaluator manifest.

For class O, convergence may color green only when the producer declares a canonical input order or supplies an exact input-order receipt. The receipt is a sequence of logical occurrence digests, never UUIDs. The generic runtime explicitly documents arrival ordering and unsupported historical `ts_orig` ordering (`crate/sinexd/src/runtime/automaton/traits.rs:173-196`, trait `Windowed`).

For class C, shadow evaluation must use a recorded logical clock or frozen cutoff. `Timestamp::now()` cannot participate in a green decision without such a receipt (`crate/sinex-primitives/src/primitives/timestamp.rs:29-33`, symbol `Timestamp::now`; `crate/sinexd/src/runtime/automaton/output.rs:122-139`, symbol `DerivedOutput::windowed_now`).

For class E, exact effect identity is mandatory: request identity, policy, output hash, and effect record. `ExplicitReevaluate` means the old effect is not replayable and therefore cannot be green solely from an old output (`crate/sinex-primitives/src/llm.rs:157-198`, symbols `ReplayPolicy` and `ModelEffectRequest`; `crate/sinex-primitives/src/llm.rs:251-265`, symbol `ModelEffectRecord::can_replay`).

### Registry classification

| Registry entry | Class | Source evidence | Changed-only rule |
|---|---|---|---|
| `canonicalizer` | D | Stateless 1:1 transform with inherited timestamp; payload embeds trigger UUID (`crate/sinexd/src/automata/canonicalizer.rs:1-11,58-80`, `TerminalCommandCanonicalizer::process`) | Green eligible after parent-reference projection and current schema validation |
| `analytics` | O + config | Environment thresholds; arrival-built vectors and counters determine boundaries and keys (`crate/sinexd/src/automata/analytics.rs:28-167,204-324`, `AnalyticsAutomaton::accumulate`/`emit`) | Requires canonical/order receipt, exact threshold manifest, and stable start-anchored window identity; current counter identity is not admissible |
| `attention-stream` | D, conditional on parent identity | Per-window mapping inherits upstream `window_id`/identity (`crate/sinexd/src/automata/attention.rs:41-68`, `AttentionStream::process`) | Eligible only after upstream analytics occurrence identity is stable and parent-derived fields are projected |
| `interval-lift` | O + C | Stateful arrival handling; IDs use parent UUIDs; ActivityWatch path reads wall clock (`crate/sinexd/src/automata/interval_lift.rs:371-440,541-621,669-743`) | Requires ordered trace, frozen clock, and `ecy`/`y8v` start identity. Current parent-ID occurrence declaration is not admissible |
| `health` | O + config + map iteration | Environment/config state; component state in `HashMap`; output components and parents are collected from map iteration (`crate/sinexd/src/automata/health.rs:27-180,234-484`, `HealthAggregator::reconcile`) | Green only after producer sorts all semantic collections and supplies input/config receipt |
| `session` | O | Arrival-built parent vector and counter-derived session key (`crate/sinexd/src/automata/session.rs:23-65,105-200`, `SessionDetector::accumulate`/`emit`) | Requires ordered trace and start-anchored session identity; current counter key blocks green |
| `hourly` | O + C | Arrival buckets, trailing wall-clock flush, parent vector, counter telemetry (`crate/sinexd/src/automata/hourly.rs:21-45,54-113,143-263`) | Requires exact bucket input order and frozen hour cutoff |
| `daily` | O + C | Arrival buckets and `flush_due(now)` (`crate/sinexd/src/automata/daily.rs:55-116,146-267`, `DailySummarizer::flush_due`) | Requires exact ordered hourly inputs and frozen day cutoff |
| `entity-extractor` | O + C fallback | Recursively traverses object iteration order, emits first match, falls back to wall clock (`crate/sinexd/src/automata/entity_extractor.rs:91-165`, `extract_text_fields` and `process`) | Require sorted field traversal and non-null inherited timestamp, or exact order/clock receipts |
| `entity-resolver` | O + C | First unresolved spelling becomes `original_name`; emission uses `windowed_now` (`crate/sinexd/src/automata/entity_resolver.rs:77-135`, `EntityResolver::accumulate`/`emit`) | Require deterministic candidate ordering and explicit timestamp; otherwise not green |
| `relation-extractor` | O | Vector/window order controls generated pairs; confidence is a floating value (`crate/sinexd/src/automata/relation_extractor.rs:28-83,119-213`) | Require ordered-window receipt; compare finite decimal output exactly |
| `entity-enricher` | O + config | Per-entity state, dirty-vector order, periodic event-time sweep, configurable interval (`crate/sinexd/src/automata/entity_enricher.rs:24-82,108-205`) | Requires ordered input receipt and exact effective configuration |
| `tag-applier` | D + C fallback | First matching rule; timestamp fallback; payload embeds trigger UUID (`crate/sinexd/src/automata/tag_applier.rs:59-84`, `TagApplier::process`) | Eligible only with inherited timestamp, rule manifest, and parent-reference projector |
| `embedding-producer` | E + O | Builds a model-effect request from `input.to_string()` and emits effect key/replay policy (`crate/sinexd/src/automata/embedding_producer.rs:31-71`, `EmbeddingProducer::process`) | Never green without an exact model-effect identity and output receipt; object-order-dependent request hashing must also be replaced by canonical input bytes |
| `document-parser` | E + C | Reads mutable live Dendron paths; timestamp fallbacks; terminal document identity derives from parent UUID (`crate/sinexd/src/automata/document_parser.rs:64-140,152-360`) | Requires content-store hash/effect receipt for external bytes, frozen timestamp, and document-identity projector; current live-path read is insufficient |
| `instruction-reconciler` | O | Pending instructions are held in arrival order, drained on observation, and matched observation UUIDs are serialized (`crate/sinexd/src/automata/instruction_reconciler.rs:32-41,77-218`) | Treat as per-instruction append-only reconciliation; require pending-order receipt and projection of matched UUIDs |

The embedding/model-effect schema is currently described as dormant and no production caller was found (`crate/sinex-schema/src/defs/model_effects.rs:1-11`, module documentation). Consequently, embedding-dependent green proof is unavailable in this package.

Map-order nondeterminism is not repaired by canonicalizing the final JSON object when map iteration has already determined an array, selected a winner, or ordered parent references. Producers must sort before semantic decisions or emit an order receipt.

---

## 4. Transitive coloring rule

### 4.1 Pairing and scope comparison

Changed-only convergence executes against an immutable operation snapshot:

1. Evaluate the selected roots and derivations under current semantics in a shadow workspace.
2. Resolve old and candidate nodes to logical occurrence keys.
3. Reject ambiguous one-to-many or many-to-one matches as `unknown`.
4. For complete-scope reconcilers, compare the complete old and candidate output bags—including cardinality and absence—before individual coloring.
5. Color matched nodes in topological parent order.
6. Only after coloring succeeds acquire the final write fence and archive/remint red nodes.

A node is green only when its own semantic envelope matches and its entire relevant lineage remains retainable.

### 4.2 Parent propagation

For an in-scope derived node:

```text
green(child) requires every immediate parent to be green.
```

If one parent is red, the child is red even when its projected payload and logical-parent commitment appear equal. The child’s persisted provenance must point to the newly minted parent interpretation. Retaining it would leave the live graph pointing at an archived parent and violate the settled “red plus affected descendants” rule (`sinex-scratchpad.xml:35573-35579`).

When only some parents are red, the reminted child receives:

- Existing UUIDs for retained green parents.
- Newly minted UUIDs for red parents.
- A new child UUID and therefore a new `ts_coided`.

The semantic parent commitment remains based on logical occurrences, so it can still compare equal to the full-replay result even though the raw UUID set is different.

### 4.3 High fan-in

Current runtime validation warns above 100 parents and rejects more than 1,000 parents (`crate/sinexd/src/runtime/automaton/adapter/output.rs:83-116`, symbol `validate_output_batch`).

For high-fan-in nodes, calculate a streaming Merkle commitment:

```text
leaf = H(parent_role, parent_logical_occurrence_digest, multiplicity)
node = H(left_child, right_child, subtree_count)
root = H(total_count, reconciliation_semantics, merkle_root)
```

Default lineage semantics are an unordered multiset because persisted source-event IDs are sorted and deduplicated. Ordered semantic input requires a separate sequence receipt and producer declaration.

The count and multiplicity must be committed. XOR or an order-insensitive sum is forbidden because duplicate cancellation and parent substitution would become possible.

### 4.4 Complete-scope reconciliation

For a declared `complete_scope` producer:

- Build old and candidate output multisets keyed by declared output slot.
- Compare slot uniqueness, output count, and absence.
- A matched slot may be green only after the entire candidate scope is known.
- Old slots absent from the candidate are red and archived.
- Candidate slots absent from the old scope are newly minted.
- Any duplicate slot or unkeyed ambiguity produces `unknown`.

The generic `ScopeReconciler` interface describes scope processing, but a registry-wide reconciliation-mode type does not yet exist. The scratch review explicitly treats per-occurrence, complete-scope, and append-only as a hypothesis still requiring proof (`sinex-scratchpad.xml:35824-35827`).

### 4.5 Append-only reconciliation outputs

For a declared `append_only` producer, including instruction/authority history:

- Match each current candidate by its contract-declared occurrence.
- Do not infer that an old historical output is obsolete merely because a bounded current reconciliation did not reproduce it.
- A new occurrence is appended.
- A correction or supersession archives only the matching occurrence and its descendants.
- Unrelated historical outputs remain live.
- Retraction requires an explicit correction/retraction authority, not candidate absence.

The instruction reconciler already declares a stable occurrence identity of `(instruction_id, desired_event_source, desired_event_type)` (`crate/sinexd/src/automata/instruction_reconciler.rs:224-238`, source contract registration). That tuple, rather than the observation UUID, is its comparison key.

### 4.6 Parents outside the convergence scope

An outside-scope parent counts as green only through a boundary certificate containing:

```text
parent event ID
logical occurrence digest
semantic envelope digest
fingerprint_version
projector manifest digest
snapshot/high-water identity
proof that the parent has not been archived, tombstoned, or replaced
```

The final commit must revalidate the certificate. A missing or changed certificate produces `unknown`; the operation may expand the scope or fall back to full replay. It must not assume that an unexamined parent is unchanged.

Cycles, unresolved parents, duplicate logical keys, unknown schema projectors, and missing archived-parent records also produce `unknown`.

---

## 5. Storage and operation record

### 5.1 Storage decision

Do not add mutable fingerprint columns to `core.events`. The events table is append-only and has a no-update trigger (`crate/sinex-schema/src/defs/events.rs:507-521`, symbol `Events::create_no_update_trigger_sql`). Green revalidation must not update an event row.

Use a companion cache table:

```text
core.event_semantic_fingerprints
--------------------------------
event_id                         UUID NOT NULL
fingerprint_version              TEXT NOT NULL
projector_manifest_digest        BYTEA NOT NULL       -- 32 bytes
logical_occurrence_digest        BYTEA NOT NULL       -- 32 bytes
canonical_payload_digest         BYTEA NOT NULL       -- 32 bytes
parent_commitment_digest         BYTEA NULL           -- 32 bytes
semantic_envelope_digest         BYTEA NOT NULL       -- 32 bytes
canonical_envelope_len           BIGINT NOT NULL
creation_evaluator_manifest_digest BYTEA NULL         -- legacy may be unknown
creation_effect_receipt_digest   BYTEA NULL
computed_at                      TIMESTAMPTZ NOT NULL
computed_by_operation_id         UUID NULL

PRIMARY KEY (event_id, fingerprint_version, projector_manifest_digest)
INDEX (fingerprint_version, logical_occurrence_digest)
INDEX (fingerprint_version, semantic_envelope_digest)
```

The table should not require a foreign key solely to `core.events`, because event IDs can move to the audit/archive plane while remaining relevant to operation evidence.

Deduplicate manifests in:

```text
core.semantic_evaluator_manifests
---------------------------------
manifest_digest       BYTEA PRIMARY KEY
fingerprint_version   TEXT NOT NULL
manifest              JSONB NOT NULL
created_at            TIMESTAMPTZ NOT NULL
```

The fingerprint table is a cache. Exact comparison reconstructs the canonical envelope from the event and projector when the digests match. Storing every full canonical envelope would multiply storage substantially and is unnecessary for routine comparisons.

### 5.2 Computation timing

Use a hybrid strategy:

- **Future events:** compute the content fingerprint after persistence, ideally from `INSERT ... RETURNING` or a same-transaction reread of the JSONB value. Never compute the authoritative value from the pre-persistence object.
- **Existing events:** compute lazily when a convergence scope needs them and retain the result as a cache.
- **Background backfill:** optional and low priority; it is not a prerequisite for convergence.
- **Evaluator manifests:** compute per operation from the effective current configuration. Store a creation-time manifest for new events when available; legacy absence remains explicitly unknown.
- **Version changes:** retain old fingerprint rows for audit and add the new version. Never overwrite a prior version.

### 5.3 Revalidation operation record

`core.operations_log` already records operation type, operator, JSON scope, scope window, result status/message, summary, and duration (`crate/sinex-schema/src/defs/operations.rs:17-34,609-688`, symbols `OperationsLog` and `OperationsLog::create_table_statement`).

Create one parent log row:

```text
operation_type = "convergence.changed_only"
scope = {
  selected roots and derivations,
  snapshot/high-water values,
  fingerprint_version,
  projector manifest digest,
  current evaluator manifest digests,
  logical clock,
  reconciliation modes
}
preview_summary = {
  green_count,
  red_count,
  new_count,
  absent_count,
  unknown_count,
  reason counts
}
```

Record per-event decisions in an operation-detail table:

```text
core.operation_event_revalidations
----------------------------------
operation_id                  UUID NOT NULL
old_event_id                  UUID NULL
replacement_event_id          UUID NULL
shadow_candidate_ref          TEXT NULL
fingerprint_version           TEXT NOT NULL
logical_occurrence_digest     BYTEA NOT NULL
old_semantic_digest           BYTEA NULL
candidate_semantic_digest     BYTEA NULL
current_evaluator_digest      BYTEA NOT NULL
parent_commitment_digest      BYTEA NULL
effect_receipt_digest         BYTEA NULL
decision                      TEXT NOT NULL
reason_code                   TEXT NOT NULL
canonical_equal               BOOLEAN NOT NULL
parents_retainable            BOOLEAN NOT NULL
schema_valid                  BOOLEAN NOT NULL
nondeterminism_proven         BOOLEAN NOT NULL
decided_at                    TIMESTAMPTZ NOT NULL

decision ∈ green | red | new | absent | blocked_unknown
PRIMARY KEY (operation_id, logical_occurrence_digest, old_event_id)
INDEX (operation_id, decision)
INDEX (old_event_id)
INDEX (replacement_event_id)
```

Representative `reason_code` values:

```text
semantic_match
payload_changed
timestamp_changed
quality_changed
parent_red
candidate_absent
candidate_new
logical_key_ambiguous
schema_invalid
projector_unavailable
missing_order_receipt
clock_receipt_mismatch
effect_receipt_mismatch
outside_scope_boundary_changed
hash_collision
fingerprint_version_unavailable
anchor_integrity_mismatch
```

Every retained green node receives a detail row. Its event, provenance, ID, `ts_coided`, and `ts_persisted` remain untouched.

### 5.4 Cost model for 70 million rows

The 70-million-row figure comes from the review brief; the package does not establish that production count.

The minimum fixed digest footprint is approximately:

- Four 32-byte digests: 128 bytes × 70M = **8.96 GB**.
- Five digests: **11.2 GB**.
- Seven digests including projector/evaluator/effect references: **15.68 GB** before nullability savings.

UUIDs, timestamps, tuple headers, alignment, indexes, and free-space overhead can plausibly bring a broad eager cache above **20–35 GB**. This is an engineering estimate, not a measured package value.

CPU and I/O are more significant than digest arithmetic:

```text
O(total JSONB bytes
  + semantic projection work
  + parent-edge lookups
  + blob metadata lookups
  + current-schema validation)
```

A full 70M-row eager pass would read the entire event plane and its lineage. Lazy scope fingerprinting makes initial cost proportional to the selected convergence scope, while eager post-persist computation amortizes future work. Therefore:

- Do not require a 70M-row backfill before launching the operation.
- Eagerly fingerprint newly persisted events.
- Lazily fingerprint legacy events as scopes are evaluated.
- Deduplicate evaluator manifests instead of copying configuration JSON to every row.

### 5.5 Declarative schema surface

Add:

```text
crate/sinex-schema/src/defs/semantic_fingerprints.rs
```

Then:

- Export the definitions from `crate/sinex-schema/src/defs/mod.rs`.
- Add table metadata to `ALL_TABLES`.
- Add create statements to `crate/sinex-schema/src/apply.rs:create_tables`.
- Add the tables to the convergence/strict-diff registry in `crate/sinex-schema/src/converge.rs`.
- Add repositories in the database crate.
- Deploy through `sinex-schema apply`.

The convergence engine states that `create_table_statement()` is the single source of truth and that the next `apply()` adds declared columns to existing databases (`crate/sinex-schema/src/converge.rs:1-64`, module contract). No migration directory should be introduced.

---

## 6. Full-replay oracle protocol

### 6.1 Differential experiment

For snapshot `S`, scope `Q`, current evaluator manifest `M`, logical clock `T`, and effect store `E`, create two isolated workspaces:

- **R — full replay:** archive/remint the complete selected interpretation graph under current semantics.
- **C — changed-only convergence:** evaluate the same graph in shadow, retain green originals, and remint only red/new nodes.

Both workspaces must use the same:

- Immutable source material and root snapshot.
- Current parser/automaton code.
- Effective contracts, schemas, policies, and configuration.
- Logical clock.
- Input-order declarations/receipts.
- Model and external-effect store.

The assertion is not raw row equality. UUID-bearing payload fields and parent provenance necessarily differ. Normalize both workspaces through the same current logical-occurrence and semantic-projection protocol.

The primary oracle assertion is:

```text
multiset(
  logical occurrence,
  semantic envelope bytes,
  parent logical commitment,
  current schema validity
) in C

==

the same multiset in R
```

This includes output absence, output cardinality, and complete-scope membership.

ID assertions:

- Every selected replay output ID in R is disjoint from the corresponding original IDs.
- Every green ID in C is exactly its original ID.
- Every red/new ID in C is disjoint from its original ID.
- R and C IDs are expected to be disjoint; equality of their IDs is not a correctness requirement.
- Every retained C node keeps its original `ts_coided`.
- Every reminted C and R node has a new interpretation ID and ID-derived clock.

Run the corpus repeatedly with changed batch sizes, process hash seeds, map insertion orders, arrival permutations, and clock perturbations. A declared deterministic producer must remain stable. An order/clock/effect-dependent producer may be compared only under its matching receipt.

This comparison function is the missing substance of `sinex-dtw5`, whose issue states that no whole-derived-plane occurrence-keyed replay/live agreement proof currently exists (`sinex-beads.xml:1222-1231`, issue `sinex-dtw5`). The scratch audit also finds incomplete material replacement proof and no derived replacement settlement (`sinex-scratchpad.xml:35771-35783`, section `What full replay currently proves poorly`).

### 6.2 Canonicalizer reference test

Corpus:

- Kitty, Atuin, Bash, Zsh, and Fish `command.executed` material events.
- Equivalent JSON with varied object-key ordering and numeric spellings.
- Distinct command, working directory, exit status, source, timestamp, and quality cases.
- Parent IDs reminted independently in R.

Required experiments:

1. **No-op evaluator change:** bump canonicalizer `semantics_version` or refactor code without changing output.  
   C must retain the canonical event; R must remint it; semantic envelopes must agree.

2. **Parent-ID churn:** shadow and replay use different parent UUIDs.  
   Projected `source_events` and parent commitments must agree; C remains green when the material parent is green.

3. **Semantic change:** alter command normalization, command text, cwd, timestamp, or event type.  
   The canonical node and every dependent must be red in C.

4. **Operational-only change:** alter operation ID, module run, host, key order, or persistence time.  
   The canonical node must remain green.

The canonicalizer is the correct first proof because its source explicitly describes it as a pure stateless per-event transform (`crate/sinexd/src/automata/canonicalizer.rs:1-11`).

### 6.3 Session and interval reference test

This test is gated on stable start-anchored identity from `sinex-ecy`, `sinex-y8v`, and `sinex-n9a`; current counter and parent-ID keys are not valid occurrence coordinates (`sinex-beads.xml:185-205`, issues `sinex-n9a`, `sinex-y8v`, and `sinex-ecy`).

Corpus:

- Hyprland focus/workspace events.
- ActivityWatch observations with growing duration and heartbeats.
- Gaps immediately below and above thresholds.
- Maximum-duration and maximum-count closures.
- Non-increasing and late arrival timestamps.
- Trailing hourly/daily flushes.
- Multiple unrelated sessions around the changed interval.
- High-fan-in windows near the current parent warning and hard limits.

Required experiments:

1. Same start occurrence and identical end/content: C retains the interval/session.
2. Same start occurrence with a later end or changed duration: C archives/remints that occurrence and all descendants; unrelated sessions remain green.
3. Same payload with only `ts_orig` resolution changed: red.
4. Same logical parents with replay-reminted UUIDs: projected semantic equality.
5. Arrival permutation without a matching order receipt: `unknown`, never green.
6. Frozen clock versus a perturbed wall clock: only the matching clock receipt may green.
7. Compare the resulting interval, session, attention, hourly, and daily semantic multisets between C and R; never compare counter-derived display IDs.

### 6.4 Instruction-reconciliation reference test

Corpus:

- Several pending workspace instructions.
- Dry-run and non-dry-run instructions.
- Matching and non-matching observations.
- Deadline-before and deadline-after cases.
- Multiple instructions drained by one observation.
- Unrelated historical instruction status events.

Key outputs by:

```text
(instruction_id, desired_event_source, desired_event_type)
```

Required experiments:

1. Remint the observation UUID in R. `matched_event_ids` must project to the same observation occurrence.
2. Preserve the same pending order with an exact order receipt. Matching status outputs may remain green in C.
3. Reorder pending instructions without a valid order contract. The scope becomes `unknown` or red; it must not be accepted based on an accidental equal bag.
4. Remove or add one output. Whole-scope cardinality comparison must detect the difference.
5. Append a new instruction occurrence without touching unrelated historical statuses.
6. Change deadline evaluation or `evaluated_at`. The affected status becomes red and remints.

---

## 7. Failure modes and abuse cases

### Fingerprint collision

A matching digest is never final proof. On a digest match:

1. Reconstruct both canonical semantic envelopes under the same `fingerprint_version`.
2. Compare their exact byte streams and lengths.
3. If digest matches but bytes differ, record `hash_collision`, block the scope, and retain neither result as green.
4. Preserve both byte lengths and diagnostic component digests in the operation record.
5. Do not silently “strengthen” identity by deriving an event UUID from either digest.

An independent secondary digest may be stored for incident triage, but exact canonical bytes remain the equality authority.

### Mixed or stale `fingerprint_version`

Never compare cached fingerprints from different versions.

For an operation using version V:

- Re-fingerprint both the old event and shadow candidate under V.
- Retain previous cached versions for audit.
- Include the projector manifest digest in the cache key.
- If an old event cannot be projected under V, return `unknown`.
- Do not translate a v1 digest into v2 through another digest.

### Semantics change that only changes `ts_orig`

Exact timestamp and quality are semantic fields. Any change to resolved instant, nanosecond remainder, presence, or quality tier turns the node red and propagates red to descendants.

This catches timestamp-resolver changes even when the JSON payload is unchanged.

### Convergence racing live ingestion

The operation needs two barriers:

1. An immutable read snapshot/high-water mark for shadow evaluation.
2. A short final write fence before archive/remint.

Before commit, verify:

- No selected root or boundary parent was archived, tombstoned, or replaced.
- No new in-scope input arrived after the captured high-water mark.
- Policy, schema/projector, evaluator manifest, logical clock, and effect store still match the recorded operation inputs.
- All outside-scope boundary certificates remain valid.

Any mismatch aborts the write phase and records `blocked_unknown`.

Current replay uses global SHARE locking that pauses inserts during archive selection (`sinex-scratchpad.xml:35785-35787`). A future scoped fence is preferable, but until it exists the operation must use the existing correctness-preserving barrier rather than race ingestion.

### Payload schema tightened since creation

Equality of payload bytes is insufficient.

Both the old row and candidate must validate against the current resolved schema and current semantic projector. The existing database validation is opt-in and can accept an event leniently when no schema is found (`crate/sinex-schema/src/defs/events.rs:524-606`, symbols `Events::create_payload_validation_function_sql` and validation trigger). Changed-only convergence must invoke strict current resolution.

Outcomes:

- Current evaluator produces a corrected valid candidate: old node is red and the candidate remints.
- Current evaluator produces no candidate because the old interpretation is no longer admissible: old node is absent/red according to its reconciliation contract.
- Current evaluator still produces the same invalid candidate or schema resolution is ambiguous: `unknown`; do not mark green.
- A new surrogate schema UUID with equivalent schema content and projection does not by itself make the node red.

### Source-anchor hash mismatch

`anchor_payload_hash` is a BLAKE3 hash of the source-material byte range (`crate/sinex-primitives/src/events/mod.rs:90-93`, field `Event::anchor_payload_hash`; `crate/sinex-primitives/src/events/builder.rs:192-213`, symbols `with_anchor_payload_hash` and `with_anchor_payload_from_bytes`).

It is an independent integrity precondition:

- A match cannot prove semantic equality.
- A mismatch does not identify which interpretation field changed.
- A mismatch blocks green and records an integrity failure; it must not be reported as a semantic fingerprint difference.
- Full replay replacement currently warns on a mismatch but can still record replacement, and a test explicitly permits that behavior (`crate/sinexd/src/replay/replay_writer.rs:148-190`, symbol `record_event_replacements`; `crate/sinexd/src/replay/execution_outcome.rs:636-718`, test `replay_anchor_payload_hash_mismatch_does_not_block_replacement`). Changed-only must be stricter because it is proposing retention.

### Random encryption and privacy effects

Equal plaintext can produce unequal ciphertext because encryption uses a random nonce. Green requires:

- Authorized decryption plus keyed plaintext-digest projection, or
- An exact privacy-effect receipt showing the same input, key identity, policy, and semantic output.

No key, stale key identity, or missing effect receipt produces `unknown`.

The exact compiled policy snapshot must be recorded. “The policy version currently in the database” is insufficient because the engine may continue using its cached snapshot after refresh failure (`crate/sinexd/src/event_engine/policy.rs:710-765`, symbols `ensure_fresh` and `redact_batch`).

### Model and external effects

A model-effect output is never green without an exact request and output identity match. Provider, model, prompt version, schema version, input hash, replay policy, and output hash all participate in the receipt.

A live file read is also an effect. The document parser’s Dendron path read cannot be replay proof merely because the same path exists. The receipt must bind the exact content-store hash. Missing or unreadable content is not equivalent to an intentional zero-output result.

### Ambiguous occurrence keys

More than one old or candidate node for the same declared logical occurrence invokes regime 3:

- Do not choose the first row.
- Do not choose the latest UUID.
- Do not compare content and select the closest one.
- Record all competing IDs and evidence.
- Return `unknown` for operator adjudication or use full replay after an explicit scope decision.

Counter-based session/window keys and parent-UUID-based occurrence declarations are known invalid examples (`sinex-scratchpad.xml:35694-35697`, section `Identity overload`; `sinex-beads.xml:203-205`, issue `sinex-ecy`).

### Projector abuse

A projector must not:

- Drop arbitrary fields merely to produce equality.
- Treat every `*_id` field as an interpretation reference.
- Sort every array.
- Suppress null-versus-absent differences.
- Rewrite timestamps within a tolerance.
- Use the old and new payloads jointly to decide which fields matter.

Projectors are deterministic functions of one event, its declared schema, and its resolvable lineage. Their code/version digest belongs in `fingerprint_version` evidence.

---

## 8. Bead surgery sketch

### Replacement text for `sinex-qky`

**Title**

> Changed-only convergence: semantic fingerprints, shadow revalidation, and red/green reminting

**Description**

> Design and implement a changed-only convergence operation distinct from full replay. Full replay remains the always-reminting, no-shortcut oracle. Changed-only convergence snapshots a selected material/derived scope, evaluates current parser and automaton semantics in an isolated shadow workspace, pairs existing and candidate outputs by declared stable occurrence/reconciliation identity, and retains an existing interpretation only when its versioned semantic envelope is exactly equal, its current schema remains valid, its nondeterminism proof is complete, and every parent is retained or covered by a verified outside-scope boundary certificate.
>
> Green interpretations keep their existing IDs, `ts_coided`, `ts_persisted`, and provenance. Revalidation is recorded in `core.operations_log` and its per-event decision ledger. Red interpretations, candidate-absent interpretations, and all affected descendants are archived and reminted. Unknown evidence blocks the affected scope or delegates it to full replay.
>
> `anchor_payload_hash` remains source-byte integrity evidence and is not a semantic equality primitive. Fingerprints are never event IDs and may not create content-derived interpretation identity.

The existing `qky` instead proposes early-cutoff replay keyed by `anchor_payload_hash` (`sinex-beads.xml:3686-3691`, issue `sinex-qky`), directly contradicting the settled source/interpretation distinction (`sinex-scratchpad.xml:35791-35802`).

### Acceptance criteria

1. `sinex-semantic-v1` specifies material and derived logical occurrence, semantic envelope, canonical JSON, transitive logical-parent commitments, and exact-byte collision verification.
2. Both old and shadow payloads are compared after the admission/privacy/NUL/JSONB boundary.
3. The active payload-schema registry owns semantic projection of interpretation references and collection semantics. Unknown projection fails closed.
4. `semantics_version`, parser version, policy/config snapshots, schema content, logical clock, and effect identities are recorded in evaluator manifests; evaluator version changes alone do not force red.
5. Every registered automaton declares D/O/C/E properties and required order, clock, model, privacy, or external-effect receipts.
6. Logical occurrence complies with `sinex-908`, `sinex-ecy`, `sinex-y8v`, and `sinex-n9a`; no counter, payload content hash, or interpretation UUID may become a covert event ID.
7. Shadow evaluation performs no archive or live-event mutation before complete scope comparison and topological coloring.
8. A red parent forces every in-scope descendant red. A green outside-scope parent requires a revalidated boundary certificate.
9. Complete-scope, per-occurrence, and append-only producers have explicit reconciliation behavior.
10. Green revalidation writes only the operation log/detail ledger. It never updates `core.events` or provenance clocks.
11. Fingerprint cache, evaluator manifests, and operation detail tables are declared through `sinex-schema` and deployed by `sinex-schema apply`.
12. `sinex-dtw5` differential tests prove semantic-plane equality between `converge(scope)` and `replay(scope)` for canonicalizer, session/interval, and instruction reconciliation, with the required disjoint ID-churn assertions.
13. Hash matches are verified by exact canonical-envelope bytes before a green decision.
14. Missing order/clock/effect evidence, ambiguous logical identity, invalid current schema, or unresolved parent projection yields `blocked_unknown`, never green.

### Design

> **Comparison:** Compute a logical occurrence key separately from a semantic envelope. Pair by the former and compare exact canonical bytes of the latter. Hashes are indexes only.
>
> **Execution:** Capture snapshot, evaluator manifests, clock, effect store, and boundary high-water values; run current semantics in a shadow workspace; compare whole reconciliation scopes; color topologically; verify the final fence; archive/remint red nodes; write every green/red/new/absent/unknown decision to operation evidence.
>
> **Identity:** Material identity is source material plus stable start anchor and declared occurrence fields. Derived identity is the producer’s declared output slot rooted transitively in designated material occurrences. Event IDs remain random UUIDv7.
>
> **Nondeterminism:** Deterministic producers may compare directly. Order-sensitive producers require canonical order or an exact logical-occurrence order receipt. Clock-sensitive producers require a frozen logical clock. Model, privacy, and external effects require exact effect identities.
>
> **Persistence:** Cache versioned fingerprints in a companion table. Never add or update mutable fingerprint state on `core.events`.
>
> **Fallback:** Unknown proof blocks changed-only handling for that scope. Full replay remains available and always remints.

### Gates

| Gate | Requirement |
|---|---|
| Full-replay oracle | `sinex-dtw5` must provide whole-derived-plane occurrence-keyed comparison and complete replacement settlement |
| Fingerprint infrastructure | Versioned canonicalizer, logical occurrence encoder, exact-byte comparator, cache tables, and operation detail ledger |
| Payload projection | Schema-owned projectors for interpretation references, stable domain IDs, encrypted values, and set/multiset arrays |
| Occurrence doctrine | `sinex-908`; interval/session paths additionally require `sinex-ecy`, `sinex-y8v`, and `sinex-n9a` |
| Determinism evidence | Producer classification plus order/clock/effect receipt infrastructure |
| Snapshot safety | Immutable read snapshot and commit-time boundary/write-fence proof |
| Reference derivations | Canonicalizer, session/interval, and instruction reconciliation must pass the differential suite |

A universal derivation-kernel proof should **not** block the canonicalizer pilot. The package says reconciliation modes are not yet a registry-wide type and must be proven on real derivations (`sinex-scratchpad.xml:35824-35827`). Gate the first implementation on a minimal execution-grade comparison and receipt interface proven on the three reference derivations. Expansion to another automaton requires its own declared reconciliation and nondeterminism contract.

Effect infrastructure also need not block deterministic pilots. It must cause effect-dependent automata to fail closed until their receipts exist.

### New beads spawned

**New design bead: Semantic payload projection contract**

> Define versioned JSON-Schema annotations and runtime projectors for interpretation UUID references, logical parent roles, UUID-derived payload identifiers, collection sequence/set/multiset semantics, stable domain UUIDs, and encrypted semantic tokens. Include projectors for canonical command `source_events`, tag parent IDs, instruction `matched_event_ids`, interval IDs, and document IDs. Unknown or unresolved projection fails closed.

This is a blocker for `qky`; otherwise raw JSON equality is replay-unstable and global field-name stripping would corrupt stable UUIDv5 domain identities.

**New design bead: Derivation reproducibility receipts**

> Add producer declarations and durable receipts for logical input order, batching equivalence, logical clock/cutoff, effective config manifest, model effects, privacy effects, and external content reads. Define which receipt classes each automaton requires and make missing evidence visible as `blocked_unknown`.

This is a blocker for green retention by O/C/E automata, but not for a deterministic canonicalizer pilot.

**New implementation bead: Versioned semantic-fingerprint cache and convergence decision ledger**

> Add the companion fingerprint, evaluator-manifest, and operation-event-revalidation tables; wire declarative schema application, repositories, lazy legacy computation, post-persist future-event computation, collision verification, and operation summaries.

---

## Claims I could not verify from the package

- **Unknown:** No changed-only convergence implementation, `fingerprint_version`, semantic-envelope canonicalizer, or green/red coloring engine exists in the supplied package.
- **Unknown:** Parser ID and parser version are not present on persisted `Event<T>` rows. The materialization path carries parser information in `ParsedEventIntent` but constructs an event without persisting those fields (`crate/sinex-primitives/src/parser/mod.rs:802-869`, symbol `ParsedEventIntent`; `crate/sinexd/src/runtime/parser/adapter_source.rs:2183-2233`, symbol `intent_to_event_with_anchor`). Historical creation-time parser manifests therefore cannot be reconstructed from event rows alone.
- **Unknown:** No durable, exact snapshot of effective automaton environment configuration was found for existing derived events.
- **Unknown:** No schema-owned semantic projection annotations or runtime projector registry were found.
- **Unknown:** No logical input-order receipts, batch receipts, wall-clock receipts, or outside-scope boundary certificates were found.
- **Unknown:** No production use of `core.model_effects` was verified, and the current embedding producer appears to emit request metadata rather than a completed embedding vector/effect.
- **Unknown:** No durable privacy-effect receipt or retained plaintext semantic digest was found for randomly encrypted values.
- **Unknown:** The package does not prove a registry-wide per-occurrence/complete-scope/append-only reconciliation declaration.
- **Unknown:** Full replay does not currently prove complete selected-root replacement or whole-derived-plane settlement; `sinex-dtw5` remains open.
- **Unknown:** The production PostgreSQL version, JSONB text-rendering details, and exact `serde_json` arbitrary-precision feature set were not available. The protocol therefore deliberately does not hash either representation directly.
- **Unknown:** A scoped convergence write fence was not found. The package documents the existing global SHARE-lock behavior only.
- **Unknown:** The 70-million-row production count, average payload size, parent fan-in distribution, index compression, and actual storage cost were not present in the package.
- **Unknown:** The interpretation-reference examples identified in source are not proven exhaustive.
- **Unknown:** Whether every archived-event path preserves enough event, schema, blob, and policy evidence to re-fingerprint old interpretations under a future `fingerprint_version` was not established.
- **Unknown:** No proof was found that current document-parser external file reads are bound to immutable content-store objects.
- **Unknown:** The exact dependency issue representing the “replay correctness gh#2194” prerequisite named by the existing `qky` text could not be identified conclusively from the package’s Bead dependency data.

---

