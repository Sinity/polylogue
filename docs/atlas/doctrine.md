# Polylogue Doctrine Index

Seven standing invariants shape most non-obvious decisions in this repository.
Before this sheet existed they were discoverable only by reading gate source
and tracker prose, so an agent that needed one either rediscovered it or broke
it. Each stanza below names the invariant, the code that owns it, the gate
that enforces it, the failure you would observe when it is violated, and how
to change it on purpose.

## How to read this sheet

This sheet is an index, not a second contract. A stanza points at the owner
and says what breaks; it does not restate an enum, a field list, or a schema.
When the sheet and the owner disagree, the owner wins and the stanza is wrong.

Each stanza carries a line of the form `**Owning gate**: ` followed by a
backticked gate name, or by `none` with the reason. That name is resolved
against the live gate registry (`devtools/gate.py:294-295`) by the atlas
checker (`devtools/verify_atlas.py:118-129`), so a stanza cannot keep an owner
that no longer exists. The denominator is the registry itself, not a copy of
it: deleting a gate makes every stanza naming it go red on the next run.

Two honest limits on that enforcement:

- `devtools gate atlas` proves that a cited file exists and that a cited line
  range lies inside it, and that a declared owning gate is a real gate. It
  makes **no judgement about whether the surrounding prose is still true**
  (`devtools/verify_atlas.py:1-10`). Prose that a change made wrong is
  re-verified or deleted by that change's author; no gate detects it.
- `devtools gate doc-commands` separately rejects a stale gate name written as
  a runnable `devtools gate` invocation anywhere in a Markdown code segment
  (`devtools/verify_doc_commands.py:309-318`) — including, as this paragraph
  had to learn, a placeholder that is not a gate at all. It never sees an
  owner written as a bare name, which is why the atlas-side resolution exists.

## Doctrine: time

**Invariant**: durable time is an INTEGER count of milliseconds since the
UTC epoch, never a TEXT string; a timestamp's provenance travels with it, and
an aggregate is reported at the confidence of its *weakest* operand; test code
does not read the host clock.

**Owner**: `polylogue/core/temporal.py:13-24` owns the storage-free
source/confidence vocabulary and `polylogue/core/temporal.py:27-35` owns the
strength ordering. Aggregate reduction goes through
`polylogue/core/temporal.py:62-67`. The durable-DDL half is
`devtools/verify_timestamp_doctrine.py:1-24`, which scans only `source.db` and
`user.db` because a violation there costs an additive migration to undo. Test
determinism is held by an autouse profile hook, not a gate
(`tests/infra/clock_guard.py:86-93`).

**Owning gate**: `timestamp-doctrine`

**Observable failure**: a TEXT-typed timestamp column merges into a durable
tier and is then only removable by migration
(`devtools/verify_timestamp_doctrine.py:63-72` is the scan that would have
refused it). Separately — and *not* covered by any gate — an aggregate whose
inputs included a materialization timestamp is reported as `recorded`, so a
derived date looks like a provider-recorded event.

**Change procedure**: durable-tier temporal columns change by additive
numbered migration behind a verified backup. Adding a member to
`TemporalSource` means extending the rank order and the confidence map in the
same edit; both are exhaustive dictionaries over the literal, so a missing
member is a type error rather than a silent `unknown`.

## Doctrine: writer ownership

**Invariant**: every archive-tier module that executes a direct SQL mutation
is inventoried, declares the tiers it writes in its own module docstring, and
owns exactly one tier unless a reviewed twin-write contract says otherwise.
At runtime the daemon is the live write owner.

**Owner**: `devtools/verify_layering.py:1-12` states the doctrine and audits
it; the inventory and its scope live in `docs/plans/layering.yaml:31-44`. Note
what that manifest says about itself: the inventory proves every facade module
*declares* what it writes, and a separate checked-in census
(`docs/plans/writer-module-census-baseline.json`) holds the "only these
modules write" half by ratchet. Neither proves process-level sole writership.

**Owning gate**: `layering`

**Observable failure**: a module mutating two tiers with one interruption
story, so a crash between the two writes leaves an archive no reader can
classify. The gate names this as `writer_module_mixed_file`; an unmarked
mutation is `writer_module_unmarked_mutation` and a new uninventoried write
path is `writer_module_uncensused_mutation`.

**Change procedure**: add the module to the manifest with its tier,
durability, and interruption semantics, and put the marker in its docstring,
in the same change that adds the mutation. A module that must span tiers needs
the twin-write contract recorded in the manifest, not a comment.

## Doctrine: finding provenance

**Invariant**: a finding is an ordinary durable assertion carrying a
`polylogue.finding.v1` value with its own evidence refs, and it is rejected at
the write boundary if those refs, its statistic, or its declared negative
controls do not hold up. Provenance is queryable, not prose.

**Owner**: the write-boundary projection and its refusals are
`polylogue/storage/sqlite/archive_tiers/user_write.py:1593-1612`; the finding
vocabulary it validates against is at
`polylogue/storage/sqlite/archive_tiers/user_write.py:1524-1526`. A declared
negative control is validated by
`polylogue/analysis/judgment/controls.py:62-72`, which refuses a deliberately
divergent baseline rather than storing it as a control. The queryable
re-derivation is `polylogue/storage/sqlite/finding_provenance.py:1-16`, whose
own docstring records what it does not yet carry.

**Owning gate**: none -- enforcement is a write-boundary refusal in the user
tier, proven by tests, not by a repository gate. See "Open questions".

**Observable failure**: a finding is stored whose `query_ref` or
`result_set_ref` names nothing, so the number survives but the evidence that
produced it cannot be re-resolved. Or an `unrelated_cohort` control is
accepted while leaving frame confounds unchecked, which makes a divergent
baseline read as a passed control.

**Change procedure**: finding fields are additive inside the
`polylogue.finding.v1` value, so a new optional field needs no user-tier
migration — but it does need its validation in the same projection function,
because the refusal *is* the contract. Adding a finding kind means extending
the frozen vocabulary set, which is a write-boundary vocabulary change, not a
durable schema change.

## Doctrine: degradation

**Invariant**: degrade loudly, once. An operation decides its own terminal
outcome from facts only it holds, and a named gap outranks an empty scope, so
zero rows behind a gap is never reported as an empty archive. Improvised
`except sqlite3` handlers that return a fabricated empty projection are a
shrinking, anchored census — not an accepted pattern.

**Owner**: `polylogue/surfaces/outcome.py:1-14` owns the terminal-outcome
contract and `polylogue/surfaces/outcome.py:64-82` owns the precedence
(error, then degraded, then empty). The census over hand-written SQLite
degradation handlers is `devtools/sqlite_degradation.py:1-24`, wired into the
layering gate through `docs/plans/layering.yaml:27-30`.

**Owning gate**: `layering`

**Observable failure**: a surface reports `empty` for a scope it could not
actually read, making a broken reader indistinguishable from an archive that
holds nothing. In the census's own words, the handler's zero or empty list is
indistinguishable from a real answer at the call site.

**Change procedure**: the census is content-anchored by normalized handler
text, so it may shrink and never grow. Converting a seam to the typed route
means dropping its anchors from the baseline in the same change. A new
degradation handler is a review decision, not a baseline bump.

## Doctrine: non-goals and revisit triggers

**Invariant**: a recorded non-goal stays recorded with the condition that
would reopen it. This section is the register; before it existed, the four
non-goals below survived only as tracker comment text after
`docs/execution-plan.md` was retired without the cross-check its own
retirement note required.

**Owner**: this section. Each entry is affirmed against current source with a
citation, or recorded as lapsed with the reason.

**Owning gate**: `atlas`

That gate covers the citations below and nothing else. No gate decides whether
a non-goal is still the right call; a reviewer does.

**Observable failure**: a non-goal is silently carried forward as if it had
been re-checked, or is quietly abandoned by a change that never argued
against it.

**Change procedure**: affirm, or record as lapsed with the reason, in the
change that makes it lapse. Do not delete an entry to make it stop being true.

### Not a general desktop-automation framework

**Status**: affirmed. Polylogue owns branch-local capture plumbing and exposes
inspectable control points; it does not drive the desktop. Browser evidence
arrives through an opt-in MV3 extension posting to a local Python receiver
(`polylogue/browser_capture/receiver.py:1-2`), and the recorded decision
explicitly rejects headless automation
(`docs/architecture-spine.md:134-142`).

**Revisit trigger**: a real cross-tool orchestration need that no existing
local tool already owns.

### OTLP export is a projection, not internal authority

**Status**: affirmed. The archive's own tables remain the source of truth and
the OTel module is an outbound mapping only; its docstring says so directly
(`polylogue/telemetry/otel_projection.py:1-6`).

**Revisit trigger**: a proposal to make OTLP an *inbound* route, or to have a
reader resolve archive truth through an exported span rather than a row.

### Context images and bundles are evidence-backed projections, never authority

**Status**: affirmed. The compiler is deliberately thin — it introduces no
durable memory store and no new handoff ontology, composing refs, rows, and
report transforms into one image (`polylogue/context/compiler.py:1-8`).
Omissions are typed rather than silent, and admitted material carries a trust
class derived from provenance, not from the bundle's own claim (see the
injected-context trust stanza).

**Revisit trigger**: any surface that presents a compiled context image as a
queryable record, or that lets a bundle's contents grant authority its source
material did not have.

### No hard-coded one-off render palette

**Status**: affirmed with a named residual. `polylogue/ui/theme.py:1-6` is the
declared single source of truth for provider, role, status, and theme colors
across CLI, HTML, and daemon web surfaces, and the WebUI consumes generated
tokens rather than a second palette (`webui/src/generated/tokens.css:1`). The
residual: nine per-tool-kind card accents are still literal CSS colour
keywords in the stylesheet rather than generated tokens
(`webui/src/styles.css:435-443`). No gate enforces palette centralization
anywhere, so this entry rests on review.

**Revisit trigger**: a second palette appearing outside `polylogue/ui/theme.py`
and the generated token file — at which point the choice is to generate it or
to retire this non-goal, not to leave both.

## Doctrine: injected-context trust

**Invariant**: trust is derived from authenticated provenance *and* the
source's own authority; content can never raise its own trust class. Assertion
prose is not eligible for `system` trust at all. A context source supplies
candidates; it cannot allocate budget or grant trust.

**Owner**: `polylogue/core/assertions.py:72-101` derives the trust class and
treats an assertion-controlled context policy as a capability cap rather than
a source of authority. The provider contract is
`polylogue/context/scheduler.py:48-53`, and only an explicitly adopted,
scoped, unexpired operator policy may enter instructions
(`polylogue/context/scheduler.py:163-176`). The preamble keeps the partition
structural rather than textual: operator guidance and quoted evidence are
different fields, not different prefixes
(`polylogue/context/preamble.py:551-567`).

**Owning gate**: none -- enforced in production code paths and proven by
tests under `tests/unit/context/`, with no repository gate. See
"Open questions".

**Observable failure**: quoted third-party content is rendered where operator
instructions go, so an archived message can issue directives to a later agent.
A deny-list of phrasings does not prevent this and is not the mechanism here;
the structural partition is.

**Change procedure**: a new context source that wants operator-class output
must earn it through authenticated provenance in the derivation, not through a
policy field it sets itself. Adding a trust class means extending the literal
and every branch that switches on it in the same change.

## Doctrine: unification

**Invariant**: before a new declaration family exists, five compatibility
dimensions are answered — identity, lifecycle, authority, access result shape,
durability. An exact match with an existing family must be reused or
explicitly justified, and asserting a *new durable* object always requires a
justification. Declarations sharing a family id must agree on all five.

**Owner**: the dimensions are
`polylogue/declarations/models.py:22-41`, including the difference report used
in refusals. The registry refuses a mismatched family member at registration
time (`polylogue/declarations/registry.py:64-78`), which means any route that
builds a registry fails — including the bindings gate. The interview and its
two refusals are `devtools/scaffold.py:36-48` and
`devtools/scaffold.py:519-527`.

**Owning gate**: `declaration-bindings`

**Observable failure**: two things that differ in authority or durability end
up in one family, so a read that is safe for one becomes a claim the other
cannot support. Worked examples, all anchored in current source:

- **Accepted reuse**: several daemon read routes share one family because they
  genuinely match on all five dimensions — same identity kind, stable
  lifecycle, daemon-read authority, read-only durability — and differ only in
  envelope shape, which is itself one of the five
  (`polylogue/daemon/route_contracts.py:165`;
  `polylogue/daemon/route_contracts.py:193`).
- **Rejected: query run into context delivery.** A query object is durable,
  content-addressed, and re-resolvable
  (`polylogue/storage/sqlite/query_objects.py:1`); a context delivery decision
  is a scheduler ledger row in the disposable ops tier
  (`polylogue/context/scheduler.py:90-98`). They differ on identity,
  durability and authority at once. Flattening them would make a disposable
  delivery record look like evidence a finding could cite.
- **Rejected: experiment into query.** Both are analysis definitions and share
  one protocol, but they stay distinct kinds
  (`polylogue/core/analysis_contracts.py:52-62`) because a causal claim
  requires a receipt naming assignment, exposure, frame, stopping and outcome
  refs, and is refused without one
  (`polylogue/core/analysis_contracts.py:505-511`). A query result cannot
  satisfy that, so unifying the two would make causal language reachable from
  an observational row.
- **Rejected: evidence value into result storage.** `EvidenceValue` is a
  storage-free protocol embedded by owning models and explicitly does not own
  persistence or a universal registry
  (`polylogue/core/evidence_value.py:1-8`); the same boundary is restated for
  analysis definitions (`polylogue/core/analysis_contracts.py:1-22`). Giving
  it a row would create the universal ledger both modules refuse.

**Change procedure**: answer the interview with `devtools schema new` rather
than hand-writing a declaration. A new durable object or registry needs a
recorded justification before generation; that refusal is the point, because
durable state is the one decision a scaffold must never make silently.

## Open questions

These are recorded as gaps rather than answered, because current source does
not establish an answer.

- **Finding provenance has no gate.** The write boundary refuses a malformed
  finding, and tests prove the refusal, but nothing at the repository level
  detects a *new* finding-shaped write path that bypasses the projection
  function. The original hook for this was mis-filed against a
  docs-publishing lane and was never re-filed.
- **Injected-context trust has no gate.** The derivation and the structural
  partition are production code proven by tests; there is no repository check
  that a newly registered context source cannot emit operator-class output.
- **Process-level sole writership is not proven by the writer gate.** The
  layering gate proves declaration and inventories mutation sites. That a
  live CLI or API caller cannot write outside the daemon coordinator is a
  separate, weaker claim; read the actual route before asserting it.
- **cpf names a sixth unification dimension that code does not carry.** The
  epic's criteria list "remaining domain semantics" alongside the five
  compatibility dimensions. In source there is no sixth dimension: the family's
  own reuse question and a free-text `justification` carry it
  (`devtools/scaffold.py:36-48`), and nothing validates that text. Either it is
  a dimension and belongs in `CompatibilityKey`, or it is review material and
  the sixth item should be dropped from the criteria.
- **A fifth salvaged candidate is still unhomed.** The same tracker comment
  that carried the four non-goals above also carried a closure-posture rule:
  an agent without the operator's live deployment and archive may patch source,
  docs and fixtures, but cannot certify deployed truth. It is not a non-goal,
  so it does not belong in the register; it would belong with a degraded-modes
  or closure-discipline doctrine that does not exist yet. Recorded here so the
  retirement that orphaned it does not orphan it twice.
- **Vocabulary drift is undetected.** No declaration surface exists that would
  let this sheet's terms be diffed against a generated vocabulary, so nothing
  catches a stanza that keeps using a term source has renamed. This is a named
  evidence gap, not scheduled work.
