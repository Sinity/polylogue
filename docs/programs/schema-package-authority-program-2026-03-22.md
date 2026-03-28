# Schema Package Authority Program

Date: 2026-03-22
Status: current execution program
Role: canonical live queue for schema package/version authority correction

See also:

- `intentional-forward-program-2026-03-21.md`
- `artifact-cohort-control-plane-program-2026-03-21.md`
- `.claude/scratch/018-wave0-schema-package-design.md`
- `.claude/scratch/026-schema-taxonomy-and-versioning.md`

This program absorbs the still-open schema package/version correction lane that
was previously split between scratch notes and broader March 19 planning docs.

## Why This Is Now The Main Frontier

Polylogue is already materially package-aware:

- schema storage is package-shaped
- runtime payload resolution is package-aware
- parser dispatch uses package element kinds
- proof/cohort reporting records resolved package versions and element kinds
- synthetic generation and schema CLI already understand package structure

The remaining problem is not lack of package support. It is incomplete package
authority.

Before this program, the repo still had several non-canonical seams:

- bundled provider schemas were still committed as flat `provider.schema.json.gz`
  files
- the registry still carried flat baseline/version fallback loaders
- direct readers and tests still bypassed the package registry and opened flat
  schema blobs
- package chronology/evidence semantics were still weaker than the runtime and
  proof layers now deserve

The goal of this program is to make schema packages the only real schema truth
surface.

## Guiding Rule

Do not maintain two schema authorities.

Clusters remain evidence.
Packages become authority.

That means:

- no flat packaged schema baseline path
- no flat runtime version fallback path
- no direct file consumers pretending package layout is optional

## Program Order

### Step 1: Bundled Package Canonicalization

Goal:

- move the committed provider schemas under `polylogue/schemas/providers/` into
  the same package layout used by generated/runtime schemas

Scope:

- `polylogue/schemas/providers/<provider>/catalog.json`
- `polylogue/schemas/providers/<provider>/versions/v1/package.json`
- `polylogue/schemas/providers/<provider>/versions/v1/elements/*.schema.json.gz`
- registry loading from bundled package roots

Status:

- executed in this working pass

Delivered:

- bundled provider schemas now live as package directories
- registry reads bundled package catalogs directly
- flat bundled schema blobs removed
- schema audit, synthetic wiring, validator tests, and direct bundled-schema
  checks now consume package layout or registry APIs

### Step 2: Flat Fallback Deletion

Goal:

- remove remaining registry fallback truth for flat baseline/versioned schema
  files

Scope:

- `SchemaRegistry.get_element_schema()`
- `SchemaRegistry.list_versions()`
- `SchemaRegistry.list_providers()`
- package age semantics

Status:

- executed in this working pass

Delivered:

- flat baseline/version fallback loaders removed
- provider listing now comes from package catalogs
- package chronology now seeds from schema generation timestamps when explicitly
  present

### Step 3: Package Chronology And Evidence Normalization

Goal:

- make package manifests reflect observed corpus chronology and evidence rather
  than generation-time placeholders

Why this is next:

- runtime package authority is now canonical
- generation evidence is the strongest remaining place where package truth is
  still weaker than it should be

Scope:

- `schemas/schema_generation.py`
- `schemas/sampling.py`
- cluster/package first-seen/last-seen windows
- exact structure ids vs profile family ids
- bundle-scope evidence in package manifests
- cluster-to-package assignment evidence

Required outcomes:

- package `first_seen` / `last_seen` come from observed artifact chronology
- package manifests record enough evidence to explain why a package exists
- record-stream providers stop looking like flat record-schema buckets

Status:

- executed in this working pass

Delivered so far:

- package manifests now expose `anchor_profile_family_id` and
  `profile_family_ids`
- element manifests now expose `first_seen`, `last_seen`,
  `bundle_scope_count`, and `bundle_scopes`
- generated element schemas now carry:
  - `x-polylogue-element-first-seen`
  - `x-polylogue-element-last-seen`
  - `x-polylogue-element-bundle-scope-count`
  - `x-polylogue-anchor-profile-family-id`
  - `x-polylogue-package-profile-family-ids`
- schema CLI list/explain output now exposes package family/scope/window
  evidence more directly

### Step 4: Operator Terminology Convergence

Goal:

- make schema CLI and related operator surfaces speak the package model
  consistently

Scope:

- `polylogue schema list`
- `polylogue schema explain`
- `polylogue schema compare`
- cluster promotion/explanation wording
- docs that still narrate “single schema per provider” or “flat versions”

Required outcomes:

- operators see package/version/element language as the default truth
- cluster language is clearly evidence/provenance language, not release
  authority language

Status:

- executed in this working pass

Delivered so far:

- cluster manifests now record `promoted_package_version` instead of the
  misleading `schema_version`
- schema CLI wording now distinguishes:
  - evidence clusters
  - evidence manifests
  - promoted packages
- cluster evidence still remains visible, but no longer presents itself as the
  release authority
- generated and promoted schemas now carry package/evidence metadata using:
  - `x-polylogue-anchor-profile-family-id`
  - `x-polylogue-profile-family-ids`
  - `x-polylogue-package-profile-family-ids`
  - `x-polylogue-observed-artifact-count`
  - `x-polylogue-evidence-confidence`
- schema explain output now surfaces those package/evidence fields directly
  instead of the stale `x-polylogue-cluster-*` metadata family

### Step 5: Package-Aware Backfill, Proof, And Synthetic Convergence

Goal:

- prove that the package model drives all downstream consumers cleanly

Scope:

- proof/cohort surfaces over resolved package authority
- package-aware synthetic generation and roundtrip checks
- backfill/migration checks for existing archives
- QA/report surfaces that expose package truth directly

Required outcomes:

- `check`, `schema`, `qa`, and synthetic tests agree on package truth
- Claude subagent streams and sidecars remain package elements, not fake
  top-level conversation types

Status:

- in progress
- partly executed in this working pass

Delivered so far:

- artifact proof summaries now aggregate:
  - resolved package versions
  - resolved element kinds
  - resolution reasons
- `polylogue check --proof` now surfaces that package-resolution truth in plain
  operator output instead of leaving it buried in provider JSON only
- QA session/report generation now carries the same package-resolution summary
  through JSON, text summary, and Markdown artifact-proof sections
- site publication manifests now embed the same aggregate package-resolution
  truth through `artifact_proof`
- proof/cohort hydration now re-resolves existing parseable artifact
  observations, so long-lived archives do not keep stale package resolution
  forever after schema authority improves

## Architectural Rules

### 1. Packages Are The Schema Runtime Contract

If a consumer needs a schema, it should ask the registry for a package element,
not inspect files directly unless it is explicitly testing the bundled assets
themselves.

### 2. Bundled And Runtime Schemas Share One Layout

Bundled provider schemas may remain committed in-repo, and runtime/operator
schemas may remain writable in the data home. But they must share the same
package shape.

### 3. Cluster Data Is Evidence, Not Release Truth

Cluster manifests still matter, but they exist to explain and audit package
assembly, not to stand beside packages as a second public schema authority.

### 4. Chronology Must Be Observed, Not Invented

Generation-time timestamps are not acceptable substitutes for observed corpus
windows once the package model is authoritative.

## Exit Criteria

- no runtime code path depends on flat provider schema files
- bundled provider schemas and operator-generated schemas share one package
  layout
- schema registry/provider listing/version lookup are package-catalog driven
- package manifests carry honest chronology and evidence
- CLI/proof/QA/synthetic surfaces all speak the package model consistently
