# Polylogue Publication Control-Plane Program

Date: 2026-03-22
Status: executed subprogram
Role: executed publication/read-model convergence slice for static-site output

## Purpose

Turn `polylogue site` from a filesystem-side-effect command that returned a
loose counts dict into a first-class publication surface with:

- typed build semantics
- durable persisted publication manifests
- shared output-manifest scanning
- explicit reuse of existing control-plane truth such as latest run and
  artifact-proof summaries

This is the concrete Step 5 execution slice that followed the already-landed:

- honest execution contract
- typed QA composition
- source-boundary cleanup
- artifact/cohort/proof control plane

## What Was Executed

### 1. Typed Site Publication Manifest

Added a typed publication model for site builds:

- `SitePublicationManifest`
- `SiteOutputSummary`
- `ArchivePublicationSummary`
- `PublicationRunSummary`
- `ArtifactProofSummary`
- shared `OutputManifest`

The site builder now returns this typed manifest instead of a loose
`{"conversations": ..., "index_pages": ...}` dict.

### 2. Honest Site Build Semantics

The site build now distinguishes:

- rendered conversation pages
- reused conversation pages
- failed conversation pages
- root/provider/dashboard page counts
- search materialization status

This makes incremental builds and search outcomes inspectable instead of only
implied.

### 3. Durable Publication Persistence

Added persisted publication records in SQLite:

- `publications` table
- backend/query-store/repository support for recording and fetching the latest
  publication manifest

The site builder now records each completed build as a durable publication
manifest.

### 4. Shared Output Manifest Scanner

The output-artifact manifest scanner is now shared instead of duplicated:

- static-site publication uses it for `site-manifest.json`
- showcase report generation uses the same scanner for
  `showcase-manifest.json`

Both explicitly exclude the manifest file itself from the captured artifact
set, so the contract is honest and stable.

### 5. Operator Surface Convergence

`polylogue site` now:

- writes `site-manifest.json`
- can emit the typed manifest via `--json`
- reports structured build truth in human output rather than re-deriving it

## Main Outcome

Publication is now a real read-model/control-plane surface rather than a
monolithic side-effecting builder with a thin CLI wrapper.

The site surface now consumes:

- repository summary/query truth
- latest run truth
- durable artifact-proof truth
- shared output-artifact manifest truth

## What Still Remains Open

This did **not** finish the entire broad “publication and repo-shape cleanup”
frontier. The main remaining work is:

- slimming `site/builder.py` internally
- deciding how much site/dashboard publication should be decomposed further
- repo-shape/document/generated-artifact cleanup beyond the already-completed
  docs regrouping
- extending publication manifests to other publication/reporting surfaces if
  they merit durable storage
