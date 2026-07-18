# Source, Beads, and History Evidence

## Authority hierarchy used

1. Supplied project-state snapshot at commit `536a53efac0cbe4a2473ad379e4db49ef3fce74d`.
2. Repository instructions and current source/tests.
3. Complete current Beads records from `.beads/issues.jsonl`.
4. Relevant Git history.
5. Prior external-attempt metadata only; no prior code was available or trusted.

Where older descriptions named stale line numbers or caller counts, the current source graph was treated as authoritative.

## Beads findings

### `polylogue-9gh1`

The epic identifies three independent structural failures: two non-interoperating configuration systems, full replacement of nested health tables, and direct environment bypasses. Its closure condition requires every inventoried TOML setting to affect its real runtime consumer and requires a fixture-driven test that catches recurrence for a new setting.

Patch response:

- one layered resolver and immutable runtime projection;
- recursive merge for nested table entries;
- caller migration plus an AST bypass guard;
- inventory-derived scalar TOML tests and real-route archive/Voyage tests.

### `polylogue-fd2s`

The decided architecture explicitly requires `load_polylogue_config` to be the sole five-layer resolver and `ResolvedRuntimeConfig`/`ResolvedArchivePaths` to be injected at CLI, daemon, MCP, API, and maintenance roots. It also forbids compatibility projections from rereading environment, cwd, home, or `polylogue.paths`.

Patch response:

- `ResolvedRuntimeConfig` captures settings, paths, source definitions, projections, bootstrap roots, and archive identity;
- `Config`, `IndexConfig`, and `DriveConfig` are projections;
- `polylogue.paths` delegates to the installed runtime;
- all named roots either resolve/install or receive the runtime;
- mutation tests prove post-construction ambient changes do not alter paths or secrets.

The Bead note says archive identity must not be inferred from archive-root text alone. The runtime therefore carries the shipped active `ArchiveIdentity` object rather than synthesizing identity solely from the path string.

### `polylogue-cxlk`

The Bead requires site/user health-table merging to preserve top-level siblings and nested family siblings, while leaving array-of-table replacement intact.

Patch response:

- inventory table entries recursively merge mappings;
- `families` recursively merges because it is a nested mapping;
- generated two-layer fixtures cover both `health_convergence_debt` and `health_cursor_lag`;
- `subscription_plans` has an explicit replacement test.

### `polylogue-uu8r`

The Bead requires every direct environment read to be classified and actual Polylogue product settings to route through the layered resolver.

Patch response:

- inventory was expanded for previously ambient runtime tunables;
- direct call sites were migrated;
- remaining reads are shell/terminal/session/systemd/external-tool or config-selection boundary inputs;
- the AST regression guard compares literal reads with every inventoried environment variable.

## Current-source findings that drove the implementation

### Parallel path authority

`polylogue/paths/_roots.py` independently computed archive and XDG paths, and compatibility `Config`/service construction consumed those helpers. This made `archive.root` in TOML diagnostics-visible but runtime-dead for many routes.

The patch removes that independent resolution logic. Path helpers now expose fields already captured by `ResolvedArchivePaths`.

### Parallel provider authority

The Voyage key had multiple ambient paths: Click env binding, `IndexConfig.from_env`, embedding-stage checks, and provider-factory fallback. This let TOML merge correctly while the real provider still behaved as if only the environment existed.

The patch carries the resolved key through `IndexConfig`, the real embedding stage, and the provider factory. The factory has no raw environment fallback.

### Provenance defect beyond the stated examples

The existing merge bookkeeping inferred a layer only when a value differed. If a user explicitly supplied a value equal to the lower layer, provenance remained incorrectly attributed to the lower layer/default.

The patch records layer provenance from key presence. A regression case supplies the same archive root through env and CLI and requires the final layer to be `cli`.

### Click default precedence defect

A composition root can violate the five layers even with a correct loader if it passes every Click default as a CLI override. The daemon and standalone browser-capture roots now inspect Click `ParameterSource` and include only explicitly supplied values in layer five.

### Source projection omission

Explicit source roots were present in settings but not carried consistently through the runtime source projection. The patch includes them in `ResolvedSourcePaths`/source construction.

### Reader template omission

`POLYLOGUE_READER_AGENT_TEMPLATES` was a real direct product setting missed by the first implementation pass. It is now inventoried as `reader.agent_templates`, supports TOML/environment precedence, and reaches the daemon reader route.

### Status/config re-resolution

Embedding status and several daemon/health consumers called the loader or ambient compatibility helpers during operation. The patch consumes settings attached to the compatibility projection, with an installed-runtime fallback only for explicit old test/library objects that lack the new field.

## Direct-read census after migration

The post-patch grep outside `polylogue/config.py` finds these categories only:

- `COMP_WORDS`: shell completion protocol;
- `POLYLOGUE_CONFIG`: explicit config-file selection/writer boundary;
- session/correlation IDs: request/process metadata;
- development-loop IDs/log directory: invocation metadata;
- demo temporary environment mutation: subprocess isolation;
- `CLAUDE_CONFIG_DIR`, `CODEX_HOME`, hook provider: external tool/provider boundary;
- `TERM` and `COLORFGBG`: terminal capability metadata;
- `INVOCATION_ID`: systemd identity.

No literal direct read of an inventoried runtime setting survives outside `config.py`; the AST test enforces this relationship from the inventory itself.

## History findings

Relevant commits showed the intent and layering evolution:

- `cf1a25ab6` split config/path ownership, creating the historical seam that later diverged;
- `8b2467629` added typed `PolylogueConfig` and environment consolidation but did not replace all legacy consumers;
- `8e6f4c74a` hardened config secret handling;
- `fc70163ad` and `d70b0d3fb` expanded effective-config debt/provenance diagnostics;
- archive/path refactors such as `ac9cfeb0b` and `beff1130b` widened use of path helpers without making TOML the authority.

This history supports replacing the parallel resolver rather than documenting it as intentional compatibility behavior.

## Prior-attempt evidence

The only present prior-attempt artifact was `result.json`. It records:

- state: rejected;
- package: `polylogue-config-closure.zip`;
- size: 35,210,095 bytes;
- note: zero-byte patch and copied repository snapshot.

Verdict: reject all implementation claims/code; reuse only the negative packaging lesson. This package contains no supplied source archive and no copied repository tree. It contains one unified patch and three handoff/evidence documents.

## Contradictions and resolutions

- Older caller counts ranged from roughly 20 to 31. Current source and inventory classification, not the stale count, determined the migration set.
- The mission mentions `IndexConfig.from_env`; current architecture retains compatibility type names but eliminates environment resolution from the projection path rather than preserving that method as authority.
- Some test plans implied every setting needs a bespoke real consumer fixture. The implemented guard uses a generated typed projection test for every scalar TOML inventory entry, then adds real production-route tests for the structural bypasses: archive roots across composition roots and Voyage through embedding execution. This avoids vacuous hand-maintained coverage while still observing the critical real routes.
- The snapshot manifest's dirty marker could not be reconstructed as a tracked patch. The named commit and embedded tracked tree matched; the deliverable therefore targets the named commit exactly and does not claim to preserve unknown ignored/runtime state.

## Patch/package evidence

At package-generation time:

- `PATCH.diff` contains 73 `diff --git` sections;
- patch SHA-256: `9223e943cc1db08f58c1b0b47bd7bc510da95727f81c340e9f7e1643e32e0d81`;
- patch byte size: 227,848 bytes;
- `git apply --check` succeeds against the named commit;
- the applied clean worktree passes `git diff --check` and the 172-test core resolver/closure suite;
- no supplied tarball, project-state archive, copied repository directory, or prior package is present in the deliverable.
