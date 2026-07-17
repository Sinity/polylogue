---
created: 2026-07-16
purpose: Decide canonical configuration composition, bootstrap, paths, and archive identity for L26-L28
status: recommended-decision
project: polylogue
---

# Configuration and path coherence

## Decision

Keep the existing five-layer precedence in `load_polylogue_config` and make it
the sole runtime resolver. Load once at each process composition root, validate
it into an immutable `ResolvedRuntimeConfig`, and inject that object into
services and path consumers.

Retire ambient parallel resolution through legacy `Config`, `get_config`,
`IndexConfig.from_env`, and `polylogue.paths` environment reads. During
migration those APIs become compatibility projections of the already resolved
configuration; they do not read environment variables again.

## Canonical precedence

Low to high:

1. built-in defaults;
2. site TOML;
3. user/project TOML selected by the existing config-path rules;
4. `POLYLOGUE_*` environment variables;
5. explicit CLI overrides.

This ordering is already documented and implemented by
`load_polylogue_config`; it needs composition proof, not redesign. The resolver
records the origin layer and physical source of each value. Secrets are redacted
in diagnostics and receipts. Secret values may originate from protected config,
environment, or credential files, but downstream code consumes the resolved
secret—not its ambient source.

## Bootstrap boundary

A tiny `BootstrapConfig` may read only what is necessary to locate site/user
config and the process's XDG roots. After full resolution, no domain service
consults `os.environ`, current working directory, or `Path.home()` for runtime
meaning.

Relative runtime paths are rejected or resolved once against their declaring
config file, then stored as absolute paths with provenance. Changing CWD after
startup cannot redirect the archive.

## Resolved archive paths

`ResolvedArchivePaths` contains:

- `archive_id` and canonical durable root;
- exact `source.db`, `user.db`, `ops.db`, blob, render, and spool paths;
- active and rollback generation pointers for `index.db` and `embeddings.db`;
- config/source provenance and existence/schema observations;
- disclosure-safe public projections.

All tier files and generation manifests must claim the same archive id. Derived
generation paths may live behind pointers, but a database anchor must not infer
a different archive root merely because it is named `index.db`. A split,
missing, or foreign tier fails closed with a typed archive-identity/path
diagnostic before writes begin.

`active_index_db_path`, `archive_root`, sibling helpers, backup, reset, daemon,
MCP, and tests consume the injected file set. Ambient convenience functions are
allowed only at the outer bootstrap boundary.

## Source discovery and temporal configuration

Source roots are resolved by the same config snapshot and expressed as typed
`Source`/`OriginSpec` entries. Provider detection remains content-driven; a
configured path name does not authorize a parser.

Time zones, cutoffs, and clock policy are explicit inputs. Equivalent instants
normalize through one temporal contract; configuration cannot reintroduce host
wall-clock or local-time differences after resolution.

## Configuration receipt

Each daemon/query/operation run may cite a redacted config snapshot digest and
archive id. Diagnostics show effective value provenance, ignored/malformed
layers, incompatible tier identities, and exact repair actions. Malformed
explicitly selected config fails; optional absent site/user files are benign.
Silently swallowing malformed present configuration is not acceptable for a
daemon start that would otherwise write to a different archive.

## Competitive alternatives

| Alternative | Advantage | Why not chosen |
| --- | --- | --- |
| Patch individual divergent call sites | Small diffs | Leaves two authorities and future drift |
| Keep legacy `Config` and `PolylogueConfig` peers | Compatibility | No principled winner; values can differ within one process |
| Global mutable singleton | Easy access | Test leakage, stale environment, hidden dependencies |
| Re-read environment on every path call | Dynamic overrides | One process can write/read different archives after env/CWD changes |
| Infer archive root from any DB path | Convenient tools | Split-brain across tier generations and foreign files |
| Store all secrets only in TOML | Simple provenance | Unnecessarily excludes protected credential/env integrations |
| Make malformed config always ignorable | Resilient CLI | A writer may silently use defaults and mutate the wrong archive |

## Migration sequence

1. Add `ResolvedRuntimeConfig`/`ResolvedArchivePaths` as projections of the
   existing loader; preserve precedence exactly.
2. Convert daemon and CLI composition roots, then storage/service constructors.
3. Make legacy `Config` a projection and deprecate internal ambient calls.
4. Convert tests to explicit loader/config fixtures; remove env-reading from
   domain objects.
5. Delete redundant path inference only after all call sites and commands use
   the injected file set.

## Required proof

- pairwise and generated layer combinations obey precedence and provenance;
- CLI/daemon/MCP/API created from one snapshot resolve byte-identical paths and
  archive id;
- changing environment or CWD after construction changes nothing;
- foreign/missing tier, symlink/generation switch, relative path, and malformed
  explicit config fail with typed diagnostics before mutation;
- equivalent instant configurations behave identically;
- removing config injection or restoring `IndexConfig.from_env` divergence
  fails a real composition test.

Primary evidence: `polylogue-fd2s`, `polylogue-nkmy`, `polylogue-9itr`;
`polylogue/config.py`, `polylogue/paths/_roots.py`, archive identity/generation
modules, and current CLI/daemon composition roots.
