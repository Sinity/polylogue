# Definition-to-production duplicate-authority census

**Job:** `analysis-05`  
**Revision:** `r01`  
**Source snapshot:** Polylogue `master` at `f654480cadb7cc4c194704e24dfd483199547b35`  
**Snapshot generated:** `2026-07-17T043202Z`

## Executive result

The census found ten decision-relevant authority families. Four are active production collisions with executable witnesses, four are currently aligned but impose broad edit cost or leave security/status claims on a separate authority, one is a legitimate compatibility-copy family with an incomplete validator, and one is a dead declared capability whose current tracker decision is to wire rather than delete.

The four active collisions are:

1. Two shipped MCP cookbook prompts instruct `query_units` to execute sessions-only expressions that the production parser and tool reject as `invalid_query`.
2. The maintenance catalog advertises seven targets, the resumable replay executor dispatches six, and the non-resumable planner already executes all seven. The omitted target fails through the CLI while the CLI exits zero.
3. The five-layer config resolver replaces nested health tables instead of composing them, so a partial user override deletes inherited site thresholds. Legacy `Config` construction and direct environment reads remain parallel effective-value authorities.
4. Two live embeddings schema writers create incompatible `vec0` metadata columns. Whichever writer creates the table first makes the other production writer fail.

The highest-value consolidation is not a generic registry framework. It is four concrete authority repairs, in this order: executable query declarations, maintenance target execution, typed config declaration/resolution, and canonical embeddings DDL. The broader `DeclarationSpec` work can then absorb already-aligned MCP, route, and API projections without creating another shadow catalog.

## Ranked consolidation portfolio

| Rank | Concept | Classification | Current production effect | Canonical decision | Tracker |
|---:|---|---|---|---|---|
| 1 | `query_units` grammar and shipped recipes | **Unresolved collision** | Two product-owned prompt recipes invoke the real MCP tool and receive `invalid_query`; a files-terminal recipe succeeds | Executable query declarations and parser semantics own recipes, schemas, completions, errors, and docs; generated recipes must compile and run | `polylogue-z9gh.3` P0 |
| 2 | Maintenance target identity and execution | **Unresolved collision** | Seven catalog targets, six replay handlers, seven legacy planner handlers; `superseded_raw_snapshots` fails in replay but completes in planner on a fresh archive; CLI returns exit 0 for failed JSON operation | One target execution declaration owns identity, preview, execute, resumability capability, destructive policy, and surface exposure | `polylogue-71ey` P1 |
| 3 | Config declaration, merge, and effective-value resolution | **Unresolved collision** | Partial nested TOML override drops inherited siblings; 16 direct reads of seven inventoried env vars occur outside `config.py`; legacy and layered config systems coexist | One typed key declaration owns default, type, path, env, CLI, secrecy, merge semantics, and provenance; legacy `Config` becomes a derived adapter | `polylogue-9gh1` P1 |
| 4 | Embeddings schema and vector writers | **Unresolved collision** | Tier-first schema rejects the legacy `source_name` insert; runtime-first schema rejects the archive-tier `origin` insert | Archive-tier embeddings DDL is canonical; runtime provider imports a dimension-parameterized canonical builder and uses the archive-tier writer contract | `polylogue-mhx.7` P2 |
| 5 | MCP tool registration, contracts, smoke inputs, and docs | **Unresolved multi-authority; currently aligned** | Admin runtime, expected-name fixture, and envelope contract each contain 104 tools; read smoke map has 66. Generated docs import the test fixture, and repo instructions still say 103 | A production `MCPToolSpec` owns registration, role, schema, result contract, minimal invocation, and docs projection | `polylogue-o21`, `polylogue-t46.8.1` P1 |
| 6 | Daemon routes, auth policy, dispatch, and OpenAPI | **Projection/execution split; currently parity-guarded** | 77 contract patterns equal 77 implemented patterns. GET dispatch consumes contracts; POST/DELETE dispatch re-declares route groups and does not read `auth_policy` | Typed `RouteSpec` drives dispatch and OpenAPI. Bind POST/DELETE entries to contracts before the planned ASGI migration | `polylogue-3utv` P3 |
| 7 | Python facade routing status | **Unresolved validation mirror** | Static status map has 141 entries versus 146 public callables; five methods are absent, including `embedding_preflight` and `embedding_status` | Stable semantic `OperationSpec` or an explicit exclusion classification drives status; method-name introspection is only a census check | `polylogue-s1kr` P2, `polylogue-9e5.31` P1 |
| 8 | Devtools command catalog and direct entry points | **Unresolved bypass** | Three production workflow/hook paths invoke modules or scripts outside the `CommandSpec` dispatcher | CI and hooks invoke catalog names, or a machine-readable sanctioned-bypass manifest records the exception | `polylogue-l8ee` P3 |
| 9 | Durable user-tier fresh schema, migrations, and assertions | **Compatibility copies plus incomplete projection** | Fresh user tier has 15 tables; the hand-written required-table test asserts 13 and omits two current holdout tables | Fresh DDL owns current schema; numbered migrations remain required compatibility copies; tests derive an exact introspected manifest and compare fresh versus upgraded databases | `polylogue-ihp0` P2 |
| 10 | `user_settings` declared storage capability | **Dead declared capability / compatibility schema copy** | Fresh DDL, migration, status probe, and docs exist, but no production read/write helper or consumer exists | Preserve the current tracker decision: keep the table, add a typed key registry and first real consumer; do not delete merely to remove textual duplication | `polylogue-at44` P3 |

## What is already sound

Not every repeated declaration is a defect. The read-view surface has 11 profiles, 11 lightweight handler-metadata entries, and 11 executable handlers. Both registry modules fail fast on missing, extra, or mismatched entries. This is a deliberate import-layer projection with an executable equality gate, so it is not a priority consolidation target.

Durable-tier migrations are also not duplicate authority to delete. They are compatibility copies required by the repository’s durability doctrine. Their current weakness is validation: the fresh schema and every supported upgraded schema need to converge on one introspected current manifest.

The MCP and daemon name/path sets are aligned today. Their problem is the direction of authority and broad edit cost, not a fabricated current count mismatch. The exception is the manual `CLAUDE.md` MCP count, which says 103 while the live admin surface and fixtures contain 104.

## Recommended sequence

First add regression tests that invoke the real consumers and fail on the four active collisions. Then repair query recipes and maintenance dispatch, because both currently hand users or agents an advertised operation that fails. Next fix nested config composition and migrate direct consumers. Consolidate embeddings DDL immediately afterward, before another writer or schema bump increases the data-rebuild blast radius.

Only after those repairs should the general declaration work generate MCP, route, facade, and docs projections. That ordering prevents a new abstraction from faithfully generating today’s contradictions.

## Limitations and residual uncertainty

This analysis used the supplied complete repository snapshot, its Git bundle, Beads export, repository instructions, and focused local execution. It did not inspect a live operator archive, a deployed daemon, browser traffic, hosted CI, or an installed out-of-repository Polylogue skill. Tracker notes state that the installed skill repeats the invalid query recipes, but that installed artifact was not available here; the source-confirmed finding is the two MCP prompts.

The snapshot manifest reports a dirty source tree. Of 3,739 tracked paths at the snapshot commit, the working-tree archive included 2,501 and omitted 1,238, primarily excluded `.agent/` history; every included tracked path matched Git HEAD byte-for-byte. No recommendation depends on an omitted tracked path.

Another iteration has **medium value**. The top four findings already have direct production-route witnesses, so their decision risk is low. A second pass would be most valuable for expanding the census to every schema-writing helper, generated docs arrow, CLI operation map, and config bypass, and for running fresh-versus-upgraded durable-tier parity across all supported versions. It is unlikely to change the first four priorities, but it could reorder ranks five through ten or uncover additional lower-level consumers.
