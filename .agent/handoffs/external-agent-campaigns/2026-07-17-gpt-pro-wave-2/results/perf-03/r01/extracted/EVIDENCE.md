# Evidence and authority record

## Authority resolution

The attached Polylogue project-state archive is the code authority for this
work. Its manifest identifies branch `master` at commit
`536a53efac0cbe4a2473ad379e4db49ef3fce74d`, generated
`2026-07-17T180950Z`, with `dirty: true`.

The same archive also contains:

- a branch-delta summary whose base and merge base are both the named commit;
- a zero-byte branch-delta patch;
- a zero-byte branch changed-file list; and
- a zero-byte branch-only commit log.

The tracked source reconstructed from the archive matches the named commit. The
implementation patch therefore targets that commit exactly. The dirty metadata
flag is reported rather than overridden, but no non-commit branch delta was
available to apply or preserve.

The supplied repository/project-state archive and its copied source are not
members of the result ZIP. The ZIP contains only the handoff documents and one
apply-ready patch.

## Repository instructions

`CLAUDE.md` and `AGENTS.md` establish these relevant constraints:

- preserve substrate-first dependency direction;
- use existing lazy Click registration and account for generated-doc parameter
  discovery;
- keep generated projections synchronized;
- run Ruff, strict mypy, and `devtools render all --check` for changed production
  work; and
- avoid parallel product contracts where a canonical surface already exists.

The implementation follows those boundaries by keeping the daemon handshake on
paths/config/schema constants/stdlib transport, moving canonical projection to a
surface module, and deferring API/model/storage layers until direct execution.

## Current-source findings

### Root and query startup

The snapshot root route reached more than Click registration. Transitive edges
included build/version resolution, query contracts and Pydantic models, API
facades, runtime services, source discovery, storage repositories, SQLite
archive-tier packages, status machinery, and grammar/model code.

The existing Click registry was already lazy at the command object level, but
package initializers and shared modules still crossed heavy boundaries. The
correct repair was therefore not another command framework; it was to make the
public package/facade, annotation, schema-version, and executor boundaries lazy.

The generator concern from the mission was checked against current source:
`devtools render cli-reference` obtains the real command parameters rather than
assuming a lazy proxy's shallow metadata. The final aggregate render check is
green and `docs/cli-reference.md` is unchanged.

### Grammar

`polylogue/archive/query/expression.py` owns the executable Lark grammar and
lowering logic. Parser construction belonged in that module, but did not need to
happen on module import. No separate grammar framework or duplicate parser was
introduced. The persistent cache uses Lark's supported cache parameter and keys
the exact current grammar/configuration.

Tests exercise parser laziness, process-to-process cache reuse, a changed grammar
selecting a different key, corrupt-cache deletion/regeneration, empty-XDG
semantics, and no-cache fallback on I/O failure.

### Archive-tier schema imports

The archive-tier package initializer was a transitive startup hotspot because
schema-version consumers loaded DDL builders for multiple databases. Schema
versions are simple substrate constants, so they were split into a leaf module.
The DDL mapping now resolves individual builders on demand. Existing DDL and
fresh-database tests verify that this import repair did not change schemas.

### Daemon discovery and health

The status path contained the established daemon-discovery/health precedent, and
the repository already had a production AF_UNIX HTTP server and client surface.
The new read adapter reuses those concepts rather than adding another daemon,
protocol, or HTTP dependency.

The non-negotiable identity tuple from current Beads/source is:

- resolved active archive root;
- current index schema version; and
- current Polylogue build version.

The configured bearer token is also part of the usable target. The client checks
socket existence before resolving expensive identity. It compares archive and
schema from health before resolving the local Git-backed build version. Client
and server socket discovery both map an unset or empty `XDG_RUNTIME_DIR` to
`/tmp`; a dedicated regression prevents either side from reverting to a
relative socket path. This
ordering is tested with sentinels that fail if config/source discovery or build
resolution happens too early.

### Query transaction and private read route

Current source has a canonical archive read context/transaction boundary and
canonical direct session/message payload construction. The new private
`POST /api/cli/read` route enters that real boundary, resolves the requested
session there, and supports only `summary` and finite `messages` views.

The route is not a parallel public API. It is a credential-gated CLI transport
adapter registered in the daemon's route contract and security matrices. It
returns the same native payload used by direct execution, allowing the CLI
process to avoid importing API facades and Pydantic payload models on a daemon
hit.

### Output projection

Earlier daemon routing had separate list/search normalization from direct CLI
projection. Expanded parity work exposed two concrete risks:

- page cursors could be shaped differently from the canonical opaque cursor;
- ranked-search rows could omit or place canonical fields differently.

`polylogue/surfaces/archive_session.py` now owns the shared projection. The MCP
archive-row contract remains separate and is explicitly tested not to widen.

The finite message parity audit also checked stored block metadata and
attachments. Current direct paginated reads deliberately emit `None` for the
retired metadata/media fields and do not hydrate attachments. The daemon path
preserves that contract; it does not "improve" one side and silently break
machine parity.

## Beads findings

### `polylogue-20d` — interactive performance epic

The epic describes cold CLI imports and cold archive I/O as structural front-door
latency and orders import deferral before/alongside the UDS daemon path. This
mission directly implements that combined fallback/hot-path architecture.

### `polylogue-20d.1` — CLI-to-daemon fast path

This record requires:

- UDS transport;
- archive/schema/version matching;
- approximately 100 ms bounded health detection;
- silent in-process fallback;
- read-only routing;
- list/read/messages/facets coverage;
- `--no-daemon` and environment escape hatches;
- verbose daemon provenance; and
- production golden parity.

The patch implements these requirements for the proven CLI shapes and extends
status/search/bare triage/public refs. Each UDS request has a 45 ms timeout, so
the normal health-plus-read sequence has a nominal sub-100 ms two-request
budget.

### `polylogue-fko9` — exact read/messages parity follow-up

This record identifies summary and messages as the missing fast-path surfaces
and requires a real UDS server test. The patch closes that technical gap for
exact summary and finite JSON/NDJSON messages. It deliberately leaves the
separately shaped raw/context/context-image/neighbors/correlation/temporal/
chronicle views direct-only.

### `polylogue-20d.2` — defer heavy CLI imports

The record is closed in Beads because earlier named help targets and gates
landed. The mission's attached snapshot still showed residual heavy imports on
root/query/status and direct-read module boundaries. Current source and measured
import graphs take precedence over the closure narrative, so this patch removes
those remaining edges without reopening or replacing the existing help-latency
framework.

### `polylogue-20d.12` — daemon result cache and warming

This is an explicitly separate feature: query/epoch-keyed in-memory results,
invalidation, warming, memory cap, and metrics. It was inspected to avoid scope
collision. No cache layer was invented here. This patch makes the already-warm
daemon cheap to reach and leaves result memoization to its owning bead.

## History findings

Relevant history includes:

- `3082c72f0 feat(cli): add hot daemon read routing (#2827)`
- `81bfedd87 perf: improve interactive CLI and coordination latency (#2784)`
- later bounded daemon-probe and parity work in the current all-refs history

That lineage established the direction: daemon-owned query execution with a
thin client, strict identity matching, and direct fallback. The present patch
extends and hardens that design rather than replacing it.

## Contradictions and resolutions

### Manifest dirty flag versus empty branch delta

The manifest says dirty, while every captured branch-delta artifact is empty and
the reconstructed tracked tree matches the commit. Resolution: name both facts,
base the patch on the commit, and avoid fabricating unavailable dirty bytes.

### Older route suggestion versus current canonical payload

The `polylogue-fko9` design text suggests reusing a workbench-style
`GET /api/sessions/:id/read?view=...` route. Current source showed that route's
envelope and imports were not the import-light canonical CLI summary/message
contract required here. Resolution: add one narrow private `POST /api/cli/read`
route inside the existing daemon framework and archive transaction boundary.
This avoids client-side API/Pydantic initialization and is covered by OpenAPI,
route-contract, auth, and real UDS tests.

### Literal byte identity versus provenance/timestamps

Existing product behavior documents daemon provenance in list/search envelopes,
and facets independently stamp generation time. Resolution: compare machine
payloads structurally after removing only the documented `source` field or the
independent facets timestamp. Exact summary, messages, and public refs compare
field-for-field. Textual formats compare bytes where provenance is not part of
the text contract.

### Maximal proxy coverage versus semantic safety

Proxying every query/read option would be broader but not coherent. Current
direct behavior owns validation, mutation modes, stream semantics, special
views, semantic-card rendering, and output destinations. Resolution: use an
explicit eligibility predicate and fall back before compilation for only the
shapes whose production parity is proven. Unsupported shapes are not errors;
they use the existing executor unchanged.

## Rejected alternatives

- Importing `httpx` or daemon/API payload models into the CLI client: rejected
  because it recreates the startup cost the fast path is meant to avoid.
- Constructing `RuntimeServices` to find the archive: rejected because it
  discovers sources and initializes unrelated service dependencies.
- Trusting socket existence alone: rejected because stale/wrong-archive daemons
  are a documented live trap.
- Resolving Git build identity before the socket/health checks: rejected because
  the common daemon-down case must remain nearly free.
- Proxying writes: rejected; user-tier writes remain direct.
- Proxying all read views without golden tests: rejected; distinct contracts
  remain direct until proven.
- Adding the `polylogue-20d.12` result cache: rejected as independently owned
  scope with separate invalidation/memory/SLO acceptance criteria.
- Shipping complete replacement files: rejected because `PATCH.diff` fully and
  unambiguously expresses the changes, including binary generated artifacts.

## Generated evidence

The source changes require regeneration of:

- `docs/openapi/search.yaml` for the private CLI read route;
- `docs/plans/topology-target.yaml` for the new modules/dependencies; and
- `docs/topology-status.md` for the topology projection.

The aggregate generator reports every target synchronized. The CLI reference is
unchanged and synchronized, proving the lazy command surfaces remain visible to
document tooling.

## Final verification evidence

On the clean-applied tree:

- `git apply --check` and `git apply --binary` succeeded against the exact commit.
- All 49 changed files matched the candidate by SHA-256.
- `ruff format --check .` reported 2,211 files already formatted.
- `ruff check .` passed.
- strict mypy passed for all 38 changed production modules.
- every `devtools render all --check` target passed.
- `git diff --check` passed.
- 1,536 tests passed and one existing optional `sqlite-vec` DDL case skipped.

The measured import, grammar, root-wall, and exact-read data are recorded in
`HANDOFF.md`, with host-dependent numbers explicitly labelled unverified.
