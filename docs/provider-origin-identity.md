# Provider, Origin, and Source Identity Map

Polylogue ingests AI-work material from exports, local agent session streams,
browser capture, sidecars, and language-server adapters. Several coordinates
can look like "where did this come from?", but they do not own the same
invariant.

This map is the vocabulary boundary for import, query, read views, refs,
provider-package completeness, browser capture, and archive debt work. It is
the Polylogue analogue of Sinex's source-identity demotion map: do not use one
source-shaped value as provider-wire family, public origin, material root,
parser identity, capture mode, disclosure policy, session identity, and runtime
context.

## Doctrine

`Provider` is a provider-wire/parser/schema coordinate. It remains valid where
Polylogue is describing raw export families, parser contracts, provider schema
packages, raw acquisition records, model/pricing-provider metadata, or
embedding-provider configuration.

`Origin` is the public source-origin coordinate for query, read, API, MCP, and
daemon surfaces. If a user asks for "Codex sessions" or filters by source
family, the public field should be `origin`.

`Source` is the source-family descriptor that joins a public family token with
its default runtime root and originating lab. It is useful for source discovery
and lab attribution, but it is not a replacement for archive refs, material
roots, parser provenance, or model identity.

Material source paths and raw artifact refs explain where bytes came from and
how to reacquire or debug them. They are not parser identity, public origin, or
privacy policy by themselves.

Capture mode describes acquisition shape: export file, live local session
stream, hook event, browser extension POST, language-server export, sidecar, or
future API capture. Capture mode affects completeness and risk posture, but it
is not provider vendor.

Archive refs are the stable product identity. Provider-native ids, logical
session roots, topology edges, runs, observed events, context snapshots, and raw
artifact refs are all useful provenance coordinates, but none should silently
replace archive object identity.

## Classification

| Coordinate | Current names and owners | Actual invariant | Public? | Decision |
| --- | --- | --- | --- | --- |
| Provider-wire family | `Provider` in `polylogue/core/enums.py`; canonical runtime/schema helpers in `polylogue/core/provider_identity.py`; parser outputs; provider schema packages | The normalized raw-export/parser/schema family used at provider-wire and older storage boundaries. | Mostly internal | Keep for parsers, schema packages, raw/provider metadata, and storage bridges that still require provider tokens. Do not use it as the public filter vocabulary. |
| Public origin | `Origin` in `polylogue/core/enums.py`; `origin` filters in query spec, CLI, MCP, daemon, and row payloads | User-facing source-origin family such as `claude-code-session`, `codex-session`, `chatgpt-export`, or `aistudio-drive`. | Yes | Use on public read/query/API/MCP/daemon surfaces. It is the preferred coordinate for user-facing source filters. |
| Source family descriptor | `Source` in `polylogue/core/sources.py`; `family`, `runtime_root`, `originating_lab`; `provider_to_source`, `origin_from_provider` bridges | A source family plus its conventional runtime root and lab attribution. | Sometimes | Use for source discovery, lab derivation, and transition bridges. Do not treat `Source` as material path, parser version, model, or archive ref. |
| Material source | configured root, source path, import path, raw artifact path, raw id, blob refs, acquired file identity | Where bytes or records were acquired from and how they can be inspected or reacquired. | Yes when redacted/safe | ImportExplain and raw/debug views should expose this separately from origin/provider. Redact paths when crossing daemon/web/MCP boundaries. |
| Capture mode | export ZIP/JSON, local session stream, hook event, sidecar, browser capture receiver, language-server export, synthetic fixture | Acquisition mechanism and completeness expectation. | Yes in ops/readiness/debt | Provider-package completeness and archive debt should classify by origin plus capture mode. Do not infer capture mode from provider vendor. |
| Parser binding | parser module, parser version, `looks_like()` detector, `LoweredPayloadSpec`, provider schema package | How raw material was interpreted into normalized sessions/messages/actions/blocks. | Debug/ops | ImportExplain should report detector and parser binding. Parser binding is not public origin, material source, or admission/completeness state. |
| Archive session identity | `session_id`, `SessionId`, content hash, `sessions.session_id`, `TargetRefPayload.session`, `session:<id>` refs | Canonical stored session object. | Yes | Use for product refs, query/read targets, context images/bundles, and durable user state. Provider-native ids remain provenance. |
| Provider-native identity | upstream thread/conversation/cascade/session ids, parent ids, language-server cascade ids | Upstream-local identity inside the raw provider product. | Yes as provenance | Store and render when useful, especially for topology and ImportExplain. Never make it a substitute for archive ids. |
| Logical session identity | `logical_session_id`, topology root, continuation/fork/subagent lineage | Work-session grouping across physical sessions. | Yes | Use for topology, context images/bundles, and context continuation. Keep distinct from physical sessions and provider-native parent ids. |
| Run identity | run refs, session-digest run projection, tool/subagent run evidence | Agent-work execution span inside one or more sessions. | Yes as refs/views | Use in context images/bundles, context compiler, OTLP projection, and query units. Run identity is evidence, not source/provider identity. |
| Observed event identity | observed-event query rows, review-injection events, tool/result events | Event-like facts observed in session material or derived projections. | Yes as refs/views | Query/read/project as evidence rows. Do not overload provider/source fields to encode event kind. |
| Context snapshot identity | context-snapshot query rows, review injection boundary, compiled context images | A bounded context image or boundary used by an agent/human. | Yes as refs/views | Address through refs/query units and context compiler surfaces. Do not encode as provider/capture mode. |
| Model/lab/runtime metadata | provider meta, model name, lab, runtime product, cost rollups | Descriptive execution context and pricing/analysis attribution. | Yes as metadata | Lab/model can be derived or reported, but should not define provider-package identity or public source-origin filters. |
| Repo/workspace context | repo name, cwd, git branch, workspace, working directories | User-work context attached to a session/action/run. | Yes as filters | Use `repo`, `cwd`, `workspace`, and related fields for query/successor-context context. They are not source identity. |
| Privacy/disclosure posture | raw-path redaction, browser capture auth, web shell raw preview policy, MCP role, export policy | What can be shown, exported, logged, or sent across a surface. | Yes as policy/status | Make disclosure explicit in route/read payloads. Do not infer safe exposure from provider, origin, or source path. |

## Current Code Invariants

- `Origin` is the public archive source-origin vocabulary. Query specs,
  terminal unit rows, daemon query parameters, and MCP query-unit filters use
  `origin`/`exclude_origin` tokens and should continue to prefer that spelling.
- `Provider` remains in parser/schema/raw-acquisition boundaries. Parsers map it
  forward once through `origin_from_provider()` when constructing normalized
  sessions. Normalized storage, search, insight, API, CLI, MCP, and daemon
  models carry `Origin` without a reverse translation or payload rewrite.
- Rebuildable storage records may retain columns named `source_name` for schema
  continuity. Their values are canonical origins, and typed hydration projects
  them directly into `origin` fields. The column spelling is not a second
  domain vocabulary.
- `Source` is richer than `Provider` and `Origin`, but it still does not name a
  concrete import path. Its `runtime_root` is a conventional root hint, not the
  actual material row or raw artifact identity.
- `canonical_acquisition_provider()` deliberately separates raw acquisition
  provider hints from configured source names. Source names such as `inbox` or
  `seeded` are operator scope, not provider truth.
- `Provider.GEMINI` and `Provider.DRIVE` both map to `Origin.AISTUDIO_DRIVE`.
  That is a semantic bridge, not a claim that Gemini export, Drive acquisition,
  and AI Studio capture mode are one coordinate.
- Provider-native parent ids and topology refs are preserved as provenance so
  late parent repair and sidechain/subagent reasoning can work even when the
  canonical parent session is not yet ingested.
- Browser capture has its own HTTP receiver/auth/raw-origin concerns. It should
  report capture mode and receiver readiness explicitly rather than pretending
  the captured page's lab/vendor explains the acquisition boundary.

## How New Features Should Cite This Map

### ImportExplain

ImportExplain payloads should separate:

```text
material source path/ref
artifact kind
detector candidates
selected provider-wire family
parser binding/version
capture mode
public origin
produced archive refs
caveats/skips
```

The explanation should answer "why did this raw material become these archive
rows?" without collapsing the answer into a single provider string.

### Provider/importer package completeness

Completeness rows should be keyed by public origin plus capture mode or package
mode. They should report provider-wire family and parser binding as fields.
They should not treat vendor/lab as enough to declare a mode complete.

Minimum row dimensions:

```text
origin
capture_mode
material_class
parser_binding
provider_wire_family
expected archive units
fixtures/tests
debt/readiness refs
operations
disclosure posture
```

### Archive debt views

Debt rows should cite the coordinate that owns the debt:

- import debt: material source, artifact kind, detector/parser binding, capture
  mode, produced archive refs;
- index/FTS debt: archive session/message/block rows and index triggers;
- embedding debt: archive rows, embedding provider config, and embedding run
  refs;
- transform/read-model debt: transform or insight refs;
- assertion debt: assertion/candidate refs and target refs;
- generated-surface debt: repo artifacts and generator inputs.

### Public ref resolution

Ref resolution should distinguish:

```text
session/message/block/action/assertion refs
run/context-snapshot/observed-event refs
raw artifact/material refs
provider-native provenance ids
logical session/topology refs
```

Provider-native ids may be searchable provenance, but they should not become
the primary product ref format.

### Query and read surfaces

Use `origin` for public source-origin filters. Use `provider` only when the
surface truly asks about provider-wire schema, parser family, provider metadata,
embedding provider, or lab/vendor metadata.

## Completed Normalized-Identity Transition

Normalized sessions, messages, actions, query units, insights, topology and
resume models, and the public API/CLI/MCP/daemon read surfaces now use
`origin`. The former recursive payload projection and origin-to-provider query
detours have been removed. New normalized features should consume that native
contract rather than introduce an adapter.

Remaining work in adjacent products is semantic, not a vocabulary sweep:
source discovery and ImportExplain should report configured material roots,
capture mode, parser binding, and provider-wire family as distinct evidence;
provider/importer completeness should use this table as its identity checklist.

## Non-Goals

- No compatibility aliases for old public spellings.
- No tests that merely assert an old word is absent.
- No static gates, allowlists, or seeded failures whose only purpose is to
  preserve identifier/import spelling after a refactor.
- No broad storage rename without a schema-owner PR and re-ingest plan.
- No attempt to turn OTel trace ids, provider-native ids, or filesystem paths
  into canonical archive refs.
