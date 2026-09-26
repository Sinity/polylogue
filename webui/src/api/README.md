# WebUI generated daemon client

`generated.ts` is committed output. Its only schema input is the generated
OpenAPI document at `docs/openapi/search.yaml`; regenerate it with:

```bash
devtools render openapi
devtools render webui-client
```

CI should run both commands in check mode. `devtools render all --check` also
includes the client because it is registered as a generated surface whose
input is the OpenAPI artifact.

Application code imports operation types and `PolylogueClient` from
`./generated.js`, and transport/page/error types from `./runtime.js`. Do not
add fetch wrappers or local response interfaces beside this directory. A
route that is absent from OpenAPI is not yet a WebUI client contract.

## Page contract

Every generated iterator yields `Page<T, TEnvelope>`:

```ts
type Page<T, TEnvelope = unknown> = {
  readonly items: ReadonlyArray<T>;
  readonly cursor: string | null;
  readonly coverage:
    | { readonly kind: "exact"; readonly total: number }
    | {
        readonly kind: "qualified";
        readonly total: number | null;
        readonly qualification: "page" | "capped" | "sampled" | "estimate" | "unknown";
      };
  readonly queryRef: string | null;
  readonly resultRef: string | null;
  readonly envelope: TEnvelope;
};
```

`client.query()` follows `QueryTransaction` continuations. The first request
contains the declared expression and filters; each later request contains
only the opaque `continuation`. Query-unit `total` is the number of rows in
that page, so its coverage is always `qualified/page`. The daemon-supplied
`query_ref` and `result_ref` are preserved.

`client.search()` follows the ranked-search `next_cursor`. Its initial
parameters may include an opaque `cursor` when continuing a server-rendered
search; subsequent requests retain the original server filters, remove the
unstable `offset`, and add the returned cursor. A numeric total is exact only
when `exactness` is absent or `exact`; `capped`, `sampled`, and `estimate`
remain qualified.

`FetchTransport` rejects absolute and protocol-relative paths, pins requests
to the browser origin (or an explicit SSR/test origin), uses same-origin
credentials, maps daemon error envelopes into `DaemonHttpError`, and supports
external abort signals, relative timeouts, and absolute deadlines.

## Vertical adoption map

All island API requests use `PolylogueClient`: overview query units and opaque
continuations, session list and message windows, observability refresh and
named-source freshness, search, and credential bootstrap. Session and
observability payload guards remain local where the OpenAPI response is a
union or a generic object. Routes without a published request schema are
documented in OpenAPI but excluded from client generation.

The optional `polylogue-browser-host` process serves manifest-governed assets
and forwards browser requests to the daemon HTTP authority. It does not open
an archive or carry a privileged daemon token. A daemon connection failure is
reported as `daemon_unavailable`; the host's own liveness and packaged assets
remain available. Direct daemon browser routes remain available while clients
are moved to the separate address.

### webui-02 session list/read

The list island uses `client.searchSessions()` and the transcript island uses
`client.readSessionView()`. Plain list mode remains offset-based and has no
opaque continuation; a server-side list continuation is still required before
the list can claim continuation-only pagination.

### webui-03 search

The search island uses `client.search()`. For explicit terminal DSL unit
queries, `client.query()` preserves server filters and opaque continuation.
Both iterators surface exact-versus-qualified coverage directly.

### webui-04 transcript rendering

The snapshot has `client.readSessionView()` for the current typed read-view
envelope, but no declared semantic-card document/detail route. Add the
server-owned card-document Pydantic model and OpenAPI path, regenerate, then
adopt the generated operation. Do not type a card-document endpoint locally.

### webui-05 insights/status

`getWebuiObservability()` and `getWebuiFreshness()` are generated request
methods with explicit route/query parameters. Their response schema is still
generic, so the observability island retains its payload guard. Typed insight
registry, freshness, and component-status models remain future contract work.

### webui-06 cost/usage

The current OpenAPI artifact has no typed cost/usage aggregate or session-usage
operation. Add the smallest server-side response models and route declarations
over existing archive operations, then regenerate. Client-side aggregation is
not an acceptable substitute.

## Declaration-kernel retarget

The renderer reads standard component schemas and operations plus one narrow
operation extension, `x-polylogue-page`. When the `DeclarationSpec` and MCP
declaration registries become the direct source of the daemon contract, their
renderer should continue emitting the same OpenAPI operation/schema shape.
Only `--schema` needs to point at a replacement artifact if the output path
changes; the runtime, committed client surface, and drift check do not depend
on Python declaration classes.
