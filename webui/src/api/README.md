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

`client.search()` follows the ranked-search `next_cursor`. It retains the
original server filters, removes the unstable `offset`, and adds the opaque
cursor. A numeric total is exact only when `exactness` is absent or `exact`;
`capped`, `sampled`, and `estimate` remain qualified.

`FetchTransport` rejects absolute and protocol-relative paths, pins requests
to the browser origin (or an explicit SSR/test origin), uses same-origin
credentials, maps daemon error envelopes into `DaemonHttpError`, and supports
external abort signals, relative timeouts, and absolute deadlines.

## Vertical adoption map

### webui-02 session list/read

Replace session search fetches with `client.search()` and single-session read
fetches with `client.readSessionView()`. Plain list mode is exposed by
`client.searchSessions()` as the declared `SessionListResponse` branch, but it
is still offset-based and has no opaque continuation in the snapshot. The
vertical's continuation-only list requirement therefore needs a server-side
list continuation before it can claim complete adoption; it must not fabricate
one in TypeScript.

### webui-03 search

Replace all `/api/sessions?query=...` calls and local search-envelope types with
`client.search()`. For explicit terminal DSL unit queries, use
`client.query()`. Both iterators preserve server ranking/filter semantics and
surface exact-versus-qualified coverage directly.

### webui-04 transcript rendering

The snapshot has `client.readSessionView()` for the current typed read-view
envelope, but no declared semantic-card document/detail route. Add the
server-owned card-document Pydantic model and OpenAPI path, regenerate, then
adopt the generated operation. Do not type a card-document endpoint locally.

### webui-05 insights/status

The current OpenAPI artifact has no typed insight registry, named-source
freshness, or component-status operation. Those routes remain declaration
gates. Once their Pydantic response models and route-backed OpenAPI paths land,
regeneration supplies the methods without runtime changes.

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
