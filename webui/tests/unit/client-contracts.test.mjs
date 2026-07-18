import assert from "node:assert/strict";
import test from "node:test";

import { PolylogueClient } from "../../.cache/client-test/api/generated.js";
import {
  ContinuationProtocolError,
  DaemonHttpError,
  DeadlineExceededError,
  FetchTransport,
  RequestAbortedError,
  SameOriginViolationError,
  iteratePages,
} from "../../.cache/client-test/api/runtime.js";

class ScriptedTransport {
  constructor(responses) {
    this.responses = [...responses];
    this.requests = [];
  }

  async request(request, options) {
    this.requests.push({ request, options });
    assert.notEqual(this.responses.length, 0, "unexpected generated-client request");
    return this.responses.shift();
  }
}

function definedQuery(query) {
  return Object.fromEntries(Object.entries(query ?? {}).filter(([, value]) => value !== undefined));
}

test("query iterator replays only the opaque continuation after the first page", async () => {
  const transport = new ScriptedTransport([
    {
      mode: "query-unit",
      unit: "message",
      query: "messages where text:hello",
      items: [{ unit: "message", message_id: "m1" }],
      total: 1,
      limit: 1,
      offset: 0,
      continuation: "q1.second",
      query_ref: "query:stable",
      result_ref: "result:stable",
    },
    {
      mode: "query-unit",
      unit: "message",
      query: "messages where text:hello",
      items: [{ unit: "message", message_id: "m2" }],
      total: 1,
      limit: 1,
      offset: 1,
      continuation: null,
      query_ref: "query:stable",
      result_ref: "result:stable",
    },
  ]);
  const client = new PolylogueClient(transport);
  const pages = [];

  for await (const page of client.query({
    expression: "messages where text:hello",
    limit: 1,
    origin: "codex-session",
  })) {
    pages.push(page);
  }

  assert.equal(pages.length, 2);
  assert.deepEqual(pages.map((page) => page.items[0].message_id), ["m1", "m2"]);
  assert.deepEqual(pages[0].coverage, {
    kind: "qualified",
    total: 1,
    qualification: "page",
  });
  assert.equal(pages[1].queryRef, "query:stable");
  assert.deepEqual(definedQuery(transport.requests[0].request.query), {
    expression: "messages where text:hello",
    limit: 1,
    origin: "codex-session",
  });
  assert.deepEqual(definedQuery(transport.requests[1].request.query), {
    continuation: "q1.second",
  });
});

test("search iterator preserves a capped total as qualified coverage", async () => {
  const transport = new ScriptedTransport([
    {
      hits: [{ session: { session_id: "s1" } }],
      total: 250,
      limit: 25,
      offset: 0,
      next_cursor: null,
      query: "timeout",
      retrieval_lane: "dialogue",
      exactness: "capped",
    },
  ]);
  const client = new PolylogueClient(transport);
  const pages = [];

  for await (const page of client.search({ query: "timeout", limit: 25, offset: 100 })) {
    pages.push(page);
  }

  assert.equal(pages.length, 1);
  assert.deepEqual(pages[0].coverage, {
    kind: "qualified",
    total: 250,
    qualification: "capped",
  });
});

test("search continuation removes the unstable offset and retains server filters", async () => {
  const transport = new ScriptedTransport([
    {
      hits: [],
      total: 2,
      limit: 1,
      offset: 9,
      next_cursor: "search.next",
      query: "cache",
      retrieval_lane: "dialogue",
      exactness: "exact",
    },
    {
      hits: [],
      total: 2,
      limit: 1,
      offset: 0,
      next_cursor: null,
      query: "cache",
      retrieval_lane: "dialogue",
      exactness: "exact",
    },
  ]);
  const client = new PolylogueClient(transport);

  for await (const _page of client.search({
    query: "cache",
    limit: 1,
    offset: 9,
    provider: "claude-code",
  })) {
    // Exhaust the iterator.
  }

  assert.deepEqual(definedQuery(transport.requests[1].request.query), {
    cursor: "search.next",
    limit: 1,
    provider: "claude-code",
    query: "cache",
  });
});

test("continuation iterator rejects a repeated cursor before yielding a duplicate page", async () => {
  let calls = 0;
  const iterable = iteratePages(async () => {
    calls += 1;
    return {
      items: [calls],
      cursor: "same-cursor",
      coverage: { kind: "qualified", total: 1, qualification: "page" },
      queryRef: null,
      resultRef: null,
      envelope: {},
    };
  });
  const seen = [];

  await assert.rejects(
    async () => {
      for await (const page of iterable) {
        seen.push(page.items[0]);
      }
    },
    (error) => error instanceof ContinuationProtocolError && error.cursor === "same-cursor",
  );
  assert.deepEqual(seen, [1]);
});

test("fetch transport maps the daemon error envelope and keeps the request same-origin", async () => {
  let observedUrl;
  let observedInit;
  const transport = new FetchTransport({
    baseUrl: "https://polylogue.test",
    fetch: async (url, init) => {
      observedUrl = String(url);
      observedInit = init;
      return new Response(
        JSON.stringify({ ok: false, error: "invalid_query", detail: "bad expression", field: "query" }),
        {
          status: 400,
          statusText: "Bad Request",
          headers: {
            "Content-Type": "application/json",
            "X-Request-ID": "request-1",
            "X-Polylogue-Web-Credential-State": "web_credential_expired",
          },
        },
      );
    },
  });

  await assert.rejects(
    () => transport.request({ method: "GET", path: "/api/sessions", query: { query: "bad" } }),
    (error) => {
      assert.ok(error instanceof DaemonHttpError);
      assert.equal(error.status, 400);
      assert.equal(error.code, "invalid_query");
      assert.equal(error.detail, "bad expression");
      assert.equal(error.field, "query");
      assert.equal(error.requestId, "request-1");
      assert.equal(error.credentialState, "web_credential_expired");
      return true;
    },
  );

  assert.equal(observedUrl, "https://polylogue.test/api/sessions?query=bad");
  assert.equal(observedInit.credentials, "same-origin");
  assert.equal(observedInit.cache, "no-store");
  assert.equal(observedInit.redirect, "error");
});

test("fetch transport reports an expired deadline", async () => {
  const transport = new FetchTransport({
    baseUrl: "https://polylogue.test",
    fetch: async (_url, init) => {
      assert.equal(init.signal.aborted, true);
      throw new DOMException("aborted", "AbortError");
    },
  });

  await assert.rejects(
    () => transport.request({ method: "GET", path: "/api/sessions" }, { deadline: Date.now() - 1 }),
    DeadlineExceededError,
  );
});

test("fetch transport preserves an external abort reason", async () => {
  const controller = new AbortController();
  const reason = new Error("route changed");
  controller.abort(reason);
  const transport = new FetchTransport({
    baseUrl: "https://polylogue.test",
    fetch: async (_url, init) => {
      assert.equal(init.signal.aborted, true);
      assert.equal(init.signal.reason, reason);
      throw new DOMException("aborted", "AbortError");
    },
  });

  await assert.rejects(
    () => transport.request({ method: "GET", path: "/api/sessions" }, { signal: controller.signal }),
    (error) => error instanceof RequestAbortedError && error.reason === reason,
  );
});

test("fetch transport refuses absolute and protocol-relative request paths", async () => {
  const transport = new FetchTransport({
    baseUrl: "https://polylogue.test",
    fetch: async () => new Response("{}"),
  });

  await assert.rejects(
    () => transport.request({ method: "GET", path: "https://elsewhere.test/api/sessions" }),
    SameOriginViolationError,
  );
  await assert.rejects(
    () => transport.request({ method: "GET", path: "//elsewhere.test/api/sessions" }),
    SameOriginViolationError,
  );
});
