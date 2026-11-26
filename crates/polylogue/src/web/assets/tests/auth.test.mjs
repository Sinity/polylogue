import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
const source = await readFile(new URL("../auth.js", import.meta.url), "utf8");
const { authenticate } = await import("data:text/javascript;base64," + Buffer.from(source).toString("base64"));
function fixture(fetch) { const events = []; const auth = { textContent: "opening" }; return { events, auth, document: { getElementById: () => auth }, location: { pathname: "/auth", hash: "#synthetic-token", replace: value => events.push(["location", value]) }, history: { replaceState: (...args) => events.push(["history", ...args]) }, fetch: async (...args) => { events.push(["fetch", ...args]); return fetch(...args); } }; }
test("fragment leaves history before network and token travels only in POST body", async () => { const env = fixture(async () => ({ ok: true })); await authenticate(env); assert.equal(env.events[0][0], "history"); assert.equal(env.events[0][3], "/auth"); const [,url,request] = env.events[1]; assert.equal(url, "/auth"); assert.equal(request.body, "token=synthetic-token"); assert.equal(request.method, "POST"); assert.equal(request.credentials, "same-origin"); assert.deepEqual(env.events[2], ["location", "/"]); });
test("failed exchanges stay on sign-in with an actionable status", async () => { for (const fetch of [async () => ({ ok: false }), async () => { throw new Error("offline"); }]) { const env = fixture(fetch); await authenticate(env); assert.match(env.auth.textContent, /polylogue web again/); assert.equal(env.events.filter(([name]) => name === "location").length, 0); } });
test("empty fragment and other pages never submit credentials", async () => { let calls = 0; const env = fixture(async () => { calls++; return { ok: true }; }); env.location.hash = ""; await authenticate(env); env.location.pathname = "/search"; env.location.hash = "#synthetic-token"; await authenticate(env); assert.equal(calls, 0); });

test("sign-in in a fresh tab returns to the requested session", async () => {
  const env = fixture(async () => ({ ok: true, headers: new Headers({ "x-polylogue-return-to": "%2Fs%2Fclaude-code%3As1%3Fview%3Ddialogue" }) }));
  await authenticate(env);
  assert.deepEqual(env.events.at(-1), ["location", "/s/claude-code:s1?view=dialogue"]);
});
test("sign-in never redirects to a foreign site", async () => {
  for (const target of ["//example.test", "/\\example.test", "https://example.test", "/\n/example.test"]) {
    const env = fixture(async () => ({ ok: true, headers: new Headers({ "x-polylogue-return-to": encodeURIComponent(target) }) }));
    await authenticate(env);
    assert.deepEqual(env.events.at(-1), ["location", "/"]);
  }
});
