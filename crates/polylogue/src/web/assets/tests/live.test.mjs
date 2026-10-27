import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
const source = await readFile(new URL("../live.js", import.meta.url), "utf8");
let serial = 0;
const flush = async () => { for (let i = 0; i < 8; i++) await new Promise(resolve => setImmediate(resolve)); };
const envelope = data => JSON.stringify({ result: "ok", data }) + "\n";
const begin = { type: "begin", replacement: false, generation_changed: false };
const change = { type: "change", component: "index", change: { ref: "session:synthetic" } };
const complete = position => ({ type: "complete", position, count: 1 });
const batch = position => new Response(envelope(begin) + envelope(change) + envelope(complete(position)));
const snapshot = (next = null) => Response.json({ rows: [{ ref: "session:current" }], read_at: { cursor: "synthetic:0:0", generation: "synthetic" }, next });
async function environment(t, fetch) {
  const saved = new Map(["document", "window", "fetch", "setTimeout", "clearTimeout", "setInterval", "clearInterval"].map(key => [key, globalThis[key]]));
  const timers = new Map(), intervals = new Map(), events = new Map();
  let next = 0;
  globalThis.document = { hidden: false, addEventListener: (name, fn) => events.set(name, fn) };
  globalThis.window = { addEventListener() {} };
  globalThis.fetch = fetch;
  globalThis.setTimeout = fn => { timers.set(++next, fn); return next; };
  globalThis.clearTimeout = id => timers.delete(id);
  globalThis.setInterval = fn => { intervals.set(++next, fn); return next; };
  globalThis.clearInterval = id => intervals.delete(id);
  t.after(() => { for (const [key, value] of saved) { if (value === undefined) delete globalThis[key]; else globalThis[key] = value; } });
  const module = await import("data:text/javascript;base64," + Buffer.from(source).toString("base64") + "#" + serial++);
  return { ...module, events, timers, intervals, async tick() {
    assert.equal(timers.size, 1);
    const [id, fn] = timers.entries().next().value; timers.delete(id); await fn(); await flush();
  }};
}
test("an interrupted batch repeats references from the last completed position", async t => {
  const requests = [], applied = [], errors = [];
  const env = await environment(t, async url => {
    if (url.startsWith("/api/v1/query")) return snapshot();
    requests.push(url);
    return requests.length === 2 ? new Response(envelope(begin) + envelope(change)) : batch("synthetic:1:0");
  });
  const stop = env.watchChanges({ apply: line => applied.push(line.change.ref), error: error => errors.push(error.message) });
  await flush(); await env.tick(); await env.tick();
  assert.deepEqual(requests, ["/api/v1/changes?since=synthetic%3A0%3A0", "/api/v1/changes?since=synthetic%3A1%3A0", "/api/v1/changes?since=synthetic%3A1%3A0"]);
  assert.equal(applied.length, 4); assert.match(errors[0], /interrupted/); stop();
});
test("joining reads every current page without resetting the shared position", async t => {
  const requests = [], pages = [], beginnings = [], completions = [];
  let snapshots = 0;
  const env = await environment(t, async url => {
    requests.push(url);
    if (url.startsWith("/api/v1/query")) return snapshot(++snapshots % 2 ? "next-page" : null);
    return batch("synthetic:5:0");
  });
  const stopA = env.watchChanges({}); await flush();
  const stopB = env.watchChanges({ snapshot: page => pages.push(page), begin: line => beginnings.push(line), complete: line => completions.push(line) }); await flush();
  assert.equal(pages.length, 2);
  assert.equal(beginnings[0].generation, "synthetic");
  assert.equal(completions[0].count, 2);
  assert.equal(requests.filter(url => url.startsWith("/api/v1/query")).length, 4);
  const feeds = requests.filter(url => url.startsWith("/api/v1/changes"));
  assert.equal(feeds.length, 2);
  assert.equal(feeds[1], "/api/v1/changes?since=synthetic%3A5%3A0");
  stopA(); stopB();
});
test("one active change request serves the tab and hidden tabs abort it", async t => {
  let calls = 0;
  const env = await environment(t, async (url, { signal }) => {
    if (url.startsWith("/api/v1/query")) return snapshot();
    calls++;
    return new Promise((resolve, reject) => signal.addEventListener("abort", () => reject(new DOMException("hidden", "AbortError"))));
  });
  const stopA = env.watchChanges({}), stopB = env.watchChanges({}); await flush();
  assert.equal(calls, 1);
  document.hidden = true; env.events.get("visibilitychange")(); await flush();
  assert.equal(env.timers.size, 0); assert.equal(env.intervals.size, 0);
  document.hidden = false; env.events.get("visibilitychange")(); assert.equal(calls, 2);
  stopA(); stopB(); await flush();
});
test("expired authentication stops requests and clock refreshes", async t => {
  let calls = 0; const errors = [];
  const env = await environment(t, async url => {
    if (url.startsWith("/api/v1/query")) return snapshot();
    calls++; return new Response("expired", { status: 401 });
  });
  const stop = env.watchChanges({ error: error => errors.push(error.message) }); await flush();
  env.events.get("visibilitychange")(); await flush();
  assert.equal(calls, 1); assert.equal(env.timers.size, 0); assert.equal(env.intervals.size, 0);
  assert.match(errors[0], /Reopen with polylogue web/); stop();
});
test("a generation change reaches views before replacement references", async t => {
  const seen = [];
  const env = await environment(t, async url => url.startsWith("/api/v1/query") ? snapshot() : new Response(envelope({ ...begin, replacement: true, generation_changed: true }) + envelope(change) + envelope(complete("new:1:0"))));
  const stop = env.watchChanges({ begin: line => seen.push(line.generation_changed), apply: () => seen.push("change"), complete: () => seen.push("complete") });
  await flush(); assert.deepEqual(seen, [false,"change","complete",true,"change","complete"]); stop();
});

test("an initial query failure retries without requesting a replacement feed", async t => {
  const requests = [], errors = [];
  const env = await environment(t, async url => {
    requests.push(url);
    if (requests.length === 1) return new Response("temporarily unavailable", { status: 503 });
    if (url.startsWith("/api/v1/query")) return snapshot();
    return batch("synthetic:1:0");
  });
  const stop = env.watchChanges({ error: error => errors.push(error.message) });
  await flush(); assert.match(errors[0], /unavailable/);
  await env.tick();
  assert.equal(requests.filter(url => url.startsWith("/api/v1/query")).length,2);
  assert.equal(requests.filter(url => url === "/api/v1/changes").length,0);
  assert.equal(requests.at(-1),"/api/v1/changes?since=synthetic%3A0%3A0");
  stop();
});
