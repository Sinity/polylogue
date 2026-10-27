import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

const source = await readFile(new URL("../live.js", import.meta.url), "utf8");
let serial = 0;
const flush = () => new Promise(resolve => setImmediate(resolve));
const envelope = data => JSON.stringify({ result: "ok", data }) + "\n";
const begin = { type: "begin", replacement: false, generation_changed: false };
const change = { type: "change", component: "index", change: { ref: "session:synthetic" } };
const complete = position => ({ type: "complete", position, count: 1 });
const batch = position => new Response(envelope(begin) + envelope(change) + envelope(complete(position)));

async function environment(t, fetch) {
  const saved = new Map(["document", "window", "fetch", "setTimeout", "clearTimeout", "setInterval", "clearInterval"].map(key => [key, globalThis[key]]));
  const timers = new Map();
  const intervals = new Map();
  const events = new Map();
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
  return {
    ...module, events, timers, intervals,
    async tick() {
      assert.equal(timers.size, 1);
      const [id, fn] = timers.entries().next().value;
      timers.delete(id);
      await fn();
      await flush();
    },
  };
}

test("an interrupted batch repeats references without advancing its position", async t => {
  const requests = [];
  const applied = [];
  const errors = [];
  const env = await environment(t, async url => {
    requests.push(url);
    if (requests.length === 2) return new Response(envelope(begin) + envelope(change));
    return batch("synthetic:1:0");
  });
  const stop = env.watchChanges({ apply: line => applied.push(line.change.ref), error: error => errors.push(error.message) });
  await flush();
  await env.tick();
  await env.tick();
  assert.deepEqual(requests, ["/api/v1/changes", "/api/v1/changes?since=synthetic%3A1%3A0", "/api/v1/changes?since=synthetic%3A1%3A0"]);
  assert.equal(applied.length, 3);
  assert.match(errors[0], /interrupted/);
  stop();
});

test("one active request serves a visible tab and hidden tabs abort it", async t => {
  let calls = 0;
  const env = await environment(t, async (url, { signal }) => {
    calls++;
    return new Promise((resolve, reject) => signal.addEventListener("abort", () => reject(new DOMException("hidden", "AbortError"))));
  });
  const stopA = env.watchChanges({});
  const stopB = env.watchChanges({});
  assert.equal(calls, 1);
  document.hidden = true;
  env.events.get("visibilitychange")();
  await flush();
  assert.equal(env.timers.size, 0);
  assert.equal(env.intervals.size, 0);
  document.hidden = false;
  env.events.get("visibilitychange")();
  assert.equal(calls, 2);
  stopA(); stopB();
  await flush();
});

test("rotated authentication stops requests and clock refreshes", async t => {
  const errors = [];
  let calls = 0;
  const env = await environment(t, async () => { calls++; return new Response("expired", { status: 401 }); });
  const stop = env.watchChanges({ error: error => errors.push(error.message) });
  await flush();
  env.events.get("visibilitychange")();
  await flush();
  assert.equal(calls, 1);
  assert.equal(env.timers.size, 0);
  assert.equal(env.intervals.size, 0);
  assert.match(errors[0], /Reopen with polylogue web/);
  stop();
});

test("generation replacement reaches the view before its reference changes", async t => {
  const seen = [];
  const env = await environment(t, async () => new Response(envelope({ ...begin, replacement: true, generation_changed: true }) + envelope(change) + envelope(complete("new:1:0"))));
  const stop = env.watchChanges({ begin: line => seen.push(line.generation_changed), apply: () => seen.push("change"), complete: () => seen.push("complete") });
  await flush();
  assert.deepEqual(seen, [true, "change", "complete"]);
  stop();
});
