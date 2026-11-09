import { expect, it } from "vitest";
import { ProviderScheduler, ProviderDeferred } from "../src/background/provider.js";
import { fakeChrome } from "./support/fake_chrome.js";

it("serializes a provider and restores Retry-After across worker restart", async () => {
  let now = Date.parse("2026-01-01T00:00:00Z");
  const storage = fakeChrome().storage.local;
  const options = { now: () => now, sleep: async (ms) => { now += ms; }, storage };
  const first = new ProviderScheduler(options);
  const events = [];
  await first.run("chatgpt", async () => ({ status: 429, retry_after: "Thu, 01 Jan 2026 00:01:00 GMT" }));
  const resumed = new ProviderScheduler(options);
  await Promise.all([1, 2].map((i) => resumed.run("chatgpt", async () => { events.push([i, now]); return { status: 200 }; })));
  expect(events).toEqual([[1, Date.parse("2026-01-01T00:01:00Z")], [2, Date.parse("2026-01-01T00:01:00Z")]]);
});

it("yields exhausted background work without limiting live capture, and resets next day", async () => {
  let now = 0;
  const scheduler = new ProviderScheduler({ dailyBudget: 1, now: () => now });
  const request = async () => { await scheduler.authorize("chatgpt"); return { status: 200 }; };
  await scheduler.run("chatgpt", request, { background: true });
  await expect(scheduler.run("chatgpt", request, { background: true })).rejects.toBeInstanceOf(ProviderDeferred);
  await expect(scheduler.run("chatgpt", request)).resolves.toEqual({ status: 200 });
  now = 86400000;
  await expect(scheduler.run("chatgpt", request, { background: true })).resolves.toEqual({ status: 200 });
});

it("rechecks ownership after waiting and never executes a rejected action", async () => {
  let now = 0;
  const scheduler = new ProviderScheduler({ now: () => now, sleep: async (ms) => { now += ms; } });
  await scheduler.run("chatgpt", async () => ({ status: 503, retry_after: "30" }));
  let ran = false;
  await expect(scheduler.run("chatgpt", async () => { ran = true; }, { before: async () => { throw new Error("claim lost"); } })).rejects.toThrow("claim lost");
  expect(ran).toBe(false);
  expect(now).toBe(30000);
});

it("runs an explicit action ahead of queued backfill once the active request settles", async () => {
  const scheduler = new ProviderScheduler();
  const order = [];
  let release;
  const held = new Promise((resolve) => { release = resolve; });
  const first = scheduler.run("chatgpt", async () => { order.push("active"); await held; }, { background: true });
  const second = scheduler.run("chatgpt", async () => { order.push("background"); }, { background: true });
  const action = scheduler.run("chatgpt", async () => { order.push("action"); }, { priority: -1 });
  release();
  await Promise.all([first, second, action]);
  expect(order).toEqual(["active", "action", "background"]);
});

it("restores the actual HTTP budget after a worker restart", async () => {
  const storage = fakeChrome().storage.local;
  const options = { storage, dailyBudget: 1, now: () => 0 };
  const first = new ProviderScheduler(options);
  await first.run("chatgpt", () => first.authorize("chatgpt"), { background: true });
  const resumed = new ProviderScheduler(options);
  await expect(resumed.run("chatgpt", () => resumed.authorize("chatgpt"), { background: true })).rejects.toBeInstanceOf(ProviderDeferred);
});

it("refuses an unrelated tab borrowing an active provider permit", async () => {
  const scheduler = new ProviderScheduler();
  await scheduler.run("chatgpt", async () => {
    await expect(scheduler.authorize("chatgpt", 2)).rejects.toThrow("no active owner");
    await expect(scheduler.authorize("chatgpt", 1)).resolves.toBeUndefined();
  }, { owner: 1 });
});
