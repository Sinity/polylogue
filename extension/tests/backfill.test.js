import { describe, expect, it } from "vitest";
import { IDBFactory } from "fake-indexeddb";
import { Backfill } from "../src/background/backfill.js";
import { Capturer } from "../src/background/capture.js";
import { Host } from "../src/background/host.js";
import { Outbox } from "../src/background/outbox.js";
import { fakeChrome, json } from "./support/fake_chrome.js";

/** A provider with two list pages and the archive's view of their conversations. */
function setup({ job, held, listing = null, acceptReport = () => true }) {
  const reports = [];
  const stored = [];
  const routes = {
    "POST /v1/backfill/claim": () => ({ json: { job } }),
    "POST /v1/backfill/j1/report": (body) => {
      const report = json(body);
      reports.push(report);
      return report.by === job.claimed_by && acceptReport(report) ? { json: { job: { ...job, status: report.kind } } } : { status: 409, json: { error: "not_the_claimant" } };
    },
    "POST /v1/conversations": (body) => {
      const { ids } = json(body);
      return { json: { daemon: "running", conversations: ids.map((id) => ({ id, ...(held[id] || { state: "missing" }) })) } };
    },
    "POST /v1/captures/files": () => ({ json: { files: [] } }),
    "POST /v1/captures": (body) => { stored.push(json(body)); return { json: { new: true } }; },
  };
  const lists = [];
  const pages = {
    "chatgpt.com": async (request) => {
      if (request.type === "list") {
        lists.push(request.position);
        if (listing) return listing(request.position);
        if (!request.position || request.position === "0:0") {
          return { status: 200, info: { conversations: [
            { id: "old-same", updated_at: "2026-01-02T00:00:00Z" },
            { id: "changed", updated_at: "2026-01-05T00:00:00Z" },
            { id: "new", updated_at: "2026-01-06T00:00:00Z" },
          ], next: "0:3", done: false, newest_first: true } };
        }
        return { status: 200, info: { conversations: [{ id: "spooled", updated_at: "2026-01-07T00:00:00Z" }], next: null, done: true, newest_first: true } };
      }
      if (request.type === "conversation") {
        return { status: 200, info: { id: request.id, url: `https://chatgpt.com/c/${request.id}` }, body: new Blob([JSON.stringify({ conversation_id: request.id })]) };
      }
      return { status: 400 };
    },
  };
  const chrome = fakeChrome({ routes, pages, tabs: [{ id: 7, url: "https://chatgpt.com/" }] });
  const host = new Host(chrome.runtime);
  const capturer = new Capturer({ chrome, host, outbox: new Outbox(new IDBFactory()), seen: chrome.storage.session });
  const backfill = new Backfill({ chrome, host, capturer, instance: async () => job.claimed_by, sleep: async () => {} });
  return { backfill, reports, stored, lists, chrome };
}

const HELD = {
  "old-same": { state: "archived", messages: 3, last_captured_at: Date.parse("2026-01-03T00:00:00Z") },
  changed: { state: "archived", messages: 3, last_captured_at: Date.parse("2026-01-04T00:00:00Z") },
  spooled: { state: "spooled" },
};

describe("a backfill job held by Polylogue", () => {
  it("captures what the archive lacks or holds an older copy of, and reports its position and totals", async () => {
    const job = { id: "j1", provider: "chatgpt", claimed_by: "ext-1", position: null, listed: 0, captured: 0, unchanged: 0, failed: 0 };
    const { backfill, reports, stored, lists } = setup({ job, held: HELD });
    await backfill.step();
    expect(lists).toEqual([null, "0:3"]);
    expect(stored.map((c) => c.session.provider_session_id)).toEqual(["changed", "new"]);
    expect(stored[0].provenance.reason).toBe("backfill");
    expect(reports.filter((r) => r.position !== null || r.kind === "finished").map((r) => [r.kind, r.position])).toEqual([["progress", "0:3"], ["progress", "0:3"], ["finished", null]]);
    expect(reports.at(-1).counts).toEqual({ listed: 4, captured: 2, unchanged: 2, failed: 0 });
  });

  it("continues from the reported position after the extension restarts", async () => {
    const job = { id: "j1", provider: "chatgpt", claimed_by: "ext-1", position: "0:3", listed: 3, captured: 2, unchanged: 1, failed: 0 };
    const { backfill, reports, lists } = setup({ job, held: HELD });
    await backfill.step();
    expect(lists).toEqual(["0:3"]);
    expect(reports.at(-1)).toMatchObject({ kind: "finished", counts: { listed: 4, captured: 2, unchanged: 2, failed: 0 } });
  });

  it("stops at the provider's wait and at conversations older than the job's date", async () => {
    const job = { id: "j1", provider: "chatgpt", claimed_by: "ext-1", position: null, since: "2026-01-04T00:00:00Z", listed: 0, captured: 0, unchanged: 0, failed: 0 };
    const { backfill, reports, stored, lists } = setup({ job, held: HELD });
    await backfill.step();
    // The first page reaches past the date, so the list ends there.
    expect(lists).toEqual([null]);
    expect(stored.map((c) => c.session.provider_session_id)).toEqual(["changed", "new"]);
    expect(reports.at(-1)).toMatchObject({ kind: "finished", counts: { captured: 2, unchanged: 0 } });
  });

  it("asks the operator to sign in when the provider refuses the list", async () => {
    const job = { id: "j1", provider: "chatgpt", claimed_by: "ext-1", position: null, listed: 0, captured: 0, unchanged: 0, failed: 0 };
    const { backfill, reports, chrome } = setup({ job, held: HELD });
    chrome.tabs.list.length = 0;
    const created = [];
    chrome.tabs.create = async ({ url }) => { const tab = { id: 9, url, status: "complete" }; chrome.tabs.list.push(tab); created.push(tab); return tab; };
    chrome.tabs.connect = (() => {
      const original = fakeChrome({ pages: { 9: async () => ({ status: 401, error: "signed_out" }) }, tabs: [{ id: 9, url: "https://chatgpt.com/" }] });
      return original.tabs.connect;
    })();
    await backfill.step();
    expect(created).toHaveLength(1);
    expect(reports.at(-1).kind).toBe("failed");
    expect(reports.at(-1).detail).toContain("Sign in");
    // The tab the backfill opened is closed again.
    expect(chrome.tabs.list).toHaveLength(0);
  });
});

it("resumes a current conversation whose attachment bytes are still downloading", async () => {
  const job = { id: "j1", provider: "chatgpt", claimed_by: "ext-1", position: null, listed: 0, captured: 0, unchanged: 0, failed: 0 };
  const { backfill, stored } = setup({ job, held: { ...HELD,
    "old-same": { ...HELD["old-same"], downloading: true },
  } });
  await backfill.step();
  expect(stored.map((c) => c.session.provider_session_id)).toEqual(["old-same", "changed", "new"]);
});

it("continues the other ChatGPT partitions after a date cutoff in each partition", async () => {
  const job = { id: "j1", provider: "chatgpt", claimed_by: "ext-1", position: null, since: "2026-01-04", listed: 0, captured: 0, unchanged: 0, failed: 0 };
  const { backfill, lists, stored, reports } = setup({ job, held: {}, listing: (position) => {
    const partition = Number((position || "0:0").split(":")[0]);
    return { status: 200, info: { conversations: [
      { id: `new-${partition}`, updated_at: "2026-01-05" },
      { id: `old-${partition}`, updated_at: "2026-01-01" },
    ], next: `${partition}:2`, next_partition: partition < 3 ? `${partition + 1}:0` : null, done: false, newest_first: true } };
  } });
  await backfill.step();
  expect(lists).toEqual([null, "1:0", "2:0", "3:0"]);
  expect(stored.map((capture) => capture.session.provider_session_id)).toEqual(["new-0", "new-1", "new-2", "new-3"]);
  expect(reports.at(-1).kind).toBe("finished");
});

it("observes cancellation within an unchanged page", async () => {
  const job = { id: "j1", provider: "chatgpt", claimed_by: "ext-1", position: null, listed: 0, captured: 0, unchanged: 0, failed: 0 };
  const conversations = Array.from({ length: 30 }, (_, i) => ({ id: String(i) }));
  let reportsSeen = 0;
  const { backfill, reports } = setup({ job, held: Object.fromEntries(conversations.map((c) => [c.id, { state: "spooled" }])),
    acceptReport: () => ++reportsSeen < 2,
    listing: () => ({ status: 200, info: { conversations, done: true, next: null } }),
  });
  await backfill.step();
  expect(reports.at(-1).counts.unchanged).toBe(10);
  expect(reports.some((report) => report.kind === "finished")).toBe(false);
});


it("does not run an unconfirmed authored job receipt", async () => {
  const receipt = { id: "job-pending", repeat: false, pending: true };
  const chrome = fakeChrome({ routes: { "POST /v1/backfill/claim": () => ({ json: receipt }) } });
  const host = new Host(chrome.runtime);
  expect(await host.claimBackfill("ext-1")).toEqual(receipt);
  const backfill = new Backfill({ chrome, host, capturer: {}, instance: async () => "ext-1" });
  await backfill.step();
  expect(chrome.tabs.list).toHaveLength(0);
});
