import { describe, expect, it } from "vitest";
import { IDBFactory } from "fake-indexeddb";
import { ImportJob } from "../src/background/import_job.js";
import { Capturer } from "../src/background/capture.js";
import { Host } from "../src/background/host.js";
import { Outbox } from "../src/background/outbox.js";
import { fakeChrome, json } from "./support/fake_chrome.js";

/** A provider with two list pages and the archive's view of their conversations. */
function setup({ job, held, listing = null, conversation = null, acceptReport = () => true }) {
  const reports = [];
  const problems = [];
  const stored = [];
  const operations = {
    "internal import claim": () => ({ json: { job } }),
    "internal import report": (body) => {
      const report = { by: body.by, ...body.report };
      reports.push(report);
      return report.by === job.claimed_by && acceptReport(report) ? { json: { job: { ...job, status: report.kind } } } : { status: 409, json: { error: "not_the_claimant" } };
    },
    "internal conversations": (body) => {
      const { ids } = body;
      return { json: { daemon: "running", conversations: ids.map((id) => ({ id, ...(held[id] || { state: "missing" }) })) } };
    },
    "internal capture-files": () => ({ json: { conversation: { provider: "chatgpt", id: "c1" }, files: [] } }),
    "capture": (body) => { stored.push(json(body)); return { json: { new: true } }; },
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
        if (conversation) return conversation(request.id);
        return { status: 200, info: { id: request.id, url: `https://chatgpt.com/c/${request.id}` }, body: new Blob([JSON.stringify({ conversation_id: request.id })]) };
      }
      return { status: 400 };
    },
  };
  const chrome = fakeChrome({ operations, pages, tabs: [{ id: 7, url: "https://chatgpt.com/" }] });
  const host = new Host(chrome.runtime);
  const capturer = new Capturer({ chrome, host, outbox: new Outbox(new IDBFactory()), seen: chrome.storage.session });
  const importJob = new ImportJob({ chrome, host, capturer, instance: async () => job.claimed_by, log: (text) => problems.push(text), sleep: async () => {} });
  return { importJob, reports, stored, lists, chrome, capturer, problems };
}

const HELD = {
  "old-same": { state: "archived", messages: 3, last_captured_at: Date.parse("2026-01-03T00:00:00Z") },
  changed: { state: "archived", messages: 3, last_captured_at: Date.parse("2026-01-04T00:00:00Z") },
  spooled: { state: "spooled" },
};

describe("an import job held by Polylogue", () => {
  it("captures what the archive lacks or holds an older copy of, and reports its position and totals", async () => {
    const job = { id: "j1", provider: "chatgpt", claimed_by: "ext-1", position: null, listed: 0, captured: 0, unchanged: 0, failed: 0 };
    const { importJob, reports, stored, lists } = setup({ job, held: HELD });
    await importJob.step();
    expect(lists).toEqual([null, "0:3"]);
    // A spooled conversation is of no known version, so it is fetched; this
    // browser has not captured it before, so it is stored.
    expect(stored.map((c) => c.session.provider_session_id)).toEqual(["changed", "new", "spooled"]);
    expect(stored[0].provenance.reason).toBe("import_job");
    expect(reports.filter((r) => r.position !== null || r.kind === "finished").map((r) => [r.kind, r.position])).toEqual([["progress", "0:3"], ["progress", "0:3"], ["progress", "0:3"], ["finished", null]]);
    expect(reports.at(-1).counts).toEqual({ listed: 4, captured: 3, unchanged: 1, failed: 0 });
  });

  it("continues from the reported position after the extension restarts", async () => {
    const job = { id: "j1", provider: "chatgpt", claimed_by: "ext-1", position: "0:3", listed: 3, captured: 2, unchanged: 1, failed: 0 };
    const { importJob, reports, lists } = setup({ job, held: HELD });
    await importJob.step();
    expect(lists).toEqual(["0:3"]);
    expect(reports.at(-1)).toMatchObject({ kind: "finished", counts: { listed: 4, captured: 3, unchanged: 1, failed: 0 } });
  });

  it("stops at the provider's wait and at conversations older than the job's date", async () => {
    const job = { id: "j1", provider: "chatgpt", claimed_by: "ext-1", position: null, since: "2026-01-04T00:00:00Z", listed: 0, captured: 0, unchanged: 0, failed: 0 };
    const { importJob, reports, stored, lists } = setup({ job, held: HELD });
    await importJob.step();
    // The first page reaches past the date, so the list ends there.
    expect(lists).toEqual([null]);
    expect(stored.map((c) => c.session.provider_session_id)).toEqual(["changed", "new"]);
    expect(reports.at(-1)).toMatchObject({ kind: "finished", counts: { captured: 2, unchanged: 0 } });
  });

  it.each([401, 403])("a_sign_in_refusal_pauses_the_import_and_raises_a_notice (%i), and a resume continues from the saved position", async (status) => {
    const job = { id: "j1", provider: "chatgpt", claimed_by: "ext-1", position: null, listed: 0, captured: 0, unchanged: 0, failed: 0 };
    const refusing = setup({ job, held: HELD, listing: (position) => (position === "0:3"
      ? { status, error: "signed_out" }
      : { status: 200, info: { conversations: [{ id: "new", updated_at: "2026-01-06T00:00:00Z" }], next: "0:3", done: false, newest_first: true } }) });
    await refusing.importJob.step();
    expect(refusing.lists).toEqual([null, "0:3"]);
    const paused = refusing.reports.at(-1);
    expect(paused).toMatchObject({ kind: "paused", position: "0:3", counts: { listed: 1, captured: 1, unchanged: 0, failed: 0 } });
    expect(paused.detail).toBe("Sign in to chatgpt in this browser, then run `polylogue import chatgpt` to resume.");
    expect(refusing.problems).toEqual([`Import of chatgpt paused: ${paused.detail}`]);

    // Starting the import again hands the job back with its saved position.
    const resumed = setup({ job: { ...job, position: paused.position, ...paused.counts }, held: HELD });
    await resumed.importJob.step();
    expect(resumed.lists).toEqual(["0:3"]);
    expect(resumed.stored.map((c) => c.session.provider_session_id)).toEqual(["spooled"]);
    expect(resumed.reports.at(-1)).toMatchObject({ kind: "finished", counts: { listed: 2, captured: 2, unchanged: 0, failed: 0 } });
  });
});

it("a_provider_edit_after_spooling_is_captured", async () => {
  const job = { id: "j1", provider: "chatgpt", claimed_by: "ext-1", position: null, listed: 0, captured: 0, unchanged: 0, failed: 0 };
  let text = "first";
  const { importJob, capturer, stored, reports } = setup({ job, held: { a: { state: "spooled" }, b: { state: "spooled" } },
    listing: () => ({ status: 200, info: { conversations: [
      { id: "a", updated_at: "2026-01-02T00:00:00Z" },
      { id: "b", updated_at: "2026-01-02T00:00:00Z" },
    ], next: null, done: true, newest_first: true } }),
    conversation: (id) => ({ status: 200, info: { id, url: `https://chatgpt.com/c/${id}` },
      body: new Blob([JSON.stringify({ conversation_id: id, text: id === "a" ? text : "same" })]) }),
  });
  // Both conversations were captured into the spool before the provider changed `a`.
  await capturer.capture(7, "chatgpt", { id: "a" });
  await capturer.capture(7, "chatgpt", { id: "b" });
  stored.length = 0;
  text = "edited";
  await importJob.step();
  expect(stored.map((c) => [c.session.provider_session_id, c.raw_provider_payload.text])).toEqual([["a", "edited"]]);
  expect(reports.at(-1)).toMatchObject({ kind: "finished", counts: { listed: 2, captured: 1, unchanged: 1, failed: 0 } });
});

it("resumes a current conversation whose attachment bytes are still downloading", async () => {
  const job = { id: "j1", provider: "chatgpt", claimed_by: "ext-1", position: null, listed: 0, captured: 0, unchanged: 0, failed: 0 };
  const { importJob, stored } = setup({ job, held: { ...HELD,
    "old-same": { ...HELD["old-same"], downloading: true },
  } });
  await importJob.step();
  expect(stored.map((c) => c.session.provider_session_id)).toEqual(["old-same", "changed", "new", "spooled"]);
});

it("continues the other ChatGPT partitions after a date cutoff in each partition", async () => {
  const job = { id: "j1", provider: "chatgpt", claimed_by: "ext-1", position: null, since: "2026-01-04", listed: 0, captured: 0, unchanged: 0, failed: 0 };
  const { importJob, lists, stored, reports } = setup({ job, held: {}, listing: (position) => {
    const partition = Number((position || "0:0").split(":")[0]);
    return { status: 200, info: { conversations: [
      { id: `new-${partition}`, updated_at: "2026-01-05" },
      { id: `old-${partition}`, updated_at: "2026-01-01" },
    ], next: `${partition}:2`, next_partition: partition < 3 ? `${partition + 1}:0` : null, done: false, newest_first: true } };
  } });
  await importJob.step();
  expect(lists).toEqual([null, "1:0", "2:0", "3:0"]);
  expect(stored.map((capture) => capture.session.provider_session_id)).toEqual(["new-0", "new-1", "new-2", "new-3"]);
  expect(reports.at(-1).kind).toBe("finished");
});

it("observes cancellation within an unchanged page", async () => {
  const job = { id: "j1", provider: "chatgpt", claimed_by: "ext-1", position: null, listed: 0, captured: 0, unchanged: 0, failed: 0 };
  const conversations = Array.from({ length: 30 }, (_, i) => ({ id: String(i), updated_at: "2026-01-02T00:00:00Z" }));
  let reportsSeen = 0;
  const { importJob, reports } = setup({ job, held: Object.fromEntries(conversations.map((c) => [c.id, HELD["old-same"]])),
    acceptReport: () => ++reportsSeen < 2,
    listing: () => ({ status: 200, info: { conversations, done: true, next: null } }),
  });
  await importJob.step();
  expect(reports).toHaveLength(2);
  expect(reports.some((report) => report.kind === "finished")).toBe(false);
});

it("a_resumed_import_counts_each_conversation_once", async () => {
  const job = { id: "j1", provider: "chatgpt", claimed_by: "ext-1", position: "0:100", listed: 100, captured: 4, unchanged: 96, failed: 0 };
  const conversations = Array.from({ length: 25 }, (_, i) => ({ id: String(i), updated_at: "2026-01-02T00:00:00Z" }));
  const held = Object.fromEntries(conversations.map((c) => [c.id, HELD["old-same"]]));
  held["15"] = { state: "missing" };
  const listing = () => ({ status: 200, info: { conversations, done: true, next: null } });
  // Reports go out before the list is read, after ten conversations and
  // before conversation 15 is fetched, where the provider refuses the sign-in.
  const first = setup({ job, held, listing, conversation: () => ({ status: 403, error: "chatgpt_http_403" }) });
  await first.importJob.step();
  const before = { listed: 100, captured: 4, unchanged: 96, failed: 0 };
  expect(first.reports.map((r) => [r.kind, r.position, r.counts])).toEqual([
    ["progress", "0:100", before], ["progress", "0:100", before], ["progress", "0:100", before], ["paused", "0:100", before],
  ]);
  const paused = first.reports.at(-1);
  const second = setup({ job: { ...job, position: paused.position, ...paused.counts }, held, listing });
  await second.importJob.step();
  expect(second.reports.at(-1)).toMatchObject({ kind: "finished", counts: { listed: 125, captured: 5, unchanged: 120, failed: 0 } });
});

it("does not run an unconfirmed authored job receipt", async () => {
  const receipt = { id: "job-pending", repeat: false, pending: true };
  const chrome = fakeChrome({ operations: { "internal import claim": () => ({ json: receipt }) } });
  const host = new Host(chrome.runtime);
  expect(await host.call("internal import claim", { by: "ext-1" })).toEqual(receipt);
  const importJob = new ImportJob({ chrome, host, capturer: {}, instance: async () => "ext-1" });
  await importJob.step();
  expect(chrome.tabs.list).toHaveLength(0);
});
