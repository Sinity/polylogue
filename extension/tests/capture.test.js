import { describe, expect, it, vi } from "vitest";
import { IDBFactory } from "fake-indexeddb";
import { start } from "../src/background/main.js";
import { Capturer, captureBody } from "../src/background/capture.js";
import { fakeChrome, json, until } from "./support/fake_chrome.js";
import "../src/wording.js";

const CONVERSATION = {
  conversation_id: "c1", title: "Synthetic", mapping: {
    n1: { id: "n1", message: { id: "m1", author: { role: "user" }, content: { parts: ["hello"] } } },
  },
};

/** A host that stores captures by their bytes, as the real one does. */
function storingHost(files = []) {
  const stored = [];
  const operations = {
    "internal capture-files": () => ({ json: { conversation: { provider: "chatgpt", id: "c1" }, files } }),
    "internal stored-files": () => ({ json: { files: files.filter((f) => f.stored).map((f) => ({ id: f.file_id, ...f.stored })) } }),
    "capture": (body) => {
      const text = new TextDecoder().decode(body);
      const known = stored.includes(text);
      if (!known) stored.push(text);
      return { json: { new: !known } };
    },
    "internal conversations": () => ({ json: { daemon: "running", conversations: [{ id: "c1", state: "archived", messages: 1,
      downloading: stored.length > 0 && JSON.parse(stored.at(-1)).session.attachments.some((a) => a.not_fetched === "downloading"),
    }] } }),
    "set": () => ({ json: { value: { type: "boolean", value: false } } }),
    "rows": () => ({ json: { reply: { rows: [] } } }),
    "internal chat claim": () => ({ json: { action: null } }),
    "internal import claim": () => ({ json: { job: null } }),
  };
  return { stored, operations };
}

function chatgptPage({ fileBytes = null } = {}) {
  const asked = [];
  const answer = async (request) => {
    asked.push(request);
    if (request.type === "page") return { status: 200, info: { id: "c1", temporary: false } };
    if (request.type === "conversation") {
      return {
        status: 200,
        info: { id: "c1", temporary: false, url: "https://chatgpt.com/c/c1" },
        body: new Blob([JSON.stringify(CONVERSATION)]),
      };
    }
    if (request.type === "file") {
      return fileBytes ? { status: 200, info: { media_type: "image/png" }, body: new Blob([fileBytes]) } : { status: 404, error: "gone" };
    }
    return { status: 400, error: "unknown_request" };
  };
  return { asked, answer };
}

describe("a capture", () => {
  it("holds the provider's JSON whole, with the bytes of the files it names", async () => {
    const body = await captureBody({
      provider: "chatgpt", conversation: "c1", url: "https://chatgpt.com/c/c1",
      raw: new Blob([JSON.stringify(CONVERSATION)]), reason: "changed", version: "1.0",
      files: [{ id: "file-1", message: "m1", name: "a.png", media_type: "image/png", body: new Blob([new Uint8Array([1, 2, 3])]) },
        { id: "file-2", message: "m1", error: "gone" },
        { id: "file-3", message: "m1", blob: "ab".repeat(32), size: 12 }],
      now: new Date("2026-10-26T00:00:00Z"),
    });
    const capture = JSON.parse(await body.text());
    expect(capture.polylogue_capture_kind).toBe("browser_llm_session");
    expect(capture.raw_provider_payload).toEqual(CONVERSATION);
    expect(capture.provenance).toEqual({ captured_at: "2026-10-26T00:00:00.000Z", source_url: "https://chatgpt.com/c/c1", extension_version: "1.0", reason: "changed" });
    expect(capture.session).toMatchObject({ provider: "chatgpt", provider_session_id: "c1", session_kind: "standard" });
    expect(capture.session.attachments[0]).toMatchObject({ provider_attachment_id: "file-1", message_provider_id: "m1", content_base64: "AQID" });
    expect(capture.session.attachments[1]).toMatchObject({ provider_attachment_id: "file-2", not_fetched: "gone" });
    expect(capture.session.attachments[2]).toMatchObject({ provider_attachment_id: "file-3", blob: "ab".repeat(32), size_bytes: 12 });
    expect(capture.session.attachments[2].content_base64).toBeUndefined();
  });

  it("encodes an empty downloaded file as present bytes", async () => {
    const body = await captureBody({
      provider: "chatgpt", conversation: "c1", url: "https://chatgpt.com/c/c1", reason: "files", version: "1.0",
      files: [{ id: "empty", message: "m1", body: new Blob([]) }],
    });
    const capture = JSON.parse(await body.text());
    expect(capture.session.attachments[0].content_base64).toBe("");
    expect(capture.session.attachments[0].not_fetched).toBeUndefined();
  });

  it("downloads and stores an empty file through the page port", async () => {
    const host = storingHost([{ id: "empty", message: "m1", inline: false }]);
    const page = chatgptPage({ fileBytes: new Uint8Array([]) });
    const chrome = fakeChrome({ operations: host.operations, pages: { 1: page.answer }, tabs: [{ id: 1, url: "https://chatgpt.com/c/c1" }] });
    const extension = start(chrome, { indexedDB: new IDBFactory() });
    expect((await extension.captureTab(1, "changed")).outcome).toBe("stored");
    await extension.capturer.downloads();
    expect(host.stored).toHaveLength(2);
    const file = JSON.parse(host.stored[1]).session.attachments[0];
    expect(file.content_base64).toBe("");
    expect(file.not_fetched).toBeUndefined();
    expect(await extension.outbox.count()).toBe(0);
  });

  it("holds a file larger than one base64 part byte for byte", async () => {
    const bytes = new Uint8Array(2 * 1024 * 1024 + 5).map((_, i) => (i * 31) % 251);
    const body = await captureBody({
      provider: "chatgpt", conversation: "c1", url: "https://chatgpt.com/c/c1", reason: "files", version: "1.0",
      raw: new Blob([JSON.stringify(CONVERSATION)]),
      files: [{ id: "big", message: "m1", body: new Blob([bytes]) }],
    });
    const capture = JSON.parse(await body.text());
    expect(Buffer.from(capture.session.attachments[0].content_base64, "base64").equals(Buffer.from(bytes))).toBe(true);
  });

  it("is stored before its files download, then again with their bytes; an unchanged conversation is not stored again", async () => {
    const host = storingHost([{ id: "file-1", message: "m1", file_id: "file-1", inline: false, stored: null }, { id: "inline", message: "m1", inline: true }]);
    const page = chatgptPage({ fileBytes: new Uint8Array([9, 9]) });
    const chrome = fakeChrome({ operations: host.operations, pages: { 1: page.answer }, tabs: [{ id: 1, url: "https://chatgpt.com/c/c1" }] });
    const extension = start(chrome, { indexedDB: new IDBFactory() });

    const first = await extension.captureTab(1, "changed");
    expect(first.outcome).toBe("stored");
    expect(host.stored).toHaveLength(1);
    const capture = JSON.parse(host.stored[0]);
    expect(capture.session.attachments.map((a) => a.provider_attachment_id)).toEqual(["file-1"]);
    expect(capture.session.attachments[0]).toMatchObject({ not_fetched: "downloading" });

    await extension.capturer.downloads();
    expect(host.stored).toHaveLength(2);
    const withFiles = JSON.parse(host.stored[1]);
    expect(withFiles.provenance.reason).toBe("files");
    expect(withFiles.raw_provider_payload).toEqual(capture.raw_provider_payload);
    expect(withFiles.session.attachments[0].content_base64).toBe("CQk=");
    expect(page.asked.filter((r) => r.type === "file")).toHaveLength(1);
    expect(await extension.outbox.count()).toBe(0);

    expect((await extension.captureTab(1, "changed")).outcome).toBe("unchanged");
    expect(host.stored).toHaveLength(2);
    const state = await extension.pageState(await chrome.tabs.get(1));
    expect(state).toMatchObject({ provider: "chatgpt", conversation: "c1", state: "archived", messages: 1 });
    expect(state.last.outcome).toBe("unchanged");
  });

  it("is not stored, and does not count as seen, when the page answer holds no conversation", async () => {
    const host = storingHost();
    let withBody = false;
    const answer = async (request) => {
      if (request.type === "page") return { status: 200, info: { id: "c1", temporary: false } };
      if (request.type === "conversation") {
        return withBody
          ? { status: 200, info: { id: "c1", temporary: false, url: "https://chatgpt.com/c/c1" }, body: new Blob([JSON.stringify(CONVERSATION)]) }
          : { status: 200, info: { id: "c1", temporary: false, url: "https://chatgpt.com/c/c1" } };
      }
      return { status: 400, error: "unknown_request" };
    };
    const chrome = fakeChrome({ operations: host.operations, pages: { 1: answer }, tabs: [{ id: 1, url: "https://chatgpt.com/c/c1" }] });
    const extension = start(chrome, { indexedDB: new IDBFactory() });

    expect(await extension.captureTab(1, "changed")).toBeNull();
    expect(host.stored).toHaveLength(0);
    const state = await extension.pageState(await chrome.tabs.get(1));
    expect(state.last).toMatchObject({ outcome: "failed", text: "no_conversation_content" });

    withBody = true;
    expect((await extension.captureTab(1, "changed")).outcome).toBe("stored");
    expect(JSON.parse(host.stored[0]).raw_provider_payload).toEqual(CONVERSATION);
  });

  it("keeps the conversation observation time when files finish later", async () => {
    vi.useFakeTimers({ toFake: ["Date"] });
    try {
      vi.setSystemTime(new Date("2026-10-27T00:00:00Z"));
      const host = storingHost([{ id: "file-1", message: "m1", inline: false }]);
      const page = chatgptPage({ fileBytes: new Uint8Array([1]) });
      let release;
      const held = new Promise((resolve) => { release = resolve; });
      const answer = (request) => request.type === "file" ? held.then(() => page.answer(request)) : page.answer(request);
      const chrome = fakeChrome({ operations: host.operations, pages: { 1: answer }, tabs: [{ id: 1, url: "https://chatgpt.com/c/c1" }] });
      const extension = start(chrome, { indexedDB: new IDBFactory() });
      await extension.captureTab(1, "changed");
      vi.setSystemTime(new Date("2026-10-28T00:00:00Z"));
      release();
      await extension.capturer.downloads();
      expect(host.stored).toHaveLength(2);
      expect(JSON.parse(host.stored[1]).provenance.captured_at).toBe("2026-10-27T00:00:00.000Z");
    } finally {
      vi.useRealTimers();
    }
  });

  it("names a file the archive holds by its hash instead of downloading it again", async () => {
    const hash = "cd".repeat(32);
    const host = storingHost([{ id: "file-1", message: "m1", file_id: "file-1", inline: false, stored: { blob: hash, size: 2 } }]);
    const page = chatgptPage({ fileBytes: new Uint8Array([9, 9]) });
    const chrome = fakeChrome({ operations: host.operations, pages: { 1: page.answer }, tabs: [{ id: 1, url: "https://chatgpt.com/c/c1" }] });
    const extension = start(chrome, { indexedDB: new IDBFactory() });
    expect((await extension.captureTab(1, "changed")).outcome).toBe("stored");
    await extension.capturer.downloads();
    expect(page.asked.filter((r) => r.type === "file")).toHaveLength(0);
    expect(host.stored).toHaveLength(1);
    expect(JSON.parse(host.stored[0]).session.attachments[0]).toMatchObject({ provider_attachment_id: "file-1", blob: hash, size_bytes: 2 });
  });

  it("downloads the files again after a restart cut their download off", async () => {
    const host = storingHost([{ id: "file-1", message: "m1", file_id: "file-1", inline: false, stored: null }]);
    let hang = true;
    const page = chatgptPage({ fileBytes: new Uint8Array([9, 9]) });
    const answer = (request) => (request.type === "file" && hang ? new Promise(() => {}) : page.answer(request));
    const chrome = fakeChrome({ operations: host.operations, pages: { 1: answer }, tabs: [{ id: 1, url: "https://chatgpt.com/c/c1" }] });
    const before = start(chrome, { indexedDB: new IDBFactory() });
    expect((await before.captureTab(1, "changed")).outcome).toBe("stored");
    expect(host.stored).toHaveLength(1);

    const restarted = start(chrome, { indexedDB: new IDBFactory() });
    const listFiles = host.operations["internal capture-files"];
    host.operations["internal capture-files"] = () => ({ status: 503, json: { error: "temporarily_unavailable" } });
    expect((await restarted.captureTab(1, "sweep")).outcome).toBe("unchanged");
    expect(host.stored).toHaveLength(1);
    expect(JSON.parse(host.stored[0]).session.attachments[0].not_fetched).toBe("downloading");
    host.operations["internal capture-files"] = listFiles;
    hang = false;
    expect((await restarted.captureTab(1, "sweep")).outcome).toBe("unchanged");
    await restarted.capturer.downloads();
    expect(host.stored).toHaveLength(2);
    expect(JSON.parse(host.stored[1]).session.attachments[0].content_base64).toBe("CQk=");
    // Once they are in, an unchanged conversation is left alone again.
    expect((await restarted.captureTab(1, "sweep")).outcome).toBe("unchanged");
    await restarted.capturer.downloads();
    expect(host.stored).toHaveLength(2);
  });

  it("keeps the newest conversation and its new files when it changes during a download", async () => {
    const files = [{ id: "first", message: "m1", inline: false }];
    const host = storingHost(files);
    const original = chatgptPage();
    let release;
    let downloading;
    const started = new Promise((resolve) => { downloading = resolve; });
    const held = new Promise((resolve) => { release = resolve; });
    let changed = false;
    const asked = [];
    const answer = async (request) => {
      if (request.type === "file") {
        asked.push(request.file.id);
        if (request.file.id === "first") { downloading(); await held; }
        return { status: 200, body: new Blob([request.file.id]) };
      }
      const reply = await original.answer(request);
      if (changed && request.type === "conversation") {
        reply.body = new Blob([JSON.stringify({ ...CONVERSATION, title: "Newest" })]);
      }
      return reply;
    };
    const chrome = fakeChrome({ operations: host.operations, pages: { 1: answer }, tabs: [{ id: 1, url: "https://chatgpt.com/c/c1" }] });
    const extension = start(chrome, { indexedDB: new IDBFactory() });
    await extension.captureTab(1, "changed");
    await started;
    changed = true;
    files.push({ id: "second", message: "m1", inline: false });
    const successor = extension.captureTab(1, "changed");
    // The newer observation cancels the capture in flight before its
    // provider request can acquire the shared lane. Release that lane first.
    release();
    await successor;
    await extension.capturer.downloads();
    const final = JSON.parse(host.stored.at(-1));
    expect(final.raw_provider_payload.title).toBe("Newest");
    expect(final.session.attachments.map((file) => file.content_base64))
      .toEqual([btoa("first"), btoa("second")]);
    expect(asked).toEqual(["first", "second"]);
    expect(host.stored.map((body) => JSON.parse(body).raw_provider_payload.title))
      .toEqual(["Synthetic", "Newest", "Newest"]);
  });

  /** A stored capture whose one file is being downloaded, held until `release`. */
  async function downloadInProgress(conversation = (request, page) => page.answer(request)) {
    const host = storingHost([{ id: "file-1", message: "m1", file_id: "file-1", inline: false, stored: null }]);
    const page = chatgptPage({ fileBytes: new Uint8Array([9, 9]) });
    let release;
    let downloading;
    const started = new Promise((resolve) => { downloading = resolve; });
    const held = new Promise((resolve) => { release = resolve; });
    const answer = async (request) => {
      if (request.type === "file") { downloading(); await held; }
      return request.type === "conversation" ? conversation(request, page) : page.answer(request);
    };
    const chrome = fakeChrome({ operations: host.operations, pages: { 1: answer }, tabs: [{ id: 1, url: "https://chatgpt.com/c/c1" }] });
    const extension = start(chrome, { indexedDB: new IDBFactory() });
    expect((await extension.captureTab(1, "changed")).outcome).toBe("stored");
    await started;
    return { host, chrome, extension, release, job: extension.capturer.downloading.get("chatgpt:c1") };
  }

  it("stores the downloaded files when a capture that overlaps the download finds the conversation unchanged", async () => {
    const { host, extension, release, job } = await downloadInProgress();
    const overlapping = extension.captureTab(1, "changed");
    await until(() => job.capturesInFlight === 1, "the overlapping capture");
    // The provider's lane is the download's until the file is in.
    release();
    expect((await overlapping).outcome).toBe("unchanged");
    await extension.capturer.downloads();
    expect(host.stored).toHaveLength(2);
    expect(JSON.parse(host.stored[1]).provenance.reason).toBe("files");
    expect(JSON.parse(host.stored[1]).session.attachments[0].content_base64).toBe("CQk=");
  });

  it("stores the downloaded files when a capture that overlaps the download fails", async () => {
    let fail = false;
    const { host, extension, release, job } = await downloadInProgress((request, page) => (
      fail ? { status: 404, error: "conversation_gone" } : page.answer(request)));
    fail = true;
    const overlapping = extension.captureTab(1, "changed");
    await until(() => job.capturesInFlight === 1, "the overlapping capture");
    release();
    expect(await overlapping).toBeNull();
    await extension.capturer.downloads();
    expect(host.stored).toHaveLength(2);
    expect(JSON.parse(host.stored[1]).session.attachments[0].content_base64).toBe("CQk=");
    const state = await extension.pageState(await extension.capturer.chrome.tabs.get(1));
    expect(state.last).toMatchObject({ outcome: "failed", text: "conversation_gone" });
  });

  it("downloads, waits for the captures in flight, stores and ends, in that order", async () => {
    const { host, extension, release, job } = await downloadInProgress();
    const phases = [];
    const enter = job.enter.bind(job);
    job.enter = (phase) => { enter(phase); phases.push(phase); };
    expect(job.phase).toBe("downloading");
    const overlapping = extension.captureTab(1, "changed");
    await until(() => job.capturesInFlight === 1, "the overlapping capture");
    release();
    await overlapping;
    await extension.capturer.downloads();
    expect(phases).toEqual(["waiting for captures", "storing", "ended"]);
    expect(extension.capturer.downloading.size).toBe(0);
    expect(host.stored).toHaveLength(2);
  });

  it("records the provider refusal when none of its files can be downloaded", async () => {
    const host = storingHost([{ id: "file-1", message: "m1", file_id: "file-1", inline: false, stored: null }]);
    const page = chatgptPage();
    const chrome = fakeChrome({ operations: host.operations, pages: { 1: page.answer }, tabs: [{ id: 1, url: "https://chatgpt.com/c/c1" }] });
    const extension = start(chrome, { indexedDB: new IDBFactory() });
    expect((await extension.captureTab(1, "changed")).outcome).toBe("stored");
    await extension.capturer.downloads();
    expect(host.stored).toHaveLength(2);
    expect(JSON.parse(host.stored[1]).session.attachments[0].not_fetched).toBe("gone");
    const { problems } = await chrome.storage.session.get({ problems: [] });
    expect(problems[0].text).toContain("could not be downloaded");
  });

  it("is not stored again over several cycles when the provider only reorders its JSON or adds null keys", async () => {
    const host = storingHost();
    let read = 0;
    const versions = [
      { conversation_id: "c1", safe_urls: ["https://a.example", "https://b.example"], mapping: {} },
      { safe_urls: ["https://b.example", "https://a.example"], mapping: {}, conversation_id: "c1" },
      { mapping: {}, conversation_id: "c1", flags: null, safe_urls: ["https://a.example", "https://b.example"] },
    ];
    const page = async (request) => (request.type === "conversation"
      ? { status: 200, info: { id: "c1", url: "https://chatgpt.com/c/c1" }, body: new Blob([JSON.stringify(versions[read++ % 3])]) }
      : { status: 200, info: { id: "c1" } });
    const chrome = fakeChrome({ operations: host.operations, pages: { 1: page }, tabs: [{ id: 1, url: "https://chatgpt.com/c/c1" }] });
    const extension = start(chrome, { indexedDB: new IDBFactory() });
    const outcomes = [];
    for (let cycle = 0; cycle < 3; cycle += 1) outcomes.push((await extension.captureTab(1, "changed")).outcome);
    expect(outcomes).toEqual(["stored", "unchanged", "unchanged"]);
    expect(host.stored).toHaveLength(1);
    // A restarted extension keeps what it last captured.
    const restarted = start(chrome, { indexedDB: new IDBFactory() });
    expect((await restarted.captureTab(1, "changed")).outcome).toBe("unchanged");
  });

  it("keeps raw bytes in the outbox when file discovery and delivery are offline", async () => {
    const host = storingHost();
    host.operations["internal capture-files"] = () => ({ status: 503, json: { error: "offline" } });
    host.operations["capture"] = () => ({ status: 503, json: { error: "offline" } });
    const page = chatgptPage();
    const indexedDB = new IDBFactory();
    const chrome = fakeChrome({ operations: host.operations, pages: { 1: page.answer }, tabs: [{ id: 1, url: "https://chatgpt.com/c/c1" }] });
    const extension = start(chrome, { indexedDB });
    expect((await extension.captureTab(1, "changed")).outcome).toBe("waiting");
    const [waiting] = await extension.outbox.all();
    expect(JSON.parse(await waiting.body.text()).raw_provider_payload).toEqual(CONVERSATION);
    expect(await new (extension.outbox.constructor)(indexedDB).count()).toBe(1);
  });

  it("captures reordered text parts as changed content", async () => {
    const host = storingHost();
    let parts = ["first", "second"];
    const original = chatgptPage();
    const answer = async (request) => {
      const reply = await original.answer(request);
      if (request.type === "conversation") {
        reply.body = new Blob([JSON.stringify({ ...CONVERSATION, parts })]);
      }
      return reply;
    };
    const chrome = fakeChrome({ operations: host.operations, pages: { 1: answer }, tabs: [{ id: 1, url: "https://chatgpt.com/c/c1" }] });
    const extension = start(chrome, { indexedDB: new IDBFactory() });
    expect((await extension.captureTab(1, "changed")).outcome).toBe("stored");
    parts = ["second", "first"];
    expect((await extension.captureTab(1, "changed")).outcome).toBe("stored");
    expect(host.stored).toHaveLength(2);
  });

  it("waits in the outbox while the host refuses it, and survives a restart of the extension", async () => {
    const indexedDB = new IDBFactory();
    const host = storingHost();
    let refusing = true;
    const store = host.operations["capture"];
    host.operations["capture"] = (body) => (refusing ? { status: 500, json: { error: "write_failed" } } : store(body));
    const page = chatgptPage();
    const chrome = fakeChrome({ operations: host.operations, pages: { 1: page.answer }, tabs: [{ id: 1, url: "https://chatgpt.com/c/c1" }] });
    const before = start(chrome, { indexedDB });
    expect((await before.captureTab(1, "changed")).outcome).toBe("waiting");
    expect(await before.outbox.count()).toBe(1);
    expect((await before.popup(1)).waiting).toBe(1);
    const [waiting] = await before.outbox.all();
    expect(waiting.last_error).toBe("write_failed");
    const held = await waiting.body.text();

    // A new version of the extension starts over the same browser storage.
    refusing = false;
    const restarted = fakeChrome({ operations: host.operations, pages: { 1: page.answer }, tabs: [{ id: 1, url: "https://chatgpt.com/c/c1" }] });
    const after = start(restarted, { indexedDB });
    restarted.runtime.onInstalled.fire({ reason: "update" });
    await until(async () => host.stored.length >= 1, "the waiting capture stored");
    await until(async () => (await after.outbox.count()) === 0, "an empty outbox");
    // The capture the old version held, byte for byte.
    expect(host.stored[0]).toBe(held);
  });

  it("is sent again when the host stored it but its answer was lost; the host keeps one copy", async () => {
    const indexedDB = new IDBFactory();
    const host = storingHost();
    const store = host.operations["capture"];
    let lose = true;
    host.operations["capture"] = (body) => {
      const answer = store(body);
      if (lose) { lose = false; return "drop"; }
      return answer;
    };
    const page = chatgptPage();
    const chrome = fakeChrome({ operations: host.operations, pages: { 1: page.answer }, tabs: [{ id: 1, url: "https://chatgpt.com/c/c1" }] });
    const extension = start(chrome, { indexedDB });
    await extension.captureTab(1, "changed");
    expect(await extension.outbox.count()).toBe(1);
    await extension.tick();
    expect(await extension.outbox.count()).toBe(0);
    const sends = chrome.calls.filter((c) => c.type === "capture");
    expect(sends).toHaveLength(2);
    expect(sends[1].capture).toEqual(sends[0].capture);
    expect(host.stored).toHaveLength(1);
  });

  it("says plainly when Polylogue is not installed for the browser", async () => {
    const chrome = fakeChrome({ installed: false, tabs: [{ id: 1, url: "https://chatgpt.com/c/c1" }], pages: { 1: chatgptPage().answer } });
    const extension = start(chrome, { indexedDB: new IDBFactory() });
    const state = await extension.popup(1);
    expect(state.status.host).toBe("missing");
    expect(state.page.host).toBe("missing");
    expect(globalThis.polylogueWording.conversation(state.page)).toMatchObject({
      tone: "bad", label: "Polylogue is not installed for this browser",
    });
    expect((await extension.captureTab(1, "changed")).outcome).toBe("waiting");
    expect(await extension.outbox.count()).toBe(1);
    const { problems } = await chrome.storage.session.get({ problems: [] });
    expect(problems[0].text).toContain("host_missing");
  });

  it("says that Polylogue did not answer when the installed host fails", async () => {
    const host = storingHost();
    host.operations["internal conversations"] = () => ({ status: 500, json: { error: "internal" } });
    const chrome = fakeChrome({ operations: host.operations, tabs: [{ id: 1, url: "https://chatgpt.com/c/c1" }], pages: { 1: chatgptPage().answer } });
    const extension = start(chrome, { indexedDB: new IDBFactory() });
    const state = await extension.pageState(await chrome.tabs.get(1));
    expect(state.host).toBe("failed");
    expect(globalThis.polylogueWording.conversation(state)).toMatchObject({
      state: "host_did_not_answer", tone: "bad", label: "Polylogue did not answer",
    });
  });

  it("is still shown as in the archive after a later capture of it fails", async () => {
    const host = storingHost();
    const page = chatgptPage();
    let fail = false;
    const answer = (request) => (fail && request.type === "conversation" ? { status: 404, error: "conversation_gone" } : page.answer(request));
    const chrome = fakeChrome({ operations: host.operations, pages: { 1: answer }, tabs: [{ id: 1, url: "https://chatgpt.com/c/c1" }] });
    const extension = start(chrome, { indexedDB: new IDBFactory() });
    expect((await extension.captureTab(1, "changed")).outcome).toBe("stored");
    fail = true;
    expect(await extension.captureTab(1, "changed")).toBeNull();
    const state = await extension.pageState(await chrome.tabs.get(1));
    expect(state.last.outcome).toBe("failed");
    expect(globalThis.polylogueWording.conversation(state)).toEqual({
      state: "in_archive", tone: "warn", label: "In the archive: 1 message",
      detail: "The last capture failed just now: conversation_gone",
    });
  });

  it("a gemini page is captured from the turns it shows", async () => {
    const host = storingHost();
    const session = { provider: "gemini", provider_session_id: "g1", title: "T", turns: [{ role: "user", text: "q", ordinal: 0 }] };
    const chrome = fakeChrome({
      operations: host.operations,
      tabs: [{ id: 2, url: "https://gemini.google.com/app/g1" }],
      pages: { 2: async () => ({ status: 200, info: { id: "g1", url: "https://gemini.google.com/app/g1", temporary: false, session } }) },
    });
    const extension = start(chrome, { indexedDB: new IDBFactory() });
    expect((await extension.captureTab(2, "changed")).outcome).toBe("stored");
    expect(chrome.calls.some((c) => c.operation === "internal capture-files")).toBe(false);
    const capture = json(new TextEncoder().encode(host.stored[0]));
    expect(capture.raw_provider_payload).toBeUndefined();
    expect(capture.session.turns).toEqual(session.turns);
  });
});

it("finishes a pending capture when the daemon already has its file bytes", async () => {
  const files = [{ id: "file-1", message: "m1", file_id: "file-1", inline: false, stored: null }];
  const host = storingHost(files);
  const page = chatgptPage();
  const answer = (request) => request.type === "file" ? new Promise(() => {}) : page.answer(request);
  const chrome = fakeChrome({ operations: host.operations, pages: { 1: answer }, tabs: [{ id: 1, url: "https://chatgpt.com/c/c1" }] });
  const before = start(chrome, { indexedDB: new IDBFactory() });
  expect((await before.captureTab(1, "changed")).outcome).toBe("stored");
  expect(JSON.parse(host.stored[0]).session.attachments[0].not_fetched).toBe("downloading");

  const hash = "cd".repeat(32);
  files[0].stored = { blob: hash, size: 2 };
  await chrome.storage.session.clear();
  const restarted = start(chrome, { indexedDB: new IDBFactory() });
  expect((await restarted.captureTab(1, "sweep")).outcome).toBe("stored");
  expect(host.stored).toHaveLength(2);
  expect(JSON.parse(host.stored[1]).session.attachments[0]).toMatchObject({ blob: hash, size_bytes: 2 });
  expect(JSON.parse(host.stored[1]).session.attachments[0].not_fetched).toBeUndefined();
  expect((await restarted.captureTab(1, "sweep")).outcome).toBe("unchanged");
  expect(host.stored).toHaveLength(2);
});

it.each([false, true])("starts another download after the loop exits (storage error: %s)", async (failed) => {
  const attempts = [];
  const capturer = new Capturer({ chrome: {}, host: {}, outbox: {}, seen: {} });
  const base = { conversation: "one", raw: new Blob(["first payload"]) };
  const newer = { conversation: "one", raw: new Blob(["new payload"]) };
  const file = { id: "file" };
  capturer.fetchFiles = async () => [{ ...file, body: new Blob(["bytes"]) }];
  capturer.keep = async (value) => {
    attempts.push(value.raw);
    if (attempts.length === 1) {
      await Promise.resolve();
      queueMicrotask(() => queueMicrotask(() => capturer.download(1, "one", newer, [file], [file])));
      if (failed) throw new Error("synthetic storage failure");
    }
    return true;
  };
  capturer.download(1, "one", base, [file], [file]);
  const first = capturer.downloading.get("one");
  await first.finished;
  await capturer.downloads();
  expect(attempts).toEqual([base.raw, newer.raw]);
  expect(first.phase).toBe("ended");
  expect(first.next).toBeNull();
  expect(first.available.size).toBe(0);
  expect(capturer.downloading.size).toBe(0);
});

it("keeps the next capture when the old page closes during a download", async () => {
  const capturer = new Capturer({ chrome: {}, host: {}, outbox: {}, seen: {} });
  const first = { conversation: "one", raw: new Blob(["first payload"]) };
  const newer = { conversation: "one", raw: new Blob(["new payload"]) };
  const file = { id: "file" };
  let fail;
  const blocked = new Promise((_, reject) => { fail = reject; });
  let began;
  const started = new Promise((resolve) => { began = resolve; });
  let calls = 0;
  capturer.fetchFiles = async () => {
    calls += 1;
    if (calls === 1) { began(); return blocked; }
    return [{ ...file, body: new Blob(["bytes"]) }];
  };
  const stored = [];
  capturer.keep = async (base) => { stored.push(base.raw); return true; };
  capturer.download(1, "one", first, [file], [file]);
  await started;
  capturer.download(2, "one", newer, [file], [file]);
  fail(new Error("page closed"));
  await capturer.downloads();
  expect(stored).toEqual([newer.raw]);
  expect(calls).toBe(2);
  expect(capturer.downloading.size).toBe(0);
});

it("keeps unchanged conversation fingerprints across a browser restart", async () => {
  const host = storingHost([]);
  const page = chatgptPage();
  const chrome = fakeChrome({ operations: host.operations, pages: { 1: page.answer }, tabs: [{ id: 1, url: "https://chatgpt.com/c/c1" }] });
  const before = start(chrome, { indexedDB: new IDBFactory() });
  expect((await before.captureTab(1, "changed")).outcome).toBe("stored");
  await chrome.storage.session.clear();
  const after = start(chrome, { indexedDB: new IDBFactory() });
  expect((await after.captureTab(1, "sweep")).outcome).toBe("unchanged");
  expect(host.stored).toHaveLength(1);
});

it("an_unchanged_conversation_is_not_fetched_on_a_sweep, while an explicit capture still fetches", async () => {
  const host = storingHost([]);
  const page = chatgptPage();
  let shown = "https://chatgpt.com/c/c1\n5\nhello";
  const answer = (request) => (request.type === "page"
    ? { status: 200, info: { id: "c1", temporary: false }, shown }
    : page.answer(request));
  const chrome = fakeChrome({ operations: host.operations, pages: { 1: answer }, tabs: [{ id: 1, url: "https://chatgpt.com/c/c1" }] });
  const fetches = () => page.asked.filter((request) => request.type === "conversation").length;
  expect((await start(chrome, { indexedDB: new IDBFactory() }).captureTab(1, "changed")).outcome).toBe("stored");
  expect(fetches()).toBe(1);
  // The service worker restarts between sweeps.
  const extension = start(chrome, { indexedDB: new IDBFactory() });
  expect((await extension.captureTab(1, "sweep")).outcome).toBe("unchanged");
  expect(fetches()).toBe(1);
  expect((await extension.captureTab(1, "asked", true)).outcome).toBe("stored");
  expect(fetches()).toBe(2);
  // The page shows another turn: the sweep fetches.
  shown = "https://chatgpt.com/c/c1\n11\nhello again";
  expect((await extension.captureTab(1, "sweep")).outcome).toBe("unchanged");
  expect(fetches()).toBe(3);
  // Only the explicit capture was stored again.
  expect(host.stored).toHaveLength(2);
});

it.each(["missing", "unknown"])("captures cached content when the daemon reports it as %s", async (state) => {
  const host = storingHost([]);
  const page = chatgptPage();
  const chrome = fakeChrome({ operations: host.operations, pages: { 1: page.answer }, tabs: [{ id: 1, url: "https://chatgpt.com/c/c1" }] });
  const before = start(chrome, { indexedDB: new IDBFactory() });
  expect((await before.captureTab(1, "changed")).outcome).toBe("stored");
  host.stored.length = 0;
  host.operations["internal conversations"] = () => state === "unknown"
    ? { status: 503, json: { error: "daemon_required", reason: { name: "daemon_did_not_answer", fields: {} } } }
    : { json: { provider: "chatgpt", conversations: [{ id: "c1", messages: null, downloading: null }] } };
  await chrome.storage.session.clear();
  const after = start(chrome, { indexedDB: new IDBFactory() });
  expect((await after.captureTab(1, "sweep")).outcome).toBe("stored");
  expect(host.stored).toHaveLength(1);
});

it("confirms a submitted turn only from a stored provider snapshot", async () => {
  const host = storingHost();
  const page = chatgptPage();
  const chrome = fakeChrome({ operations: host.operations, pages: { 1: page.answer }, tabs: [{ id: 1, url: "https://chatgpt.com/c/c1" }] });
  const extension = start(chrome, { indexedDB: new IDBFactory() });
  const capture = (turn, text) => extension.capturer.capture(1, "chatgpt", { force: true, confirm: { turn, text } });
  expect(await capture("m1", "hello")).toMatchObject({ outcome: "stored", confirmedTurn: "m1" });
  expect(await capture("other", "hello")).toMatchObject({ confirmedTurn: null });
  expect(await capture("m1", "different")).toMatchObject({ confirmedTurn: null });
});

describe("the background worker's own messages", () => {
  function worker(extra = {}) {
    const host = storingHost();
    Object.assign(host.operations, extra);
    const page = chatgptPage();
    const chrome = fakeChrome({ operations: host.operations, pages: { 1: page.answer }, tabs: [{ id: 1, url: "https://chatgpt.com/c/c1" }] });
    return { host, chrome, extension: start(chrome, { indexedDB: new IDBFactory() }) };
  }

  it("does not capture a tab that closed while its capture waited out the spacing", async () => {
    const { host, chrome, extension } = worker();
    await extension.changed(1, "changed");
    expect(host.stored).toHaveLength(1);
    vi.useFakeTimers({ toFake: ["setTimeout"] });
    try {
      const second = extension.changed(1, "changed");
      for (let i = 0; i < 1000 && vi.getTimerCount() === 0; i += 1) await Promise.resolve();
      chrome.tabs.list.length = 0;
      await vi.advanceTimersByTimeAsync(10_000);
      await expect(second).resolves.toBeUndefined();
    } finally {
      vi.useRealTimers();
    }
    expect(host.stored).toHaveLength(1);
  });

  it("stops automatic captures when they are switched off, and hides and shows the panel per site", async () => {
    const { host, extension } = worker();
    const { handlers } = extension;
    const tab = { id: 1, url: "https://chatgpt.com/c/c1" };
    expect(await handlers["polylogue.settings"]({ automatic: false })).toMatchObject({ automatic: false, panel: true });
    await extension.changed(1, "changed");
    expect(host.stored).toHaveLength(0);
    await handlers["polylogue.hide-panel"]({ site: "chatgpt.com" });
    expect((await extension.pageState(tab)).panel).toBe(false);
    await handlers["polylogue.settings"]({ show_site: "chatgpt.com" });
    expect((await extension.pageState(tab)).panel).toBe(true);
  });

  it("shows the running import jobs, the last three ended ones and the actions awaiting approval", async () => {
    const jobs = [
      { id: "e4", state: "finished" }, { id: "e3", state: "cancelled" }, { id: "e2", state: "finished" },
      { id: "r1", state: "running" }, { id: "e1", state: "finished" },
    ];
    const actions = [{ id: "a1", state: "awaiting_approval" }, { id: "a2", state: "submitted" }];
    const { extension } = worker({
      "rows": ({ selection }) => ({ json: { reply: { rows: selection === "import_jobs" ? jobs : actions } } }),
    });
    const shown = await extension.popup(null);
    expect(shown.importJob.map((job) => job.id)).toEqual(["r1", "e4", "e3", "e2"]);
    expect(shown.approvals.map((action) => action.id)).toEqual(["a1"]);
    expect(shown.version).toBe("9.9.9");
  });
});
