import { describe, expect, it } from "vitest";
import { IDBFactory } from "fake-indexeddb";
import { start } from "../src/background/main.js";
import { captureBody } from "../src/background/capture.js";
import { fakeChrome, json, until } from "./support/fake_chrome.js";

const CONVERSATION = {
  conversation_id: "c1", title: "Synthetic", mapping: {
    n1: { id: "n1", message: { id: "m1", author: { role: "user" }, content: { parts: ["hello"] } } },
  },
};

/** A host that stores captures by their bytes, as the real one does. */
function storingHost(files = []) {
  const stored = [];
  const routes = {
    "POST /v1/captures/files": () => ({ json: { conversation: { id: "c1" }, files } }),
    "POST /v1/captures": (body) => {
      const text = new TextDecoder().decode(body);
      const known = stored.includes(text);
      if (!known) stored.push(text);
      return { json: { new: !known } };
    },
    "POST /v1/conversations": () => ({ json: { daemon: "running", conversations: [{ id: "c1", state: "archived", messages: 1 }] } }),
    "GET /v1/status": () => ({ json: { daemon: "running", chat_actions: false } }),
    "GET /v1/backfill": () => ({ json: { jobs: [] } }),
    "GET /v1/chat-actions": () => ({ json: { actions: [] } }),
    "POST /v1/backfill/claim": () => ({ json: { job: null } }),
  };
  return { stored, routes };
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
    const body = captureBody({
      provider: "chatgpt", conversation: "c1", url: "https://chatgpt.com/c/c1",
      raw: new Blob([JSON.stringify(CONVERSATION)]), reason: "changed", version: "1.0",
      files: [{ id: "file-1", message: "m1", name: "a.png", media_type: "image/png", bytes: new Uint8Array([1, 2, 3]) },
        { id: "file-2", message: "m1", error: "gone" }],
      now: new Date("2026-10-26T00:00:00Z"),
    });
    const capture = JSON.parse(await body.text());
    expect(capture.polylogue_capture_kind).toBe("browser_llm_session");
    expect(capture.raw_provider_payload).toEqual(CONVERSATION);
    expect(capture.provenance).toEqual({ captured_at: "2026-10-26T00:00:00.000Z", source_url: "https://chatgpt.com/c/c1", extension_version: "1.0", reason: "changed" });
    expect(capture.session).toMatchObject({ provider: "chatgpt", provider_session_id: "c1", session_kind: "standard" });
    expect(capture.session.attachments[0]).toMatchObject({ provider_attachment_id: "file-1", message_provider_id: "m1", content_base64: "AQID" });
    expect(capture.session.attachments[1]).toMatchObject({ provider_attachment_id: "file-2", not_fetched: "gone" });
  });

  it("is stored once the host acknowledges it, and an unchanged conversation is not stored again", async () => {
    const host = storingHost([{ id: "file-1", message: "m1", file_id: "file-1", inline: false }, { id: "inline", message: "m1", inline: true }]);
    const page = chatgptPage({ fileBytes: new Uint8Array([9, 9]) });
    const chrome = fakeChrome({ routes: host.routes, pages: { 1: page.answer }, tabs: [{ id: 1, url: "https://chatgpt.com/c/c1" }] });
    const extension = start(chrome, { indexedDB: new IDBFactory() });

    const first = await extension.captureTab(1, "changed");
    expect(first.outcome).toBe("stored");
    expect(host.stored).toHaveLength(1);
    const capture = JSON.parse(host.stored[0]);
    expect(capture.session.attachments.map((a) => a.provider_attachment_id)).toEqual(["file-1"]);
    expect(capture.session.attachments[0].content_base64).toBe("CQk=");
    expect(page.asked.filter((r) => r.type === "file")).toHaveLength(1);
    expect(await extension.outbox.count()).toBe(0);

    expect((await extension.captureTab(1, "changed")).outcome).toBe("unchanged");
    expect(host.stored).toHaveLength(1);
    const state = await extension.pageState(await chrome.tabs.get(1));
    expect(state).toMatchObject({ provider: "chatgpt", conversation: "c1", state: "archived", messages: 1 });
    expect(state.last.outcome).toBe("unchanged");
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
    const chrome = fakeChrome({ routes: host.routes, pages: { 1: page }, tabs: [{ id: 1, url: "https://chatgpt.com/c/c1" }] });
    const extension = start(chrome, { indexedDB: new IDBFactory() });
    const outcomes = [];
    for (let cycle = 0; cycle < 3; cycle += 1) outcomes.push((await extension.captureTab(1, "changed")).outcome);
    expect(outcomes).toEqual(["stored", "unchanged", "unchanged"]);
    expect(host.stored).toHaveLength(1);
    // A restarted extension keeps what it last captured.
    const restarted = start(chrome, { indexedDB: new IDBFactory() });
    expect((await restarted.captureTab(1, "changed")).outcome).toBe("unchanged");
  });

  it("waits in the outbox while the host refuses it, and survives a restart of the extension", async () => {
    const indexedDB = new IDBFactory();
    const host = storingHost();
    let refusing = true;
    const store = host.routes["POST /v1/captures"];
    host.routes["POST /v1/captures"] = (body) => (refusing ? { status: 500, json: { error: "write_failed" } } : store(body));
    const page = chatgptPage();
    const chrome = fakeChrome({ routes: host.routes, pages: { 1: page.answer }, tabs: [{ id: 1, url: "https://chatgpt.com/c/c1" }] });
    const before = start(chrome, { indexedDB });
    expect((await before.captureTab(1, "changed")).outcome).toBe("waiting");
    expect(await before.outbox.count()).toBe(1);
    expect((await before.popup(1)).waiting).toBe(1);
    const [waiting] = await before.outbox.all();
    expect(waiting.last_error).toBe("write_failed");
    const held = await waiting.body.text();

    // A new version of the extension starts over the same browser storage.
    refusing = false;
    const restarted = fakeChrome({ routes: host.routes, pages: { 1: page.answer }, tabs: [{ id: 1, url: "https://chatgpt.com/c/c1" }] });
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
    const store = host.routes["POST /v1/captures"];
    let lose = true;
    host.routes["POST /v1/captures"] = (body) => {
      const answer = store(body);
      if (lose) { lose = false; return "drop"; }
      return answer;
    };
    const page = chatgptPage();
    const chrome = fakeChrome({ routes: host.routes, pages: { 1: page.answer }, tabs: [{ id: 1, url: "https://chatgpt.com/c/c1" }] });
    const extension = start(chrome, { indexedDB });
    await extension.captureTab(1, "changed");
    expect(await extension.outbox.count()).toBe(1);
    await extension.tick();
    expect(await extension.outbox.count()).toBe(0);
    const sends = chrome.calls.filter((c) => c.path === "/v1/captures");
    expect(sends).toHaveLength(2);
    expect(sends[1].body).toEqual(sends[0].body);
    expect(host.stored).toHaveLength(1);
  });

  it("says plainly when Polylogue is not installed for the browser", async () => {
    const chrome = fakeChrome({ installed: false, tabs: [{ id: 1, url: "https://chatgpt.com/c/c1" }], pages: { 1: chatgptPage().answer } });
    const extension = start(chrome, { indexedDB: new IDBFactory() });
    const state = await extension.popup(1);
    expect(state.status.host).toBe("missing");
    expect(state.page.host).toBe("missing");
    expect(await extension.captureTab(1, "changed")).toBeNull();
    const { problems } = await chrome.storage.session.get({ problems: [] });
    expect(problems[0].text).toContain("host_missing");
  });

  it("a gemini page is captured from the turns it shows", async () => {
    const host = storingHost();
    const session = { provider: "gemini", provider_session_id: "g1", title: "T", turns: [{ role: "user", text: "q", ordinal: 0 }] };
    const chrome = fakeChrome({
      routes: host.routes,
      tabs: [{ id: 2, url: "https://gemini.google.com/app/g1" }],
      pages: { 2: async () => ({ status: 200, info: { id: "g1", url: "https://gemini.google.com/app/g1", temporary: false, session } }) },
    });
    const extension = start(chrome, { indexedDB: new IDBFactory() });
    expect((await extension.captureTab(2, "changed")).outcome).toBe("stored");
    expect(chrome.calls.some((c) => c.path === "/v1/captures/files")).toBe(false);
    const capture = json(new TextEncoder().encode(host.stored[0]));
    expect(capture.raw_provider_payload).toBeUndefined();
    expect(capture.session.turns).toEqual(session.turns);
  });
});
