import { describe, expect, it } from "vitest";
import { IDBFactory } from "fake-indexeddb";
import { webcrypto } from "node:crypto";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { ProviderScheduler } from "../src/background/provider.js";
import { start } from "../src/background/main.js";
import { chatActionInPage } from "../src/actions/chatgpt.js";
import { PROVIDERS } from "../src/background/tabs.js";
import { JSDOM } from "jsdom";
import { event, fakeChrome, json, portPair, until } from "./support/fake_chrome.js";

const SRC = join(dirname(fileURLToPath(import.meta.url)), "../src");
const source = (path) => readFileSync(join(SRC, path), "utf8");
const chatgptSession = ["https://chatgpt.com/api/auth/session", [200, { accessToken: "token", account: { id: "acct" } }]];

/** A chat page with the extension's page scripts, its fetches answered by `routes`. */
function page(url, scripts, { html = "<body></body>", routes = {}, resources = [], storage = {} } = {}) {
  const dom = new JSDOM(html, { url, runScripts: "outside-only", pretendToBeVisual: true });
  const { window } = dom;
  for (const [key, value] of Object.entries(storage)) window.localStorage.setItem(key, value);
  // Chrome's Blob reads its bytes; jsdom's does not.
  window.Blob = Blob;
  window.TextEncoder = TextEncoder;
  Object.defineProperty(window.crypto, "subtle", { value: webcrypto.subtle });
  const sent = [];
  const fetched = [];
  window.chrome = {
    runtime: { id: "x", onConnect: event(), onMessage: event(), sendMessage: async (m) => { sent.push(m); return { ok: true }; } },
  };
  // A response as the page's fetch gives it, with the page's own Blob.
  const respond = (status, text, headers = {}) => ({
    ok: status >= 200 && status < 300,
    status,
    headers: { get: (name) => headers[name.toLowerCase()] ?? null },
    json: async () => JSON.parse(text),
    text: async () => text,
    blob: async () => new window.Blob([text]),
    body: new Blob([text]).stream(),
  });
  window.fetch = async (path, init = {}) => {
    const href = new URL(path, url).href;
    fetched.push({ href, init });
    const route = Object.entries(routes).find(([prefix]) => href.startsWith(prefix));
    if (!route) return respond(404, "{}");
    const [status, body, headers] = route[1];
    return respond(status, typeof body === "string" ? body : JSON.stringify(body), headers);
  };
  window.performance.getEntriesByType = () => resources.map((name) => ({ name }));
  for (const script of scripts) window.eval(source(script));
  return { window, sent, fetched };
}

describe("the ChatGPT page script", () => {
  it("reads the conversation with the page's sign-in", async () => {
    const { window, fetched } = page("https://chatgpt.com/c/c1", ["content/chatgpt.js"], {
      routes: Object.fromEntries([chatgptSession, ["https://chatgpt.com/backend-api/conversation/c1", [200, { conversation_id: "c1" }]]]),
    });
    const answer = await window.polylogueProvider.fetchConversation(null);
    expect(answer.status).toBe(200);
    expect(answer.info).toMatchObject({ id: "c1", temporary: false });
    expect(JSON.parse(await answer.body.text())).toEqual({ conversation_id: "c1" });
    expect(fetched[1].init.headers).toEqual({ Authorization: "Bearer token", "ChatGPT-Account-Id": "acct" });
  });

  it("finds a temporary chat's ID in the requests the page made", async () => {
    const { window } = page("https://chatgpt.com/?temporary-chat=true", ["content/chatgpt.js"], {
      resources: ["https://chatgpt.com/backend-api/conversation/tmp-1", "https://chatgpt.com/backend-api/me"],
    });
    expect(await window.polylogueProvider.currentConversation()).toEqual({ id: "tmp-1", temporary: true });
  });

  it("pages through the four partitions of the conversation list", async () => {
    const { window } = page("https://chatgpt.com/", ["content/chatgpt.js"], {
      routes: Object.fromEntries([chatgptSession,
        ["https://chatgpt.com/backend-api/conversations?offset=0&limit=28&order=updated&is_archived=false&is_starred=false",
          [200, { items: [{ id: "a", update_time: 1767225600 }], total: 1 }]],
        ["https://chatgpt.com/backend-api/conversations?offset=0&limit=28&order=updated&is_archived=true&is_starred=true",
          [200, { items: [], total: 0 }]],
      ]),
    });
    const first = await window.polylogueProvider.list(null);
    expect(first.info).toMatchObject({ next: "1:0", done: false, newest_first: true });
    expect(first.info.conversations).toEqual([{ id: "a", title: null, updated_at: "2026-01-01T00:00:00.000Z" }]);
    const last = await window.polylogueProvider.list("3:0");
    expect(last.info).toMatchObject({ next: null, done: true });
  });

  it("keeps listing a partition whose reply has no total until a short page", async () => {
    const full = Array.from({ length: 28 }, (_, i) => ({ id: `c${i}` }));
    const { window } = page("https://chatgpt.com/", ["content/chatgpt.js"], {
      routes: Object.fromEntries([chatgptSession,
        ["https://chatgpt.com/backend-api/conversations?offset=0&limit=28&order=updated&is_archived=false&is_starred=false", [200, { items: full }]],
        ["https://chatgpt.com/backend-api/conversations?offset=28&limit=28&order=updated&is_archived=false&is_starred=false", [200, { items: [{ id: "last" }] }]],
      ]),
    });
    const first = await window.polylogueProvider.list(null);
    expect(first.info).toMatchObject({ next: "0:28", done: false });
    const second = await window.polylogueProvider.list("0:28");
    expect(second.info).toMatchObject({ next: "1:0", next_partition: "1:0", done: false });
  });

  it("reports a refused sign-in check as the refusal, and an empty one as signed out", async () => {
    const refused = page("https://chatgpt.com/c/c1", ["content/chatgpt.js"], {
      routes: { "https://chatgpt.com/api/auth/session": [403, "{}"] },
    });
    expect(await refused.window.polylogueProvider.fetchConversation(null)).toEqual({ status: 403, error: "chatgpt_http_403" });
    const signedOut = page("https://chatgpt.com/c/c1", ["content/chatgpt.js"], {
      routes: { "https://chatgpt.com/api/auth/session": [200, {}] },
    });
    expect(await signedOut.window.polylogueProvider.fetchConversation(null)).toEqual({ status: 401, error: "signed_out" });
  });

  it("answers a provider's request to wait with its delay", async () => {
    const { window } = page("https://chatgpt.com/c/c1", ["content/chatgpt.js"], {
      routes: { "https://chatgpt.com/api/auth/session": [429, "", { "retry-after": "30" }] },
    });
    expect(await window.polylogueProvider.fetchConversation(null)).toEqual({ status: 429, retry_after: "30" });
  });
});

describe("the Claude page script", () => {
  const org = "0b1c2d3e-0000-4000-8000-000000000001";
  const list = `https://claude.ai/api/organizations/${org}/chat_conversations`;
  const selected = { "omelette-org-settings-cache": JSON.stringify({ orgUuid: org }) };
  const open = (routes) => {
    const { window } = page("https://claude.ai/recents", ["content/claude.js"], { routes, storage: selected });
    return window.polylogueProvider;
  };

  it("an_unknown_selected_organization_captures_nothing", async () => {
    // The page has requested another workspace's API, but claude.ai has no
    // selected organization stored: the script must not borrow that ID.
    const other = "https://claude.ai/api/organizations/9f8e7d6c-0000-4000-8000-000000000002/chat_conversations";
    const { window, fetched } = page("https://claude.ai/chat/c1", ["content/claude.js"], {
      routes: { [other]: [200, [{ uuid: "theirs", name: "Not ours" }]] },
      resources: [`${other}?limit=30`],
      storage: { "omelette-org-settings-cache": JSON.stringify({ orgUuid: "not-a-uuid" }) },
    });
    const provider = window.polylogueProvider;
    expect(await provider.list(null)).toEqual({ status: 401, error: "signed_out" });
    expect(await provider.fetchConversation(null)).toEqual({ status: 401, error: "signed_out" });
    expect(fetched).toEqual([]);
  });

  it("reads the list of the organization claude.ai has selected", async () => {
    const provider = open({ [`${list}?limit=100&offset=0`]: [200, [{ uuid: "u1", name: "One", updated_at: "2026-01-01T00:00:00Z" }]] });
    expect((await provider.list(null)).info.conversations).toEqual([{ id: "u1", title: "One", updated_at: "2026-01-01T00:00:00Z" }]);
  });

  it("ends the list on an empty page, not a short one", async () => {
    const provider = open({ [`${list}?limit=100&offset=0`]: [200, [{ uuid: "u1", name: "One" }]] });
    expect((await provider.list(null)).info).toMatchObject({ next: "1", done: false });
  });

  it("refuses a list reply that is not a list", async () => {
    const provider = open({ [`${list}?limit=100&offset=0`]: [200, { error: "x" }] });
    expect(await provider.list(null)).toEqual({ status: 502, error: "claude_list_unreadable" });
  });
});

describe("the Grok page script", () => {
  it("joins a conversation's three answers into one JSON object", async () => {
    const base = "https://grok.com/rest/app-chat/conversations/g1";
    const { window } = page("https://grok.com/c/g1", ["content/grok.js"], {
      routes: { [`${base}/responses`]: [200, { responses: [] }], [`${base}/response-node`]: [404, ""], [base]: [200, { conversationId: "g1" }] },
    });
    const answer = await window.polylogueProvider.fetchConversation(null);
    expect(JSON.parse(await answer.body.text())).toEqual({ conversation: { conversationId: "g1" }, responses: { responses: [] } });
  });

  it("fails a capture when the optional response nodes are refused for a reason other than being absent", async () => {
    const base = "https://grok.com/rest/app-chat/conversations/g1";
    const { window } = page("https://grok.com/c/g1", ["content/grok.js"], {
      routes: { [`${base}/responses`]: [200, { responses: [] }], [`${base}/response-node`]: [403, ""], [base]: [200, { conversationId: "g1" }] },
    });
    expect(await window.polylogueProvider.fetchConversation(null)).toEqual({ status: 403, error: "grok_http_403" });
  });

  it("fetches a file only from Grok's asset host", async () => {
    const { window } = page("https://grok.com/c/g1", ["content/grok.js"]);
    expect(await window.polylogueProvider.fetchFile({ url: "\\\\attacker.example/x" })).toMatchObject({ status: 400 });
  });
});

describe("the Gemini page script", () => {
  it("captures the prompts and answers the page shows, in order", async () => {
    const { window } = page("https://gemini.google.com/app/abc", ["content/gemini.js"], {
      html: "<body><user-query><div class='message-content'>question</div></user-query><model-response><message-content>answer</message-content></model-response></body>",
    });
    const answer = await window.polylogueProvider.fetchConversation(null);
    expect(answer.info.session.turns).toEqual([
      { provider_turn_id: null, role: "user", text: "question", ordinal: 0 },
      { provider_turn_id: null, role: "assistant", text: "answer", ordinal: 1 },
    ]);
  });
});

describe("the shared page script", () => {
  async function exchanged(window) {
    const received = [];
    const port = { name: "polylogue", onMessage: event(), onDisconnect: event(), postMessage: (m) => received.push(m) };
    window.chrome.runtime.onConnect.fire(port);
    port.onMessage.fire({ type: "conversation" });
    await until(() => received.at(-1)?.type === "end", "the end of the answer");
    return received;
  }

  it("answers through a port with the facts, then the body in chunks", async () => {
    const { window } = page("https://claude.ai/chat/c1", ["content/claude.js", "content/page.js"]);
    const big = "x".repeat(5 * 1024 * 1024);
    window.polylogueProvider.fetchConversation = async () => ({ status: 200, info: { id: "c1" }, body: new window.Blob([big]) });
    const received = await exchanged(window);
    expect(received[0]).toEqual({ type: "facts", status: 200, info: { id: "c1" }, has_body: true });
    const chunk = received.filter((m) => m.type === "chunk");
    expect(chunk.length).toBe(20);
    expect(chunk.map((p) => Buffer.from(p.data, "base64").toString()).join("")).toBe(big);
  });

  it("distinguishes empty bodies from answers without a body", async () => {
    for (const body of [null, new Blob([]), new Response(""), new Response(null, { status: 204 })]) {
      const { window } = page("https://claude.ai/chat/c1", ["content/claude.js", "content/page.js"]);
      window.polylogueProvider.fetchConversation = async () => ({ status: 200, info: { id: "c1" }, body });
      const received = await exchanged(window);
      expect(received[0].has_body).toBe(body !== null);
      expect(received.filter((message) => message.type === "chunk")).toHaveLength(0);
      window.close();
    }
  });

  it("sends a download as appended bytes as it arrives, before it has finished", async () => {
    const { window } = page("https://claude.ai/chat/c1", ["content/claude.js", "content/page.js"]);
    const chunk = new Uint8Array(1024).fill(7);
    let release;
    const held = new Promise((resolve) => { release = resolve; });
    const stream = new ReadableStream({
      async start(controller) {
        controller.enqueue(chunk);
        await held;
        controller.enqueue(chunk);
        controller.close();
      },
    });
    window.polylogueProvider.fetchConversation = async () => ({ status: 200, info: { id: "c1" }, body: new Response(stream) });
    const received = [];
    const port = { name: "polylogue", onMessage: event(), onDisconnect: event(), postMessage: (m) => received.push(m) };
    window.chrome.runtime.onConnect.fire(port);
    port.onMessage.fire({ type: "conversation" });
    // The first bytes arrive while the rest of the download is held back.
    await until(() => received.some((m) => m.type === "chunk"), "the first chunk");
    expect(received.at(-1).type).toBe("chunk");
    release();
    await until(() => received.at(-1)?.type === "end", "the end of the answer");
    const bytes = Buffer.concat(received.filter((m) => m.type === "chunk").map((p) => Buffer.from(p.data, "base64")));
    expect(bytes.length).toBe(2 * chunk.length);
    expect(bytes.every((b) => b === 7)).toBe(true);
  });

  it("tells the background when the conversation's text changes, not when the markup does", async () => {
    const { window, sent } = page("https://claude.ai/chat/c1", ["content/claude.js", "content/page.js"], {
      html: "<body><main><p>first answer</p></main><div id='menu'></div></body>",
    });
    const pause = () => new Promise((resolve) => setTimeout(resolve, 3300));
    await pause();
    expect(sent).toEqual([{ type: "polylogue.changed", provider: "claude-ai", reason: "opened" }]);
    // A menu opening and closing, as pages do all the time.
    window.document.getElementById("menu").append(window.document.createElement("span"));
    await pause();
    window.document.getElementById("menu").replaceChildren();
    await pause();
    expect(sent).toHaveLength(1);
    window.document.querySelector("main").append(window.document.createTextNode(" and a new turn"));
    await pause();
    expect(sent).toHaveLength(2);
    expect(sent[1].reason).toBe("changed");
  }, 20_000);

  it("captures an early same-length edit once and ignores an unrelated menu", async () => {
    const url = "https://chatgpt.com/c/c1";
    const tab = { id: 1, url };
    const nativeUrl = "https://chatgpt.com/backend-api/conversation/c1";
    const tail = "z".repeat(2100);
    const snapshot = (text) => ({ conversation_id: "c1", mapping: {
      first: { message: { id: "m1", author: { role: "user" }, content: { parts: [text] } } },
      second: { message: { id: "m2", author: { role: "assistant" }, content: { parts: [tail] } } },
    } });
    const routes = Object.fromEntries([chatgptSession, [nativeUrl, [200, snapshot("first")]]]);
    const { window, sent, fetched } = page(url, ["content/chatgpt.js"], {
      routes,
      html: `<main><p data-message-id="m1">first</p><p data-message-id="m2">${tail}</p><div role="menu" id="menu"></div></main>`,
    });
    const stored = [];
    const chrome = fakeChrome({ tabs: [tab], operations: {
      capture: (bytes) => { stored.push(json(bytes)); return { json: { new: true } }; },
      "internal capture-files": () => ({ json: { conversation: { provider: "chatgpt", id: "c1" }, files: [] } }),
      "internal conversations": () => ({ json: { daemon: "running", conversations: [{ id: "c1", state: "archived", messages: 2 }] } }),
    } });
    chrome.tabs.connect = () => {
      const [worker, contentPage] = portPair("polylogue");
      window.chrome.runtime.onConnect.fire(contentPage);
      return worker;
    };
    let timer = null;
    window.setTimeout = (fn) => { timer = fn; };
    window.clearTimeout = () => { timer = null; };
    const settle = async () => {
      await Promise.resolve();
      const scheduled = timer;
      timer = null;
      scheduled?.();
    };
    try {
      window.eval(source("content/page.js"));
      await settle();
      const extension = start(chrome, { indexedDB: new IDBFactory() });
      const pageMessages = window.chrome.runtime.sendMessage;
      window.chrome.runtime.sendMessage = (message) => {
        void pageMessages(message);
        return chrome.runtime.sendMessage(message, { tab });
      };
      expect((await extension.captureTab(tab.id, "opened")).outcome).toBe("stored");
      const shown = "shown chatgpt:c1";
      const original = (await chrome.storage.local.get(shown))[shown];

      routes[nativeUrl][1] = snapshot("other");
      window.document.querySelector('[data-message-id="m1"]').firstChild.data = "other";
      await settle();
      expect(sent.filter((message) => message.type === "polylogue.changed").map((message) => message.reason)).toEqual(["opened", "changed"]);
      await until(() => stored.length === 2, "both native snapshots to be archived");
      // Wait for the automatic capture to publish its observed page state.
      await until(async () => (await chrome.storage.local.get(shown))[shown] !== original, "the renewed page state");

      window.document.getElementById("menu").textContent = "Rename conversation";
      await settle();
      expect(sent.filter((message) => message.type === "polylogue.changed")).toHaveLength(2);
      expect((await extension.captureTab(tab.id, "sweep")).outcome).toBe("unchanged");
      expect(fetched.filter(({ href }) => href === nativeUrl)).toHaveLength(2);
      expect(stored.map((capture) => capture.raw_provider_payload)).toEqual([snapshot("first"), snapshot("other")]);
    } finally { window.close(); }
  });
});

describe("the in-page panel", () => {
  const html = "<body><div data-message-id='m1'><p id='a'>first part</p><p id='b'>second part</p></div><div data-message-id='m2'><p id='c'>other</p></div></body>";

  it("makes a note from a selection inside one message", () => {
    const { window } = page("https://chatgpt.com/c/c1", ["content/chatgpt.js", "capture_state.js", "wording.js", "content/panel.js"], { html });
    const { selectionNote } = window.polylogueTestHooks.panel;
    const select = (from, to) => {
      const range = window.document.createRange();
      range.setStart(window.document.getElementById(from).firstChild, 0);
      range.setEnd(window.document.getElementById(to).firstChild, 5);
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      return selection;
    };
    const note = selectionNote(select("a", "b"), window.location.href, window.polylogueProvider);
    expect(note.source).toMatchObject({ provider: "chatgpt", conversation_id: "c1", message_id: "m1" });
    expect(note.quote).toContain("first part");
    expect(selectionNote(select("a", "c"), window.location.href, window.polylogueProvider)).toBeNull();
  });
});

it("stops a MAIN-world action immediately on a retryable provider failure", async () => {
  const { window, fetched } = page("https://chatgpt.com/c/c1", [], {
    routes: { "https://chatgpt.com/backend-api/conversation/c1": [503, "", { "retry-after": "Thu, 01 Jan 2026 00:01:00 GMT" }] },
  });
  const execute = window.eval(`(${chatActionInPage.toString()})`);
  const result = await execute({ provider: "chatgpt", conversation: "c1", text: "hello", model: "X", submit: true }, "run");
  expect(result).toMatchObject({ ok: false, status: 503, retry_after: "Thu, 01 Jan 2026 00:01:00 GMT", submission_may_have_occurred: false });
  expect(fetched).toHaveLength(1);
  window.close();
});

it("refuses a requested effort on an account whose model switcher has no effort menu", async () => {
  const { window } = page("https://chatgpt.com/", [], {
    html: "<body><div id='prompt-textarea'></div><button data-testid='model-switcher-dropdown-button'>Model</button></body>",
  });
  const execute = window.eval(`(${chatActionInPage.toString()})`);
  const result = await execute({ provider: "chatgpt", conversation: null, text: "hello", model: "X", effort: "Extended", submit: false }, "run");
  expect(result).toMatchObject({ ok: false, submission_may_have_occurred: false });
  expect(result.detail).toContain("unsupported effort");
  window.close();
});

it("charges each background HTTP request before fetch, including hidden authentication", async () => {
  const scheduler = new ProviderScheduler({ dailyBudget: 1, now: () => 0 });
  const { window, fetched } = page("https://chatgpt.com/c/c1", ["content/chatgpt.js", "content/page.js"], {
    routes: { "https://chatgpt.com/api/auth/session": [200, { accessToken: "synthetic" }] },
  });
  window.chrome.runtime.sendMessage = async () => {
    try { await scheduler.authorize("chatgpt"); return { ok: true }; }
    catch (error) { return { error: error.message, until: error.until }; }
  };
  await expect(scheduler.run("chatgpt", () => window.polylogueProvider.fetchConversation("c1"), { background: true }))
    .rejects.toMatchObject({ until: 86400000 });
  expect(fetched).toHaveLength(1);
  window.close();
});

it("the content fetch wrapper stops an operation on the first retryable response", async () => {
  const scheduler = new ProviderScheduler();
  const { window, fetched } = page("https://chatgpt.com/c/c1", ["content/chatgpt.js", "content/page.js"], {
    routes: { "https://chatgpt.com/api/auth/session": [503, "", { "retry-after": "30" }] },
  });
  window.chrome.runtime.sendMessage = async () => {
    await scheduler.authorize("chatgpt", 1); return { ok: true };
  };
  await expect(scheduler.run("chatgpt", () => window.polylogueProvider.fetchConversation("c1"), { owner: 1 }))
    .rejects.toMatchObject({ status: 503, retryAfter: "30" });
  expect(fetched).toHaveLength(1);
  window.close();
});

it("has a page script and host permission for every host of every provider it reads", () => {
  const manifest = JSON.parse(readFileSync(join(SRC, "../manifest.json"), "utf8"));
  const scripted = manifest.content_scripts.flatMap((entry) => entry.matches);
  const hosts = Object.values(PROVIDERS).flatMap((site) => site.hosts.map((host) => `https://${host}/*`));
  expect([...hosts].sort()).toEqual([...scripted].sort());
  expect([...hosts].sort()).toEqual([...manifest.host_permissions].sort());
});
