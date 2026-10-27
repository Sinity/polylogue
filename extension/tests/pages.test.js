import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { JSDOM } from "jsdom";

const SRC = join(dirname(fileURLToPath(import.meta.url)), "../src");
const source = (path) => readFileSync(join(SRC, path), "utf8");

function event() {
  const listeners = [];
  return { addListener: (f) => listeners.push(f), removeListener: () => {}, fire: (...a) => listeners.map((f) => f(...a)) };
}

/** A chat page with the extension's page scripts, its fetches answered by `routes`. */
function page(url, scripts, { html = "<body></body>", routes = {}, resources = [] } = {}) {
  const dom = new JSDOM(html, { url, runScripts: "outside-only", pretendToBeVisual: true });
  const { window } = dom;
  // Chrome's Blob reads its bytes; jsdom's does not.
  window.Blob = Blob;
  const sent = [];
  const fetched = [];
  window.chrome = {
    runtime: { id: "x", onConnect: event(), onMessage: event(), sendMessage: async (m) => { sent.push(m); return {}; } },
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
  const session = ["https://chatgpt.com/api/auth/session", [200, { accessToken: "token", account: { id: "acct" } }]];

  it("reads the conversation with the page's sign-in", async () => {
    const { window, fetched } = page("https://chatgpt.com/c/c1", ["content/chatgpt.js"], {
      routes: Object.fromEntries([session, ["https://chatgpt.com/backend-api/conversation/c1", [200, { conversation_id: "c1" }]]]),
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

  it("pages through the four parts of the conversation list", async () => {
    const { window } = page("https://chatgpt.com/", ["content/chatgpt.js"], {
      routes: Object.fromEntries([session,
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

  it("answers a provider's request to wait with its delay", async () => {
    const { window } = page("https://chatgpt.com/c/c1", ["content/chatgpt.js"], {
      routes: { "https://chatgpt.com/api/auth/session": [429, "", { "retry-after": "30" }] },
    });
    expect(await window.polylogueProvider.fetchConversation(null)).toEqual({ status: 429, retry_after: "30" });
  });
});

describe("the Grok page script", () => {
  it("joins a conversation's three answers into one JSON object", async () => {
    const base = "https://grok.com/rest/app-chat/conversations/g1";
    const { window } = page("https://grok.com/c/g1", ["content/grok.js"], {
      routes: { [`${base}/responses`]: [200, { responses: [] }], [`${base}/response-node`]: [500, ""], [base]: [200, { conversationId: "g1" }] },
    });
    const answer = await window.polylogueProvider.fetchConversation(null);
    expect(JSON.parse(await answer.body.text())).toEqual({ conversation: { conversationId: "g1" }, responses: { responses: [] } });
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
    await new Promise((resolve) => { const check = () => (received.at(-1)?.type === "end" ? resolve() : setTimeout(check, 5)); check(); });
    return received;
  }

  it("answers through a port with the facts, then the body as appended bytes", async () => {
    const { window } = page("https://claude.ai/chat/c1", ["content/claude.js", "content/page.js"]);
    const big = "x".repeat(5 * 1024 * 1024);
    window.polylogueProvider.fetchConversation = async () => ({ status: 200, info: { id: "c1" }, body: new window.Blob([big]) });
    const received = await exchanged(window);
    expect(received[0]).toEqual({ type: "facts", status: 200, info: { id: "c1" }, has_body: true });
    const appendedBytes = received.filter((m) => m.type === "appended_bytes");
    expect(appendedBytes.length).toBe(20);
    expect(appendedBytes.map((p) => Buffer.from(p.data, "base64").toString()).join("")).toBe(big);
  });

  it("distinguishes empty bodies from answers without a body", async () => {
    for (const body of [null, new Blob([]), new Response("")]) {
      const { window } = page("https://claude.ai/chat/c1", ["content/claude.js", "content/page.js"]);
      window.polylogueProvider.fetchConversation = async () => ({ status: 200, info: { id: "c1" }, body });
      const received = await exchanged(window);
      expect(received[0].has_body).toBe(body !== null);
      expect(received.filter((message) => message.type === "appended_bytes")).toHaveLength(0);
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
    await new Promise((resolve) => { const check = () => (received.some((m) => m.type === "appended_bytes") ? resolve() : setTimeout(check, 5)); check(); });
    expect(received.at(-1).type).toBe("appended_bytes");
    release();
    await new Promise((resolve) => { const check = () => (received.at(-1)?.type === "end" ? resolve() : setTimeout(check, 5)); check(); });
    const bytes = Buffer.concat(received.filter((m) => m.type === "appended_bytes").map((p) => Buffer.from(p.data, "base64")));
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
});

describe("the in-page panel", () => {
  const html = "<body><div data-message-id='m1'><p id='a'>first part</p><p id='b'>second part</p></div><div data-message-id='m2'><p id='c'>other</p></div></body>";

  it("makes a note from a selection inside one message", () => {
    const { window } = page("https://chatgpt.com/c/c1", ["content/chatgpt.js", "wording.js", "content/panel.js"], { html });
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

  it("says what the archive holds of the conversation", () => {
    const { window } = page("https://chatgpt.com/c/c1", ["content/chatgpt.js", "wording.js"], { html });
    window.eval(source("content/panel.js"));
    const said = window.polylogueWording.conversation({ provider: "chatgpt", conversation: "c1", state: "archived", messages: 12, daemon: "running" });
    expect(said).toMatchObject({ tone: "ok", label: "In the archive: 12 messages" });
    expect(window.polylogueWording.conversation({ provider: "chatgpt", conversation: "c1", state: "unknown", daemon: "no_answer" }).label)
      .toBe("The daemon did not answer");
    expect(window.polylogueWording.conversation({ provider: "chatgpt", conversation: "c1", state: "spooled", daemon: "running" }).label)
      .toBe("Stored; Polylogue is reading it");
  });
});
