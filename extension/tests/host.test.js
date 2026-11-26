import { describe, expect, it } from "vitest";
import { Host, HostError, exchange } from "../src/background/host.js";
import { notePayload } from "../src/background/notes.js";
import { fakeChrome } from "./support/fake_chrome.js";

describe("the native messaging host", () => {
  it("receives a body of any size in acknowledged chunks and answers in chunks", async () => {
    const chrome = fakeChrome({ routes: { "POST /v1/captures": (body) => ({ bytes: body }) } });
    const big = new Uint8Array(200 * 1024).map((_, i) => i % 251);
    const answer = await exchange(chrome.runtime, "POST", "/v1/captures", new Blob([big]));
    expect(answer.status).toBe(200);
    expect(new Uint8Array(await answer.body.arrayBuffer())).toEqual(big);
    expect(chrome.calls[0].body).toEqual(big);
  });

  it("turns a refusal into an error with its code", async () => {
    const chrome = fakeChrome({ routes: { "GET /v1/reader": () => ({ status: 503, json: { error: "reader_unavailable", message: "no reader" } }) } });
    const error = await new Host(chrome.runtime).reader().catch((e) => e);
    expect(error).toBeInstanceOf(HostError);
    expect([error.code, error.status, error.message]).toEqual(["reader_unavailable", 503, "no reader"]);
  });

  it("does not take an answer that is not JSON for a success", async () => {
    const chrome = fakeChrome({ routes: {
      "POST /v1/captures": () => ({ bytes: new TextEncoder().encode("<html>not json"), type: "text/html" }),
      "GET /v1/reader": () => ({ status: 502, bytes: new TextEncoder().encode("bad gateway"), type: "text/plain" }),
    } });
    const host = new Host(chrome.runtime);
    expect((await host.storeCapture(new Blob(["x"])).catch((e) => e)).code).toBe("host_answer_unreadable");
    expect((await host.reader().catch((e) => e)).code).toBe("host_http_502");
  });

  it("extension_popup_decision_uses_answer: carries the popup's Allow or Deny to the answer route", async () => {
    const chrome = fakeChrome({ routes: { "POST /v1/chat-actions/a%2F1/answer": (body) => ({ json: { answered: JSON.parse(new TextDecoder().decode(body)) } }) } });
    const answer = await new Host(chrome.runtime).answerChatAction("a/1", "deny");
    expect(answer).toEqual({ answered: { value: "deny" } });
  });

  it("tells a missing host apart from one that stopped", async () => {
    const missing = fakeChrome({ installed: false });
    expect((await new Host(missing.runtime).status().catch((e) => e)).code).toBe("host_missing");
    const stopping = fakeChrome({ routes: { "GET /v1/status": () => "drop" } });
    expect((await new Host(stopping.runtime).status().catch((e) => e)).code).toBe("host_stopped");
  });
});

describe("a note on a selected passage", () => {
  const sender = { tab: { url: "https://chatgpt.com/c/c1" }, url: "https://chatgpt.com/c/c1", frameId: 0 };
  const note = {
    request_id: "019c6e27-e55b-73d1-87d8-4e01f1f75043", body: "mine", quote: "theirs",
    source: { provider: "chatgpt", conversation_id: "c1", message_id: "m1", page_url: "https://chatgpt.com/c/c1", selected_at: "2026-10-26T00:00:00Z" },
  };

  it("is accepted only from the conversation page it names", () => {
    expect(notePayload(note, sender)).toEqual(note);
    expect(() => notePayload({ ...note, source: { ...note.source, conversation_id: "other" } }, sender)).toThrow("the_page_is_not_that_conversation");
    expect(() => notePayload(note, { ...sender, url: "https://chatgpt.com/c/c2" })).toThrow("page_changed_since_the_selection");
    const inProject = "https://chatgpt.com/g/g-p-abc/c/c1";
    expect(notePayload({ ...note, source: { ...note.source, page_url: inProject } }, { ...sender, url: inProject })).toMatchObject({ source: { conversation_id: "c1" } });
    const query = "https://gemini.google.com/?conversation=c1";
    expect(notePayload({ ...note, source: { ...note.source, provider: "gemini", page_url: query } }, { ...sender, url: query }).source.provider).toBe("gemini");
    expect(() => notePayload(note, { ...sender, frameId: 3 })).toThrow("note_not_from_a_chat_page");
    expect(() => notePayload({ ...note, body: " " }, sender)).toThrow("the_note_needs_text_and_a_quote");
  });

  it("is refused from a page that is not an https page of the provider's own host", () => {
    const refuse = (provider, url) => {
      const named = { ...note, source: { ...note.source, provider, page_url: url } };
      return () => notePayload(named, { ...sender, url });
    };
    expect(refuse("chatgpt", "http://chatgpt.com/c/c1")).toThrow("the_page_is_not_that_conversation");
    expect(refuse("chatgpt", "https://chatgpt.example/c/c1")).toThrow("the_page_is_not_that_conversation");
    expect(refuse("claude-ai", "https://chatgpt.com/c/c1")).toThrow("the_page_is_not_that_conversation");
    expect(refuse("unknown", "https://chatgpt.com/c/c1")).toThrow("the_page_is_not_that_conversation");
  });

  it("reads the conversation from each provider's own path", () => {
    const accept = (provider, url) => notePayload(
      { ...note, source: { ...note.source, provider, page_url: url } },
      { ...sender, url },
    ).source.conversation_id;
    expect(accept("claude-ai", "https://claude.ai/chat/c1")).toBe("c1");
    expect(accept("grok", "https://grok.com/c/c1")).toBe("c1");
    expect(accept("gemini", "https://gemini.google.com/app/c1")).toBe("c1");
    expect(() => accept("claude-ai", "https://claude.ai/c/c1")).toThrow("the_page_is_not_that_conversation");
    expect(() => accept("gemini", "https://gemini.google.com/c/c1")).toThrow("the_page_is_not_that_conversation");
  });

  it("must name the message it was selected in", () => {
    expect(() => notePayload({ ...note, source: { ...note.source, message_id: "" } }, sender)).toThrow("the_selection_names_no_message");
  });
});

function scriptedPort(reply) {
  const listeners = new Set();
  const disconnected = new Set();
  const port = {
    onMessage: { addListener: (fn) => listeners.add(fn), removeListener: (fn) => listeners.delete(fn) },
    onDisconnect: { addListener: (fn) => disconnected.add(fn), removeListener: (fn) => disconnected.delete(fn) },
    postMessage: (message) => reply(message, (frame) => queueMicrotask(() => { for (const fn of listeners) fn(frame); })),
    disconnect: () => {},
  };
  return port;
}

it("rejects a response before upload completion and an incorrect body acknowledgment", async () => {
  const early = { connectNative: () => scriptedPort((message, emit) => {
    if (message.type === "request") emit({ type: "response", status: 200 });
  }) };
  await expect(exchange(early, "POST", "/", new Blob(["x"]))).rejects.toMatchObject({ code: "host_answer_unreadable" });
  const badAck = { connectNative: () => scriptedPort((message, emit) => {
    if (message.type === "body") emit({ type: "body_ack", sequence: message.sequence + 1 });
  }) };
  await expect(exchange(badAck, "POST", "/", new Blob(["x"]))).rejects.toMatchObject({ code: "host_answer_unreadable" });
});

it("rejects duplicate acknowledgments while preparing the next chunk", async () => {
  const runtime = { connectNative: () => scriptedPort((message, emit) => {
    if (message.type === "body") {
      emit({ type: "body_ack", sequence: message.sequence });
      emit({ type: "body_ack", sequence: message.sequence });
    }
  }) };
  await expect(exchange(runtime, "POST", "/", new Blob([new Uint8Array(200000)]))).rejects.toMatchObject({ code: "host_answer_unreadable" });
});
