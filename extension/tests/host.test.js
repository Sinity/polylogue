import { describe, expect, it } from "vitest";
import { Host, HostError, exchange } from "../src/background/host.js";
import { start } from "../src/background/main.js";
import { IDBFactory } from "fake-indexeddb";
import { notePayload } from "../src/background/notes.js";
import { fakeChrome } from "./support/fake_chrome.js";

describe("the native messaging host", () => {
  it("sends an operation by name and joins a large Unicode answer", async () => {
    const text = "A conversation 🦀 ".repeat(70_000);
    const chrome = fakeChrome({ operations: { "internal echo": (args) => ({ json: args }) } });
    const answer = await new Host(chrome.runtime).call("internal echo", { text }, "caller-id");
    expect(answer).toEqual({ text });
    expect(chrome.calls).toEqual([{ version: 2, type: "call", operation: "internal echo", arguments: { text }, request_id: "caller-id" }]);
  });

  it("carries a daemon refusal's code and typed reason unchanged", async () => {
    const reason = { name: "capture_unreadable", fields: { problem: "no conversation" } };
    const chrome = fakeChrome({ operations: { "internal capture-files": () => ({ status: 400, json: { error: "bad_request", message: "cannot read", reason } }) } });
    const error = await new Host(chrome.runtime).call("internal capture-files", { capture: "{}" }).catch((e) => e);
    expect(error).toBeInstanceOf(HostError);
    expect([error.code, error.message, error.reason]).toEqual(["bad_request", "cannot read", reason]);
  });

  it("sends capture JSON without changing its bytes", async () => {
    const body = '{ "text": "' + "x".repeat(1024 * 1024) + '" }';
    const chrome = fakeChrome({ operations: { capture: () => ({ json: { new: true } }) } });
    expect(await new Host(chrome.runtime).storeCapture(new Blob([body]))).toEqual({ new: true });
    expect(chrome.calls).toEqual([{ version: 2, type: "capture", capture: body }]);
  });

  it("tells a missing host apart from one that stopped", async () => {
    const missing = fakeChrome({ installed: false });
    expect((await new Host(missing.runtime).call("set").catch((e) => e)).code).toBe("host_missing");
    const stopping = fakeChrome({ operations: { set: () => "drop" } });
    expect((await new Host(stopping.runtime).call("set").catch((e) => e)).code).toBe("host_stopped");
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

it("rejects out-of-order parts and unreadable answers", async () => {
  for (const frame of [
    { version: 2, type: "part", sequence: 1, last: true, data: "e30=" },
    { version: 2, type: "answer", answer: null },
  ]) {
    const runtime = { connectNative: () => scriptedPort((_message, emit) => emit(frame)) };
    await expect(exchange(runtime, { type: "call", operation: "set", arguments: {} })).rejects.toMatchObject({ code: "host_answer_unreadable" });
  }
});

it("asks for an extension reload on another host version", async () => {
  const runtime = { connectNative: () => scriptedPort((_message, emit) => emit({ version: 1, type: "answer" })) };
  await expect(exchange(runtime, { type: "spooled", provider: "chatgpt", ids: [] }))
    .rejects.toMatchObject({ code: "native_version_mismatch", message: "Reload the extension to use this browser host." });
});

it("joins spool and archive state while the daemon is down", async () => {
  const chrome = fakeChrome({ operations: {
    spooled: () => ({ json: { spooled: [true, false] } }),
    "internal conversations": () => ({ status: 503, json: { error: "daemon_required", reason: { name: "daemon_did_not_answer", fields: {} } } }),
  } });
  const answer = await globalThis.polylogueCaptureState.conversations(new Host(chrome.runtime), "chatgpt", ["saved", "absent"]);
  expect(answer).toEqual({ daemon: "no_answer", conversations: [{ id: "saved", state: "spooled" }, { id: "absent", state: "unknown" }] });
  expect(globalThis.polylogueCaptureState.ofArchiveAnswer("chatgpt", answer)).toEqual(["stored_daemon_not_answering", "daemon_did_not_answer"]);
});

it("forwards a selection's request ID and the popup's answer by operation name", async () => {
  const chrome = fakeChrome({ operations: {
    "internal selection-note": () => ({ json: { id: "note-id", repeat: true } }),
    "internal chat answer": () => ({ json: { action: { status: "denied" } } }),
  } });
  const { handlers } = start(chrome, { indexedDB: new IDBFactory() });
  const source = { provider: "chatgpt", conversation_id: "c1", message_id: "m1", page_url: "https://chatgpt.com/c/c1", selected_at: "2026-10-26T00:00:00Z" };
  const note = { request_id: "note-id", body: "my note", quote: "selected text", source };
  const sender = { tab: { url: source.page_url }, url: source.page_url, frameId: 0 };
  for (let i = 0; i < 2; i += 1) {
    expect(await handlers["polylogue.note"]({ note }, sender)).toEqual({ note: { id: "note-id", repeat: true } });
  }
  expect(chrome.calls.slice(0, 2)).toEqual(Array.from({ length: 2 }, () => ({ version: 2, type: "call", operation: "internal selection-note",
    arguments: { body: note.body, quote: note.quote, ...source }, request_id: "note-id" })));
  expect(await handlers["polylogue.answer"]({ id: "a/1", value: "deny" })).toEqual({ action: { status: "denied" } });
  expect(chrome.calls[2]).toMatchObject({ operation: "internal chat answer", arguments: { id: "a/1", value: "deny" } });
  await expect(handlers["polylogue.reader"]()).rejects.toMatchObject({ code: "reader_unavailable" });
});
