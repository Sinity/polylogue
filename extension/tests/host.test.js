import { describe, expect, it } from "vitest";
import { Host, HostError, exchange } from "../src/background/host.js";
import { notePayload } from "../src/background/notes.js";
import { fakeChrome } from "./support/fake_chrome.js";

describe("the native messaging host", () => {
  it("receives a body of any size in acknowledged appended bytes and answers in appended bytes", async () => {
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
    expect(() => notePayload(note, { ...sender, frameId: 3 })).toThrow("note_not_from_a_chat_page");
    expect(() => notePayload({ ...note, body: " " }, sender)).toThrow("the_note_needs_text_and_a_quote");
  });
});
