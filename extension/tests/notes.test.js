import { describe, expect, it } from "vitest";
import { selectionNotePayload } from "../src/notes.js";
const source = { provider: "chatgpt", conversation_id: "c1", message_id: "m1", page_url: "https://chatgpt.com/c/c1", selected_at: "2026-10-24T00:00:00Z" };
const note = { request_id: "019c6e27-e55b-73d1-87d8-4e01f1f75043", body: "My note", quote: "Exact  provider\nquote", source };
const sender = { tab: { id: 1, url: source.page_url }, frameId: 0, url: source.page_url };
describe("selected passage attribution", () => {
  it("preserves explicit quote, edited body, draft ID and native identity", () => {
    expect(selectionNotePayload(note, sender)).toEqual(note);
  });
  it("refuses different documents, frames and missing native identity", () => {
    expect(() => selectionNotePayload(note, {})).toThrow("provider_document_required");
    expect(() => selectionNotePayload(note, { ...sender, frameId: 1 })).toThrow("provider_document_required");
    expect(() => selectionNotePayload(note, { ...sender, url: "https://chatgpt.com/c/c2" })).toThrow("selection_document_changed");
    expect(() => selectionNotePayload({ ...note, source: { ...source, message_id: "" } }, sender)).toThrow("native_message_required");
    expect(() => selectionNotePayload({ ...note, source: { ...source, conversation_id: "c2" } }, sender)).toThrow("exact_provider_conversation_required");
  });
});
