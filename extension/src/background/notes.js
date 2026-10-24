/** Validate the originating provider document before a native note write. */
export function selectionNotePayload(draft, sender) {
  if (!sender?.tab || (sender.frameId !== undefined && sender.frameId !== 0)) throw new Error("provider_document_required");
  const source = draft?.source;
  const documentUrl = sender.url || sender.tab.url;
  if (!source || source.page_url !== documentUrl) throw new Error("selection_document_changed");
  const url = new URL(documentUrl);
  const sites = { chatgpt: ["chatgpt.com", "c"], "claude-ai": ["claude.ai", "chat"], grok: ["grok.com", "c"], gemini: ["gemini.google.com", "app"] };
  const site = sites[source.provider];
  const parts = url.pathname.split("/").filter(Boolean);
  if (!site || url.protocol !== "https:" || url.host !== site[0] || parts.length !== 2 || parts[0] !== site[1] || parts[1] !== source.conversation_id) {
    throw new Error("exact_provider_conversation_required");
  }
  if (typeof source.message_id !== "string" || !source.message_id || source.message_id.length > 512 ||
      typeof draft.request_id !== "string" || !/^[0-9a-f-]{36}$/i.test(draft.request_id) ||
      typeof draft.body !== "string" || !draft.body.trim() || draft.body.length > 10000 ||
      typeof draft.quote !== "string" || !draft.quote.trim() || draft.quote.length > 2000) throw new Error("bounded_note_and_native_message_required");
  return { request_id: draft.request_id, body: draft.body, quote: draft.quote, source: {
    provider: source.provider, conversation_id: source.conversation_id, message_id: source.message_id,
    page_url: source.page_url, selected_at: source.selected_at,
  } };
}
