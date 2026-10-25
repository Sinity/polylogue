// A note on a selected passage, as the in-page panel sends it. The note is
// accepted only from the chat page it names, so a page cannot write a note
// about another conversation; the host then checks the message belongs to
// that archived conversation.

import { PROVIDERS } from "./tabs.js";

// Where each provider shows one conversation: `/<path>/<id>`.
const PATHS = { chatgpt: "c", "claude-ai": "chat", grok: "c", gemini: "app" };

export function notePayload(note, sender) {
  if (!sender?.tab || (sender.frameId !== undefined && sender.frameId !== 0)) throw new Error("note_not_from_a_chat_page");
  const source = note?.source;
  const pageUrl = sender.url || sender.tab.url;
  if (!source || source.page_url !== pageUrl) throw new Error("page_changed_since_the_selection");
  const url = new URL(pageUrl);
  const parts = url.pathname.split("/").filter(Boolean);
  const site = PROVIDERS[source.provider];
  if (!site || url.protocol !== "https:" || !site.hosts.includes(url.host) || parts.length !== 2
      || parts[0] !== PATHS[source.provider] || parts[1] !== source.conversation_id) {
    throw new Error("the_page_is_not_that_conversation");
  }
  if (typeof source.message_id !== "string" || !source.message_id) throw new Error("the_selection_names_no_message");
  if (typeof note.request_id !== "string" || typeof note.body !== "string" || !note.body.trim()
      || typeof note.quote !== "string" || !note.quote.trim()) {
    throw new Error("the_note_needs_text_and_a_quote");
  }
  return {
    request_id: note.request_id,
    body: note.body,
    quote: note.quote,
    source: {
      provider: source.provider,
      conversation_id: source.conversation_id,
      message_id: source.message_id,
      page_url: source.page_url,
      selected_at: source.selected_at,
    },
  };
}
