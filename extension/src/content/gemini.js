// Gemini has no conversation API the page can read, so its capture is the
// conversation as the page shows it: each prompt and answer in order. The
// page's own elements for a prompt and an answer are the boundary; when
// Gemini changes them, the capture finds no turns and says so.
(function () {
  const TURNS = "user-query, model-response";

  function conversationId(url) {
    const parsed = new URL(url);
    return parsed.pathname.match(/\/app\/([A-Za-z0-9_-]+)/)?.[1]
      || parsed.searchParams.get("conversation") || null;
  }

  async function currentConversation() {
    return { id: conversationId(location.href), temporary: false };
  }

  function turns() {
    return [...document.querySelectorAll(TURNS)].map((element) => {
      const content = element.querySelector("message-content, .message-content, [data-message-content]") || element;
      return {
        provider_turn_id: element.getAttribute("data-message-id") || null,
        role: element.localName === "user-query" ? "user" : "assistant",
        text: (content.innerText || content.textContent || "").replace(/[ \t]+\n/g, "\n").trim(),
      };
    }).filter((turn) => turn.text);
  }

  // The whole capture, built here: there is no provider JSON to add.
  async function fetchConversation(id) {
    const conversation = conversationId(location.href);
    if (!conversation || (id && id !== conversation)) return { status: 404, error: "no_conversation_on_page" };
    const found = turns();
    if (!found.length) return { status: 422, error: "no_turns_on_page" };
    const session = {
      provider: "gemini",
      provider_session_id: conversation,
      title: document.title || null,
      turns: found.map((turn, ordinal) => ({ ...turn, ordinal })),
    };
    return {
      status: 200,
      info: { id: conversation, temporary: false, url: location.href, session },
    };
  }

  async function list() {
    return { status: 404, error: "no_conversation_list" };
  }

  async function fetchFile() {
    return { status: 404, error: "no_download_for_file" };
  }

  function generating() {
    return Boolean(document.querySelector('button[aria-label="Stop response"]'));
  }

  globalThis.polylogueProvider = {
    name: "gemini", conversationId, currentConversation, fetchConversation, list, fetchFile, generating,
  };
})();
