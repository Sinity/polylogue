// claude.ai, read from its own API with the page's sign-in: the
// conversation JSON its export holds and the conversation list of the
// organization the page has selected.
(function () {
  const PAGE_SIZE = 100;
  const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

  function conversationId(url) {
    const parts = new URL(url).pathname.split("/").filter(Boolean);
    return parts[0] === "chat" && parts[1] ? parts[1] : null;
  }

  // The organization the page works in, as claude.ai itself remembers it,
  // or as the page's own API requests name it.
  function organization() {
    try {
      const selected = JSON.parse(localStorage.getItem("omelette-org-settings-cache") || "null")?.orgUuid;
      if (UUID.test(selected || "")) return selected;
    } catch { /* Fall back to the page's requests. */ }
    for (const entry of performance.getEntriesByType("resource").reverse()) {
      const id = new URL(entry.name, location.href).pathname.match(/^\/api\/organizations\/([^/]+)\//)?.[1];
      if (UUID.test(id || "")) return id;
    }
    return null;
  }

  async function currentConversation() {
    return { id: conversationId(location.href), temporary: false };
  }

  async function get(path) {
    const response = await fetch(path, { credentials: "include", cache: "no-store" });
    if (response.status === 429) return { failed: { status: 429, retry_after: response.headers.get("retry-after") } };
    if (!response.ok) return { failed: { status: response.status, error: `claude_http_${response.status}` } };
    return { response };
  }

  async function fetchConversation(id) {
    const conversation = id || conversationId(location.href);
    if (!conversation) return { status: 404, error: "no_conversation_on_page" };
    const org = organization();
    if (!org) return { status: 401, error: "signed_out" };
    const query = new URLSearchParams({
      tree: "True", rendering_mode: "messages", render_all_tools: "true", consistency: "strong",
    });
    const { response, failed } = await get(
      `/api/organizations/${org}/chat_conversations/${encodeURIComponent(conversation)}?${query}`,
    );
    if (failed) return failed;
    return {
      status: 200,
      info: { id: conversation, temporary: false, url: location.href },
      body: await response.blob(),
    };
  }

  async function list(position) {
    const org = organization();
    if (!org) return { status: 401, error: "signed_out" };
    const offset = Number.parseInt(position || "0", 10);
    if (!Number.isSafeInteger(offset)) return { status: 400, error: "unknown_list_position" };
    const { response, failed } = await get(
      `/api/organizations/${org}/chat_conversations?limit=${PAGE_SIZE}&offset=${offset}`,
    );
    if (failed) return failed;
    const items = await response.json();
    const rows = Array.isArray(items) ? items : [];
    const conversations = rows.filter((item) => typeof item.uuid === "string").map((item) => ({
      id: item.uuid,
      title: typeof item.name === "string" ? item.name : null,
      updated_at: typeof item.updated_at === "string" ? item.updated_at : null,
    }));
    const done = rows.length < PAGE_SIZE;
    return { status: 200, info: { conversations, next: done ? null : String(offset + rows.length), done, newest_first: false } };
  }

  // Files in a claude.ai conversation arrive inside its JSON; there is no
  // separate download to make.
  async function fetchFile() {
    return { status: 404, error: "no_download_for_file" };
  }

  function generating() {
    return Boolean(document.querySelector('button[aria-label="Stop response"]'));
  }

  globalThis.polylogueProvider = {
    name: "claude-ai", conversationId, currentConversation, fetchConversation, list, fetchFile, generating,
  };
})();
