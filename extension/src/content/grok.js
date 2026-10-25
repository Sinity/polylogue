// Grok, read from its own API with the page's sign-in. A conversation is
// three answers (its metadata, its responses and their tree), joined into
// one JSON object for the capture.
(function () {
  const PAGE_SIZE = 60;

  function conversationId(url) {
    const parts = new URL(url).pathname.split("/").filter(Boolean);
    const at = parts.indexOf("c");
    return at >= 0 && parts[at + 1] ? parts[at + 1] : null;
  }

  async function currentConversation() {
    return { id: conversationId(location.href), temporary: false };
  }

  async function get(url, required = true) {
    const response = await fetch(url, { credentials: "include", cache: "no-store" });
    if (response.status === 429) return { failed: { status: 429, retry_after: response.headers.get("retry-after") } };
    if (!response.ok) {
      return required ? { failed: { status: response.status, error: `grok_http_${response.status}` } } : { response: null };
    }
    return { response };
  }

  async function fetchConversation(id) {
    const conversation = id || conversationId(location.href);
    if (!conversation) return { status: 404, error: "no_conversation_on_page" };
    const base = `/rest/app-chat/conversations/${encodeURIComponent(conversation)}`;
    const parts = [];
    for (const [name, path, required] of [
      ["conversation", base, true],
      ["responses", `${base}/responses`, true],
      ["response_nodes", `${base}/response-node`, false],
    ]) {
      const { response, failed } = await get(path, required);
      if (failed) return failed;
      if (response) parts.push(`${parts.length ? "," : "{"}${JSON.stringify(name)}:`, await response.blob());
    }
    parts.push("}");
    return {
      status: 200,
      info: { id: conversation, temporary: false, url: location.href },
      body: new Blob(parts, { type: "application/json" }),
    };
  }

  async function list(position) {
    const url = new URL("/rest/app-chat/conversations", location.origin);
    url.searchParams.set("pageSize", String(PAGE_SIZE));
    if (position) url.searchParams.set("pageToken", position);
    const { response, failed } = await get(url.href);
    if (failed) return failed;
    const page = await response.json();
    const rows = Array.isArray(page.conversations) ? page.conversations : [];
    const conversations = rows.filter((item) => typeof item.conversationId === "string").map((item) => ({
      id: item.conversationId,
      title: typeof item.title === "string" ? item.title : null,
      updated_at: typeof item.modifyTime === "string" ? item.modifyTime : null,
    }));
    const next = typeof page.nextPageToken === "string" && page.nextPageToken ? page.nextPageToken : null;
    return { status: 200, info: { conversations, next, done: !next, newest_first: true } };
  }

  // Attached files live on Grok's asset host, which takes the same sign-in.
  async function fetchFile(file) {
    if (typeof file.url !== "string" || !file.url) return { status: 404, error: "no_download_for_file" };
    const target = new URL(`/${file.url.replace(/^\/+/, "")}`, "https://assets.grok.com");
    // The key comes from the provider's data; whatever it spells, the
    // request goes to the asset host only.
    if (target.origin !== "https://assets.grok.com") return { status: 400, error: "file_not_on_asset_host" };
    const { response, failed } = await get(target.href);
    if (failed) return failed;
    return {
      status: 200,
      info: { name: file.name || null, media_type: response.headers.get("content-type") },
      body: await response.blob(),
    };
  }

  function generating() {
    return Boolean(document.querySelector('button[aria-label="Stop model response"], button[aria-label="Stop"]'));
  }

  globalThis.polylogueProvider = {
    name: "grok", conversationId, currentConversation, fetchConversation, list, fetchFile, generating,
  };
})();
