// ChatGPT, read from its own API with the page's sign-in: the conversation
// JSON the export holds, the conversation list, and the files a
// conversation names. Requests from a content script go out as the page's.
(function () {
  const PAGE_SIZE = 28;
  // The conversation list comes in four parts: archived or not, starred or
  // not. A list position is `<part>:<offset>`.
  const PARTS = [
    { archived: false, starred: false },
    { archived: false, starred: true },
    { archived: true, starred: false },
    { archived: true, starred: true },
  ];
  let auth = null;

  function conversationId(url) {
    const parts = new URL(url).pathname.split("/").filter(Boolean);
    const at = parts.indexOf("c");
    return at >= 0 && parts[at + 1] ? parts[at + 1] : null;
  }

  function temporary(url) {
    return new URL(url).searchParams.get("temporary-chat") === "true";
  }

  // A temporary chat never shows its ID in the address; the page fetched
  // its conversation by ID, and the browser's resource list names it.
  function temporaryId() {
    const ids = performance.getEntriesByType("resource")
      .map((entry) => new URL(entry.name, location.href))
      .filter((url) => url.origin === location.origin)
      .map((url) => url.pathname.match(/^\/backend-api\/conversation\/([A-Za-z0-9-]+)\/?$/)?.[1])
      .filter(Boolean);
    return ids.at(-1) || null;
  }

  // The conversation's own page: this one, or where it would open.
  function pageUrl(id) {
    return conversationId(location.href) === id ? location.href : `${location.origin}/c/${encodeURIComponent(id)}`;
  }

  async function currentConversation() {
    const id = conversationId(location.href);
    if (id) return { id, temporary: false };
    if (temporary(location.href)) return { id: temporaryId(), temporary: true };
    return { id: null, temporary: false };
  }

  function wait(response) {
    return { status: 429, retry_after: response.headers.get("retry-after") };
  }

  async function headers() {
    if (auth && auth.until > Date.now()) return auth.headers;
    const response = await fetch("/api/auth/session", { credentials: "include", cache: "no-store" });
    if (response.status === 429) return wait(response);
    let session = response.ok ? await response.json().catch(() => null) : null;
    if (!session?.accessToken) {
      try {
        const page = JSON.parse(document.getElementById("client-bootstrap")?.textContent || "null");
        const fromPage = page?.session || page;
        if (fromPage?.accessToken) session = fromPage;
      } catch { /* The page's own startup data is optional. */ }
    }
    const token = session?.accessToken;
    if (!token) return { status: 401, error: "signed_out" };
    const found = { Authorization: `Bearer ${token}` };
    if (session.account?.id) found["ChatGPT-Account-Id"] = session.account.id;
    auth = { headers: found, until: Date.now() + 60_000 };
    return found;
  }

  async function api(path) {
    const sent = await headers();
    if (typeof sent.status === "number") return { failed: sent };
    const response = await fetch(path, { credentials: "include", cache: "no-store", headers: sent });
    if (response.status === 429) return { failed: wait(response) };
    if (!response.ok) return { failed: { status: response.status, error: `chatgpt_http_${response.status}` } };
    return { response };
  }

  async function fetchConversation(id) {
    const current = id ? { id, temporary: false } : await currentConversation();
    if (!current.id) return { status: 404, error: "no_conversation_on_page" };
    const { response, failed } = await api(`/backend-api/conversation/${encodeURIComponent(current.id)}`);
    if (failed) return failed;
    return {
      status: 200,
      info: { id: current.id, temporary: current.temporary, url: current.temporary ? location.href : pageUrl(current.id) },
      body: response,
    };
  }

  async function list(position) {
    const [part, offset] = String(position || "0:0").split(":").map((n) => Number.parseInt(n, 10));
    const flags = PARTS[part];
    if (!flags || !Number.isSafeInteger(offset)) return { status: 400, error: "unknown_list_position" };
    const query = new URLSearchParams({
      offset: String(offset), limit: String(PAGE_SIZE), order: "updated",
      is_archived: String(flags.archived), is_starred: String(flags.starred),
    });
    const { response, failed } = await api(`/backend-api/conversations?${query}`);
    if (failed) return failed;
    const page = await response.json();
    const items = Array.isArray(page.items) ? page.items : [];
    const conversations = items.filter((item) => typeof item.id === "string").map((item) => ({
      id: item.id,
      title: typeof item.title === "string" ? item.title : null,
      updated_at: time(item.update_time),
    }));
    const next = offset + items.length;
    const total = Number.isFinite(page.total) ? page.total : next;
    const partDone = items.length === 0 || next >= total;
    const done = partDone && part === PARTS.length - 1;
    return {
      status: 200,
      // Each part is listed newest change first.
      info: { conversations, next: partDone ? (done ? null : `${part + 1}:0`) : `${part}:${next}`, done, newest_first: true },
    };
  }

  function time(value) {
    if (typeof value === "number") return new Date(value < 1e10 ? value * 1000 : value).toISOString();
    return typeof value === "string" && value ? value : null;
  }

  // A file is reached through a download link the API hands out: for
  // files the code interpreter wrote (`sandbox:` paths), or for uploaded
  // and generated files by their ID.
  async function fetchFile(file, conversation) {
    let path;
    if (file.url?.startsWith("sandbox:")) {
      const query = new URLSearchParams({ message_id: file.message, sandbox_path: file.url.slice("sandbox:".length) });
      path = `/backend-api/conversation/${encodeURIComponent(conversation)}/interpreter/download?${query}`;
    } else if (file.file_id) {
      path = `/backend-api/files/${encodeURIComponent(file.file_id)}/download`;
    } else {
      return { status: 404, error: "no_download_for_file" };
    }
    const { response, failed } = await api(path);
    if (failed) return failed;
    const link = await response.json().catch(() => null);
    const href = link?.download_url || link?.url;
    if (typeof href !== "string") return { status: 404, error: "no_download_for_file" };
    const target = new URL(href, location.origin);
    if (target.protocol !== "https:") return { status: 400, error: "download_not_https" };
    // Page cookies go only to ChatGPT itself, never to a storage host.
    const bytes = await fetch(target.href, {
      credentials: target.origin === location.origin ? "include" : "omit", cache: "no-store",
    });
    if (bytes.status === 429) return wait(bytes);
    if (!bytes.ok) return { status: bytes.status, error: `download_http_${bytes.status}` };
    return {
      status: 200,
      info: { name: link.file_name || file.name || null, media_type: bytes.headers.get("content-type") },
      body: bytes,
    };
  }

  function generating() {
    return Boolean(document.querySelector('[data-testid="stop-button"], button[aria-label="Stop streaming"]'));
  }

  globalThis.polylogueProvider = {
    name: "chatgpt", conversationId, temporary, currentConversation, fetchConversation, list, fetchFile, generating,
  };
})();
