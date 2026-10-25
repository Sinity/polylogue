// The extension's side of a chat page, shared by every provider. The
// provider script loaded before this one sets `globalThis.polylogueProvider`:
// how to find the conversation on the page and fetch it, the provider's
// conversation list and its files.
//
// The background asks through a port (`chrome.tabs.connect`) and gets the
// answer's facts first, then any body as base64 pieces, so a conversation of
// any size crosses without one oversized message. The page tells the
// background when its conversation may have changed; the background decides
// whether to capture.
(function () {
  if (globalThis.polylogueContent) return;
  const provider = globalThis.polylogueProvider;
  if (!provider) return;
  globalThis.polylogueContent = true;

  const PIECE_BYTES = 4 * 1024 * 1024;
  // The page counts as settled once its messages stop changing for this long.
  const SETTLED_MS = 3000;

  function base64(bytes) {
    let text = "";
    for (let at = 0; at < bytes.length; at += 0x8000) {
      text += String.fromCharCode(...bytes.subarray(at, at + 0x8000));
    }
    return btoa(text);
  }

  async function sendBody(port, body) {
    for (let at = 0; at < body.size; at += PIECE_BYTES) {
      const piece = new Uint8Array(await body.slice(at, at + PIECE_BYTES).arrayBuffer());
      port.postMessage({ type: "piece", data: base64(piece) });
    }
  }

  async function answer(request) {
    switch (request.type) {
      case "conversation":
        return provider.fetchConversation(request.id || null);
      case "list":
        return provider.list(request.position || null);
      case "file":
        return provider.fetchFile(request.file, request.conversation);
      case "page":
        return { status: 200, info: await provider.currentConversation() };
      default:
        return { status: 400, error: "unknown_request" };
    }
  }

  chrome.runtime.onConnect.addListener((port) => {
    if (port.name !== "polylogue") return;
    let open = true;
    port.onDisconnect.addListener(() => { open = false; });
    port.onMessage.addListener(async (request) => {
      try {
        const { body = null, ...facts } = await answer(request);
        if (!open) return;
        port.postMessage({ type: "facts", ...facts, size: body ? body.size : null });
        if (body) await sendBody(port, body);
        if (open) port.postMessage({ type: "end" });
      } catch (error) {
        if (open) port.postMessage({ type: "failed", error: String(error?.message || error) });
      }
    });
  });

  // Change notices. A conversation that is still being written (the
  // provider shows its stop button) is announced once it settles.
  let timer = null;
  let lastUrl = location.href;
  function announce(reason) {
    const id = provider.conversationId(location.href);
    if (!id && !provider.temporary?.(location.href)) return;
    chrome.runtime.sendMessage({ type: "polylogue.changed", provider: provider.name, reason }).catch(() => undefined);
  }
  function settleThenAnnounce(reason) {
    clearTimeout(timer);
    timer = setTimeout(() => {
      if (provider.generating?.()) {
        settleThenAnnounce(reason);
        return;
      }
      announce(reason);
    }, SETTLED_MS);
  }
  const observer = new MutationObserver(() => {
    if (location.href !== lastUrl) {
      lastUrl = location.href;
      settleThenAnnounce("opened");
      return;
    }
    settleThenAnnounce("changed");
  });
  observer.observe(document.documentElement, { childList: true, subtree: true, characterData: true });
  window.addEventListener("pagehide", () => { clearTimeout(timer); observer.disconnect(); });
  window.addEventListener("pageshow", (event) => {
    if (!event.persisted) return;
    observer.observe(document.documentElement, { childList: true, subtree: true, characterData: true });
    settleThenAnnounce("opened");
  });
  settleThenAnnounce("opened");
})();
