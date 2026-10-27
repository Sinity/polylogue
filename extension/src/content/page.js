// The extension's side of a chat page, shared by every provider. The
// provider script loaded before this one sets `globalThis.polylogueProvider`:
// how to find the conversation on the page and fetch it, the provider's
// conversation list and its files.
//
// The background asks through a port (`chrome.tabs.connect`) and gets the
// answer's facts first, then any body as base64 parts, so a conversation of
// any size crosses without one oversized message. The page tells the
// background when its conversation may have changed; the background decides
// whether to capture.
(function () {
  if (globalThis.polylogueContent) return;
  const provider = globalThis.polylogueProvider;
  if (!provider) return;
  globalThis.polylogueContent = true;

  // A body is sent as appended bytes as it arrives, so a slow download keeps
  // talking to the background, which Chrome would otherwise stop as idle.
  const APPENDED_BYTES = 256 * 1024;
  // Announce a change after the messages stop changing for this long.
  const CHANGE_PAUSE_MS = 3000;

  function base64(bytes) {
    let text = "";
    for (let at = 0; at < bytes.length; at += 0x8000) {
      text += String.fromCharCode(...bytes.subarray(at, at + 0x8000));
    }
    return btoa(text);
  }

  /** Sends a body (a Blob, or a fetch Response read as it arrives). */
  async function sendBody(port, body) {
    if (body instanceof Blob) {
      for (let at = 0; at < body.size; at += APPENDED_BYTES) {
        const appendedBytes = new Uint8Array(await body.slice(at, at + APPENDED_BYTES).arrayBuffer());
        port.postMessage({ type: "appended_bytes", data: base64(appendedBytes) });
      }
      return;
    }
    const reader = body.body.getReader();
    let held = [];
    let size = 0;
    const flush = () => {
      if (!size) return;
      const appendedBytes = new Uint8Array(size);
      let at = 0;
      for (const chunk of held) {
        appendedBytes.set(chunk, at);
        at += chunk.length;
      }
      port.postMessage({ type: "appended_bytes", data: base64(appendedBytes) });
      held = [];
      size = 0;
    };
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      held.push(value);
      size += value.length;
      if (size >= APPENDED_BYTES) flush();
    }
    flush();
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
        port.postMessage({ type: "facts", ...facts });
        if (body) await sendBody(port, body);
        if (open) port.postMessage({ type: "end" });
      } catch (error) {
        if (open) port.postMessage({ type: "failed", error: String(error?.message || error) });
      }
    });
  });

  // Change notices. A conversation that is still being written (the
  // provider shows its stop button) is announced once it stops changing. Pages
  // change their markup all the time (menus, animations); only a change of
  // the conversation's text, or another page, is announced.
  let timer = null;
  let lastUrl = location.href;
  let announced = null;
  function shownText() {
    const shown = document.querySelector("main") || document.body;
    const text = shown?.innerText ?? shown?.textContent ?? "";
    return `${location.href}\n${text.length}\n${text.slice(-2000)}`;
  }
  function announce(reason) {
    const id = provider.conversationId(location.href);
    if (!id && !provider.temporary?.(location.href)) return;
    const shown = shownText();
    if (shown === announced) return;
    announced = shown;
    chrome.runtime.sendMessage({ type: "polylogue.changed", provider: provider.name, reason }).catch(() => undefined);
  }
  function announceAfterPause(reason) {
    clearTimeout(timer);
    timer = setTimeout(() => {
      if (provider.generating?.()) {
        announceAfterPause(reason);
        return;
      }
      announce(reason);
    }, CHANGE_PAUSE_MS);
  }
  const observer = new MutationObserver(() => {
    if (location.href !== lastUrl) {
      lastUrl = location.href;
      announceAfterPause("opened");
      return;
    }
    announceAfterPause("changed");
  });
  observer.observe(document.documentElement, { childList: true, subtree: true, characterData: true });
  window.addEventListener("pagehide", () => { clearTimeout(timer); observer.disconnect(); });
  window.addEventListener("pageshow", (event) => {
    if (!event.persisted) return;
    observer.observe(document.documentElement, { childList: true, subtree: true, characterData: true });
    announceAfterPause("opened");
  });
  announceAfterPause("opened");
})();
