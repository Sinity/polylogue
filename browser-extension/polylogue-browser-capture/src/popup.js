const DEFAULT_RECEIVER = "http://127.0.0.1:8765";

function hostLabel(url) {
  try {
    const parsed = new URL(url);
    if (parsed.hostname.includes("chatgpt.com")) return "ChatGPT";
    if (parsed.hostname.includes("claude.ai")) return "Claude.ai";
    return parsed.hostname;
  } catch {
    return "Unknown";
  }
}

function setBadge(kind, text) {
  const badge = document.getElementById("badge");
  badge.className = `badge ${kind}`;
  badge.textContent = text;
}

async function activeTab() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  return tab;
}

async function render() {
  const stateNode = document.getElementById("state");
  const stored = await chrome.storage.local.get({
    polylogueState: null,
    receiverBaseUrl: DEFAULT_RECEIVER
  });
  document.getElementById("receiver-url").value = stored.receiverBaseUrl;
  document.getElementById("receiver").textContent = stored.receiverBaseUrl;
  const tab = await activeTab();
  document.getElementById("page").textContent = hostLabel(tab?.url || "");
  const state = stored.polylogueState;
  if (!state) {
    stateNode.textContent = "No capture state yet.";
    setBadge("warn", "idle");
    return;
  }
  if (!state.online) {
    stateNode.textContent = `Receiver offline. Start: polylogue browser-capture serve\n${state.error || ""}`.trim();
    document.getElementById("archive").textContent = "Offline";
    setBadge("bad", "offline");
    return;
  }
  document.getElementById("archive").textContent = state.captured ? "Captured" : "Receiver online";
  if (state.last_capture) {
    stateNode.textContent = `Last capture: ${state.last_capture.provider} / ${state.last_capture.provider_session_id}`;
  } else {
    stateNode.textContent = "Receiver online. Open ChatGPT or Claude.ai to capture.";
  }
  setBadge(state.captured ? "ok" : "warn", state.captured ? "captured" : "online");
}

document.getElementById("check").addEventListener("click", async () => {
  await chrome.runtime.sendMessage({ type: "polylogue.status" });
  await render();
});

document.getElementById("save").addEventListener("click", async () => {
  const receiverBaseUrl = document.getElementById("receiver-url").value;
  await chrome.runtime.sendMessage({ type: "polylogue.configureReceiver", receiverBaseUrl });
  await render();
});

document.getElementById("capture").addEventListener("click", async () => {
  const tab = await activeTab();
  if (!tab?.id) return;
  const result = await chrome.tabs.sendMessage(tab.id, { type: "polylogue.capturePage" }).catch((error) => ({
    ok: false,
    error: String(error.message || error)
  }));
  if (!result?.ok) {
    await chrome.storage.local.set({
      polylogueState: {
        online: false,
        captured: false,
        error: result?.error || "This page is not supported.",
        updated_at: new Date().toISOString()
      }
    });
  }
  await render();
});

render();
