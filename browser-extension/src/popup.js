const DEFAULT_RECEIVER = "http://127.0.0.1:8765";

function hostLabel(url) {
  try {
    const parsed = new URL(url);
    if (parsed.hostname.includes("chatgpt.com")) return "ChatGPT";
    if (parsed.hostname.includes("claude.ai")) return "Claude.ai";
    if (parsed.hostname.includes("grok.com")) return "Grok";
    if (parsed.hostname === "x.com" || parsed.hostname.endsWith(".x.com")) return "Grok / X";
    if (parsed.hostname === "twitter.com" || parsed.hostname.endsWith(".twitter.com")) return "Grok / X";
    return parsed.hostname;
  } catch {
    return "Unknown";
  }
}

function providerFromUrl(url) {
  try {
    const parsed = new URL(url || "");
    if (parsed.hostname.includes("chatgpt.com")) return "chatgpt";
    if (parsed.hostname.includes("claude.ai")) return "claude-ai";
    if (parsed.hostname.includes("grok.com") || parsed.hostname.includes("x.com") || parsed.hostname.includes("twitter.com")) {
      return "grok";
    }
  } catch {
    return "unknown";
  }
  return "unknown";
}

function providerLogo(provider) {
  const labels = {
    chatgpt: "GPT",
    "claude-ai": "C",
    grok: "G",
    unknown: "?",
  };
  return `<span class="provider-logo ${provider || "unknown"}">${labels[provider] || "?"}</span>`;
}

function contentScriptFiles(url) {
  try {
    const parsed = new URL(url || "");
    if (parsed.hostname === "chatgpt.com" || parsed.hostname.endsWith(".chatgpt.com")) {
      return ["src/common.js", "src/content/chatgpt.js"];
    }
    if (parsed.hostname === "claude.ai" || parsed.hostname.endsWith(".claude.ai")) {
      return ["src/common.js", "src/content/claude.js"];
    }
    if (
      parsed.hostname === "grok.com" ||
      parsed.hostname.endsWith(".grok.com") ||
      parsed.hostname === "x.com" ||
      parsed.hostname.endsWith(".x.com") ||
      parsed.hostname === "twitter.com" ||
      parsed.hostname.endsWith(".twitter.com")
    ) {
      return ["src/common.js", "src/content/grok.js"];
    }
  } catch {
    return [];
  }
  return [];
}

async function ensureCaptureScripts(tab) {
  const files = contentScriptFiles(tab?.url || "");
  if (!tab?.id || !files.length) return false;
  for (const file of files) {
    await chrome.scripting.executeScript({ target: { tabId: tab.id }, files: [file] });
  }
  return true;
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

function relativeAge(iso) {
  const then = Date.parse(iso || "");
  if (!Number.isFinite(then)) return "--";
  const seconds = Math.max(0, Math.round((Date.now() - then) / 1000));
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes}m`;
  return `${Math.round(minutes / 60)}h`;
}

function renderLog(items) {
  const log = document.getElementById("log");
  const safeItems = Array.isArray(items) ? items.slice(0, 8) : [];
  document.getElementById("log-count").textContent = String(Array.isArray(items) ? items.length : 0);
  if (!safeItems.length) {
    log.innerHTML = '<div class="log-meta">No capture attempts recorded yet.</div>';
    return;
  }
  log.innerHTML = safeItems
    .map((entry) => {
      const provider = entry.provider || "unknown";
      const title = entry.ok
        ? `${provider} ${entry.provider_session_id || ""}`.trim()
        : entry.error || "capture failed";
      const meta = [entry.archive_state, entry.capture_mode, entry.receiver_request_id].filter(Boolean).join(" · ");
      return `<div class="log-item"><div class="log-time">${relativeAge(entry.at)}</div><div><div class="log-title">${providerLogo(provider)} ${title}</div><div class="log-meta">${meta || entry.reason || ""}</div></div></div>`;
    })
    .join("");
}

async function render() {
  const stateNode = document.getElementById("state");
  const stored = await chrome.storage.local.get({
    polylogueCaptureLog: [],
    polylogueState: null,
    receiverAuthToken: "",
    receiverBaseUrl: DEFAULT_RECEIVER
  });
  renderLog(stored.polylogueCaptureLog);
  document.getElementById("receiver-url").value = stored.receiverBaseUrl;
  document.getElementById("receiver-token").value = stored.receiverAuthToken || "";
  document.getElementById("receiver").textContent = stored.receiverBaseUrl;
  const tab = await activeTab();
  const currentProvider = providerFromUrl(tab?.url || "");
  document.getElementById("page").innerHTML = `${providerLogo(currentProvider)} <span>${hostLabel(tab?.url || "")}</span>`;
  const state = stored.polylogueState;
  const requestNode = document.getElementById("receiver-request");
  const modeNode = document.getElementById("mode");
  const turnsNode = document.getElementById("turns");
  if (!state) {
    stateNode.textContent = "No capture state yet.";
    requestNode.textContent = "--";
    modeNode.textContent = "--";
    turnsNode.textContent = "--";
    setBadge("warn", "idle");
    return;
  }
  requestNode.textContent = state.last_receiver_request_id || "--";
  modeNode.textContent = state.capture_mode || state.archive_state?.state || "--";
  const lastSession = state.last_capture || {};
  const turnCount = state.archive_state?.indexed_message_count || lastSession.turn_count || "--";
  const attachmentCount = state.archive_state?.attachment_count || lastSession.attachment_count || null;
  turnsNode.textContent = attachmentCount === null ? String(turnCount) : `${turnCount} / ${attachmentCount} files`;
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

document.getElementById("sync-open-tabs").addEventListener("click", async () => {
  await chrome.runtime.sendMessage({ type: "polylogue.captureSupportedTabs", reason: "popup_sync_open_tabs" });
  await render();
});

document.getElementById("copy-ref").addEventListener("click", async () => {
  const stored = await chrome.storage.local.get({ polylogueState: null });
  const state = stored.polylogueState || {};
  const provider = state.provider || state.last_capture?.provider;
  const providerSessionId = state.provider_session_id || state.last_capture?.provider_session_id;
  const ref = provider && providerSessionId ? `${provider}:${providerSessionId}` : "";
  if (ref && window.navigator.clipboard?.writeText) await window.navigator.clipboard.writeText(ref);
});

document.getElementById("open-polylogue").addEventListener("click", async () => {
  const stored = await chrome.storage.local.get({ polylogueState: null, receiverBaseUrl: DEFAULT_RECEIVER });
  const providerSessionId = stored.polylogueState?.provider_session_id || stored.polylogueState?.last_capture?.provider_session_id;
  const url = `${String(stored.receiverBaseUrl || DEFAULT_RECEIVER).replace(/\/+$/, "")}/?q=${encodeURIComponent(providerSessionId || "")}`;
  await chrome.tabs.create({ url });
});

document.getElementById("save").addEventListener("click", async () => {
  const receiverBaseUrl = document.getElementById("receiver-url").value;
  const receiverAuthToken = document.getElementById("receiver-token").value;
  await chrome.runtime.sendMessage({ type: "polylogue.configureReceiver", receiverBaseUrl, receiverAuthToken });
  await render();
});

document.getElementById("capture").addEventListener("click", async () => {
  const tab = await activeTab();
  if (!tab?.id) return;
  let result = await chrome.tabs.sendMessage(tab.id, { type: "polylogue.capturePage" }).catch((error) => ({
    ok: false,
    error: String(error.message || error)
  }));
  if (!result?.ok && (await ensureCaptureScripts(tab))) {
    result = await chrome.tabs.sendMessage(tab.id, { type: "polylogue.capturePage" }).catch((error) => ({
      ok: false,
      error: String(error.message || error)
    }));
  }
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
