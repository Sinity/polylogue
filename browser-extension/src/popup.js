const DEFAULT_RECEIVER = "http://127.0.0.1:8765";
const AUTO_REFRESH_MS = 8000;
const CAPTURE_MESSAGE_TIMEOUT_MS = 15000;

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

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
  const safeProvider = provider || "unknown";
  return `<span class="provider-logo ${escapeHtml(safeProvider)}">${escapeHtml(labels[safeProvider] || "?")}</span>`;
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

function timeoutError(label, timeoutMs) {
  const error = new Error(`${label}_timeout_after_${timeoutMs}ms`);
  error.name = "PolylogueTimeoutError";
  return error;
}

function withTimeout(promise, timeoutMs, label) {
  let timer = 0;
  const timeout = new Promise((_resolve, reject) => {
    timer = window.setTimeout(() => reject(timeoutError(label, timeoutMs)), timeoutMs);
  });
  return Promise.race([promise, timeout]).finally(() => {
    if (timer) window.clearTimeout(timer);
  });
}

async function capturePageFromTab(tab) {
  return withTimeout(
    chrome.tabs.sendMessage(tab.id, { type: "polylogue.capturePage" }),
    CAPTURE_MESSAGE_TIMEOUT_MS,
    "capture_message",
  );
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

function stateExplanation(state) {
  if (!state) {
    return {
      badge: ["warn", "idle"],
      archive: "Not checked",
      headline: "No receiver state yet.",
      detail: "The popup refreshes automatically on open. If this stays idle, the service worker has not returned a status payload.",
    };
  }
  if (state.active_page_state === "unsupported") {
    return {
      badge: ["warn", "idle"],
      archive: "Unsupported",
      headline: "This page is not a supported conversation.",
      detail: "Open a ChatGPT, Claude.ai, or Grok/X conversation tab. The extension will update this state when the active tab changes.",
    };
  }
  if (!state.online) {
    return {
      badge: ["bad", "offline"],
      archive: "Offline",
      headline: "Receiver offline.",
      detail: `Start the local receiver, then refresh. ${state.error || ""}`.trim(),
    };
  }
  const archiveState = state.archive_state?.state || null;
  if (archiveState === "failed" || state.error) {
    return {
      badge: ["bad", "failed"],
      archive: "Failed",
      headline: "Capture was rejected or could not be parsed.",
      detail: state.archive_state?.latest_failure || state.error || "Open the debug log and match the request id in the receiver log.",
    };
  }
  if (archiveState === "archived" || state.captured) {
    return {
      badge: ["ok", "captured"],
      archive: "Archived",
      headline: state.last_capture
        ? `Last capture: ${state.last_capture.provider} / ${state.last_capture.provider_session_id}`
        : "The latest capture is visible in the archive.",
      detail: "Archive evidence includes receiver spool, source raw row, indexed session, and indexed messages.",
    };
  }
  if (archiveState === "stale") {
    return {
      badge: ["warn", "stale"],
      archive: "Stale",
      headline: "Receiver spool is newer than the indexed archive.",
      detail: "The daemon has not caught up to the latest browser capture yet. Keep the daemon running; this should converge without a manual repair step.",
    };
  }
  if (archiveState === "ingest_pending") {
    return {
      badge: ["warn", "pending"],
      archive: "Ingest pending",
      headline: "Capture reached source.db but is not queryable yet.",
      detail: "The daemon still needs to materialize the indexed session and messages.",
    };
  }
  if (archiveState === "spooled_only") {
    return {
      badge: ["warn", "spooled"],
      archive: "Spooled",
      headline: "Receiver wrote the capture artifact.",
      detail: "The daemon has not acquired the spool artifact into source.db yet.",
    };
  }
  if (archiveState === "missing") {
    return {
      badge: ["warn", "missing"],
      archive: "Not archived",
      headline: "No capture exists for this conversation yet.",
      detail: "Use Capture page for the active conversation. The extension checks archive state automatically when tabs activate or finish loading.",
    };
  }
  if (state.capture_mode === "dom_degraded") {
    return {
      badge: ["warn", "dom"],
      archive: "DOM fallback",
      headline: "Captured from visible DOM, not provider-native app data.",
      detail: "DOM fallback is useful but is not provider-native app data; it may omit branches, provider ids, timestamps, or attachments. Reload the page, wait for the conversation API response, then capture again.",
    };
  }
  if (state.active_page_state === "supported_no_session") {
    return {
      badge: ["warn", "ready"],
      archive: "Ready",
      headline: "Supported site open, but no conversation id is visible.",
      detail: "Open or select a concrete conversation. The extension does not read page content until Capture page or Sync open tabs is used.",
    };
  }
  return {
    badge: ["warn", "online"],
    archive: "Receiver online",
    headline: "Receiver online. Open a supported conversation to capture.",
    detail: "Supported pages are ChatGPT, Claude.ai, and Grok/X conversation routes.",
  };
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
      return `<div class="log-item"><div class="log-time">${escapeHtml(relativeAge(entry.at))}</div><div><div class="log-title">${providerLogo(provider)} ${escapeHtml(title)}</div><div class="log-meta">${escapeHtml(meta || entry.reason || "")}</div></div></div>`;
    })
    .join("");
}

function renderDebugLog(items) {
  const debug = document.getElementById("debug-log");
  const safeItems = Array.isArray(items) ? items.slice(0, 24) : [];
  document.getElementById("debug-count").textContent = String(Array.isArray(items) ? items.length : 0);
  if (!safeItems.length) {
    debug.innerHTML = '<div class="log-meta">No debug events yet.</div>';
    return;
  }
  debug.innerHTML = safeItems
    .map((entry) => {
      const title = [entry.stage, entry.method, entry.path].filter(Boolean).join(" ");
      const meta = [
        entry.ok === false ? "error" : entry.ok === true ? "ok" : null,
        entry.status,
        entry.archive_state,
        entry.capture_mode,
        entry.provider,
        entry.provider_session_id,
        entry.receiver_request_id || entry.request_id,
      ]
        .filter(Boolean)
        .join(" · ");
      return `<div class="log-item"><div class="log-time">${escapeHtml(relativeAge(entry.at))}</div><div><div class="log-title">${escapeHtml(title || "event")}</div><div class="log-meta">${escapeHtml(meta || entry.error || "")}</div></div></div>`;
    })
    .join("");
}

function setActionState(button, state, text = "") {
  if (!button) return;
  button.dataset.state = state;
  button.disabled = state === "busy";
  const status = button.querySelector(".button-status");
  if (status) status.textContent = text;
}

async function withAction(buttonId, fn, labels = {}) {
  const button = document.getElementById(buttonId);
  const busyText = labels.busy || "Working";
  const okText = labels.ok || "Done";
  const badText = labels.bad || "Failed";
  setActionState(button, "busy", busyText);
  try {
    const result = await fn();
    setActionState(button, "ok", okText);
    window.setTimeout(() => setActionState(button, "idle", ""), 1600);
    return result;
  } catch (error) {
    setActionState(button, "bad", badText);
    window.setTimeout(() => setActionState(button, "idle", ""), 3000);
    throw error;
  }
}

async function render() {
  const stored = await chrome.storage.local.get({
    polylogueCaptureLog: [],
    polylogueDebugLog: [],
    polylogueState: null,
    receiverAuthToken: "",
    receiverBaseUrl: DEFAULT_RECEIVER
  });
  renderLog(stored.polylogueCaptureLog);
  renderDebugLog(stored.polylogueDebugLog);
  document.getElementById("receiver-url").value = stored.receiverBaseUrl;
  document.getElementById("receiver-token").value = stored.receiverAuthToken || "";
  document.getElementById("receiver").textContent = stored.receiverBaseUrl;
  const tab = await activeTab();
  const currentProvider = providerFromUrl(tab?.url || "");
  document.getElementById("page").innerHTML = `${providerLogo(currentProvider)} <span>${escapeHtml(hostLabel(tab?.url || ""))}</span>`;
  const state = stored.polylogueState;
  const requestNode = document.getElementById("receiver-request");
  const modeNode = document.getElementById("mode");
  const turnsNode = document.getElementById("turns");
  const updatedNode = document.getElementById("updated");
  requestNode.textContent = state?.last_receiver_request_id || "--";
  modeNode.textContent = state?.capture_mode || state?.archive_state?.state || "--";
  const lastSession = state?.last_capture || {};
  const turnCount = state?.archive_state?.indexed_message_count || lastSession.turn_count || "--";
  const attachmentCount = state?.archive_state?.attachment_count || lastSession.attachment_count || null;
  turnsNode.textContent = attachmentCount === null ? String(turnCount) : `${turnCount} / ${attachmentCount} files`;
  updatedNode.textContent = state?.updated_at ? `${relativeAge(state.updated_at)} ago` : "--";

  const explanation = stateExplanation(state);
  const [badgeKind, badgeText] = explanation.badge;
  document.getElementById("archive").textContent = explanation.archive;
  document.getElementById("state").textContent = explanation.headline;
  document.getElementById("state-detail").textContent = explanation.detail;
  setBadge(badgeKind, badgeText);
}

async function refreshStatus(reason = "popup_manual") {
  await chrome.runtime.sendMessage({ type: "polylogue.status", reason });
  await render();
}

document.getElementById("check").addEventListener("click", async () => {
  await withAction("check", () => refreshStatus("popup_manual"), { busy: "Checking" });
});

document.getElementById("sync-open-tabs").addEventListener("click", async () => {
  await withAction("sync-open-tabs", async () => {
    await chrome.runtime.sendMessage({ type: "polylogue.captureSupportedTabs", reason: "popup_sync_open_tabs" });
    await render();
  }, { busy: "Syncing" });
});

document.getElementById("copy-ref").addEventListener("click", async () => {
  await withAction("copy-ref", async () => {
    const stored = await chrome.storage.local.get({ polylogueState: null });
    const state = stored.polylogueState || {};
    const provider = state.provider || state.last_capture?.provider;
    const providerSessionId = state.provider_session_id || state.last_capture?.provider_session_id;
    const ref = provider && providerSessionId ? `${provider}:${providerSessionId}` : "";
    if (ref && window.navigator.clipboard?.writeText) await window.navigator.clipboard.writeText(ref);
  }, { busy: "Copying", ok: "Copied" });
});

document.getElementById("open-polylogue").addEventListener("click", async () => {
  await withAction("open-polylogue", async () => {
    const stored = await chrome.storage.local.get({ polylogueState: null, receiverBaseUrl: DEFAULT_RECEIVER });
    const providerSessionId = stored.polylogueState?.provider_session_id || stored.polylogueState?.last_capture?.provider_session_id;
    const url = `${String(stored.receiverBaseUrl || DEFAULT_RECEIVER).replace(/\/+$/, "")}/?q=${encodeURIComponent(providerSessionId || "")}`;
    await chrome.tabs.create({ url });
  }, { busy: "Opening" });
});

document.getElementById("save").addEventListener("click", async () => {
  await withAction("save", async () => {
    const receiverBaseUrl = document.getElementById("receiver-url").value;
    const receiverAuthToken = document.getElementById("receiver-token").value;
    await chrome.runtime.sendMessage({ type: "polylogue.configureReceiver", receiverBaseUrl, receiverAuthToken });
    await refreshStatus("popup_configure_receiver");
  }, { busy: "Saving", ok: "Saved" });
});

document.getElementById("capture").addEventListener("click", async () => {
  await withAction("capture", async () => {
    const tab = await activeTab();
    if (!tab?.id) return;
    let result = await capturePageFromTab(tab).catch((error) => ({
      ok: false,
      error: String(error.message || error)
    }));
    if (!result?.ok && (await ensureCaptureScripts(tab))) {
      result = await capturePageFromTab(tab).catch((error) => ({
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
  }, { busy: "Capturing" });
});

document.getElementById("debug-toggle").addEventListener("click", () => {
  document.getElementById("debug-panel").toggleAttribute("hidden");
});

document.getElementById("debug-export").addEventListener("click", async () => {
  await withAction("debug-export", async () => {
    const stored = await chrome.storage.local.get({ polylogueDebugLog: [], polylogueCaptureLog: [], polylogueState: null });
    const payload = {
      exported_at: new Date().toISOString(),
      state: stored.polylogueState || null,
      debug_log: Array.isArray(stored.polylogueDebugLog) ? stored.polylogueDebugLog : [],
      capture_log: Array.isArray(stored.polylogueCaptureLog) ? stored.polylogueCaptureLog : [],
    };
    const blob = new globalThis.Blob([`${JSON.stringify(payload, null, 2)}\n`], { type: "application/json" });
    const url = globalThis.URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `polylogue-browser-capture-debug-${Date.now()}.json`;
    link.click();
    globalThis.URL.revokeObjectURL(url);
  }, { busy: "Exporting", ok: "Exported" });
});

void (async () => {
  await render();
  void refreshStatus("popup_open").catch(() => render());
  const refreshTimer = window.setInterval(() => {
    void refreshStatus("popup_auto").catch(() => render());
  }, AUTO_REFRESH_MS);
  refreshTimer.unref?.();
})();
