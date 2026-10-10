// Proof-only authority. The production worker receives this browser surface
// before registering handlers; content scripts are never registered statically.
function refused() { return new Error("proof_owned_tab_refused"); }

export async function createOwnedProviderBrowser(browser, declarations) {
  if (!Array.isArray(declarations) || declarations.length === 0) throw refused();
  const windows = new Map();
  const owned = new Map();
  const windowNativeIds = new Map();
  const automaticCaptures = new Map();
  function revoke(tabId) { automaticCaptures.delete(tabId); return owned.delete(tabId); }
  for (const declaration of declarations) {
    if (!Number.isInteger(declaration.windowId) || declaration.windowId < 0 || windows.has(declaration.windowId)
        || typeof declaration.url !== "string" || typeof declaration.nativeId !== "string" || !declaration.nativeId) throw refused();
    const url = new URL(declaration.url);
    if (url.protocol !== "https:" || !["chatgpt.com", "claude.ai"].includes(url.hostname)
        || url.username || url.password || url.hash || url.search || url.toString() !== declaration.url
        || url.pathname !== `/${url.hostname === "chatgpt.com" ? "c" : "chat"}/${declaration.nativeId}`) throw refused();
    windows.set(declaration.windowId, declaration.url);
    windowNativeIds.set(declaration.windowId, declaration.nativeId);
    // Never enumerate the operator's windows to discover a provider tab.
    const tabs = await browser.tabs.query({ windowId: declaration.windowId });
    if (tabs.length !== 1 || !valid(tabs[0], declaration.windowId, declaration.url)) throw refused();
    owned.set(tabs[0].id, Object.freeze({ windowId: declaration.windowId, url: declaration.url, nativeId: declaration.nativeId }));
  }
  function valid(tab, windowId, url) {
    return Number.isInteger(tab?.id) && tab.windowId === windowId && tab.url === url
      && (!tab.pendingUrl || tab.pendingUrl === url) && tab.pinned !== true;
  }
  async function current(tabId) {
    const declaration = owned.get(tabId);
    if (!declaration) throw refused();
    const tab = await browser.tabs.get(tabId);
    if (!valid(tab, declaration.windowId, declaration.url)) { revoke(tabId); throw refused(); }
    return tab;
  }
  async function documentFor(tabId) {
    const tab = await current(tabId);
    const rows = await browser.scripting.executeScript({ target: { tabId }, world: "ISOLATED",
      func: expectedUrl => globalThis.location.href === expectedUrl, args: [tab.url] });
    if (rows.length !== 1 || rows[0].frameId !== 0 || rows[0].result !== true
        || typeof rows[0].documentId !== "string" || !rows[0].documentId) { revoke(tabId); throw refused(); }
    // The asynchronous document probe can overlap a tab move or SPA navigation.
    await current(tabId);
    return rows[0].documentId;
  }
  function event(source, listenerWrapper) {
    const wrappers = new Map();
    return Object.freeze({
      addListener(listener) { const wrapped = listenerWrapper(listener); wrappers.set(listener, wrapped); source.addListener(wrapped); },
      removeListener(listener) { const wrapped = wrappers.get(listener); if (wrapped) source.removeListener(wrapped); wrappers.delete(listener); },
    });
  }
  const tabs = Object.freeze({
    async query(query = {}) {
      if (Object.keys(query).some(key => !["windowId", "active", "currentWindow"].includes(key))) throw refused();
      const selected = query.windowId !== undefined ? [query.windowId]
        : query.currentWindow === true ? [[...windows.keys()][0]] : [...windows.keys()];
      if (selected.some(id => !windows.has(id))) throw refused();
      const result = [];
      for (const windowId of selected) {
        const rows = await browser.tabs.query({ windowId, ...(query.active === undefined ? {} : { active: query.active }) });
        for (const tab of rows) {
          if (!owned.has(tab.id)) continue;
          try { result.push(await current(tab.id)); }
          catch (error) { if (error.message !== "proof_owned_tab_refused") throw error; }
        }
      }
      return result;
    },
    get: current,
    async sendMessage(tabId, message, options = {}) {
      const documentId = await documentFor(tabId);
      if (options.documentId !== undefined && options.documentId !== documentId) throw refused();
      const capture = message.type === "polylogue.capturePage" ? { documentId, settled: false } : null;
      if (capture) automaticCaptures.set(tabId, capture);
      const result = await browser.tabs.sendMessage(tabId, message, { ...options, documentId });
      if (capture) {
        if (documentId !== await documentFor(tabId)) { revoke(tabId); throw refused(); }
        if (automaticCaptures.get(tabId) === capture) {
          capture.result = result; capture.settled = true;
        }
      }
      return result;
    },
    async update(tabId, properties) {
      const tab = await current(tabId);
      if ((properties.url !== undefined && properties.url !== tab.url) || properties.pinned === true) throw refused();
      const updated = await browser.tabs.update(tabId, properties);
      await current(tabId);
      return updated;
    },
    async remove(tabId) { await current(tabId); await browser.tabs.remove(tabId); revoke(tabId); },
    async create(properties) {
      if (!windows.has(properties.windowId) || properties.url !== windows.get(properties.windowId)
          || properties.pinned === true) throw refused();
      const tab = await browser.tabs.create(properties);
      if (!valid(tab, properties.windowId, properties.url)) {
        // This creation is ours even if Chrome returned an unexpected location.
        if (Number.isInteger(tab?.id)) await browser.tabs.remove(tab.id);
        throw refused();
      }
      owned.set(tab.id, Object.freeze({ windowId: tab.windowId, url: tab.url, nativeId: windowNativeIds.get(tab.windowId) }));
      return tab;
    },
    onActivated: event(browser.tabs.onActivated, listener => info => {
      const declaration = owned.get(info.tabId);
      if (!declaration) return;
      if (info.windowId !== undefined && info.windowId !== declaration.windowId) { revoke(info.tabId); return; }
      void current(info.tabId).then(() => listener(info), () => undefined);
    }),
    onUpdated: event(browser.tabs.onUpdated, listener => (tabId, change, tab) => {
      const declaration = owned.get(tabId);
      if (!declaration) return;
      if ((change.url !== undefined && change.url !== declaration.url) || !valid(tab, declaration.windowId, declaration.url)) {
        revoke(tabId); return;
      }
      void current(tabId).then(() => listener(tabId, change, tab), () => undefined);
    }),
    onRemoved: event(browser.tabs.onRemoved, listener => (tabId, ...args) => {
      if (!revoke(tabId)) return;
      listener(tabId, ...args);
    }),
  });
  const messageListeners = new Set();
  const runtime = Object.freeze({
    id: browser.runtime.id,
    getManifest: () => browser.runtime.getManifest(),
    connectNative: name => browser.runtime.connectNative(name),
    get lastError() { return browser.runtime.lastError; },
    onInstalled: browser.runtime.onInstalled,
    onStartup: browser.runtime.onStartup,
    onMessage: event(browser.runtime.onMessage, listener => {
      messageListeners.add(listener);
      return (message, sender, sendResponse) => {
        void (async () => {
          if (sender.id !== browser.runtime.id) throw refused();
          if (sender.tab) {
            const tab = await current(sender.tab.id);
            const declaration = owned.get(tab.id);
            if (sender.url !== tab.url || !valid(sender.tab, declaration.windowId, declaration.url)
                || sender.documentId !== await documentFor(tab.id)) throw refused();
          } else if (sender.url !== `chrome-extension://${browser.runtime.id}/proof.html`) throw refused();
          listener(message, sender, sendResponse);
        })().catch(() => sendResponse({ ok: false, error: "proof_owned_tab_refused" }));
        return true;
      };
    }),
  });
  const scripting = Object.freeze({
    async executeScript(details) {
      if (!Number.isInteger(details?.target?.tabId) || details.target.allFrames === true
          || details.target.frameIds !== undefined) throw refused();
      const documentId = await documentFor(details.target.tabId);
      if (details.target.documentIds !== undefined
          && JSON.stringify(details.target.documentIds) !== JSON.stringify([documentId])) throw refused();
      return browser.scripting.executeScript({ ...details, target: { tabId: details.target.tabId, documentIds: [documentId] } });
    },
  });
  const action = Object.freeze(Object.fromEntries(["setBadgeText", "setBadgeBackgroundColor"].map(method => [method, async details => {
    if (details.tabId !== undefined) await current(details.tabId);
    return browser.action[method](details);
  }])));
  const restricted = Object.freeze({ storage: browser.storage, alarms: browser.alarms, permissions: browser.permissions,
    tabs, scripting, runtime, action });
  return Object.freeze({ browser: restricted,
    async ownedTabs() { return tabs.query({}); },
    async consumeCapture(tabId, nativeId) {
      const declaration = owned.get(tabId);
      if (!declaration || nativeId !== declaration.nativeId) throw refused();
      const capture = automaticCaptures.get(tabId);
      if (!capture?.settled) throw refused();
      if (capture.documentId !== await documentFor(tabId)) { revoke(tabId); throw refused(); }
      if (automaticCaptures.get(tabId) !== capture) throw refused();
      automaticCaptures.delete(tabId);
      return capture.result;
    },
    async startCapture() {
      if (messageListeners.size !== 1) throw refused();
      const listener = [...messageListeners][0];
      return new Promise(resolve => listener({ type: "polylogue.captureSupportedTabs", reason: "owned_provider_proof" },
        { id: browser.runtime.id, url: `chrome-extension://${browser.runtime.id}/proof.html` }, resolve));
    },
  });
}
