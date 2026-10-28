// Chat pages: which provider a tab shows, asking its page script for a
// conversation, a list page or a file, and finding or opening a tab to
// work in for backfill.

export const PROVIDERS = {
  chatgpt: { hosts: ["chatgpt.com", "chat.openai.com"], home: "https://chatgpt.com/" },
  "claude-ai": { hosts: ["claude.ai"], home: "https://claude.ai/recents" },
  grok: { hosts: ["grok.com"], home: "https://grok.com/" },
  gemini: { hosts: ["gemini.google.com"], home: "https://gemini.google.com/app" },
};

export function providerOf(url) {
  let host;
  try { host = new URL(url).hostname; } catch { return null; }
  return Object.keys(PROVIDERS).find((name) => PROVIDERS[name].hosts.includes(host)) || null;
}

export class PageError extends Error {
  constructor(code) {
    super(code);
    this.code = code;
  }
}

function decode(data) {
  return Uint8Array.from(atob(data), (c) => c.charCodeAt(0));
}

function once(chrome, tabId, request) {
  return new Promise((resolve, reject) => {
    const port = chrome.tabs.connect(tabId, { name: "polylogue" });
    let facts = null;
    let hasBody = false;
    const chunk = [];
    port.onMessage.addListener((message) => {
      if (message.type === "facts") {
        const { type: _type, has_body: present, ...rest } = message;
        if (typeof present !== "boolean") {
          reject(new PageError("invalid_page_reply"));
          port.disconnect();
          return;
        }
        hasBody = present;
        facts = rest;
      } else if (message.type === "chunk") {
        chunk.push(decode(message.data));
      } else if (message.type === "end") {
        port.disconnect();
        resolve({ ...facts, body: hasBody ? new Blob(chunk) : null });
      } else if (message.type === "failed") {
        port.disconnect();
        reject(new PageError(message.error));
      }
    });
    port.onDisconnect.addListener(() => {
      // Chrome sets lastError when no page script listens.
      const missing = Boolean(chrome.runtime.lastError) && !facts;
      reject(new PageError(missing ? "no_page_script" : "page_closed"));
    });
    port.postMessage(request);
  });
}

/** Runs the extension's page scripts in a tab opened before it loaded. */
export async function addScripts(chrome, tab) {
  const host = (() => { try { return new URL(tab.url).hostname; } catch { return ""; } })();
  for (const entry of chrome.runtime.getManifest().content_scripts || []) {
    const matches = entry.matches.some((pattern) => new URL(pattern.replace("*", "x")).hostname === host);
    if (!matches) continue;
    await chrome.scripting.executeScript({ target: { tabId: tab.id }, files: entry.js });
  }
}

/** Asks a tab's page script; adds the scripts once if the tab has none. */
export async function askPage(chrome, tabId, request) {
  try {
    return await once(chrome, tabId, request);
  } catch (error) {
    if (error.code !== "no_page_script") throw error;
    await addScripts(chrome, await chrome.tabs.get(tabId));
    return once(chrome, tabId, request);
  }
}

/** The tab once it has finished loading. */
export async function loaded(chrome, tabId) {
  for (let i = 0; i < 120; i += 1) {
    const tab = await chrome.tabs.get(tabId);
    if (tab.status === "complete") return tab;
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  throw new PageError("page_did_not_load");
}

/**
 * A tab of `provider` to work in: an open one, or a new background tab the
 * extension opened (`opened` is then true, and the caller closes it).
 */
export async function workTab(chrome, provider) {
  const site = PROVIDERS[provider];
  const tabs = await chrome.tabs.query({ url: site.hosts.map((host) => `https://${host}/*`) });
  const open = tabs.find((tab) => tab.status === "complete") || tabs[0];
  if (open) return { tab: await loaded(chrome, open.id), opened: false };
  const tab = await chrome.tabs.create({ url: site.home, active: false });
  return { tab: await loaded(chrome, tab.id), opened: true };
}

/** Every open tab of a supported provider. */
export async function chatTabs(chrome) {
  const urls = Object.values(PROVIDERS).flatMap((site) => site.hosts.map((host) => `https://${host}/*`));
  return chrome.tabs.query({ url: urls });
}
