// A stand-in for the parts of Chrome the extension uses, with a fake
// Polylogue host behind native messaging and fake chat pages behind tab
// ports. The host speaks the real native messaging frames, so the
// extension's transport runs unchanged.

export function event() {
  const listeners = [];
  return {
    addListener: (f) => listeners.push(f),
    removeListener: (f) => { const i = listeners.indexOf(f); if (i >= 0) listeners.splice(i, 1); },
    // Chrome dispatches to a copy, so a listener may remove itself.
    fire: (...args) => [...listeners].map((f) => f(...args)),
    listeners,
  };
}

function storageArea() {
  let data = {};
  return {
    // The forms Chrome takes: no argument, a key, a list of keys, or an
    // object of defaults.
    async get(keys) {
      if (keys === undefined || keys === null) return structuredClone(data);
      const defaults = typeof keys === "string" ? { [keys]: undefined }
        : Array.isArray(keys) ? Object.fromEntries(keys.map((k) => [k, undefined])) : keys;
      return Object.fromEntries(Object.entries(defaults)
        .filter(([k, v]) => k in data || v !== undefined)
        .map(([k, v]) => [k, k in data ? structuredClone(data[k]) : v]));
    },
    async set(values) { data = { ...data, ...structuredClone(values) }; },
    async clear() { data = {}; },
    dump: () => data,
  };
}

const encode = (bytes) => Buffer.from(bytes).toString("base64");
const decode = (text) => new Uint8Array(Buffer.from(text, "base64"));

/** A port pair: what one side posts, the other receives. */
function portPair(name) {
  const a = { name, onMessage: event(), onDisconnect: event(), connected: true };
  const b = { name, onMessage: event(), onDisconnect: event(), connected: true };
  const link = (from, to) => {
    from.postMessage = (message) => {
      if (!from.connected) throw new Error("port closed");
      const copy = JSON.parse(JSON.stringify(message));
      queueMicrotask(() => { if (to.connected) to.onMessage.fire(copy); });
    };
    from.disconnect = () => {
      if (!from.connected) return;
      from.connected = false;
      to.connected = false;
      queueMicrotask(() => to.onDisconnect.fire());
    };
  };
  link(a, b);
  link(b, a);
  return [a, b];
}

/** A host that answers logical requests through the real framing. */
function hostPort(operations, calls) {
  const [extension, host] = portPair("host");
  const parts = [];
  host.onMessage.addListener(async (message) => {
    if (message.type === "part") {
      parts.push(decode(message.data));
      if (!message.last) return;
      message = JSON.parse(Buffer.concat(parts));
    }
    calls.push(message);
    const name = message.type === "call" ? message.operation : message.type;
    const operation = operations[name];
    const args = message.type === "capture" ? new TextEncoder().encode(message.capture) : message.arguments || message;
    let result;
    if (operation) result = await operation(args, message);
    else if (name === "spooled") result = { json: { spooled: message.ids.map(() => false) } };
    else if (name === "internal stored-files") result = { json: { files: [] } };
    else result = { status: 404, json: { error: "not_found" } };
    if (result === "drop") { host.disconnect(); return; }
    const answer = result.answer || (result.status >= 400
      ? { outcome: "error", error: { code: result.json.error, message: result.json.message || result.json.error, reason: result.json.reason } }
      : { outcome: "ok", value: result.json ?? {} });
    const value = { version: 2, type: "answer", answer };
    const bytes = new TextEncoder().encode(JSON.stringify(value));
    if (bytes.length <= 64 * 1024) host.postMessage(value);
    else for (let at = 0; at < bytes.length; at += 64 * 1024) {
      host.postMessage({ version: 2, type: "part", sequence: at / (64 * 1024),
        data: encode(bytes.subarray(at, at + 64 * 1024)), last: at + 64 * 1024 >= bytes.length });
    }
  });
  return extension;
}

/**
 * A fake chat page answering the page script's port protocol; `answer`
 * takes the request and returns `{status, info, body (Blob)}`.
 */
function pagePort(answer) {
  const [extension, page] = portPair("polylogue");
  page.onMessage.addListener(async (request) => {
    try {
      const { body = null, ...facts } = await answer(request);
      page.postMessage({ type: "facts", ...facts, has_body: body !== null });
      if (body) {
        const bytes = new Uint8Array(await body.arrayBuffer());
        for (let at = 0; at < bytes.length; at += 1000) page.postMessage({ type: "chunk", data: encode(bytes.subarray(at, at + 1000)) });
      }
      page.postMessage({ type: "end" });
    } catch (error) {
      page.postMessage({ type: "failed", error: String(error.message || error) });
    }
  });
  return extension;
}

export function fakeChrome({ operations = {}, pages = {}, tabs = [], installed = true } = {}) {
  const calls = [];
  const created = [];
  const scripts = [];
  let nextTab = 100;
  const tabList = tabs.map((t) => ({ status: "complete", active: false, ...t }));
  const chrome = {
    calls,
    created,
    scripts,
    runtime: {
      id: "abcdefghijklmnopabcdefghijklmnop",
      lastError: undefined,
      onMessage: event(),
      onInstalled: event(),
      onStartup: event(),
      getManifest: () => ({
        version: "9.9.9",
        content_scripts: [{ matches: ["https://chatgpt.com/*"], js: ["src/content/chatgpt.js", "src/content/page.js"] }],
      }),
      connectNative: () => {
        if (!installed) throw new Error("Specified native messaging host not found.");
        return hostPort(operations, calls);
      },
      // Resolves with the first response a listener sends, as Chrome does.
      sendMessage: (message) => new Promise((resolve) => {
        const answers = chrome.runtime.onMessage.fire(message, {}, resolve);
        if (!answers.includes(true)) resolve(undefined);
      }),
    },
    storage: { local: storageArea(), session: storageArea(), onChanged: event() },
    alarms: { created: [], create: async (name, info) => { chrome.alarms.created.push([name, info]); }, onAlarm: event() },
    tabs: {
      list: tabList,
      get: async (id) => {
        const tab = tabList.find((t) => t.id === id);
        if (!tab) throw new Error(`No tab with id: ${id}`);
        return { ...tab };
      },
      query: async ({ url } = {}) => tabList.filter((t) => !url || [url].flat().some((pattern) => t.url.startsWith(pattern.replace("/*", "/")))),
      create: async ({ url, active }) => {
        const tab = { id: nextTab++, url, active, status: "complete" };
        tabList.push(tab);
        created.push(tab);
        return { ...tab };
      },
      remove: async (id) => { const i = tabList.findIndex((t) => t.id === id); if (i >= 0) tabList.splice(i, 1); },
      connect: (tabId) => {
        const answer = pages[tabId] || pages[tabList.find((t) => t.id === tabId)?.url?.split("/")[2]];
        if (!answer) {
          const [extension] = portPair("polylogue");
          chrome.runtime.lastError = { message: "Receiving end does not exist." };
          queueMicrotask(() => { extension.connected = false; extension.onDisconnect.fire(); chrome.runtime.lastError = undefined; });
          extension.postMessage = () => {};
          return extension;
        }
        return pagePort(answer);
      },
    },
    scripting: {
      executeScript: async (details) => {
        scripts.push(details);
        if (details.func) return [{ result: await details.func(...(details.args || [])) }];
        return [];
      },
    },
  };
  return chrome;
}

/** Waits until `check` holds, letting the fake ports deliver. */
export async function until(check, what = "the expected state") {
  for (let i = 0; i < 400; i += 1) {
    if (await check()) return;
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
  throw new Error(`never reached ${what}`);
}

export function json(bytes) {
  return JSON.parse(new TextDecoder().decode(bytes));
}
