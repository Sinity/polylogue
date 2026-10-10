// @vitest-environment node
import { expect, it, vi } from "vitest";
import { createOwnedProviderBrowser } from "../scripts/owned_provider_browser.mjs";
import { registerBackgroundEvents } from "../src/background/events.js";
import { createBackgroundAdapters } from "../src/background/adapters.js";
import { IDBFactory, IDBKeyRange } from "fake-indexeddb";
import { memoryOriginStorage } from "./infra/capture-staging.js";

const firstUrl = "https://chatgpt.com/c/neutral-first";
const secondUrl = "https://claude.ai/chat/neutral-second";
function event() {
  const listeners = new Set();
  return { addListener: fn => listeners.add(fn), removeListener: fn => listeners.delete(fn),
    emit: (...args) => [...listeners].map(fn => fn(...args)) };
}
function fixture() {
  const rows = new Map([
    [11, { id: 11, windowId: 21, url: firstUrl, active: true }],
    [12, { id: 12, windowId: 22, url: secondUrl, active: true }],
    [99, { id: 99, windowId: 90, url: firstUrl, active: true }],
  ]);
  const browser = {
    tabs: {
      query: vi.fn(async query => [...rows.values()].filter(tab => tab.windowId === query.windowId
        && (query.active === undefined || query.active === tab.active))),
      get: vi.fn(async id => rows.get(id)), sendMessage: vi.fn(async () => ({ ok: true })),
      update: vi.fn(async (id, patch) => { const tab = { ...rows.get(id), ...patch }; rows.set(id, tab); return tab; }),
      remove: vi.fn(async id => rows.delete(id)),
      create: vi.fn(async properties => { const tab = { id: 13, ...properties }; rows.set(13, tab); return tab; }),
      onActivated: event(), onUpdated: event(), onRemoved: event(),
    },
    scripting: { executeScript: vi.fn(async details => [{ frameId: 0, documentId: "neutral-document",
      result: details.files ? undefined : details.args ? rows.get(details.target.tabId).url === details.args[0] : true }]) },
    runtime: { id: "a".repeat(32), getManifest: () => ({ version: "0.3.0" }), connectNative: vi.fn(),
      onMessage: event(), onInstalled: event(), onStartup: event() },
    action: { setBadgeText: vi.fn(), setBadgeBackgroundColor: vi.fn() },
    storage: {}, permissions: {}, alarms: { onAlarm: event() },
  };
  return { rows, browser, declarations: [{ windowId: 21, url: firstUrl, nativeId: "neutral-first" }, { windowId: 22, url: secondUrl, nativeId: "neutral-second" }] };
}
const settle = () => new Promise(resolve => globalThis.setTimeout(resolve, 0));

it("never enumerates operator windows, including production currentWindow queries", async () => {
  const { browser, declarations } = fixture();
  const owner = await createOwnedProviderBrowser(browser, declarations);
  expect((await owner.browser.tabs.query({})).map(tab => tab.id)).toEqual([11, 12]);
  expect((await owner.browser.tabs.query({ active: true, currentWindow: true })).map(tab => tab.id)).toEqual([11]);
  expect(browser.tabs.query.mock.calls.every(([query]) => [21, 22].includes(query.windowId))).toBe(true);
  await expect(owner.browser.tabs.query({ windowId: 90 })).rejects.toThrow("proof_owned_tab_refused");
  await expect(owner.browser.tabs.get(99)).rejects.toThrow("proof_owned_tab_refused");
  expect(browser.tabs.get.mock.calls.some(([id]) => id === 99)).toBe(false);
});

it("refuses every tab effect for unknown, moved, pinned or navigated tabs", async () => {
  for (const patch of [null, { windowId: 90 }, { pinned: true }, { url: "https://chatgpt.com/c/outside" }, { pendingUrl: "https://claude.ai/chat/outside" }]) {
    const { browser, declarations, rows } = fixture();
    const owner = await createOwnedProviderBrowser(browser, declarations);
    const id = patch === null ? 99 : 11;
    if (patch) rows.set(id, { ...rows.get(id), ...patch });
    for (const effect of [() => owner.browser.tabs.sendMessage(id, {}), () => owner.browser.tabs.update(id, { active: true }),
      () => owner.browser.tabs.remove(id), () => owner.browser.scripting.executeScript({ target: { tabId: id }, files: ["src/content/chatgpt.js"] }),
      () => owner.browser.action.setBadgeText({ tabId: id, text: "x" })]) {
      await expect(effect()).rejects.toThrow("proof_owned_tab_refused");
    }
    expect(browser.tabs.sendMessage).not.toHaveBeenCalled();
    expect(browser.tabs.update).not.toHaveBeenCalled();
    expect(browser.tabs.remove).not.toHaveBeenCalled();
    expect(browser.scripting.executeScript).not.toHaveBeenCalled();
    expect(browser.action.setBadgeText).not.toHaveBeenCalled();
  }
});

it("pins provider script and message effects to the checked top-frame document", async () => {
  const { browser, declarations } = fixture();
  const owner = await createOwnedProviderBrowser(browser, declarations);
  await owner.browser.scripting.executeScript({ target: { tabId: 11 }, world: "MAIN", files: ["src/content/chatgpt_bridge.js"] });
  expect(browser.scripting.executeScript).toHaveBeenLastCalledWith({ target: { tabId: 11, documentIds: ["neutral-document"] },
    world: "MAIN", files: ["src/content/chatgpt_bridge.js"] });
  await owner.browser.tabs.sendMessage(11, { type: "polylogue.capturePage" });
  expect(browser.tabs.sendMessage).toHaveBeenCalledWith(11, { type: "polylogue.capturePage" }, { documentId: "neutral-document" });
  await expect(owner.browser.tabs.sendMessage(11, {}, { documentId: "stale-document" })).rejects.toThrow("proof_owned_tab_refused");
  await expect(owner.browser.scripting.executeScript({ target: { tabId: 11, allFrames: true }, files: [] })).rejects.toThrow("proof_owned_tab_refused");
  browser.scripting.executeScript.mockResolvedValueOnce([{ frameId: 0, documentId: "new-document", result: false }]);
  await expect(owner.browser.scripting.executeScript({ target: { tabId: 11 }, files: ["src/content/chatgpt.js"] })).rejects.toThrow("proof_owned_tab_refused");
  expect(browser.scripting.executeScript.mock.calls.filter(([details]) => details.files)).toHaveLength(1);
});

it("revokes ownership when a tab moves or navigates during the document probe", async () => {
  for (const patch of [{ windowId: 90 }, { url: "https://chatgpt.com/c/outside" }]) {
    const { browser, declarations, rows } = fixture();
    const owner = await createOwnedProviderBrowser(browser, declarations);
    browser.scripting.executeScript.mockImplementationOnce(async () => {
      rows.set(11, { ...rows.get(11), ...patch });
      return [{ frameId: 0, documentId: "neutral-document", result: true }];
    });
    await expect(owner.browser.scripting.executeScript({ target: { tabId: 11 }, files: ["src/content/chatgpt.js"] }))
      .rejects.toThrow("proof_owned_tab_refused");
    expect(browser.scripting.executeScript.mock.calls.filter(([details]) => details.files)).toHaveLength(0);
    await expect(owner.browser.tabs.sendMessage(11, { type: "polylogue.capturePage" })).rejects.toThrow("proof_owned_tab_refused");
    expect(browser.tabs.sendMessage).not.toHaveBeenCalled();
  }
});

it("consumes the exact automatic capture once and refuses stale or wrong-target results", async () => {
  for (const patch of [{ windowId: 90 }, { url: "https://chatgpt.com/c/outside" }, { documentId: "new-document" }]) {
    const { browser, declarations, rows } = fixture();
    const owner = await createOwnedProviderBrowser(browser, declarations);
    const result = { ok: true, envelope: { neutral: true }, captureResult: { artifact_ref: "neutral-artifact" } };
    browser.tabs.sendMessage.mockResolvedValueOnce(result);
    await owner.browser.tabs.sendMessage(11, { type: "polylogue.capturePage", reason: "auto_capture_missing" });
    await expect(owner.consumeCapture(99, "neutral-first")).rejects.toThrow("proof_owned_tab_refused");
    await expect(owner.consumeCapture(11, "neutral-second")).rejects.toThrow("proof_owned_tab_refused");
    if (patch.documentId) browser.scripting.executeScript.mockResolvedValueOnce([{ frameId: 0, documentId: patch.documentId, result: true }]);
    else rows.set(11, { ...rows.get(11), ...patch });
    await expect(owner.consumeCapture(11, "neutral-first")).rejects.toThrow("proof_owned_tab_refused");
    expect(browser.tabs.sendMessage).toHaveBeenCalledTimes(1);
  }
  for (const result of [{ ok: true, envelope: { neutral: true }, captureResult: { artifact_ref: "neutral-artifact" } },
    { ok: false, error: "capture_cancelled", outcome: "cancelled" }]) {
    const { browser, declarations } = fixture();
    const owner = await createOwnedProviderBrowser(browser, declarations);
    browser.tabs.sendMessage.mockResolvedValueOnce(result);
    await owner.browser.tabs.sendMessage(11, { type: "polylogue.capturePage", reason: "auto_capture_missing" });
    expect(await owner.consumeCapture(11, "neutral-first")).toBe(result);
    await expect(owner.consumeCapture(11, "neutral-first")).rejects.toThrow("proof_automatic_capture_missing");
    expect(browser.tabs.sendMessage).toHaveBeenCalledTimes(1);
  }
});

it("never consumes an older completion while a newer automatic capture is pending", async () => {
  const { browser, declarations } = fixture();
  const owner = await createOwnedProviderBrowser(browser, declarations);
  let first; let second;
  browser.tabs.sendMessage.mockImplementationOnce(() => new Promise(resolve => { first = resolve; }))
    .mockImplementationOnce(() => new Promise(resolve => { second = resolve; }));
  const older = owner.browser.tabs.sendMessage(11, { type: "polylogue.capturePage", reason: "auto_capture_missing" });
  await vi.waitFor(() => expect(first).toBeTypeOf("function"));
  const newer = owner.browser.tabs.sendMessage(11, { type: "polylogue.capturePage", reason: "auto_capture_unconverged_provider" });
  await vi.waitFor(() => expect(second).toBeTypeOf("function"));
  first({ ok: true, revision: "older" }); await older;
  await expect(owner.consumeCapture(11, "neutral-first")).rejects.toThrow("proof_automatic_capture_pending");
  const result = { ok: true, revision: "newer" };
  second(result); await newer;
  expect(await owner.consumeCapture(11, "neutral-first")).toBe(result);
  expect(browser.tabs.sendMessage).toHaveBeenCalledTimes(2);
});

it("preserves a newer capture when it starts during old result validation", async () => {
  const { browser, declarations } = fixture();
  const owner = await createOwnedProviderBrowser(browser, declarations);
  browser.tabs.sendMessage.mockResolvedValueOnce({ ok: true, revision: "older" });
  await owner.browser.tabs.sendMessage(11, { type: "polylogue.capturePage", reason: "auto_capture_missing" });
  let resolveNewer; let newer;
  browser.tabs.sendMessage.mockImplementationOnce(() => new Promise(resolve => { resolveNewer = resolve; }));
  browser.scripting.executeScript.mockImplementationOnce(async () => {
    newer = owner.browser.tabs.sendMessage(11, { type: "polylogue.capturePage", reason: "auto_capture_unconverged_provider" });
    await vi.waitFor(() => expect(resolveNewer).toBeTypeOf("function"));
    return [{ frameId: 0, documentId: "neutral-document", result: true }];
  });
  await expect(owner.consumeCapture(11, "neutral-first")).rejects.toThrow("proof_owned_tab_refused");
  expect((await owner.browser.tabs.get(11)).id).toBe(11);
  const result = { ok: true, revision: "newer" };
  resolveNewer(result); await newer;
  expect(await owner.consumeCapture(11, "neutral-first")).toBe(result);
  expect(browser.tabs.sendMessage).toHaveBeenCalledTimes(2);
});

it("filters the actual production event router before unknown or moved tabs reach handlers", async () => {
  const { browser, declarations, rows } = fixture();
  const owner = await createOwnedProviderBrowser(browser, declarations);
  const handlers = { activated: vi.fn(), updated: vi.fn(), removed: vi.fn() };
  registerBackgroundEvents(createBackgroundAdapters(owner.browser, vi.fn()), handlers);
  browser.tabs.onActivated.emit({ tabId: 99 });
  browser.tabs.onUpdated.emit(99, { status: "complete" }, rows.get(99));
  browser.tabs.onRemoved.emit(99);
  await settle();
  expect(handlers.activated).not.toHaveBeenCalled(); expect(handlers.updated).not.toHaveBeenCalled(); expect(handlers.removed).not.toHaveBeenCalled();
  browser.tabs.onActivated.emit({ tabId: 11 }); await settle(); expect(handlers.activated).toHaveBeenCalledOnce();
  rows.set(11, { ...rows.get(11), windowId: 90 });
  browser.tabs.onUpdated.emit(11, { status: "complete" }, rows.get(11)); await settle();
  expect(handlers.updated).not.toHaveBeenCalled();
  rows.set(11, { ...rows.get(11), windowId: 21 });
  await expect(owner.browser.tabs.get(11)).rejects.toThrow("proof_owned_tab_refused");
  browser.tabs.onUpdated.emit(12, { url: "https://claude.ai/chat/outside" }, rows.get(12)); await settle();
  expect(handlers.updated).not.toHaveBeenCalled();
  await expect(owner.browser.tabs.get(12)).rejects.toThrow("proof_owned_tab_refused");
  browser.tabs.onRemoved.emit(12); expect(handlers.removed).not.toHaveBeenCalled();
});

it("admits only the current owned content document and the owned control page to runtime messaging", async () => {
  const { browser, declarations, rows } = fixture();
  const owner = await createOwnedProviderBrowser(browser, declarations);
  const handler = vi.fn((_message, _sender, respond) => respond({ ok: true }));
  owner.browser.runtime.onMessage.addListener(handler);
  const send = sender => new Promise(resolve => browser.runtime.onMessage.emit({ type: "neutral" }, sender, resolve));
  for (const sender of [{ id: browser.runtime.id, tab: rows.get(99), url: firstUrl, documentId: "neutral-document" },
    { id: browser.runtime.id, tab: rows.get(11), url: firstUrl, documentId: "stale-document" },
    { id: "foreign", tab: rows.get(11), url: firstUrl, documentId: "neutral-document" },
    { id: browser.runtime.id, url: `chrome-extension://${browser.runtime.id}/src/popup.html` }]) {
    expect(await send(sender)).toEqual({ ok: false, error: "proof_owned_tab_refused" });
  }
  expect(handler).not.toHaveBeenCalled();
  expect(await send({ id: browser.runtime.id, tab: rows.get(11), url: firstUrl, documentId: "neutral-document" })).toEqual({ ok: true });
  expect(await send({ id: browser.runtime.id, url: `chrome-extension://${browser.runtime.id}/proof.html` })).toEqual({ ok: true });
  expect(handler).toHaveBeenCalledTimes(2);
});

it("creates and updates only declared conversation targets inside owned windows", async () => {
  const { browser, declarations } = fixture();
  const owner = await createOwnedProviderBrowser(browser, declarations);
  await expect(owner.browser.tabs.create({ url: firstUrl })).rejects.toThrow("proof_owned_tab_refused");
  await expect(owner.browser.tabs.create({ windowId: 90, url: firstUrl })).rejects.toThrow("proof_owned_tab_refused");
  await expect(owner.browser.tabs.update(11, { url: "https://chatgpt.com/c/outside" })).rejects.toThrow("proof_owned_tab_refused");
  expect(browser.tabs.create).not.toHaveBeenCalled(); expect(browser.tabs.update).not.toHaveBeenCalled();
  const tab = await owner.browser.tabs.create({ windowId: 21, url: firstUrl, active: false });
  expect(await owner.browser.tabs.get(tab.id)).toEqual(tab);
  await owner.browser.tabs.remove(tab.id);
  await expect(owner.browser.tabs.get(tab.id)).rejects.toThrow("proof_owned_tab_refused");
});

it("refuses ambiguous initial ownership without registering any production handler", async () => {
  const { browser, declarations, rows } = fixture();
  rows.set(14, { id: 14, windowId: 21, url: firstUrl });
  await expect(createOwnedProviderBrowser(browser, declarations)).rejects.toThrow("proof_owned_tab_refused");
  expect(browser.runtime.onMessage.emit({})).toEqual([]);
  expect(browser.scripting.executeScript).not.toHaveBeenCalled();
});

it("runs automatic production capture and injection only through the admitted browser authority", async () => {
  vi.resetModules();
  globalThis.indexedDB = new IDBFactory(); globalThis.IDBKeyRange = IDBKeyRange;
  Object.defineProperty(globalThis.navigator, "storage", { configurable: true, value: memoryOriginStorage() });
  const { browser, declarations, rows } = fixture();
  const stored = { receiverBaseUrl: "http://127.0.0.1:18765",
    polylogueReceiverPairing: { receiver_id: "neutral-receiver", api_schema: "polylogue-browser-capture/v1", state: "online", dev_override: true },
    polylogueAmbientSettings: { enabled: true, automatic_capture_enabled: true, disabled_sites: {} } };
  const storage = values => ({
    async get(keys) {
      if (Array.isArray(keys)) return Object.fromEntries(keys.filter(key => Object.hasOwn(values, key)).map(key => [key, values[key]]));
      return { ...keys, ...values };
    },
    async set(patch) { Object.assign(values, patch); },
    async remove(keys) { for (const key of Array.isArray(keys) ? keys : [keys]) delete values[key]; },
  });
  browser.storage = { local: storage(stored), session: storage({}) };
  browser.alarms.create = vi.fn(); browser.alarms.clear = vi.fn();
  browser.permissions.contains = async () => true;
  browser.tabs.sendMessage.mockResolvedValue({ ok: false, error: "neutral_capture_refused" });
  const network = vi.fn(async url => new globalThis.Response(JSON.stringify(new URL(url).pathname === "/v1/status"
    ? { ok: true, receiver_id: "neutral-receiver", api_schema: "polylogue-browser-capture/v1" }
    : { state: "missing", captured: false }), { status: 200, headers: { "Content-Type": "application/json" } }));
  const owner = await createOwnedProviderBrowser(browser, declarations);
  const { startBackgroundRuntime } = await import("../src/background/runtime.js");
  startBackgroundRuntime(createBackgroundAdapters(owner.browser, network));
  browser.tabs.onActivated.emit({ tabId: 99 });
  browser.tabs.onUpdated.emit(99, { status: "complete" }, rows.get(99));
  await owner.startCapture();
  expect(stored.polylogueAmbientSettings.automatic_capture_enabled).toBe(true);
  expect(browser.action.setBadgeText).toHaveBeenCalled();
  expect(browser.action.setBadgeBackgroundColor).toHaveBeenCalled();
  expect(browser.action.setBadgeText.mock.calls.every(([details]) => details.tabId === undefined || [11, 12].includes(details.tabId))).toBe(true);
  await expect(owner.consumeCapture(11, "neutral-second")).rejects.toThrow("proof_owned_tab_refused");
  expect(await owner.consumeCapture(11, "neutral-first")).toEqual({ ok: false, error: "neutral_capture_refused" });
  expect(await owner.consumeCapture(12, "neutral-second")).toEqual({ ok: false, error: "neutral_capture_refused" });
  await expect(owner.consumeCapture(11, "neutral-first")).rejects.toThrow("proof_automatic_capture_missing");
  const captures = browser.tabs.sendMessage.mock.calls.filter(([, message]) => message.type === "polylogue.capturePage");
  expect(captures.map(([id]) => id).sort()).toEqual([11, 12]);
  expect(captures.every(([, , options]) => options.documentId === "neutral-document")).toBe(true);
  const injections = browser.scripting.executeScript.mock.calls.filter(([details]) => details.files);
  expect(injections.length).toBeGreaterThan(0);
  expect(injections.every(([details]) => [11, 12].includes(details.target.tabId)
    && JSON.stringify(details.target.documentIds) === '["neutral-document"]')).toBe(true);
  expect(browser.tabs.get.mock.calls.some(([id]) => id === 99)).toBe(false);
  expect(browser.tabs.query.mock.calls.every(([query]) => [21, 22].includes(query.windowId))).toBe(true);
  expect(network.mock.calls.every(([url]) => new URL(url).origin === "http://127.0.0.1:18765")).toBe(true);
});


it("distinguishes absent automatic work from an invalid listener without requesting capture", async () => {
  const { browser, declarations } = fixture();
  const owner = await createOwnedProviderBrowser(browser, declarations);
  await expect(owner.consumeCapture(11, "neutral-first")).rejects.toThrow("proof_automatic_capture_missing");
  await expect(owner.startCapture()).rejects.toThrow("proof_capture_listener_invalid");
  expect(browser.tabs.sendMessage).not.toHaveBeenCalled();
});


it("retains owned injection failures and existing worker state without requesting capture", async () => {
  const { browser, declarations } = fixture();
  browser.storage = { local: { get: vi.fn(async () => ({ polylogueDebugLog: [{ stage: "neutral_stage" }], polylogueState: { error: "neutral_error" } })) } };
  const owner = await createOwnedProviderBrowser(browser, declarations);
  browser.scripting.executeScript.mockImplementation(async details => {
    if (details.files) throw new Error("neutral injection failure");
    return [{ frameId: 0, documentId: "neutral-document", result: true }];
  });
  await expect(owner.browser.scripting.executeScript({ target: { tabId: 11 }, files: ["src/content/chatgpt.js"] }))
    .rejects.toThrow("neutral injection failure");
  const details = await owner.captureFailureDetails();
  expect(details).toEqual({ boundary_failures: [{ stage: "script_injection", tabId: 11, error: "neutral injection failure", code: null }],
    debug_log: [{ stage: "neutral_stage" }], state: { error: "neutral_error" } });
  expect(browser.storage.local.get).toHaveBeenCalledWith({ polylogueDebugLog: [], polylogueState: null });
  expect(browser.tabs.sendMessage).not.toHaveBeenCalled();
});
