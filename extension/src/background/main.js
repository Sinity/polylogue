// The service worker: captures chat pages as they change, keeps captures in
// the outbox until the host has them, and runs the importJob jobs and chat
// actions Polylogue hands out. Chrome stops the worker when it is idle; the
// alarms wake it, and everything it needs again is in Polylogue or the
// outbox.

import { ImportJob } from "./import_job.js";
import { Capturer } from "./capture.js";
import { ChatActions } from "./chat_actions.js";
import { Host, HostError } from "./host.js";
import "../capture_state.js";
import { notePayload } from "./notes.js";
import { ProviderScheduler } from "./provider.js";
import { Outbox } from "./outbox.js";
import { addScripts, askPage, chatTabs, providerOf } from "./tabs.js";

const captureState = globalThis.polylogueCaptureState;

const TICK = "polylogue.tick";
const SWEEP = "polylogue.sweep";
// The least time between two automatic captures of one tab.
const SPACING_MS = 10_000;
const PROBLEMS_KEPT = 20;
// ImportJob job states that are over.
const JOB_ENDED = ["finished", "failed", "cancelled"];
const JOBS_ENDED_SHOWN = 3;
const DEFAULT_SETTINGS = { automatic: true, panel: true, hidden_sites: [] };

export function start(chrome, { indexedDB = globalThis.indexedDB } = {}) {
  const host = new Host(chrome.runtime);
  const outbox = new Outbox(indexedDB);
  const session = chrome.storage.session;
  const local = chrome.storage.local;

  async function settings() {
    return { ...DEFAULT_SETTINGS, ...(await local.get({ settings: {} })).settings };
  }

  // Recent problems, for the popup; cleared when the browser restarts.
  async function log(text) {
    const { problems } = await session.get({ problems: [] });
    await session.set({ problems: [{ at: new Date().toISOString(), text }, ...problems].slice(0, PROBLEMS_KEPT) });
  }

  let instancePromise = null;
  // This browser profile's extension, as Polylogue knows it in claims.
  function instance() {
    instancePromise ||= (async () => {
      const { instance: known } = await local.get({ instance: null });
      if (known) return known;
      const made = `extension-${crypto.randomUUID()}`;
      await local.set({ instance: made });
      return made;
    })();
    return instancePromise;
  }

  const capturer = new Capturer({ chrome, host, outbox, seen: local, scheduler: new ProviderScheduler({ storage: local }), log });
  const importJob = new ImportJob({ chrome, host, capturer, instance, log });
  const chatActions = new ChatActions({
    chrome, host, instance, log, scheduler: capturer.scheduler,
    capture: (tabId, confirm) => capturer.capture(tabId, "chatgpt", { reason: "chat action", force: true, confirm }),
  });

  async function remember(provider, conversation, last) {
    if (!conversation) return;
    const { results } = await session.get({ results: {} });
    await session.set({ results: { ...results, [`${provider}:${conversation}`]: last } });
  }

  const running = new Map();

  /** Captures a tab's conversation now; answers what happened. */
  async function captureTab(tabId, reason, force = false) {
    // The tab may have closed since the capture was asked for.
    const tab = await chrome.tabs.get(tabId).catch(() => null);
    const provider = providerOf(tab?.url);
    if (!provider) return null;
    try {
      const done = await capturer.capture(tabId, provider, { reason, force });
      const text = {
        stored: "stored",
        waiting: "waiting for Polylogue to store it",
        unchanged: "no change since the last capture",
      }[done.outcome];
      await remember(provider, done.conversation, { at: new Date().toISOString(), outcome: done.outcome, text });
      return done;
    } catch (error) {
      if (error.code === "no_conversation_on_page") return null;
      const page = await askPage(chrome, tabId, { type: "page" }).catch(() => null);
      const text = error.status === 429 ? "the provider asked to wait" : String(error.code || error.message);
      await remember(provider, page?.info?.id, { at: new Date().toISOString(), outcome: "failed", text });
      await log(`Capture on ${provider} failed: ${text}`);
      return null;
    }
  }

  /** Asks for a capture of a tab after a change, spacing them out. */
  async function changed(tabId, reason) {
    if (!(await settings()).automatic) return;
    const now = Date.now();
    const tab = await chrome.tabs.get(tabId).catch(() => null);
    const provider = tab && providerOf(tab.url);
    if (!provider) return;
    const state = running.get(tabId);
    if (state?.busy) {
      state.again = reason;
      return;
    }
    const next = { busy: true, again: null, last: now };
    running.set(tabId, next);
    const delay = state ? Math.max(0, state.last + SPACING_MS - now) : 0;
    try {
      await new Promise((resolve) => setTimeout(resolve, delay));
      await captureTab(tabId, reason);
    } finally {
      next.busy = false;
      next.last = Date.now();
      if (next.again) void changed(tabId, next.again);
    }
  }

  async function pageState(tab) {
    const provider = providerOf(tab?.url || "");
    const shown = await settings();
    const panel = shown.panel && !shown.hidden_sites.includes(new URL(tab?.url || "about:blank").hostname);
    if (!provider) return { provider: null, panel };
    const page = await askPage(chrome, tab.id, { type: "page" }).catch(() => null);
    const conversation = page?.info?.id || null;
    const { results } = await session.get({ results: {} });
    const last = conversation ? results[`${provider}:${conversation}`] || null : null;
    let held = null;
    let hostState = "ok";
    if (conversation) {
      try {
        const answer = await captureState.conversations(host, provider, [conversation]);
        held = { ...answer.conversations[0], daemon: answer.daemon };
      } catch (error) {
        hostState = error.code === "host_missing" ? "missing" : "failed";
      }
    }
    return {
      provider, conversation, panel, last, host: hostState,
      state: held?.state || "unknown", messages: held?.messages ?? null, daemon: held?.daemon || "no_answer",
      downloading: held?.downloading === true,
    };
  }

  async function popup(tabId) {
    const tab = tabId ? await chrome.tabs.get(tabId).catch(() => null) : null;
    let status;
    try {
      status = { ...(await captureState.status(host)), host: "ok" };
    } catch (error) {
      status = { host: error.code === "host_missing" ? "missing" : "failed", daemon: "unknown" };
    }
    const daemonUp = status.daemon === "running";
    const jobs = daemonUp ? (await host.call("rows", { selection: "import_jobs" }).catch(() => null))?.reply?.rows || [] : [];
    const actions = daemonUp ? (await host.call("rows", { selection: "chat_actions" }).catch(() => null))?.reply?.rows || [] : [];
    const { problems } = await session.get({ problems: [] });
    return {
      status,
      page: tab ? await pageState(tab) : null,
      waiting: await outbox.count(),
      importJob: [
        // The host lists jobs newest first.
        ...jobs.filter((job) => !JOB_ENDED.includes(job.state)),
        ...jobs.filter((job) => JOB_ENDED.includes(job.state)).slice(0, JOBS_ENDED_SHOWN),
      ],
      approvals: actions.filter((action) => action.state === "awaiting_approval"),
      problems,
      settings: await settings(),
      version: chrome.runtime.getManifest().version,
    };
  }

  async function tick() {
    await capturer.deliver().catch((error) => log(`Captures could not be stored: ${error.message}`));
    await Promise.all([
      importJob.step().catch((error) => log(`Provider import: ${error.message}`)),
      chatActions.step().catch((error) => log(`Chat action: ${error.message}`)),
    ]);
  }

  async function sweep(reason) {
    for (const tab of await chatTabs(chrome)) void changed(tab.id, reason);
  }

  async function wake(reason) {
    await chrome.alarms.create(TICK, { periodInMinutes: 1 });
    await chrome.alarms.create(SWEEP, { periodInMinutes: 15 });
    if (reason === "installed") {
      // Pages opened before this version loaded have no page script yet.
      for (const tab of await chatTabs(chrome)) await addScripts(chrome, tab).catch(() => undefined);
    }
    await tick();
    await sweep("opened");
  }

  const handlers = {
    "polylogue.provider-request": async (_message, sender) => {
      const provider = providerOf(sender.tab?.url || "");
      if (!provider || sender.frameId !== 0) return { error: "provider request is not from a chat page" };
      try { await capturer.scheduler.authorize(provider, sender.tab.id); return { ok: true }; }
      catch (error) { return { error: error.message, until: error.until ?? null }; }
    },
    "polylogue.changed": (message, sender) => {
      if (sender.tab?.id !== undefined) void changed(sender.tab.id, message.reason);
      return { ok: true };
    },
    "polylogue.page": (_message, sender) => pageState(sender.tab),
    "polylogue.capture": async (message, sender) => {
      const tabId = message.tabId ?? sender.tab?.id;
      await captureTab(tabId, "asked", true);
      return pageState(await chrome.tabs.get(tabId));
    },
    "polylogue.note": async (message, sender) => {
      const { request_id, body, quote, source } = notePayload(message.note, sender);
      return { note: await host.call("internal selection-note", { body, quote, ...source }, request_id) };
    },
    "polylogue.hide-panel": async (message) => {
      const now = await settings();
      await local.set({ settings: { ...now, hidden_sites: [...new Set([...now.hidden_sites, message.site])] } });
      return { ok: true };
    },
    "polylogue.popup": (message) => popup(message.tabId),
    "polylogue.settings": async (message) => {
      const now = await settings();
      const next = { ...now };
      if (typeof message.automatic === "boolean") next.automatic = message.automatic;
      if (typeof message.panel === "boolean") next.panel = message.panel;
      if (message.show_site) next.hidden_sites = now.hidden_sites.filter((site) => site !== message.show_site);
      await local.set({ settings: next });
      return next;
    },
    "polylogue.answer": async (message) => host.call("internal chat answer", { id: message.id, value: message.value === "allow" ? "allow" : "deny" }),
    "polylogue.reader": async () => {
      throw new HostError("reader_unavailable", null, "The daemon has no operation for the reader address; open Polylogue with `polylogue web`.");
    },
  };

  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    const handler = handlers[message?.type];
    if (!handler) return false;
    Promise.resolve()
      .then(() => handler(message, sender))
      .then(sendResponse, (error) => sendResponse({ error: String(error.code || error.message || error) }));
    return true;
  });
  chrome.alarms.onAlarm.addListener((alarm) => {
    if (alarm.name === TICK) void tick();
    if (alarm.name === SWEEP) void sweep("sweep");
  });
  chrome.tabs.onRemoved?.addListener((tabId) => running.delete(tabId));
  chrome.runtime.onInstalled.addListener(() => void wake("installed"));
  chrome.runtime.onStartup.addListener(() => void wake("started"));

  return { capturer, outbox, tick, changed, captureTab, pageState, popup, handlers };
}
