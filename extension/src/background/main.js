// The service worker: captures chat pages as they change, keeps captures in
// the outbox until the host has them, and runs the backfill jobs and chat
// actions Polylogue hands out. Chrome stops the worker when it is idle; the
// alarms wake it, and everything it needs again is in Polylogue or the
// outbox.

import { Backfill } from "./backfill.js";
import { Capturer } from "./capture.js";
import { ChatActions } from "./chat_actions.js";
import { Host } from "./host.js";
import { notePayload } from "./notes.js";
import { Outbox } from "./outbox.js";
import { addScripts, askPage, chatTabs, providerOf } from "./tabs.js";

const TICK = "polylogue.tick";
const SWEEP = "polylogue.sweep";
// The least time between two automatic captures of one tab.
const SPACING_MS = 10_000;
const PROBLEMS_KEPT = 20;
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

  const capturer = new Capturer({ chrome, host, outbox, seen: local, log });
  const backfill = new Backfill({ chrome, host, capturer, instance, log });
  const chatActions = new ChatActions({
    chrome, host, instance, log,
    capture: (tabId) => captureTab(tabId, "chat action", true),
  });

  async function remember(provider, conversation, last) {
    if (!conversation) return;
    const { results } = await session.get({ results: {} });
    await session.set({ results: { ...results, [`${provider}:${conversation}`]: last } });
  }

  const running = new Map();
  const waitUntil = new Map();

  /** Captures a tab's conversation now; answers what happened. */
  async function captureTab(tabId, reason, force = false) {
    const tab = await chrome.tabs.get(tabId);
    const provider = providerOf(tab.url);
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
      if (error.status === 429) {
        const seconds = Number(error.retryAfter);
        waitUntil.set(provider, Date.now() + Math.max(60_000, Number.isFinite(seconds) ? seconds * 1000 : 0));
      }
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
    if (!provider || (waitUntil.get(provider) || 0) > now) return;
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
        const answer = await host.conversations(provider, [conversation]);
        held = { ...answer.conversations[0], daemon: answer.daemon };
      } catch (error) {
        hostState = error.code === "host_missing" ? "missing" : "failed";
      }
    }
    return {
      provider, conversation, panel, last, host: hostState,
      state: held?.state || "unknown", messages: held?.messages ?? null, daemon: held?.daemon || "no_answer",
    };
  }

  async function popup(tabId) {
    const tab = tabId ? await chrome.tabs.get(tabId).catch(() => null) : null;
    let status;
    try {
      status = { ...(await host.status()), host: "ok" };
    } catch (error) {
      status = { host: error.code === "host_missing" ? "missing" : "failed", daemon: "unknown" };
    }
    const daemonUp = status.daemon === "running";
    const jobs = daemonUp ? (await host.backfillJobs().catch(() => ({ jobs: [] }))).jobs : [];
    const actions = daemonUp ? (await host.chatActions().catch(() => ({ actions: [] }))).actions : [];
    const { problems } = await session.get({ problems: [] });
    return {
      status,
      page: tab ? await pageState(tab) : null,
      waiting: await outbox.count(),
      backfill: jobs.filter((job) => !["finished", "cancelled"].includes(job.status)).concat(
        jobs.filter((job) => ["finished", "cancelled"].includes(job.status)).slice(-3)),
      approvals: actions.filter((action) => action.status === "awaiting_approval"),
      problems,
      settings: await settings(),
      version: chrome.runtime.getManifest().version,
    };
  }

  async function tick() {
    await capturer.deliver().catch((error) => log(`Captures could not be stored: ${error.message}`));
    await Promise.all([
      backfill.step().catch((error) => log(`Backfill: ${error.message}`)),
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
    "polylogue.note": async (message, sender) => host.saveNote(notePayload(message.note, sender)),
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
    "polylogue.decide": async (message) => host.decideChatAction(message.id, "operator", message.approve === true),
    "polylogue.reader": async () => {
      const { url } = await host.reader();
      await chrome.tabs.create({ url });
      return { url };
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

  return { capturer, backfill, chatActions, outbox, host, tick, wake, changed, captureTab, pageState, popup, handlers };
}
