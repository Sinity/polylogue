// Chat actions: turns the operator queued with `polylogue chat send`. The
// extension claims one at a time through the host, opens the conversation in
// a background tab, drafts the turn or sends it, and reports what happened.
// Replies come back through ordinary capture.

import { attachmentInPage, chatActionInPage, failureOutcome } from "../actions/chatgpt.js";
import { loaded } from "./tabs.js";

const CHUNK_BYTES = 256 * 1024;
const RENEW_MS = 60_000;

function base64(bytes) {
  let text = "";
  for (let at = 0; at < bytes.length; at += 0x8000) text += String.fromCharCode(...bytes.subarray(at, at + 0x8000));
  return btoa(text);
}

/** Where an action's turn is written. */
export function actionUrl(action) {
  if (action.conversation) return `https://chatgpt.com/c/${encodeURIComponent(action.conversation)}`;
  if (action.project) {
    const project = String(action.project).replace(/^https:\/\/chatgpt\.com\/g\//, "").replace(/^\/+|\/+$/g, "");
    return `https://chatgpt.com/g/${project}/project`;
  }
  return "https://chatgpt.com/";
}

export class ChatActions {
  constructor({ chrome, host, instance, capture, log = () => {} }) {
    this.chrome = chrome;
    this.host = host;
    this.instance = instance;
    this.capture = capture;
    this.log = log;
    this.running = null;
  }

  /** Claims an action and runs it, unless one is running here already. */
  step() {
    if (!this.running) {
      this.running = (async () => {
        try {
          const by = await this.instance();
          const { action } = await this.host.claimChatAction(by);
          if (action) await this.run(action, by);
        } finally {
          this.running = null;
        }
      })();
    }
    return this.running;
  }

  async inPage(tabId, func, args) {
    const [result] = await this.chrome.scripting.executeScript({ target: { tabId }, world: "MAIN", func, args });
    return result?.result;
  }

  async passFiles(action, runId, tabId) {
    await this.inPage(tabId, attachmentInPage, [action.id, runId, "begin"]);
    for (const item of action.attachments || []) {
      await this.inPage(tabId, attachmentInPage, [action.id, runId, "start", item]);
      const bytes = new Uint8Array(await (await this.host.chatAttachment(action.id, item.id)).arrayBuffer());
      if (bytes.length !== item.size) throw new Error("protocol_attachment_size_mismatch");
      for (let at = 0; at < bytes.length; at += CHUNK_BYTES) {
        await this.inPage(tabId, attachmentInPage, [action.id, runId, "append", item, at, base64(bytes.subarray(at, at + CHUNK_BYTES))]);
      }
      await this.inPage(tabId, attachmentInPage, [action.id, runId, "finish", item]);
    }
  }

  async run(action, by) {
    const report = (update) => this.host.reportChatAction(action.id, { by, ...update });
    const runId = crypto.randomUUID();
    let sending = false;
    let tab = null;
    // Progress reports renew the claim while the page works.
    const renew = setInterval(() => {
      void report({ outcome: "progress", phase: sending ? "sending" : "preparing" }).catch(() => undefined);
    }, RENEW_MS);
    try {
      tab = await this.chrome.tabs.create({ url: actionUrl(action), active: false });
      await loaded(this.chrome, tab.id);
      if (action.attachments?.length) await this.passFiles(action, runId, tab.id);
      sending = action.submit;
      await report({ outcome: "progress", phase: sending ? "sending" : "preparing" });
      const result = await this.inPage(tab.id, chatActionInPage, [action, runId]);
      if (!result?.ok) {
        const error = new Error(result?.detail || "the page gave no answer");
        error.mayHaveSent = Boolean(result?.submission_may_have_occurred);
        error.retryAfterSeconds = result?.retry_after_seconds || null;
        throw error;
      }
      await report({
        outcome: result.outcome,
        phase: result.outcome,
        detail: result.outcome === "submitted" ? "sent; the conversation shows the new turn" : "drafted in the composer",
        receipt: {
          conversation: result.provider_conversation_id || null,
          url: result.provider_conversation_url || null,
          turn: result.provider_turn_id || null,
          model: result.observed_model || null,
          effort: result.observed_effort || null,
          project: result.observed_project_ref || null,
          evidence: result.provider_evidence || {},
          at: new Date().toISOString(),
        },
      });
      if (result.outcome === "submitted") {
        await this.capture(tab.id).catch(() => undefined);
        await this.chrome.tabs.remove(tab.id).catch(() => undefined);
      }
      // A draft stays open in its tab for the operator.
    } catch (error) {
      // Once the send button may have been pressed, a failure cannot say
      // whether the turn went out.
      const unknown = sending && error.mayHaveSent !== false;
      const outcome = unknown
        ? { outcome: "outcome_unknown", detail: `sending ended without a confirmation: ${error.message}` }
        : failureOutcome(error, error.retryAfterSeconds || null);
      await report({ outcome: outcome.outcome, phase: unknown ? "outcome_unknown" : "failed", detail: outcome.detail })
        .catch(() => undefined);
      this.log(`Chat action ${action.id} ended: ${outcome.detail || outcome.outcome}`);
      if (tab && !unknown) await this.inPage(tab.id, attachmentInPage, [action.id, runId, "discard"]).catch(() => undefined);
    } finally {
      clearInterval(renew);
    }
  }
}
