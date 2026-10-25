// Backfill: capturing a web chat's older conversations. The job lives in
// Polylogue (started with `polylogue backfill start`); this side claims it
// through the host, pages through the provider's conversation list, captures
// each conversation the archive lacks or holds an older copy of, and reports
// its position and totals. A restarted extension claims its job again and
// continues from the last reported position.

import { askPage, workTab } from "./tabs.js";

// Pause between two conversations, to stay well inside what the providers
// allow a person's browser.
const PAUSE_MS = 2000;
// Captures between two progress reports, which also renew the claim.
const REPORT_EVERY = 10;

export class Backfill {
  constructor({ chrome, host, capturer, instance, log = () => {}, sleep = (ms) => new Promise((r) => setTimeout(r, ms)) }) {
    this.chrome = chrome;
    this.host = host;
    this.capturer = capturer;
    this.instance = instance;
    this.log = log;
    this.sleep = sleep;
    this.running = null;
    this.waitUntil = 0;
  }

  /**
   * Claims a job and runs it, unless one is running here already or the
   * provider asked to wait.
   */
  step() {
    if (!this.running && Date.now() >= this.waitUntil) {
      this.running = (async () => {
        try {
          const { job } = await this.host.claimBackfill(await this.instance());
          if (job) await this.run(job);
        } finally {
          this.running = null;
        }
      })();
    }
    return this.running;
  }

  wait(retryAfter) {
    this.waitUntil = Date.now() + Math.max(60_000, retryMs(retryAfter));
  }

  async run(job) {
    const by = await this.instance();
    const counts = { listed: job.listed, captured: job.captured, unchanged: job.unchanged, failed: job.failed };
    let position = job.position;
    const since = job.since ? Date.parse(job.since) : null;
    const report = async (kind, detail = null) => {
      try {
        await this.host.reportBackfill(job.id, by, { kind, position, counts: { ...counts }, detail });
        return true;
      } catch (error) {
        // The operator cancelled the job, or another instance holds it.
        this.log(`Backfill of ${job.provider} stopped: ${error.message}`);
        return false;
      }
    };
    const { tab, opened } = await workTab(this.chrome, job.provider);
    try {
      for (;;) {
        const page = await askPage(this.chrome, tab.id, { type: "list", position });
        if (page.status === 429) {
          this.wait(page.retry_after);
          await report("progress", "The provider asked to wait; continuing later.");
          return;
        }
        if (page.status === 401 || page.status === 403) {
          await report("failed", `Sign in to ${job.provider} in this browser, then start the backfill again.`);
          return;
        }
        if (page.status !== 200) {
          await report("failed", `The provider's conversation list could not be read (${page.error || page.status}).`);
          return;
        }
        const listed = page.info.conversations.filter((c) => !since || !c.updated_at || Date.parse(c.updated_at) >= since);
        const held = listed.length
          ? (await this.host.conversations(job.provider, listed.map((c) => c.id))).conversations
          : [];
        let sinceReport = 0;
        for (const [i, conversation] of listed.entries()) {
          if (current(held[i], conversation)) {
            counts.unchanged += 1;
            continue;
          }
          try {
            const done = await this.capturer.capture(tab.id, job.provider, { id: conversation.id, reason: "backfill" });
            counts[done.outcome === "stored" ? "captured" : "unchanged"] += 1;
          } catch (error) {
            if (error.status === 429) {
              this.wait(error.retryAfter);
              await report("progress", "The provider asked to wait; continuing later.");
              return;
            }
            counts.failed += 1;
            this.log(`Backfill could not capture ${job.provider} conversation ${conversation.id}: ${error.message}`);
          }
          sinceReport += 1;
          if (sinceReport >= REPORT_EVERY) {
            sinceReport = 0;
            if (!(await report("progress"))) return;
          }
          await this.sleep(PAUSE_MS);
        }
        counts.listed += page.info.conversations.length;
        // A list ordered by change time ends where it passes `since`.
        const passedSince = since && page.info.newest_first
          && page.info.conversations.some((c) => c.updated_at && Date.parse(c.updated_at) < since);
        if (page.info.done || passedSince) {
          position = page.info.next;
          await report("finished");
          return;
        }
        position = page.info.next;
        if (!(await report("progress"))) return;
      }
    } finally {
      if (opened) await this.chrome.tabs.remove(tab.id).catch(() => undefined);
    }
  }
}

/** Seconds or an HTTP date, as a provider's Retry-After gives it. */
function retryMs(value) {
  const seconds = Number(value);
  if (Number.isFinite(seconds)) return seconds * 1000;
  const at = Date.parse(value || "");
  return Number.isFinite(at) ? at - Date.now() : 0;
}

/** Whether the archive holds this conversation as the provider has it. */
function current(held, conversation) {
  if (!held || held.state === "missing" || held.state === "unknown") return false;
  if (held.state === "spooled") return true;
  if (!conversation.updated_at || !held.last_captured_at) return false;
  return held.last_captured_at >= Date.parse(conversation.updated_at);
}
