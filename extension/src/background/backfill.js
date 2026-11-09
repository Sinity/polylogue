// Backfill: capturing a web chat's older conversations. The job lives in
// Polylogue (started with `polylogue backfill start`); this side claims it
// through the host, pages through the provider's conversation list, captures
// each conversation the archive lacks or holds an older copy of, and reports
// its position and totals. A restarted extension claims its job again and
// continues from the last reported position.

import { ProviderDeferred } from "./provider.js";
import { askPage, workTab } from "./tabs.js";

// Pause between two conversations, to stay well inside what the providers
// allow a person's browser.
const PAUSE_MS = 2000;
// Captures between two progress reports, which also renew the claim.
const REPORT_EVERY = 10;

class ClaimLost extends Error {}

export class Backfill {
  constructor({ chrome, host, capturer, instance, log = () => {}, sleep = (ms) => new Promise((r) => setTimeout(r, ms)) }) {
    this.chrome = chrome;
    this.host = host;
    this.capturer = capturer;
    this.instance = instance;
    this.log = log;
    this.sleep = sleep;
    this.running = null;
    this.resumeAt = 0;
  }

  /**
   * Claims a job and runs it, unless one is running here already or the
   * provider asked to wait.
   */
  step() {
    if (!this.running && Date.now() >= this.resumeAt) {
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
        const page = await this.capturer.scheduler.run(job.provider, () => askPage(this.chrome, tab.id, { type: "list", position }), { background: true, owner: tab.id, before: async () => { if (!(await report("progress"))) throw new ClaimLost("backfill claim lost"); } });
        if (page.status === 429 || page.status >= 500) {
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
            sinceReport += 1;
            if (sinceReport >= REPORT_EVERY) {
              sinceReport = 0;
              if (!(await report("progress"))) return;
            }
            continue;
          }
          try {
            const done = await this.capturer.capture(tab.id, job.provider, { id: conversation.id, reason: "backfill", before: async () => { if (!(await report("progress"))) throw new ClaimLost("backfill claim lost"); } });
            counts[done.outcome === "unchanged" ? "unchanged" : "captured"] += 1;
          } catch (error) {
            if (error instanceof ClaimLost) throw error;
            if (error.until) throw new ProviderDeferred(error.until, error.message);
            if (error.status === 429 || error.status >= 500) {
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
        if (page.info.done || (passedSince && !page.info.next_partition)) {
          position = page.info.next;
          await report("finished");
          return;
        }
        position = passedSince ? page.info.next_partition : page.info.next;
        if (!(await report("progress"))) return;
      }
    } catch (error) {
      if (error instanceof ClaimLost) return;
      if (!(error instanceof ProviderDeferred) && !error.until) throw error;
      this.resumeAt = error.until;
      await report("progress", error.message);
    } finally {
      // Files download through the tab, after their conversation is stored.
      if (opened) {
        await this.capturer.downloads();
        await this.chrome.tabs.remove(tab.id).catch(() => undefined);
      }
    }
  }
}

/** Whether the archive holds this conversation as the provider has it. */
function current(held, conversation) {
  if (held?.downloading) return false;
  if (!held || held.state === "missing" || held.state === "unknown") return false;
  if (held.state === "spooled") return true;
  if (!conversation.updated_at || !held.last_captured_at) return false;
  return held.last_captured_at >= Date.parse(conversation.updated_at);
}
