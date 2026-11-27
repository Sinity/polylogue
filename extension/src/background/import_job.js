// Provider import: capturing a web chat's older conversations. The job lives in
// Polylogue (started with `polylogue import PROVIDER`); this side claims it
// through the host, pages through the provider's conversation list, captures
// each conversation the archive lacks or holds an older copy of, and reports
// its position and totals. A restarted extension claims its job again and
// continues from the last reported position.

import "../capture_state.js";
import { ProviderDeferred } from "./provider.js";
import { askPage, workTab } from "./tabs.js";

const captureState = globalThis.polylogueCaptureState;

// Pause between two conversations, to stay well inside what the providers
// allow a person's browser.
const PAUSE_MS = 2000;
// Captures between two progress reports, which also renew the claim.
const REPORT_EVERY = 10;

class ClaimLost extends Error {}

export class ImportJob {
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
          const { job } = await this.host.call("internal import claim", { by: await this.instance() });
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
        await this.host.call("internal import report", { id: job.id, by, report: { kind, position, counts: { ...counts }, detail } });
        return true;
      } catch (error) {
        // The operator cancelled the job, or another instance holds it.
        this.log(`Provider import of ${job.provider} stopped: ${error.message}`);
        return false;
      }
    };
    // A refused sign-in pauses the job at the page's start: Polylogue keeps
    // it from being claimed until the operator starts the import again.
    const pause = async () => {
      const notice = `Sign in to ${job.provider} in this browser, then run \`polylogue import ${job.provider}\` to resume.`;
      this.log(`Import of ${job.provider} paused: ${notice}`);
      await report("paused", notice);
    };
    const { tab, opened } = await workTab(this.chrome, job.provider);
    try {
      for (;;) {
        const page = await this.capturer.scheduler.run(job.provider, () => askPage(this.chrome, tab.id, { type: "list", position }), { background: true, owner: tab.id, before: async () => { if (!(await report("progress"))) throw new ClaimLost("import_job claim lost"); } });
        if (page.status === 429 || page.status >= 500) {
          await report("progress", "The provider asked to wait; continuing later.");
          return;
        }
        if (signInRefused(page.status)) {
          await pause();
          return;
        }
        if (page.status !== 200) {
          await report("failed", `The provider's conversation list could not be read (${page.error || page.status}).`);
          return;
        }
        const listed = page.info.conversations.filter((c) => !since || !c.updated_at || Date.parse(c.updated_at) >= since);
        const answer = listed.length
          ? await captureState.conversations(this.host, job.provider, listed.map((c) => c.id))
          : { conversations: [] };
        const held = captureState.ofArchiveAnswer(job.provider, answer);
        // Reports name the position of the page's start, so the page's
        // outcomes join the totals only once the page is done; a resumed job
        // lists the page again and must not count it twice.
        const tally = { captured: 0, unchanged: 0, failed: 0 };
        let sinceReport = 0;
        for (const [i, conversation] of listed.entries()) {
          if (current(held[i], answer.conversations[i], conversation)) {
            tally.unchanged += 1;
            sinceReport += 1;
            if (sinceReport >= REPORT_EVERY) {
              sinceReport = 0;
              if (!(await report("progress"))) return;
            }
            continue;
          }
          try {
            const done = await this.capturer.capture(tab.id, job.provider, { id: conversation.id, reason: "import_job", before: async () => { if (!(await report("progress"))) throw new ClaimLost("import_job claim lost"); } });
            tally[done.outcome === "unchanged" ? "unchanged" : "captured"] += 1;
          } catch (error) {
            if (error instanceof ClaimLost) throw error;
            if (error.until) throw new ProviderDeferred(error.until, error.message);
            if (error.status === 429 || error.status >= 500) {
              await report("progress", "The provider asked to wait; continuing later.");
              return;
            }
            if (signInRefused(error.status)) {
              await pause();
              return;
            }
            tally.failed += 1;
            this.log(`Provider import could not capture ${job.provider} conversation ${conversation.id}: ${error.message}`);
          }
          sinceReport += 1;
          if (sinceReport >= REPORT_EVERY) {
            sinceReport = 0;
            if (!(await report("progress"))) return;
          }
          await this.sleep(PAUSE_MS);
        }
        counts.listed += page.info.conversations.length;
        for (const [name, number] of Object.entries(tally)) counts[name] += number;
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

function signInRefused(status) {
  return status === 401 || status === 403;
}

/**
 * Whether the archive holds this conversation as the provider has it: read
 * by the daemon, with its files, from a capture taken after the provider's
 * last change. `held` is its capture state and `row` the host's row for it.
 * A conversation that is only stored has some capture in the spool, of no
 * known version, so it is fetched again; the capturer stores it only if its
 * content changed.
 */
function current(held, row, conversation) {
  if (held !== captureState.IN_ARCHIVE) return false;
  if (!conversation.updated_at || !row.last_captured_at) return false;
  return row.last_captured_at >= Date.parse(conversation.updated_at);
}
