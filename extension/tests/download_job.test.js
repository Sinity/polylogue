import { describe, expect, it } from "vitest";
import {
  DOWNLOADING, DownloadJob, ENDED, FOLLOWING_PHASES, QUEUED, STORING, WAITING_FOR_CAPTURES,
} from "../src/background/download_job.js";

const work = (name) => ({ tabId: 1, base: { name }, files: [], missing: [{ id: "file-1" }] });

/** A job brought to `phase` through its own transitions. */
function jobIn(phase) {
  const job = new DownloadJob(work("first"));
  if (phase === QUEUED) return job;
  job.startNext();
  if (phase === DOWNLOADING) return job;
  void job.capturesEnded();
  if (phase === WAITING_FOR_CAPTURES) return job;
  job.startStoring();
  if (phase === STORING) return job;
  job.end();
  return job;
}

describe("a download job", () => {
  it("enters only the phases that may follow its own", () => {
    const allowed = {
      [QUEUED]: [DOWNLOADING, ENDED],
      [DOWNLOADING]: [DOWNLOADING, WAITING_FOR_CAPTURES, ENDED],
      [WAITING_FOR_CAPTURES]: [DOWNLOADING, STORING, ENDED],
      [STORING]: [DOWNLOADING, WAITING_FOR_CAPTURES, ENDED],
      [ENDED]: [],
    };
    const phases = Object.keys(allowed);
    expect(Object.keys(FOLLOWING_PHASES).sort()).toEqual([...phases].sort());
    for (const from of phases) {
      for (const to of phases) {
        const job = jobIn(from);
        expect(job.phase).toBe(from);
        if (allowed[from].includes(to)) {
          job.enter(to);
          expect(job.phase).toBe(to);
        } else {
          expect(() => job.enter(to)).toThrow(`a download job cannot go from ${from} to ${to}`);
          expect(job.phase).toBe(from);
        }
      }
    }
  });

  it("stores nothing while a capture is in flight, and resumes when every one has ended", async () => {
    const job = new DownloadJob(work("first"));
    job.startNext();
    job.captureBegan();
    job.captureBegan();
    let resumed = false;
    const waited = job.capturesEnded().then(() => { resumed = true; });
    expect(job.phase).toBe(WAITING_FOR_CAPTURES);
    expect(job.mayStore()).toBe(false);
    job.captureEnded();
    await Promise.resolve();
    expect(resumed).toBe(false);
    job.captureEnded();
    await waited;
    // The captures ended without storing a newer capture.
    expect(job.replaced).toBe(false);
    expect(job.mayStore()).toBe(true);
  });

  it("takes the newer capture next when one replaces the capture it downloads for", () => {
    const job = new DownloadJob(work("first"));
    expect(job.startNext().base.name).toBe("first");
    job.replace(work("second"));
    expect(job.replaced).toBe(true);
    expect(job.mayStore()).toBe(false);
    expect(job.startNext().base.name).toBe("second");
    expect(job.replaced).toBe(false);
    expect(job.startNext()).toBeNull();
    expect(job.phase).toBe(ENDED);
  });

  it("ends without downloading when a capture whose files are unknown replaces the queued one", () => {
    const job = new DownloadJob(work("first"));
    job.replace(null);
    expect(job.startNext()).toBeNull();
    expect(job.phase).toBe(ENDED);
  });
});
