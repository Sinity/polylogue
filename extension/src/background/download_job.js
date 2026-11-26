// The background download of one conversation's files, as explicit state.
// A job downloads the files a stored capture lacks and then stores that
// capture again with their bytes. Captures of the same conversation may run
// meanwhile: nothing is stored while one is in flight, and one that stores a
// newer capture replaces the capture the download was for.

export const QUEUED = "queued";
export const DOWNLOADING = "downloading";
export const WAITING_FOR_CAPTURES = "waiting for captures";
export const STORING = "storing";
export const ENDED = "ended";

// The phases a job may enter from each phase.
export const FOLLOWING_PHASES = Object.freeze({
  // Ended from the queue when a newer capture whose files could not be
  // listed replaced the queued one.
  [QUEUED]: [DOWNLOADING, ENDED],
  // A failed download goes straight to the next queued capture, or ends.
  [DOWNLOADING]: [WAITING_FOR_CAPTURES, DOWNLOADING, ENDED],
  // Replaced captures and captures with nothing to download are not stored.
  [WAITING_FOR_CAPTURES]: [STORING, DOWNLOADING, ENDED],
  // Back to waiting when a capture began before the bytes reached the outbox.
  [STORING]: [WAITING_FOR_CAPTURES, DOWNLOADING, ENDED],
  [ENDED]: [],
});

export class DownloadJob {
  /** `work` is `{tabId, base, files, missing}`: a stored capture and the files it lacks. */
  constructor(work) {
    this.phase = QUEUED;
    // The newest capture whose files are still to download, or null.
    this.next = work;
    // Captures of this conversation that began and have not ended.
    this.capturesInFlight = 0;
    // A newer capture was stored since the current download began, so the
    // capture being downloaded for no longer holds the newest text.
    this.replaced = false;
    // Downloaded files by ID, reused by a newer capture that names them.
    this.available = new Map();
    this.waiting = [];
    // Resolves when the job has ended; set by whoever runs the job.
    this.finished = null;
  }

  enter(phase) {
    if (!FOLLOWING_PHASES[this.phase].includes(phase)) {
      throw new Error(`a download job cannot go from ${this.phase} to ${phase}`);
    }
    this.phase = phase;
  }

  captureBegan() {
    this.capturesInFlight += 1;
  }

  captureEnded() {
    this.capturesInFlight -= 1;
    if (this.capturesInFlight === 0) for (const resume of this.waiting.splice(0)) resume();
  }

  /**
   * A newer capture of the conversation was stored. `work` is that capture
   * and the files it lacks, or null when its files could not be listed.
   */
  replace(work) {
    this.replaced = true;
    this.next = work;
  }

  /** Starts downloading for the queued capture and answers it; ends the job and answers null when none is queued. */
  startNext() {
    const work = this.next;
    if (!work) {
      this.end();
      return null;
    }
    this.next = null;
    this.replaced = false;
    this.enter(DOWNLOADING);
    return work;
  }

  /** Resolves once no capture of the conversation is in flight. */
  async capturesEnded() {
    this.enter(WAITING_FOR_CAPTURES);
    while (this.capturesInFlight > 0) await new Promise((resume) => { this.waiting.push(resume); });
  }

  startStoring() {
    this.enter(STORING);
  }

  /** Whether the downloaded bytes may still be stored with the capture they were downloaded for. */
  mayStore() {
    return !this.replaced && this.capturesInFlight === 0;
  }

  end() {
    if (this.phase !== ENDED) this.enter(ENDED);
    this.next = null;
    this.available.clear();
  }
}
