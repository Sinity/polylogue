/** Explicit browser seams used by the background composition root. */
export function createBackgroundAdapters(browser = globalThis.chrome, network = null, clock = Date) {
  return Object.freeze({
    storage: browser.storage,
    alarms: browser.alarms,
    tabs: browser.tabs,
    scripting: browser.scripting,
    runtime: browser.runtime,
    action: browser.action,
    // Resolve fetch at call time: test profiles and browser harnesses replace
    // the network seam between service-worker starts.
    network: (...args) => (network || globalThis.fetch)(...args),
    now: () => clock.now(),
    log: (...args) => console.debug("[polylogue-background]", ...args),
  });
}

export const BACKGROUND_ALARMS = Object.freeze({
  browserActions: "polylogueBrowserActionWake",
  captureFreshness: "polylogueCaptureFreshnessWake",
  captureFreshnessSweep: "polylogueCaptureFreshnessSweep",
  captureRetry: "polylogueCaptureRetry",
  backfillTransportCleanup: "polylogueBackfillTransportCleanup",
  backfill: "polylogueBackfillWake",
});

export const BACKGROUND_ALARM_OWNERS = Object.freeze({
  [BACKGROUND_ALARMS.browserActions]: "browser-actions",
  [BACKGROUND_ALARMS.captureFreshness]: "capture-freshness",
  [BACKGROUND_ALARMS.captureFreshnessSweep]: "capture-freshness",
  [BACKGROUND_ALARMS.captureRetry]: "capture-retry",
  [`${BACKGROUND_ALARMS.backfillTransportCleanup}:*`]: "backfill-provider-transport",
  [`${BACKGROUND_ALARMS.backfill}:*`]: "backfill",
});
