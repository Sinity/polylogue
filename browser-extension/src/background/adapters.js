/** Explicit browser seams used by the background composition root. */
export function createBackgroundAdapters(browser = globalThis.chrome, _network = globalThis.fetch, clock = Date) {
  return Object.freeze({
    storage: browser.storage,
    alarms: browser.alarms,
    tabs: browser.tabs,
    scripting: browser.scripting,
    runtime: browser.runtime,
    action: browser.action,
    // Resolve fetch at call time: test profiles and browser harnesses replace
    // the network seam between service-worker starts.
    network: (...args) => globalThis.fetch(...args),
    now: () => clock.now(),
    log: (...args) => console.debug("[polylogue-background]", ...args),
  });
}

export const BACKGROUND_ALARM_OWNERS = Object.freeze({
  polylogueBrowserActionWake: "browser-actions",
  polylogueCaptureFreshnessWake: "capture-freshness",
  polylogueCaptureFreshnessSweep: "capture-freshness",
  polylogueCaptureRetry: "capture-retry",
  "polylogueBackfillTransportCleanup:*": "backfill-provider-transport",
  "polylogueBackfill:*": "backfill",
});
