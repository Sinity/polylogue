import { BACKGROUND_ALARM_OWNERS, BACKGROUND_ALARMS } from "./adapters.js";

/**
 * Chrome wakeups are routing signals only. Durable state decides whether work
 * is still due after a duplicate wake or service-worker restart.
 */
export function registerBackgroundEvents({ alarms, runtime, tabs }, handlers) {
  alarms?.onAlarm?.addListener((alarm) => {
    const name = alarm?.name || "";
    if (name === BACKGROUND_ALARMS.browserActions) {
      void handlers.browserActions();
      return;
    }
    if (name === BACKGROUND_ALARMS.captureFreshness) {
      void handlers.captureFreshness();
      return;
    }
    if (name === BACKGROUND_ALARMS.captureFreshnessSweep) {
      void handlers.captureFreshnessSweep();
      return;
    }
    if (name.startsWith(`${BACKGROUND_ALARMS.backfillTransportCleanup}:`)) {
      void handlers.providerTransportCleanup(name);
      return;
    }
    if (name === BACKGROUND_ALARMS.captureRetry) {
      void handlers.captureRetry();
      return;
    }
    if (name.startsWith(`${BACKGROUND_ALARMS.backfill}:`)) {
      void handlers.backfill(name.slice(`${BACKGROUND_ALARMS.backfill}:`.length));
    }
  });

  runtime?.onInstalled?.addListener(() => {
    void handlers.installed();
  });

  runtime?.onStartup?.addListener(() => {
    void handlers.startup();
  });

  tabs?.onActivated?.addListener((activeInfo) => {
    void handlers.activated(activeInfo);
  });

  tabs?.onUpdated?.addListener((tabId, changeInfo, tab) => {
    if (changeInfo?.status !== "complete" && !changeInfo?.url) return;
    void handlers.updated(tabId, tab);
  });

  tabs?.onRemoved?.addListener((tabId) => {
    void handlers.removed(tabId);
  });
}

export function alarmOwner(name) {
  if (Object.hasOwn(BACKGROUND_ALARM_OWNERS, name)) return BACKGROUND_ALARM_OWNERS[name];
  if (name?.startsWith(`${BACKGROUND_ALARMS.backfillTransportCleanup}:`)) {
    return BACKGROUND_ALARM_OWNERS[`${BACKGROUND_ALARMS.backfillTransportCleanup}:*`];
  }
  if (name?.startsWith(`${BACKGROUND_ALARMS.backfill}:`)) {
    return BACKGROUND_ALARM_OWNERS[`${BACKGROUND_ALARMS.backfill}:*`];
  }
  return null;
}
