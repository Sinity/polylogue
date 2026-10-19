import { describe, expect, it, vi } from "vitest";
import { BACKGROUND_ALARMS, BACKGROUND_ALARM_OWNERS } from "../src/background/adapters.js";
import { alarmOwner, registerBackgroundEvents } from "../src/background/events.js";

function eventHarness() {
  const listeners = {};
  const add = (name) => ({ addListener: vi.fn((listener) => { listeners[name] = listener; }) });
  const adapters = {
    alarms: { onAlarm: add("alarm") },
    runtime: { onInstalled: add("installed"), onStartup: add("startup") },
    tabs: { onActivated: add("activated"), onUpdated: add("updated"), onRemoved: add("removed") },
  };
  const handlers = Object.fromEntries([
    "browserActions", "captureFreshness", "captureFreshnessSweep", "providerTransportCleanup",
    "captureRetry", "backfill", "installed", "startup", "activated", "updated", "removed",
  ].map((name) => [name, vi.fn()]));
  registerBackgroundEvents(adapters, handlers);
  return { handlers, listeners };
}

describe("background event ownership", () => {
  it("declares one owner for each exact and patterned wakeup", () => {
    expect(alarmOwner(BACKGROUND_ALARMS.browserActions)).toBe("browser-actions");
    expect(alarmOwner(BACKGROUND_ALARMS.captureFreshness)).toBe("capture-freshness");
    expect(alarmOwner(BACKGROUND_ALARMS.captureFreshnessSweep)).toBe("capture-freshness");
    expect(alarmOwner(BACKGROUND_ALARMS.captureRetry)).toBe("capture-retry");
    expect(alarmOwner(`${BACKGROUND_ALARMS.backfillTransportCleanup}:chatgpt:42`))
      .toBe("backfill-provider-transport");
    expect(alarmOwner(`${BACKGROUND_ALARMS.backfill}:job-1`)).toBe("backfill");
    expect(alarmOwner("unowned-alarm")).toBeNull();
    expect(Object.values(BACKGROUND_ALARM_OWNERS)).toEqual([
      "browser-actions", "capture-freshness", "capture-freshness", "capture-retry",
      "backfill-provider-transport", "backfill",
    ]);
  });

  it("routes duplicate wakeups to the single owning handler", () => {
    const { handlers, listeners } = eventHarness();

    listeners.alarm({ name: BACKGROUND_ALARMS.browserActions });
    listeners.alarm({ name: `${BACKGROUND_ALARMS.backfill}:job-1` });
    listeners.alarm({ name: `${BACKGROUND_ALARMS.backfill}:job-1` });
    listeners.alarm({ name: `${BACKGROUND_ALARMS.backfillTransportCleanup}:chatgpt:42` });
    listeners.alarm({ name: "unknown" });

    expect(handlers.browserActions).toHaveBeenCalledTimes(1);
    expect(handlers.backfill).toHaveBeenCalledTimes(2);
    expect(handlers.backfill).toHaveBeenNthCalledWith(1, "job-1");
    expect(handlers.backfill).toHaveBeenNthCalledWith(2, "job-1");
    expect(handlers.providerTransportCleanup).toHaveBeenCalledWith(
      `${BACKGROUND_ALARMS.backfillTransportCleanup}:chatgpt:42`,
    );
    expect(handlers.captureRetry).not.toHaveBeenCalled();
  });

  it("filters tab updates before handing them to the presentation owner", () => {
    const { handlers, listeners } = eventHarness();

    listeners.updated(42, { status: "loading" }, { id: 42 });
    listeners.updated(42, { status: "complete" }, { id: 42 });
    listeners.activated({ tabId: 42 });
    listeners.removed(42);
    listeners.installed();
    listeners.startup();

    expect(handlers.updated).toHaveBeenCalledTimes(1);
    expect(handlers.updated).toHaveBeenCalledWith(42, { id: 42 });
    expect(handlers.activated).toHaveBeenCalledWith({ tabId: 42 });
    expect(handlers.removed).toHaveBeenCalledWith(42);
    expect(handlers.installed).toHaveBeenCalledTimes(1);
    expect(handlers.startup).toHaveBeenCalledTimes(1);
  });
});
