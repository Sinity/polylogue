// One finite change request per visible tab. Subscribers apply references
// idempotently; only the final complete record advances the shared position.
const subscribers = new Set();
let position;
let controller;
let timer;
let stopped = false;
let revision = 0;

function report(error) {
  for (const subscriber of subscribers) subscriber.error?.(error);
}

async function batch() {
  if (controller || stopped || document.hidden || !subscribers.size) return;
  controller = new AbortController();
  const audience = [...subscribers];
  const version = revision;
  try {
    const query = position ? "?since=" + encodeURIComponent(position) : "";
    const response = await fetch("/api/v1/changes" + query, {
      credentials: "same-origin", signal: controller.signal,
    });
    if (response.status === 401) {
      stopped = true;
      clearInterval(clock);
      throw new Error("Sign-in expired. Reopen with polylogue web.");
    }
    if (!response.ok) throw new Error("Changes unavailable (" + response.status + ")");
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let pending = "";
    let began = false;
    let complete;
    let count = 0;
    try {
      for (;;) {
        const { value, done } = await reader.read();
        pending += decoder.decode(value, { stream: !done });
        let newline;
        while ((newline = pending.indexOf("\n")) >= 0) {
          const raw = pending.slice(0, newline);
          pending = pending.slice(newline + 1);
          const response = JSON.parse(raw);
          if (response.result !== "ok") throw new Error("Change batch failed");
          const line = response.data;
          if (complete) throw new Error("Records after change completion");
          if (line.type === "begin" && !began) {
            began = true;
            for (const subscriber of audience) await subscriber.begin?.(line);
          } else if (line.type === "change" && began) {
            for (const subscriber of audience) await subscriber.apply?.(line);
            count++;
          } else if (line.type === "complete" && began && line.count === count) {
            complete = line;
          } else throw new Error("Invalid change batch");
        }
        if (done) break;
      }
      if (pending || !complete) throw new Error("Change batch interrupted; retrying from the last complete position");
      for (const subscriber of audience) await subscriber.complete?.(complete);
      position = version === revision ? complete.position : undefined;
    } finally {
      await reader.cancel();
    }
  } catch (error) {
    if (error.name !== "AbortError") report(error);
  } finally {
    controller = undefined;
    if (!stopped && !document.hidden && subscribers.size) {
      timer = setTimeout(batch, 2000);
    }
  }
}

// Clock-driven live states can change without an archive write. The callback
// asks the existing API to refresh; it never infers a state from elapsed time.
let clock;
function visibility() {
  clearTimeout(timer);
  clearInterval(clock);
  if (document.hidden || stopped || !subscribers.size) controller?.abort();
  else {
    batch();
    clock = setInterval(() => {
      for (const subscriber of subscribers) Promise.resolve().then(() => subscriber.refresh?.()).catch(report);
    }, 30000);
  }
}
document.addEventListener("visibilitychange", visibility);
window.addEventListener("pagehide", () => {
  clearTimeout(timer);
  clearInterval(clock);
  controller?.abort();
});
window.addEventListener("pageshow", visibility);

export function watchChanges(subscriber) {
  subscribers.add(subscriber);
  revision++;
  // A joining view needs the full reference set, not another view's checkpoint.
  position = undefined;
  visibility();
  return () => {
    subscribers.delete(subscriber);
    if (!subscribers.size) visibility();
  };
}
