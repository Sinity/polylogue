// One finite change request per visible tab. Subscribers apply references
// idempotently; only the final complete record advances the shared position.
const subscribers = new Set();
let position;
let controller;
let timer;
let stopped = false;
const ready = new Set();
const initializing = new Map();

function report(error) {
  for (const subscriber of subscribers) subscriber.error?.(error);
}

async function batch() {
  if (controller || stopped || document.hidden || !ready.size || initializing.size !== 0) return;
  controller = new AbortController();
  const audience = [...ready];
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
          if (response.outcome !== "ok") throw new Error("Change batch failed");
          const line = response.value;
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
      if (initializing.size === 0 && [...ready].every(subscriber => audience.includes(subscriber))) position = complete.position;
    } finally {
      await reader.cancel();
    }
  } catch (error) {
    if (error.name !== "AbortError") report(error);
  } finally {
    controller = undefined;
    if (!stopped && !document.hidden && subscribers.size) {
      clearTimeout(timer);
      timer = setTimeout(resume, 2000);
    }
  }
}

// Clock-driven live states can change without an archive write. The callback
// asks the existing API to refresh; it never infers a state from elapsed time.
let clock;
function visibility() {
  clearTimeout(timer);
  clearInterval(clock);
  if (document.hidden || stopped || !subscribers.size) {
    controller?.abort();
    for (const request of initializing.values()) request.abort();
  } else {
    resume();
    clock = setInterval(() => {
      for (const subscriber of ready) Promise.resolve().then(() => subscriber.refresh?.()).catch(report);
    }, 30000);
  }
}
document.addEventListener("visibilitychange", visibility);
window.addEventListener("pagehide", () => {
  clearTimeout(timer);
  clearInterval(clock);
  controller?.abort();
  for (const request of initializing.values()) request.abort();
});
window.addEventListener("pageshow", visibility);

// Each joining view reads current references in pages. Joining never resets
// the position that existing views have already completed.
async function readCurrent(subscriber, signal) {
  let cursor;
  let initial;
  let count = 0;
  do {
    const query = new URLSearchParams({ query: subscriber.query, limit: "50" });
    if (cursor) query.set("cursor", cursor);
    const response = await fetch("/api/v1/query?" + query, { credentials: "same-origin", signal });
    if (response.status === 401) { stopped = true; throw new Error("Sign-in expired. Reopen with polylogue web."); }
    if (!response.ok) throw new Error("Current references unavailable (" + response.status + ")");
    const answer = await response.json();
    if (answer.outcome === "error") throw new Error(answer.error.message);
    const page = answer.value;
    if (!subscribers.has(subscriber)) return;
    if (!initial) {
      initial = page.read_at.cursor;
      position ??= initial;
      await subscriber.begin?.({ type: "begin", position: null, generation: page.read_at.generation, replacement: true, generation_changed: false });
    }
    if (subscriber.page) await subscriber.page(page);
    else for (const row of page.rows) await subscriber.apply?.({ type: "change", component: ["note", "excision"].includes(row.kind) ? "archive" : "index", change: row });
    count += page.rows.length;
    cursor = page.next;
  } while (cursor && subscribers.has(subscriber));
  if (!subscribers.has(subscriber)) return;
  await subscriber.complete?.({ type: "complete", position: initial, count, replacement: true });
  ready.add(subscriber);
  initializing.delete(subscriber);
  visibility();
}

function resume() {
  if (document.hidden || stopped || !subscribers.size) return;
  for (const subscriber of subscribers) {
    if (ready.has(subscriber) || initializing.has(subscriber)) continue;
    const request = new AbortController();
    initializing.set(subscriber, request);
    readCurrent(subscriber, request.signal).catch(error => {
      initializing.delete(subscriber);
      if (error.name !== "AbortError") subscriber.error?.(error);
      batch();
      if (stopped) visibility();
      else if (!document.hidden && subscribers.has(subscriber)) {
        clearTimeout(timer);
        timer = setTimeout(resume, 2000);
      }
    });
  }
  batch();
}

export function watchChanges(subscriber) {
  if (!subscriber.query?.trim()) throw new Error("A live view must supply its current query");
  subscribers.add(subscriber);
  resume();
  return () => {
    subscribers.delete(subscriber);
    ready.delete(subscriber);
    initializing.get(subscriber)?.abort();
    initializing.delete(subscriber);
    visibility();
  };
}
