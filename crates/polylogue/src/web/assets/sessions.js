import "/assets/search.js";
import { watchChanges } from "/assets/live.js";

const list = document.getElementById("session-list");
const announcement = document.getElementById("live-announcement");
const apply = document.getElementById("apply-updates");
const pause = document.getElementById("pause-updates");
let pending;
let paused = false;
let refreshing = false;
let refreshAgain = false;
let focusedSession;
let previous;
let updated = 0;
document.addEventListener("focusin", event => {
  if (!list.contains(event.target) && event.target !== apply) focusedSession = undefined;
});
list.addEventListener("focusin", event => {
  focusedSession = event.target.closest("[data-session]")?.dataset.session;
});
let blocked = false;
pause.hidden = false;
pause.addEventListener("click", () => {
  paused = !paused;
  pause.textContent = paused ? "Resume updates" : "Pause updates";
  if (!paused) refresh();
});

function commit() {
  if (!pending) return;
  // Never remove the focused row, including when it stopped matching.
  const focused = document.activeElement === apply ? focusedSession : document.activeElement?.closest("[data-session]")?.dataset.session;
  if (focused) {
    const replacement = [...pending.querySelectorAll("[data-session]")].find(row => row.dataset.session === focused);
    if (!replacement) {
      announcement.textContent = "The focused session left this list. Open it or move focus before applying updates.";
      return;
    }
  }
  list.replaceChildren(...pending.childNodes);
  for (const button of list.querySelectorAll("[data-previous]")) {
    button.hidden = false;
    button.disabled = history.length < 2;
    button.addEventListener("click", () => history.back());
  }
  if (focused) [...list.querySelectorAll("[data-session]")].find(row => row.dataset.session === focused)?.querySelector("a")?.focus();
  pending = undefined;
  apply.hidden = true;
  announcement.textContent = updated + " sessions updated at " + new Date().toLocaleTimeString();
}
apply.addEventListener("click", commit);

function content(root) {
  return JSON.stringify([...root.querySelectorAll("[data-session]")].map(row => [row.dataset.session, row.dataset.content])) + Boolean(root.querySelector("[data-next]"));
}

async function refresh() {
  if (paused || blocked || document.hidden) return;
  if (refreshing) {
    refreshAgain = true;
    return;
  }
  refreshing = true;
  try {
    const response = await fetch(location.href, { credentials: "same-origin" });
    if (response.status === 401 || response.redirected) {
      blocked = true;
      throw new Error("Sign-in expired. Reopen with polylogue web.");
    }
    if (!response.ok) throw new Error("Session update unavailable (" + response.status + ")");
    const next = new DOMParser().parseFromString(await response.text(), "text/html").getElementById("session-list");
    if (!next) throw new Error("Session list unavailable");
    refreshNotices();
    if (content(next) === content(list)) {
      if (pending) announcement.textContent = "Sessions are current.";
      pending = undefined;
      apply.hidden = true;
      return;
    }
    const oldRows = new Map([...list.querySelectorAll("[data-session]")].map(row => [row.dataset.session, row.dataset.content]));
    const newRows = new Map([...next.querySelectorAll("[data-session]")].map(row => [row.dataset.session, row.dataset.content]));
    updated = [...new Set([...oldRows.keys(), ...newRows.keys()])].filter(id => oldRows.get(id) !== newRows.get(id)).length;
    pending = next;
    if (list.contains(document.activeElement) || document.activeElement === apply) {
      apply.hidden = false;
      announcement.textContent = updated ? updated + " sessions updated. Apply updates when ready." : "List pages changed. Apply updates when ready.";
    } else commit();
  } catch (error) { announcement.textContent = error.message; }
  finally {
    refreshing = false;
    if (refreshAgain) {
      refreshAgain = false;
      refresh();
    }
  }
}
watchChanges({
  async complete(line) { if (previous !== line.position) await refresh(); previous = line.position; },
  refresh,
  error(error) { announcement.textContent = error.message; },
});
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) refresh();
});
const load = document.getElementById("load-notices");
load.hidden = false;
async function refreshNotices() {
  const output = document.getElementById("notices");
  try {
    const response = await fetch("/api/v1/status", { credentials: "same-origin" });
    if (!response.ok) throw new Error("Notices unavailable (" + response.status + ")");
    const data = await response.json();
    if (data.notices?.error || !Number.isInteger(data.notices?.last_day)) throw new Error(data.notices?.error ?? "Notice count unavailable");
    output.textContent = data.notices.last_day + " notices in the last day. ";
    for (const notice of data.notices.latest ?? []) {
      const p = document.createElement("p");
      p.textContent = notice.text + " ";
      const link = document.createElement("a");
      link.href = "/search?q=" + encodeURIComponent('deliveries where id:"' + notice.id.replaceAll("\\", "\\\\").replaceAll('"', '\\"') + '"');
      link.textContent = "View delivery";
      p.append(link);
      output.append(p);
    }
  } catch (error) { output.textContent = error.message; }
}
load.addEventListener("click", refreshNotices);
refreshNotices();
