// The reader's keys and its sign-in exchange. Pages work without this
// script; it adds: `/` search focus, `j`/`k` next and previous result or
// message, `o` or Enter expand the current tool call or open the result,
// `y` copy the page as a CLI command.

const auth = document.getElementById("auth");
if (auth && location.pathname === "/auth") {
  // `polylogue web` opens /auth#<token>. The fragment never reaches the
  // server or a log; it is dropped from history before anything else.
  const token = location.hash.slice(1);
  history.replaceState(null, "", "/auth");
  if (token) {
    fetch("/auth", {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: "token=" + encodeURIComponent(token),
      credentials: "same-origin",
    }).then((r) => {
      if (r.ok) location.replace("/");
      else auth.textContent = "The access token is stale. Run `polylogue web` again.";
    });
  }
}

const items = () => [...document.querySelectorAll("tr.hit, section.message")];
let current = -1;

function focusItem(i) {
  const all = items();
  if (!all.length) return;
  current = Math.max(0, Math.min(all.length - 1, i));
  all.forEach((el, n) => el.classList.toggle("current", n === current));
  const el = all[current];
  el.scrollIntoView({ block: "nearest" });
  const target = el.querySelector("a, summary") || el;
  if (target.tabIndex < 0 && !target.matches("a, summary")) target.tabIndex = -1;
  target.focus({ preventScroll: true });
}

document.addEventListener("focusin", (event) => {
  const item = event.target.closest("tr.hit, section.message");
  if (item) current = items().indexOf(item);
});

document.addEventListener("keydown", (e) => {
  if (e.ctrlKey || e.metaKey || e.altKey || e.isComposing) return;
  const typing = e.target.closest("input, textarea, [contenteditable]:not([contenteditable=false])");
  if (typing) {
    if (e.key === "Escape") e.target.blur();
    return;
  }
  if (document.getSelection()?.toString()) return;
  if (!document.getElementById("shortcuts")?.checked) return;
  switch (e.key) {
    case "/": {
      e.preventDefault();
      const q = document.getElementById("q");
      q.focus();
      q.select();
      break;
    }
    case "j":
      focusItem(current + 1);
      break;
    case "k":
      focusItem(current - 1);
      break;
    case "o": {
      const el = items()[current];
      const d = el && el.querySelector("details");
      if (d) d.open = !d.open;
      else el?.querySelector("a")?.click();
      break;
    }
    case "Escape": {
      const details = e.target.closest("details");
      if (details) {
        details.open = false;
        details.querySelector("summary")?.focus();
      }
      break;
    }
    case "[":
      document.querySelector("[data-previous]")?.click();
      break;
    case "]":
      document.querySelector("[data-next]")?.click();
      break;
    case "?": {
      const help = document.getElementById("key-help");
      if (help) help.open = !help.open;
      break;
    }
    case "y":
      copyCli();
      break;
  }
});

function copyCli() {
  const button = document.querySelector("button.copy-cli");
  if (!button) return;
  Promise.resolve().then(() => navigator.clipboard.writeText(button.dataset.cli)).then(() => {
    const was = button.textContent;
    button.textContent = "copied";
    setTimeout(() => (button.textContent = was), 1200);
  }).catch(() => {
    const command = document.createElement("pre");
    command.textContent = button.dataset.cli;
    button.after(command);
    button.textContent = "Select and copy the command below";
  });
}

document.querySelector("button.copy-cli")?.addEventListener("click", copyCli);
