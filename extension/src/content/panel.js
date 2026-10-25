// The in-page panel: a small button at the page's corner that shows whether
// this conversation is in the archive, captures it on request, and saves a
// selected passage as a note. Alt+P opens it. It lives in a closed shadow
// root and acts only on real clicks and key presses, because page scripts
// share the document and can fire synthetic events.
(function (root) {
  const HOST_ID = "polylogue-panel";
  const REFRESH_MS = 15_000;
  // Elements that hold one message, by the provider's own markup.
  const MESSAGE = "[data-message-id]";

  function messageOf(node) {
    const element = node?.nodeType === 1 ? node : node?.parentElement;
    return element?.closest?.(MESSAGE) || null;
  }

  // A selection inside one message, with what identifies that message.
  function selectionNote(selection, url, provider) {
    const quote = String(selection?.toString?.() || "");
    if (!quote.trim() || !provider || selection.rangeCount < 1) return null;
    const message = messageOf(selection.anchorNode);
    if (!message || message !== messageOf(selection.focusNode)) return null;
    return {
      quote,
      source: {
        provider: provider.name,
        conversation_id: provider.conversationId(url),
        message_id: message.getAttribute("data-message-id"),
        page_url: url,
        selected_at: new Date().toISOString(),
      },
    };
  }

  const CSS = `
    :host { all: initial; color-scheme: light dark; }
    *, *::before, *::after { box-sizing: border-box; }
    .root { --bg: #fff; --panel: #f5f7f8; --ink: #172026; --muted: #5f6b76; --line: #d7dde2;
      --accent: #325d8f; --ok: #14764e; --warn: #9a5b00; --bad: #ad2f2f; --neutral: #64748b;
      position: fixed; right: 16px; bottom: 16px; z-index: 2147483646; pointer-events: none;
      font: 14px/1.45 system-ui, sans-serif; color: var(--ink); }
    button, textarea { font: inherit; }
    .chip { pointer-events: auto; float: right; display: inline-flex; align-items: center; gap: 8px;
      min-height: 40px; padding: 8px 11px; border: 1px solid var(--line); border-radius: 999px;
      color: var(--ink); background: var(--bg); box-shadow: 0 8px 28px rgba(0,0,0,.18); cursor: pointer; }
    .mark { display: grid; place-items: center; width: 22px; height: 22px; border-radius: 6px;
      color: #fff; background: var(--accent); font-weight: 700; }
    .dot { width: 9px; height: 9px; border-radius: 999px; background: var(--neutral); }
    .dot.ok { background: var(--ok); } .dot.warn { background: var(--warn); } .dot.bad { background: var(--bad); }
    .panel { pointer-events: auto; width: min(340px, calc(100vw - 24px)); max-height: calc(100vh - 84px);
      margin-bottom: 10px; overflow: auto; border: 1px solid var(--line); border-radius: 12px;
      background: var(--bg); box-shadow: 0 18px 54px rgba(0,0,0,.24); }
    .panel[hidden] { display: none; }
    section { display: grid; gap: 8px; padding: 12px 14px; border-bottom: 1px solid var(--line); }
    section:last-child { border-bottom: 0; }
    h2, h3, p { margin: 0; } h2 { font-size: 15px; }
    h3 { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }
    .line { display: flex; align-items: center; gap: 8px; font-weight: 650; }
    .meta { color: var(--muted); font-size: 12px; overflow-wrap: anywhere; }
    .quote { max-height: 96px; overflow: auto; padding: 8px; border-left: 3px solid var(--accent);
      background: var(--panel); white-space: pre-wrap; }
    textarea { width: 100%; border: 1px solid var(--line); border-radius: 8px; padding: 8px; color: var(--ink); background: var(--bg); }
    .buttons { display: flex; gap: 8px; flex-wrap: wrap; }
    .buttons button { min-height: 34px; padding: 6px 10px; border: 1px solid var(--line); border-radius: 8px;
      color: var(--ink); background: var(--panel); cursor: pointer; }
    .buttons button.primary { color: #fff; background: var(--accent); border-color: var(--accent); }
    .buttons button:disabled { opacity: .5; cursor: not-allowed; }
    @media (prefers-color-scheme: dark) { .root { --bg: #17191c; --panel: #22262a; --ink: #f2f4f5;
      --muted: #aab2b9; --line: #3a4148; --accent: #78a6d8; --ok: #3aa979; --warn: #c68a2c; --bad: #d66565; } }
  `;

  function element(doc, tag, className, text) {
    const node = doc.createElement(tag);
    if (className) node.className = className;
    if (text) node.textContent = text;
    return node;
  }

  function mount({ doc = root.document, runtime = root.chrome?.runtime, provider = root.polylogueProvider } = {}) {
    if (!doc?.documentElement || !runtime?.sendMessage || !provider || doc.getElementById(HOST_ID)) return null;
    const host = doc.createElement("div");
    host.id = HOST_ID;
    host.style.cssText = "all:initial;position:fixed;right:0;bottom:0;width:0;height:0;overflow:visible;z-index:2147483646;";
    const shadow = host.attachShadow({ mode: "closed" });
    const style = doc.createElement("style");
    style.textContent = CSS;
    const shell = element(doc, "div", "root");
    shell.hidden = true;

    const panel = element(doc, "div", "panel");
    panel.setAttribute("role", "dialog");
    panel.setAttribute("aria-label", "Polylogue");
    panel.hidden = true;
    const head = element(doc, "section");
    head.append(element(doc, "h2", "", "Polylogue"));
    const line = element(doc, "div", "line");
    const dot = element(doc, "span", "dot");
    const label = element(doc, "span", "", "Checking…");
    line.append(dot, label);
    const detail = element(doc, "p", "meta");
    const pageButtons = element(doc, "div", "buttons");
    const captureButton = element(doc, "button", "primary", "Capture now");
    const hideButton = element(doc, "button", "", "Hide on this site");
    pageButtons.append(captureButton, hideButton);
    head.append(line, detail, pageButtons);

    const noteSection = element(doc, "section");
    noteSection.append(element(doc, "h3", "", "Note on a selected passage"));
    const quote = element(doc, "div", "quote", "Select text inside one message to write a note about it.");
    const body = doc.createElement("textarea");
    body.rows = 4;
    body.setAttribute("aria-label", "Note");
    body.hidden = true;
    const noteButtons = element(doc, "div", "buttons");
    const saveButton = element(doc, "button", "primary", "Save note");
    saveButton.disabled = true;
    noteButtons.append(saveButton);
    const noteStatus = element(doc, "p", "meta");
    noteStatus.setAttribute("role", "status");
    noteSection.append(quote, body, noteButtons, noteStatus);
    panel.append(head, noteSection);

    const chip = element(doc, "button", "chip");
    chip.setAttribute("aria-label", "Open Polylogue");
    const chipDot = element(doc, "span", "dot");
    chip.append(element(doc, "span", "mark", "P"), chipDot);
    shell.append(panel, chip);
    shadow.append(style, shell);
    doc.documentElement.appendChild(host);

    let stopped = false;
    let note = null;
    let draft = null;
    let saving = false;
    let timer = null;

    function render(page) {
      if (page?.panel === false) {
        stop();
        return;
      }
      const said = root.polylogueWording.conversation(page);
      label.textContent = said.label;
      detail.textContent = said.detail;
      dot.className = `dot ${said.tone}`;
      chipDot.className = `dot ${said.tone}`;
      chip.setAttribute("aria-label", `Open Polylogue: ${said.label}`);
      captureButton.disabled = !page?.conversation;
      shell.hidden = false;
    }

    async function refresh() {
      if (stopped) return;
      try {
        render(await runtime.sendMessage({ type: "polylogue.page" }));
      } catch {
        stop();
      }
    }

    function updateSelection() {
      if (saving || shadow.activeElement === body) return;
      const next = selectionNote(root.getSelection?.(), root.location.href, provider);
      if (!next) return;
      note = next;
      draft = null;
      quote.textContent = next.quote;
      body.value = "";
      body.hidden = false;
      saveButton.disabled = false;
      noteStatus.textContent = "";
    }

    async function save() {
      if (!note || saving) return;
      if (!body.value.trim()) {
        noteStatus.textContent = "Write the note first.";
        return;
      }
      const content = JSON.stringify([note, body.value]);
      // Retrying the same draft reuses its ID, so it is never saved twice.
      if (draft?.content !== content) {
        draft = { content, request_id: root.crypto.randomUUID() };
      }
      saving = true;
      saveButton.disabled = true;
      noteStatus.textContent = "Saving…";
      try {
        const answer = await runtime.sendMessage({
          type: "polylogue.note",
          note: { request_id: draft.request_id, body: body.value, quote: note.quote, source: note.source },
        });
        if (answer?.note?.id !== draft.request_id) throw new Error(answer?.error || "no answer");
        noteStatus.textContent = "Note saved.";
      } catch (error) {
        noteStatus.textContent = `Not saved (${String(error?.message || error)}). Save again to retry.`;
      } finally {
        saving = false;
        saveButton.disabled = false;
      }
    }

    function open() {
      panel.hidden = false;
      void refresh();
    }

    function close() {
      panel.hidden = true;
    }

    function stop() {
      if (stopped) return;
      stopped = true;
      root.clearInterval(timer);
      doc.removeEventListener("selectionchange", updateSelection);
      doc.removeEventListener("keydown", toggle);
      host.remove();
    }

    function toggle(event) {
      if (!event.isTrusted || !event.altKey || String(event.key).toLowerCase() !== "p") return;
      event.preventDefault();
      if (panel.hidden) open(); else close();
    }

    const onClick = (handler) => (event) => { if (event.isTrusted) handler(); };
    chip.addEventListener("click", onClick(() => (panel.hidden ? open() : close())));
    captureButton.addEventListener("click", onClick(async () => {
      label.textContent = "Capturing…";
      render(await runtime.sendMessage({ type: "polylogue.capture" }).catch(() => null));
    }));
    hideButton.addEventListener("click", onClick(async () => {
      await runtime.sendMessage({ type: "polylogue.hide-panel", site: root.location.hostname });
      stop();
    }));
    saveButton.addEventListener("click", onClick(save));
    shadow.addEventListener("keydown", (event) => {
      if (event.isTrusted && event.key === "Escape") close();
    });
    doc.addEventListener("selectionchange", updateSelection);
    doc.addEventListener("keydown", toggle);
    timer = root.setInterval(refresh, REFRESH_MS);
    void refresh();
    return { host, shadow, refresh, stop, open, close, render };
  }

  root.polylogueTestHooks = { ...root.polylogueTestHooks, panel: { mount, selectionNote } };
  if (root.top === root.self) mount();
})(globalThis);
