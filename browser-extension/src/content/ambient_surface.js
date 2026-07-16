(function (root) {
  const HOST_ID = "polylogue-ambient-surface";
  const REFRESH_INTERVAL_MS = 15000;
  const MAX_SELECTION_CHARS = 2000;
  const MESSAGE_SELECTORS = [
    "[data-message-author-role]",
    "[data-testid^='conversation-turn']",
    "[data-testid='user-message']",
    "[data-testid^='chat-message']",
    ".font-claude-message",
  ];

  function providerForUrl(value) {
    try {
      const hostname = new URL(value).hostname;
      if (hostname === "chatgpt.com" || hostname.endsWith(".chatgpt.com")) return "chatgpt";
      if (hostname === "claude.ai" || hostname.endsWith(".claude.ai")) return "claude-ai";
    } catch {
      return null;
    }
    return null;
  }

  function messageElementForNode(node) {
    const element = node?.nodeType === 1 ? node : node?.parentElement || null;
    if (!element?.closest) return null;
    for (const selector of MESSAGE_SELECTORS) {
      const match = element.closest(selector);
      if (match) return match;
    }
    return null;
  }

  function selectedMessageElement(selection) {
    if (!selection || selection.rangeCount < 1 || selection.isCollapsed) return null;
    const anchorMessage = messageElementForNode(selection.anchorNode);
    const focusMessage = messageElementForNode(selection.focusNode);
    return anchorMessage && anchorMessage === focusMessage ? anchorMessage : null;
  }

  function deriveSelectionCandidate(selection, {
    url = root.location?.href || "",
    provider = providerForUrl(url),
    capturedAt = new Date().toISOString(),
  } = {}) {
    const text = String(selection?.toString?.() || "").replace(/\s+/g, " ").trim();
    if (!text || !provider || !selectedMessageElement(selection)) return null;
    return {
      kind: "selection_assertion_candidate",
      provider,
      source_url: url,
      captured_at: capturedAt,
      text: text.slice(0, MAX_SELECTION_CHARS),
      truncated: text.length > MAX_SELECTION_CHARS,
      persistence: "not_supported",
    };
  }

  function createElement(doc, tag, className = "", text = "") {
    const node = doc.createElement(tag);
    if (className) node.className = className;
    if (text) node.textContent = text;
    return node;
  }

  function cssText() {
    return `
      :host { all: initial; color-scheme: light dark; }
      *, *::before, *::after { box-sizing: border-box; }
      .root {
        --pl-bg: #ffffff;
        --pl-panel: #f5f7f8;
        --pl-ink: #172026;
        --pl-muted: #5f6b76;
        --pl-line: #d7dde2;
        --pl-accent: #325d8f;
        --pl-ok: #14764e;
        --pl-warn: #9a5b00;
        --pl-neutral: #64748b;
        --pl-bad: #ad2f2f;
        position: fixed;
        right: 16px;
        bottom: 16px;
        z-index: 2147483646;
        font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        color: var(--pl-ink);
        pointer-events: none;
      }
      button { font: inherit; }
      .chip {
        pointer-events: auto;
        display: inline-flex;
        float: right;
        align-items: center;
        gap: 8px;
        min-width: 44px;
        min-height: 44px;
        padding: 8px 11px;
        border: 1px solid var(--pl-line);
        border-radius: 999px;
        color: var(--pl-ink);
        background: var(--pl-bg);
        box-shadow: 0 8px 28px rgba(0, 0, 0, .18);
        cursor: pointer;
      }
      .chip:hover { border-color: var(--pl-accent); }
      button:focus-visible { outline: 3px solid color-mix(in srgb, var(--pl-accent), transparent 55%); outline-offset: 2px; }
      .mark { display: grid; place-items: center; width: 24px; height: 24px; border-radius: 7px; color: white; background: var(--pl-accent); font-weight: 750; }
      .dot { width: 9px; height: 9px; border-radius: 999px; background: var(--pl-muted); }
      .dot.ok { background: var(--pl-ok); }
      .dot.warn { background: var(--pl-warn); }
      .dot.neutral { background: var(--pl-neutral); }
      .dot.bad { background: var(--pl-bad); }
      .count { min-width: 18px; color: var(--pl-muted); font-size: 12px; text-align: center; }
      .panel {
        pointer-events: auto;
        width: min(380px, calc(100vw - 24px));
        max-height: min(720px, calc(100vh - 84px));
        margin-bottom: 10px;
        overflow: auto;
        border: 1px solid var(--pl-line);
        border-radius: 14px;
        background: var(--pl-bg);
        box-shadow: 0 18px 54px rgba(0, 0, 0, .24);
      }
      .panel[hidden] { display: none; }
      .head { position: sticky; top: 0; z-index: 1; display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 13px 14px; border-bottom: 1px solid var(--pl-line); background: var(--pl-bg); }
      h2, h3, p { margin: 0; }
      h2 { font-size: 16px; }
      h3 { font-size: 12px; letter-spacing: .04em; text-transform: uppercase; color: var(--pl-muted); }
      .close, .secondary {
        border: 1px solid var(--pl-line);
        border-radius: 8px;
        color: var(--pl-ink);
        background: var(--pl-panel);
        cursor: pointer;
      }
      .close { width: 34px; height: 34px; }
      .secondary { min-height: 36px; padding: 7px 10px; }
      .section { display: grid; gap: 8px; padding: 13px 14px; border-bottom: 1px solid var(--pl-line); }
      .section:last-child { border-bottom: 0; }
      .status-line { display: flex; align-items: center; gap: 8px; font-weight: 700; }
      .meta { color: var(--pl-muted); font-size: 12px; overflow-wrap: anywhere; }
      .list { display: grid; gap: 8px; }
      .item { display: grid; gap: 3px; padding: 9px; border: 1px solid var(--pl-line); border-radius: 9px; background: var(--pl-panel); }
      .item-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 8px; }
      .item-title { min-width: 0; font-weight: 650; overflow-wrap: anywhere; }
      .pill { flex: 0 0 auto; padding: 2px 7px; border-radius: 999px; color: white; background: var(--pl-muted); font-size: 11px; font-weight: 700; }
      .pill.ok { background: var(--pl-ok); }
      .pill.warn { background: var(--pl-warn); }
      .pill.neutral { background: var(--pl-neutral); }
      .pill.bad { background: var(--pl-bad); }
      .empty { color: var(--pl-muted); font-size: 12px; }
      .selection { max-height: 96px; overflow: auto; padding: 8px; border-left: 3px solid var(--pl-accent); background: var(--pl-panel); white-space: pre-wrap; }
      .disabled { width: 100%; min-height: 38px; border: 1px dashed var(--pl-line); border-radius: 8px; color: var(--pl-muted); background: transparent; cursor: not-allowed; }
      .actions { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
      @media (prefers-color-scheme: dark) {
        .root {
          --pl-bg: #17191c;
          --pl-panel: #22262a;
          --pl-ink: #f2f4f5;
          --pl-muted: #aab2b9;
          --pl-line: #3a4148;
          --pl-accent: #78a6d8;
          --pl-ok: #3aa979;
          --pl-warn: #c68a2c;
          --pl-neutral: #9aa4ad;
          --pl-bad: #d66565;
        }
      }
      @media (max-width: 520px) {
        .root { right: 8px; bottom: 8px; }
        .panel { width: calc(100vw - 16px); max-height: calc(100vh - 68px); }
      }
      @media (prefers-reduced-motion: reduce) { * { scroll-behavior: auto !important; transition: none !important; } }
    `;
  }

  function mount({
    doc = root.document,
    runtime = root.chrome?.runtime,
    selectionSource = root,
    locationSource = root.location,
  } = {}) {
    if (!doc?.documentElement || !runtime?.sendMessage) return null;
    const prior = doc.getElementById(HOST_ID);
    if (prior) return root.polylogueAmbientSurfaceMounted || null;

    const host = doc.createElement("div");
    host.id = HOST_ID;
    host.setAttribute("data-polylogue-surface", "ambient-mission-control");
    // Fixed positioning keeps the host outside provider layout. Overflow must
    // remain visible: paint containment would clip the panel to this zero-size
    // anchor and make a visually correct DOM tree render as an invisible UI.
    host.style.cssText = "all:initial;position:fixed;right:0;bottom:0;width:0;height:0;overflow:visible;pointer-events:none;z-index:2147483646;contain:style;";
    const shadow = host.attachShadow({ mode: "closed" });
    const style = doc.createElement("style");
    style.textContent = cssText();
    const shell = createElement(doc, "div", "root");
    // Stay visually quiet until the first settings/status response confirms
    // that the surface is enabled for this site.
    shell.hidden = true;
    const panel = createElement(doc, "section", "panel");
    panel.id = "polylogue-ambient-panel";
    panel.setAttribute("role", "dialog");
    panel.setAttribute("aria-labelledby", "polylogue-ambient-title");
    panel.setAttribute("aria-modal", "false");
    panel.hidden = true;

    const head = createElement(doc, "div", "head");
    const title = createElement(doc, "h2", "", "Polylogue mission control");
    title.id = "polylogue-ambient-title";
    const close = createElement(doc, "button", "close", "×");
    close.type = "button";
    close.setAttribute("aria-label", "Close Polylogue mission control");
    head.append(title, close);
    panel.appendChild(head);

    const conversationSection = createElement(doc, "section", "section");
    conversationSection.appendChild(createElement(doc, "h3", "", "This conversation"));
    const conversationStatus = createElement(doc, "div", "status-line");
    const conversationDot = createElement(doc, "span", "dot");
    const conversationLabel = createElement(doc, "span", "", "Checking");
    conversationStatus.append(conversationDot, conversationLabel);
    const conversationDetail = createElement(doc, "p", "meta", "Reading receiver state…");
    conversationSection.append(conversationStatus, conversationDetail);
    panel.appendChild(conversationSection);

    const receiverSection = createElement(doc, "section", "section");
    receiverSection.appendChild(createElement(doc, "h3", "", "Receiver pairing"));
    const receiverHeadline = createElement(doc, "div", "item-title", "Checking receiver…");
    const receiverDetail = createElement(doc, "p", "meta");
    receiverSection.append(receiverHeadline, receiverDetail);
    panel.appendChild(receiverSection);

    const timelineSection = createElement(doc, "section", "section");
    timelineSection.appendChild(createElement(doc, "h3", "", "What Polylogue did here"));
    const timelineList = createElement(doc, "div", "list");
    timelineSection.appendChild(timelineList);
    panel.appendChild(timelineSection);

    const workSection = createElement(doc, "section", "section");
    workSection.appendChild(createElement(doc, "h3", "", "Work queue"));
    const workMeta = createElement(doc, "p", "meta");
    const workList = createElement(doc, "div", "list");
    workSection.append(workMeta, workList);
    panel.appendChild(workSection);

    const selectionSection = createElement(doc, "section", "section");
    selectionSection.appendChild(createElement(doc, "h3", "", "Selection to assertion"));
    const selectionText = createElement(doc, "div", "selection", "Select text inside a conversation message to prepare an assertion candidate.");
    const assertionButton = createElement(doc, "button", "disabled", "Save assertion — receiver API unavailable");
    assertionButton.type = "button";
    assertionButton.disabled = true;
    const assertionDetail = createElement(doc, "p", "meta", "Candidates stay in this page only; nothing is persisted or sent.");
    selectionSection.append(selectionText, assertionButton, assertionDetail);
    panel.appendChild(selectionSection);

    const actionSection = createElement(doc, "section", "section");
    const actions = createElement(doc, "div", "actions");
    const refreshButton = createElement(doc, "button", "secondary", "Refresh");
    refreshButton.type = "button";
    const hideButton = createElement(doc, "button", "secondary", "Hide on this site");
    hideButton.type = "button";
    actions.append(refreshButton, hideButton);
    actionSection.appendChild(actions);
    panel.appendChild(actionSection);

    const chip = createElement(doc, "button", "chip");
    chip.type = "button";
    chip.setAttribute("aria-controls", panel.id);
    chip.setAttribute("aria-expanded", "false");
    chip.setAttribute("aria-label", "Open Polylogue mission control; checking status");
    const mark = createElement(doc, "span", "mark", "P");
    const chipDot = createElement(doc, "span", "dot");
    const count = createElement(doc, "span", "count", "—");
    chip.append(mark, chipDot, count);
    shell.append(panel, chip);
    shadow.append(style, shell);
    doc.documentElement.appendChild(host);

    let stopped = false;
    let snapshot = null;
    let selectionCandidate = null;
    let timer = null;

    function setTone(node, tone) {
      node.className = `dot ${tone || "neutral"}`;
    }

    function clearNode(node) {
      while (node.firstChild) node.firstChild.remove();
    }

    function renderTimeline(items) {
      clearNode(timelineList);
      const events = Array.isArray(items) ? items.slice(0, 5) : [];
      if (!events.length) {
        timelineList.appendChild(createElement(doc, "p", "empty", "No decisions recorded for this conversation yet."));
        return;
      }
      for (const event of events) {
        const presentation = root.PolylogueOperatorStatus?.eventPresentation?.(event) || {
          label: event.event || "Observed",
          detail: [event.reason, event.detail].filter(Boolean).join(" · "),
        };
        const item = createElement(doc, "div", "item");
        item.append(createElement(doc, "div", "item-title", presentation.label));
        if (presentation.detail) item.append(createElement(doc, "p", "meta", presentation.detail));
        item.append(createElement(doc, "p", "meta", event.at ? new Date(event.at).toLocaleString() : ""));
        timelineList.appendChild(item);
      }
    }

    function renderWork(work) {
      clearNode(workList);
      workMeta.textContent = "";
      const items = root.PolylogueOperatorStatus?.normalizeWorkItems?.({
        captureQueue: work?.capture_queue,
        freshnessQueue: work?.freshness_queue,
        backfillJobs: work?.backfill_jobs,
        receiverOnline: snapshot?.state?.online !== false,
      }) || [];
      count.textContent = items.length > 99 ? "99+" : String(items.length);
      if (!items.length) {
        workList.appendChild(createElement(doc, "p", "empty", "No capture work is waiting."));
        return;
      }
      for (const workItem of items.slice(0, 6)) {
        const item = createElement(doc, "div", "item");
        const itemHead = createElement(doc, "div", "item-head");
        itemHead.append(createElement(doc, "div", "item-title", workItem.title));
        const pill = createElement(doc, "span", `pill ${workItem.status.tone}`, workItem.status.label);
        itemHead.appendChild(pill);
        item.append(itemHead);
        item.append(createElement(doc, "p", "meta", `${workItem.phase} · ${workItem.cadence}`));
        item.append(createElement(doc, "p", "meta", `Owner: ${workItem.owner}`));
        if (workItem.cooldown) item.append(createElement(doc, "p", "meta", `Cooldown/backoff: ${workItem.cooldown}`));
        workList.appendChild(item);
      }
    }

    function render(nextSnapshot) {
      snapshot = nextSnapshot;
      const state = nextSnapshot?.state || { online: false, error: "receiver_unavailable" };
      const status = root.PolylogueOperatorStatus?.operatorStatusForState?.(state) || {
        label: state.online === false ? "Receiver offline" : "Needs attention",
        tone: "warn",
      };
      conversationLabel.textContent = status.label;
      conversationDetail.textContent = status.detail || "Current conversation status unavailable.";
      setTone(conversationDot, status.tone);
      setTone(chipDot, status.tone);
      chip.setAttribute("aria-label", `Open Polylogue mission control; ${status.label}`);

      const pairing = root.PolylogueOperatorStatus?.receiverPairingPresentation?.({
        pairing: nextSnapshot?.receiver?.pairing,
        health: nextSnapshot?.receiver?.health,
        configuredUrl: nextSnapshot?.receiver?.configured_url,
      });
      receiverHeadline.textContent = pairing?.headline || "Receiver state unavailable";
      receiverDetail.textContent = pairing?.detail || "";

      renderTimeline(nextSnapshot?.timeline);
      renderWork(nextSnapshot?.work);

      const assertionRouteAdvertised = Boolean(nextSnapshot?.assertions?.persistence_supported);
      assertionButton.disabled = true;
      assertionButton.textContent = assertionRouteAdvertised
        ? "Save assertion — extension handler unavailable"
        : "Save assertion — receiver API unavailable";
      assertionDetail.textContent = assertionRouteAdvertised
        ? "The receiver advertises persistence, but this extension build has no authenticated write handler; nothing is sent."
        : "Candidate is ephemeral; the authenticated receiver exposes no assertion route, so nothing is sent.";

      if (nextSnapshot?.ambient?.enabled === false || nextSnapshot?.ambient?.site_enabled === false) {
        stop();
        return;
      }
      shell.hidden = false;
    }

    async function refresh() {
      if (stopped) return null;
      try {
        const response = await runtime.sendMessage({ type: "polylogue.missionControl.status", refresh: true });
        if (response?.ok) render(response);
        return response;
      } catch (error) {
        render({
          ok: false,
          state: { online: false, error: String(error.message || error) },
          receiver: { health: { status: "unreachable", detail: String(error.message || error) } },
          work: {},
          timeline: [],
          assertions: { persistence_supported: false },
          ambient: { enabled: true, site_enabled: true },
        });
        return null;
      }
    }

    function openPanel() {
      panel.hidden = false;
      chip.setAttribute("aria-expanded", "true");
      close.focus();
    }

    function closePanel() {
      panel.hidden = true;
      chip.setAttribute("aria-expanded", "false");
      chip.focus();
    }

    function updateSelection() {
      selectionCandidate = deriveSelectionCandidate(selectionSource.getSelection?.(), {
        url: locationSource?.href || "",
      });
      if (!selectionCandidate) {
        selectionText.textContent = "Select text inside a conversation message to prepare an assertion candidate.";
        return;
      }
      selectionText.textContent = selectionCandidate.text;
      if (selectionCandidate.truncated) selectionText.textContent += "…";
    }

    async function hideOnSite() {
      const hostname = providerForUrl(locationSource?.href || "") ? new URL(locationSource.href).hostname : "";
      await runtime.sendMessage({
        type: "polylogue.ambient.configure",
        hostname,
        site_enabled: false,
      });
      stop();
    }

    function stop() {
      if (stopped) return;
      stopped = true;
      if (timer) root.clearInterval(timer);
      doc.removeEventListener("selectionchange", updateSelection);
      host.remove();
      if (root.polylogueAmbientSurfaceMounted === api) root.polylogueAmbientSurfaceMounted = null;
    }

    chip.addEventListener("click", () => panel.hidden ? openPanel() : closePanel());
    close.addEventListener("click", closePanel);
    refreshButton.addEventListener("click", () => { void refresh(); });
    hideButton.addEventListener("click", () => { void hideOnSite(); });
    shadow.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && !panel.hidden) {
        event.preventDefault();
        closePanel();
      }
    });
    doc.addEventListener("selectionchange", updateSelection);
    timer = root.setInterval(() => { void refresh(); }, REFRESH_INTERVAL_MS);
    timer.unref?.();
    void refresh();

    const api = {
      host,
      shadow,
      refresh,
      render,
      stop,
      open: openPanel,
      close: closePanel,
      getSnapshot: () => snapshot,
      getSelectionCandidate: () => selectionCandidate,
    };
    root.polylogueAmbientSurfaceMounted = api;
    return api;
  }

  root.PolylogueAmbientSurface = Object.freeze({
    deriveSelectionCandidate,
    mount,
    providerForUrl,
  });

  if (root.top === root.self && !root.polylogueAmbientSurfaceMounted) {
    const start = () => mount();
    if (root.document?.documentElement) start();
    else root.document?.addEventListener?.("DOMContentLoaded", start, { once: true });
  }
})(globalThis);
