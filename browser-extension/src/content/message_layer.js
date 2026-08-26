(function () {
  // In-page Layer 1: a blended per-message capture-status dot + save action.
  //
  // Boundary rule (polylogue-yyvg): per-message state blends into the host's
  // existing message row (this module); cross-conversation intelligence is a
  // separate floating surface (Layer 2, not this module). This module never
  // removes or rewrites host DOM/content — it only appends an isolated
  // Shadow DOM badge next to each detected message container and, at most,
  // sets `position: relative` on a message container that has no existing
  // positioning context (a single non-destructive style property, never
  // touching host classes, text, or listeners).
  //
  // The archive's capture unit is a whole session, not a single message —
  // there is no per-message receiver endpoint. "Save" therefore triggers the
  // same whole-session capture the popup/auto-capture path already uses, and
  // this module reflects that outcome onto every currently-mounted badge.
  // Per-message identity comes from the provider adapter. DOM order and text
  // are hints only and can never authorize a captured state.

  const STATE_ATTR = "data-polylogue-state";

  const STATE_COLORS = {
    captured: "#14764e",
    pending: "#9a5b00",
    failed: "#ad2f2f",
    unknown: "#6b7280",
    "not-seen": "#9ca3af",
  };

  const STATE_LABELS = {
    captured: "Saved to Polylogue",
    pending: "Saving to Polylogue…",
    failed: "Save to Polylogue failed — click to retry",
    unknown: "Polylogue capture status unknown",
    "not-seen": "Save to Polylogue",
  };

  const VALID_STATES = new Set(Object.keys(STATE_COLORS));

  function normalizeState(state) {
    return VALID_STATES.has(state) ? state : "unknown";
  }

  // Pure state-derivation, exported for direct unit testing. Keys are stable
  // provider/canonical identity keys, never DOM ordinals.
  function deriveStateMap({ identityKeys = [], priorMap = {}, capture = null, pending = false }) {
    const next = {};
    for (const key of identityKeys) {
      next[key] = normalizeState(priorMap[key] ?? "not-seen");
    }
    if (pending) {
      for (const key of identityKeys) next[key] = "pending";
      return next;
    }
    if (capture) {
      const ok = Boolean(capture.ok);
      if (!ok) {
        for (const key of identityKeys) next[key] = "failed";
        return next;
      }
      for (const key of identityKeys) next[key] = "unknown";
    }
    return next;
  }

  function canonicalMessageRef(observation) {
    if (!observation?.origin || !observation.provider_conversation_id || !observation.provider_message_id) return null;
    return `${observation.origin}:${observation.provider_conversation_id}:n:${observation.provider_message_id}`;
  }

  function acceptedMessageRefs(acceptedIdentities, ok) {
    if (!ok || !Array.isArray(acceptedIdentities)) return new Map();
    return new Map(acceptedIdentities
      .filter((item) => item?.fidelity === "native" && typeof item.message_ref === "string")
      .map((item) => [item.message_ref, item]));
  }

  function ensurePositioned(node) {
    try {
      const view = node.ownerDocument && node.ownerDocument.defaultView;
      const computed = view && view.getComputedStyle ? view.getComputedStyle(node).position : node.style.position;
      if (!computed || computed === "static") node.style.position = "relative";
    } catch {
      // Reading computed style can throw on a detached/foreign node; leaving
      // positioning untouched is the safe (native-control-preserving) choice.
    }
  }

  function buildBadge({ state, onActivate, doc }) {
    const host = doc.createElement("span");
    host.style.cssText =
      "all:initial;position:absolute;top:4px;right:4px;width:28px;height:28px;" +
      "pointer-events:auto;z-index:2147483000;display:block;";
    const shadow = host.attachShadow({ mode: "open" });
    const style = doc.createElement("style");
    style.textContent = [
      ":host{all:initial;}",
      "button{all:unset;display:inline-flex;align-items:center;justify-content:center;",
      "width:28px;height:28px;border-radius:999px;cursor:pointer;box-sizing:border-box;}",
      "button:focus-visible{outline:2px solid #4b8bf5;outline-offset:1px;}",
      ".dot{width:9px;height:9px;border-radius:999px;background:var(--polylogue-dot-color,#9ca3af);",
      "box-shadow:0 0 0 1px rgba(0,0,0,0.25);}",
    ].join("");
    const button = doc.createElement("button");
    button.type = "button";
    button.setAttribute("role", "button");
    button.tabIndex = 0;
    const dot = doc.createElement("span");
    dot.className = "dot";
    button.appendChild(dot);
    shadow.appendChild(style);
    shadow.appendChild(button);

    function activate(event) {
      event.preventDefault();
      event.stopPropagation();
      onActivate();
    }
    button.addEventListener("click", activate);
    button.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") activate(event);
    });

    const api = {
      host,
      button,
      setState(nextState) {
        const normalized = normalizeState(nextState);
        dot.style.setProperty("--polylogue-dot-color", STATE_COLORS[normalized]);
        const sharedPresentation = window.PolylogueOperatorStatus?.captureStatePresentation?.(normalized);
        const label = sharedPresentation?.label || STATE_LABELS[normalized];
        button.setAttribute("aria-label", label);
        button.setAttribute("aria-pressed", normalized === "captured" ? "true" : "false");
        button.setAttribute("title", label);
        host.setAttribute(STATE_ATTR, normalized);
      },
    };
    api.setState(state);
    return api;
  }

  // Mount the per-message layer under `root` (default: the whole document
  // element, safe even before <body> exists at document_start). Returns a
  // handle with `reportPending()` / `reportOutcome()` for the owning content
  // script to call around its whole-session capture, plus `stop()` for
  // tests/teardown. Every DOM operation is wrapped so a selector/DOM surprise
  // never throws into the host page — worst case, no badges mount.
  function mount({ containerSelector, onSave, identityForNode = null, doc = document, root = null }) {
    const mountedRoot = root || doc.documentElement;
    const mounted = new Map(); // container node -> badge api
    let nodeOrder = [];
    let stateMap = {};
    let pending = false;

    function observationFor(node) {
      try {
        return identityForNode?.(node) || null;
      } catch {
        return null;
      }
    }

    function keyFor(node) {
      return canonicalMessageRef(observationFor(node));
    }

    function identityKeys() {
      return nodeOrder.map((node, index) => keyFor(node) || `unknown:${index}`);
    }

    function currentNodes() {
      try {
        return [...doc.querySelectorAll(containerSelector)];
      } catch {
        return [];
      }
    }

    function reconcile() {
      nodeOrder = currentNodes();
      for (const [node, badge] of [...mounted.entries()]) {
        if (!nodeOrder.includes(node)) {
          try {
            badge.host.remove();
          } catch {
            /* already detached */
          }
          mounted.delete(node);
        }
      }
      nodeOrder.forEach((node, index) => {
        let badge = mounted.get(node);
        try {
          if (!badge) {
            ensurePositioned(node);
            badge = buildBadge({
              state: stateMap[keyFor(node) || `unknown:${index}`] ?? "not-seen",
              onActivate: () => {
                pending = true;
                stateMap = deriveStateMap({ identityKeys: identityKeys(), priorMap: stateMap, pending: true });
                reconcile();
                onSave();
              },
              doc,
            });
            node.appendChild(badge.host);
            mounted.set(node, badge);
          }
          badge.setState(stateMap[keyFor(node) || `unknown:${index}`] ?? "not-seen");
        } catch {
          // Fail closed: an unsupported/foreign node never breaks the host
          // page or the rest of the reconciliation pass.
        }
      });
    }

    function reportPending() {
      pending = true;
      stateMap = deriveStateMap({ identityKeys: identityKeys(), priorMap: stateMap, pending: true });
      reconcile();
    }

    function reportOutcome({ ok, acceptedIdentities = [] }) {
      pending = false;
      if (!identityForNode) {
        stateMap = deriveStateMap({ identityKeys: identityKeys(), priorMap: stateMap, capture: { ok } });
        reconcile();
        return;
      }
      const accepted = acceptedMessageRefs(acceptedIdentities, ok);
      const keyCounts = new Map();
      nodeOrder.forEach((node) => {
        const key = keyFor(node);
        if (key) keyCounts.set(key, (keyCounts.get(key) || 0) + 1);
      });
      const next = {};
      nodeOrder.forEach((node, index) => {
        const key = keyFor(node) || `unknown:${index}`;
        const observation = observationFor(node);
        const acknowledgement = accepted.get(key);
        const adapterMatches = acknowledgement &&
          (!acknowledgement.adapter_version || !observation?.adapter_version ||
            acknowledgement.adapter_version === observation.adapter_version);
        next[key] = acknowledgement && keyCounts.get(key) === 1 && adapterMatches
          ? "captured" : (ok ? "unknown" : "failed");
      });
      stateMap = next;
      reconcile();
    }

    let observer = null;
    if (typeof MutationObserver !== "undefined") {
      observer = new MutationObserver(() => reconcile());
      try {
        observer.observe(mountedRoot, { childList: true, subtree: true });
      } catch {
        observer = null;
      }
    }
    reconcile();

    return {
      reconcile,
      reportPending,
      reportOutcome,
      stop() {
        if (observer) observer.disconnect();
        for (const badge of mounted.values()) {
          try {
            badge.host.remove();
          } catch {
            /* already detached */
          }
        }
        mounted.clear();
      },
      isPending: () => pending,
      debugState: () => ({ ...stateMap }),
      debugMountedCount: () => mounted.size,
    };
  }

  const existing = window.polylogueMessageLayer || {};
  window.polylogueMessageLayer = {
    ...existing,
    deriveStateMap,
    mount,
  };
})();
