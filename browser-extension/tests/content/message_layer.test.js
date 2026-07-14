/**
 * Tests for src/content/message_layer.js (polylogue-ys30: in-page Layer 1 —
 * blended per-message capture-status dot + save action).
 *
 * Unlike the chatgpt.test.js/claude.test.js convention of duplicating pure
 * DOM-detection helpers inline, this suite evaluates the real production
 * file inside a JSDOM window (the same pattern used by
 * tests/content/grok.test.js and tests/content/chatgpt_bridge.test.js), so a
 * regression in the shipped module fails these tests directly.
 */

import { readFileSync } from "node:fs";

import { JSDOM } from "jsdom";
import { afterEach, describe, expect, it } from "vitest";

const moduleSource = readFileSync("src/content/message_layer.js", "utf8");
const openDoms = [];

function freshDom(html = "<!DOCTYPE html><html><body></body></html>") {
  const dom = new JSDOM(html, { runScripts: "outside-only" });
  openDoms.push(dom);
  dom.window.eval(moduleSource);
  return dom;
}

afterEach(() => {
  while (openDoms.length) openDoms.pop().window.close();
});

describe("deriveStateMap (pure)", () => {
  it("defaults every index to not-seen with no prior state", () => {
    const dom = freshDom();
    const map = dom.window.polylogueMessageLayer.deriveStateMap({ nodeCount: 3 });
    expect(map).toEqual({ 0: "not-seen", 1: "not-seen", 2: "not-seen" });
  });

  it("preserves a known-valid prior state and normalizes an invalid one to unknown", () => {
    const dom = freshDom();
    const map = dom.window.polylogueMessageLayer.deriveStateMap({
      nodeCount: 2,
      priorMap: { 0: "captured", 1: "not-a-real-state" },
    });
    expect(map).toEqual({ 0: "captured", 1: "unknown" });
  });

  it("marks every index pending regardless of prior state", () => {
    const dom = freshDom();
    const map = dom.window.polylogueMessageLayer.deriveStateMap({
      nodeCount: 2,
      priorMap: { 0: "captured", 1: "failed" },
      pending: true,
    });
    expect(map).toEqual({ 0: "pending", 1: "pending" });
  });

  it("marks every index captured when the capture turn count matches the DOM node count", () => {
    const dom = freshDom();
    const map = dom.window.polylogueMessageLayer.deriveStateMap({
      nodeCount: 3,
      capture: { ok: true, turnCount: 3 },
    });
    expect(map).toEqual({ 0: "captured", 1: "captured", 2: "captured" });
  });

  it("falls back to unknown (never a false captured claim) when turn/node counts disagree", () => {
    const dom = freshDom();
    const map = dom.window.polylogueMessageLayer.deriveStateMap({
      nodeCount: 3,
      capture: { ok: true, turnCount: 2 },
    });
    expect(map).toEqual({ 0: "unknown", 1: "unknown", 2: "unknown" });
  });

  it("marks every index failed on an unsuccessful capture", () => {
    const dom = freshDom();
    const map = dom.window.polylogueMessageLayer.deriveStateMap({
      nodeCount: 2,
      capture: { ok: false, turnCount: null },
    });
    expect(map).toEqual({ 0: "failed", 1: "failed" });
  });
});

describe("mount() DOM behavior", () => {
  it("mounts one shadow-DOM badge per matching container with zero host-node duplication", () => {
    const dom = freshDom(
      '<!DOCTYPE html><html><body>' +
        '<article data-message-author-role="user">Hi</article>' +
        '<article data-message-author-role="assistant">Hello</article>' +
        "</body></html>",
    );
    const { document } = dom.window;
    const containers = [...document.querySelectorAll("article")];
    expect(containers).toHaveLength(2);

    const handle = dom.window.polylogueMessageLayer.mount({
      containerSelector: "article",
      onSave: () => {},
      doc: document,
    });

    expect(handle.debugMountedCount()).toBe(2);
    for (const container of containers) {
      const badges = [...container.children].filter((child) => child.shadowRoot);
      expect(badges).toHaveLength(1);
      const button = badges[0].shadowRoot.querySelector("button");
      expect(button.getAttribute("role")).toBe("button");
      expect(button.getAttribute("aria-label")).toBe("Save to Polylogue");
      expect(button.tabIndex).toBe(0);
    }
    handle.stop();
  });

  it("never removes or rewrites existing host content — only appends a badge", () => {
    const dom = freshDom(
      '<!DOCTYPE html><html><body>' +
        '<article data-message-author-role="user"><button class="native-action">Copy</button>Hi</article>' +
        "</body></html>",
    );
    const { document } = dom.window;
    const before = document.querySelector("article").innerHTML;

    const handle = dom.window.polylogueMessageLayer.mount({
      containerSelector: "article",
      onSave: () => {},
      doc: document,
    });

    const nativeButton = document.querySelector(".native-action");
    expect(nativeButton).not.toBeNull();
    expect(nativeButton.textContent).toBe("Copy");
    // The badge host is additive: original markup is still a prefix/subset of
    // the container's content, nothing native was deleted or rewritten.
    expect(document.querySelector("article").innerHTML.startsWith(before)).toBe(true);
    handle.stop();
  });

  it("sets pending on click and reflects a matching captured outcome afterwards", () => {
    const dom = freshDom(
      '<!DOCTYPE html><html><body>' +
        '<article data-message-author-role="user">Hi</article>' +
        "</body></html>",
    );
    const { document } = dom.window;
    let saveCalls = 0;
    const handle = dom.window.polylogueMessageLayer.mount({
      containerSelector: "article",
      onSave: () => {
        saveCalls += 1;
      },
      doc: document,
    });

    const container = document.querySelector("article");
    const badgeHost = [...container.children].find((child) => child.shadowRoot);
    const button = badgeHost.shadowRoot.querySelector("button");
    button.dispatchEvent(new dom.window.MouseEvent("click", { bubbles: true, cancelable: true }));

    expect(saveCalls).toBe(1);
    expect(handle.isPending()).toBe(true);
    expect(badgeHost.getAttribute("data-polylogue-state")).toBe("pending");

    handle.reportOutcome({ ok: true, turnCount: 1 });
    expect(handle.isPending()).toBe(false);
    const badgeAfter = [...document.querySelector("article").children].find((child) => child.shadowRoot);
    expect(badgeAfter.getAttribute("data-polylogue-state")).toBe("captured");
    expect(badgeAfter.shadowRoot.querySelector("button").getAttribute("aria-pressed")).toBe("true");
    handle.stop();
  });

  it("activates on Enter and Space keydown, not other keys", () => {
    const dom = freshDom('<!DOCTYPE html><html><body><article>Hi</article></body></html>');
    const { document } = dom.window;
    let saveCalls = 0;
    const handle = dom.window.polylogueMessageLayer.mount({
      containerSelector: "article",
      onSave: () => {
        saveCalls += 1;
      },
      doc: document,
    });
    const badgeHost = [...document.querySelector("article").children].find((child) => child.shadowRoot);
    const button = badgeHost.shadowRoot.querySelector("button");

    button.dispatchEvent(new dom.window.KeyboardEvent("keydown", { key: "Tab", bubbles: true, cancelable: true }));
    expect(saveCalls).toBe(0);

    button.dispatchEvent(new dom.window.KeyboardEvent("keydown", { key: "Enter", bubbles: true, cancelable: true }));
    expect(saveCalls).toBe(1);

    button.dispatchEvent(new dom.window.KeyboardEvent("keydown", { key: " ", bubbles: true, cancelable: true }));
    expect(saveCalls).toBe(2);
    handle.stop();
  });

  it("marks failed state on an unsuccessful capture without breaking the badge", () => {
    const dom = freshDom('<!DOCTYPE html><html><body><article>Hi</article></body></html>');
    const { document } = dom.window;
    const handle = dom.window.polylogueMessageLayer.mount({
      containerSelector: "article",
      onSave: () => {},
      doc: document,
    });
    handle.reportOutcome({ ok: false, turnCount: null });
    const badgeHost = [...document.querySelector("article").children].find((child) => child.shadowRoot);
    expect(badgeHost.getAttribute("data-polylogue-state")).toBe("failed");
    expect(badgeHost.shadowRoot.querySelector("button").getAttribute("aria-label")).toMatch(/failed/i);
    handle.stop();
  });

  it("tracks DOM churn: mounts a badge for a node added after mount and drops it for a removed node", async () => {
    const dom = freshDom('<!DOCTYPE html><html><body><div id="root"><article>Hi</article></div></body></html>');
    const { document } = dom.window;
    const handle = dom.window.polylogueMessageLayer.mount({
      containerSelector: "article",
      onSave: () => {},
      doc: document,
      root: document.getElementById("root"),
    });
    expect(handle.debugMountedCount()).toBe(1);

    const second = document.createElement("article");
    second.textContent = "New message";
    document.getElementById("root").appendChild(second);

    await new Promise((resolve) => dom.window.setTimeout(resolve, 0));
    handle.reconcile();
    expect(handle.debugMountedCount()).toBe(2);

    document.querySelectorAll("article")[0].remove();
    handle.reconcile();
    expect(handle.debugMountedCount()).toBe(1);
    handle.stop();
  });

  it("fails closed on an invalid selector: mounts nothing and never throws", () => {
    const dom = freshDom('<!DOCTYPE html><html><body><article>Hi</article></body></html>');
    const { document } = dom.window;
    expect(() => {
      const handle = dom.window.polylogueMessageLayer.mount({
        containerSelector: "article[[[not-a-valid-selector",
        onSave: () => {},
        doc: document,
      });
      expect(handle.debugMountedCount()).toBe(0);
      handle.stop();
    }).not.toThrow();
  });

  it("stop() removes every mounted badge and disconnects the observer", () => {
    const dom = freshDom(
      '<!DOCTYPE html><html><body><article>A</article><article>B</article></body></html>',
    );
    const { document } = dom.window;
    const handle = dom.window.polylogueMessageLayer.mount({
      containerSelector: "article",
      onSave: () => {},
      doc: document,
    });
    expect(handle.debugMountedCount()).toBe(2);
    handle.stop();
    for (const container of document.querySelectorAll("article")) {
      const badges = [...container.children].filter((child) => child.shadowRoot);
      expect(badges).toHaveLength(0);
    }
  });

  it("establishes a positioning context only when the container has none", () => {
    const dom = freshDom(
      '<!DOCTYPE html><html><body>' +
        '<article style="position:absolute">A</article>' +
        "<article>B</article>" +
        "</body></html>",
    );
    const { document } = dom.window;
    const handle = dom.window.polylogueMessageLayer.mount({
      containerSelector: "article",
      onSave: () => {},
      doc: document,
    });
    const [positioned, unpositioned] = document.querySelectorAll("article");
    // Pre-existing positioning is never overwritten.
    expect(positioned.style.position).toBe("absolute");
    // A static container gets a non-destructive positioning context so the
    // badge can be placed without shifting sibling layout.
    expect(unpositioned.style.position).toBe("relative");
    handle.stop();
  });
});
