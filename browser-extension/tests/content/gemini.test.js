import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { Script } from "node:vm";
import { JSDOM } from "jsdom";
import { describe, expect, it } from "vitest";

const dir = dirname(fileURLToPath(import.meta.url));
const source = readFileSync(resolve(dir, "../../src/content/gemini.js"), "utf8");

const DEFAULT_BODY = `<user-query data-message-id="u1"><message-content>Hello</message-content></user-query><model-response data-message-id="a1"><message-content>Hi there</message-content></model-response>`;

function harness(body = DEFAULT_BODY) {
  const dom = new JSDOM(`<!doctype html><title>Gemini fixture</title>${body}`, {
    url: "https://gemini.google.com/app/fixture-chat",
    runScripts: "outside-only",
  });
  const messages = [];
  const listeners = [];
  const chrome = {
    runtime: {
      id: "fixture-extension",
      getManifest: () => ({ version: "0.1.0" }),
      onMessage: { addListener: (listener) => listeners.push(listener) },
      sendMessage: async (message) => {
        messages.push(message);
        if (message.type === "polylogue.capture") return { ok: true };
        if (message.type === "polylogue.archiveState") return { captured: true, state: "archived" };
        return { ok: true };
      },
    },
  };
  Object.defineProperty(dom.window, "chrome", { value: chrome });
  new Script(readFileSync(resolve(dir, "../../src/common.js"), "utf8")).runInContext(dom.getInternalVMContext());
  new Script(source).runInContext(dom.getInternalVMContext());
  return { dom, messages, listener: listeners[0] };
}

describe("Gemini DOM capture contract", () => {
  it("captures provider turn elements in document order with stable ids", async () => {
    const { dom, messages } = harness();
    const result = await dom.window.polylogueCapture.capturePage("fixture");
    const capture = messages.find((message) => message.type === "polylogue.capture").envelope;
    expect(result.ok).toBe(true);
    expect(capture.session.provider).toBe("gemini");
    expect(capture.session.provider_session_id).toBe("fixture-chat");
    expect(capture.session.turns.map((turn) => [turn.provider_turn_id, turn.role, turn.text])).toEqual([
      ["u1", "user", "Hello"],
      ["a1", "assistant", "Hi there"],
    ]);
    dom.window.close();
  });

  it("anti-vacuity: an inserted earlier turn does not move a fallback id onto other content", async () => {
    // Positional fallbacks (gemini-dom-<ordinal>) would rename "Hello" from index 0 to index 1 here.
    const { dom, messages } = harness(`<user-query><message-content>Hello</message-content></user-query><model-response><message-content>Hi there</message-content></model-response>`);
    await dom.window.polylogueCapture.capturePage("fixture");
    const idsBefore = Object.fromEntries(messages.filter((m) => m.type === "polylogue.capture").at(-1).envelope.session.turns.map((turn) => [turn.text, turn.provider_turn_id]));
    const earlier = dom.window.document.createElement("user-query");
    earlier.innerHTML = "<message-content>Earlier</message-content>";
    dom.window.document.body.prepend(earlier);
    await dom.window.polylogueCapture.capturePage("fixture");
    const idsAfter = Object.fromEntries(messages.filter((m) => m.type === "polylogue.capture").at(-1).envelope.session.turns.map((turn) => [turn.text, turn.provider_turn_id]));
    expect(idsAfter.Hello).toBe(idsBefore.Hello);
    expect(idsAfter["Hi there"]).toBe(idsBefore["Hi there"]);
    expect(new Set(Object.values(idsAfter)).size).toBe(3);
    dom.window.close();
  });

  it("anti-vacuity: emits capture health when the provider has no readable turns", async () => {
    const { dom, messages } = harness();
    dom.window.document.querySelector("user-query").remove();
    dom.window.document.querySelector("model-response").remove();
    await expect(dom.window.polylogueCapture.capturePage()).resolves.toMatchObject({ ok: false, error: "no_turns" });
    expect(messages).toContainEqual(expect.objectContaining({
      type: "polylogue.captureHealth", event: "capture_error", provider: "gemini", reason: "no_turns",
    }));
    dom.window.close();
  });

  it("reports a capture gap when a visible provider turn has no readable text", async () => {
    const { dom, messages } = harness();
    const blank = dom.window.document.createElement("model-response");
    blank.setAttribute("data-message-id", "a2");
    dom.window.document.body.append(blank);
    await dom.window.polylogueCapture.capturePage("fixture");
    expect(messages.find((message) => message.type === "polylogue.captureHealth")).toMatchObject({
      event: "capture_gap",
      provider: "gemini",
      visible_count: 3,
      captured_count: 2,
    });
    dom.window.close();
  });

  it("anti-vacuity: keeps the tab title marked as page provenance", async () => {
    const { dom, messages } = harness();
    await dom.window.polylogueCapture.capturePage("fixture");
    const capture = messages.find((message) => message.type === "polylogue.capture").envelope;
    expect(capture.session.title_source).toBe("page");
    dom.window.close();
  });

  it("anti-vacuity: a changed turn sends a policy-routed freshness hint, not a direct capture", async () => {
    const { dom, messages } = harness();
    await dom.window.polylogueCapture.capturePage("initial");
    const response = dom.window.document.querySelector("model-response message-content");
    response.textContent = "Updated answer";
    await new Promise((resolve) => setTimeout(resolve, 600));
    // Capturing directly here would bypass the automatic-capture opt-out.
    expect(messages.filter((message) => message.type === "polylogue.capture")).toHaveLength(1);
    const hints = messages.filter((message) => message.type === "polylogue.captureFreshnessHint");
    expect(hints).toHaveLength(1);
    expect(hints[0]).toMatchObject({ provider: "gemini", reason: "gemini_dom_changed" });
    dom.window.close();
  });
});
