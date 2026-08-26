import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { Script } from "node:vm";
import { JSDOM } from "jsdom";
import { describe, expect, it } from "vitest";

const dir = dirname(fileURLToPath(import.meta.url));
const source = readFileSync(resolve(dir, "../../src/content/gemini.js"), "utf8");

function harness() {
  const dom = new JSDOM(`<!doctype html><title>Gemini fixture</title><user-query data-message-id="u1"><message-content>Hello</message-content></user-query><model-response data-message-id="a1"><message-content>Hi there</message-content></model-response>`, {
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

  it("fails loudly when the provider page has no supported turn boundary", async () => {
    const { dom } = harness();
    dom.window.document.querySelector("user-query").remove();
    dom.window.document.querySelector("model-response").remove();
    await expect(dom.window.polylogueCapture.capturePage()).resolves.toMatchObject({ ok: false, error: "no_turns" });
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
});
