import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { JSDOM } from "jsdom";
import { until } from "./support/fake_chrome.js";

const SRC = join(dirname(fileURLToPath(import.meta.url)), "../src");
const source = (path) => readFileSync(join(SRC, path), "utf8");

const state = (changes = {}) => ({
  status: { host: "ok", daemon: "running", chat_actions: true },
  page: { provider: "chatgpt", conversation: "c1", host: "ok", state: "archived", messages: 1, daemon: "running" },
  waiting: 0, approvals: [], importJob: [], problems: [],
  settings: { automatic: true, panel: true }, version: "1.2.3",
  ...changes,
});

/** The popup page with the background's answers given by `answer(message)`. */
async function open(answer) {
  const dom = new JSDOM(source("popup.html"), { url: "chrome-extension://x/popup.html", runScripts: "outside-only" });
  const { window } = dom;
  const sent = [];
  window.chrome = {
    runtime: { id: "x", sendMessage: async (message) => { sent.push(message); return answer(message); } },
    tabs: { query: async () => [{ id: 7 }] },
  };
  window.eval(source("capture_state.js"));
  window.eval(source("wording.js"));
  window.eval(source("popup.js"));
  await until(() => window.document.getElementById("version").textContent !== "", "the first answer");
  const text = (id) => window.document.getElementById(id).textContent;
  return { window, sent, text };
}

describe("the popup", () => {
  it("says what the archive holds of the page and how many captures wait", async () => {
    const one = await open(() => state({ waiting: 1 }));
    expect(one.text("page-state")).toBe("In the archive: 1 message");
    expect(one.text("waiting")).toBe("1 capture waiting to be stored.");
    expect(one.window.document.getElementById("waiting").hidden).toBe(false);
    const many = await open(() => state({ waiting: 3 }));
    expect(many.text("waiting")).toBe("3 captures waiting to be stored.");
    const none = await open(() => state());
    expect(none.window.document.getElementById("waiting").hidden).toBe(true);
  });

  it("names the daemon's state", async () => {
    expect((await open(() => state())).text("polylogue-state")).toBe("Running");
    const stopped = await open(() => state({ status: { host: "ok", daemon: "stopped" } }));
    expect(stopped.text("polylogue-state")).toBe("The daemon did not answer");
    const missing = await open(() => state({ status: { host: "missing", daemon: "unknown" } }));
    expect(missing.text("polylogue-detail")).toBe("Run: polylogue install chrome --extension-id x");
  });

  it("lists five problems and counts the rest", async () => {
    const problems = Array.from({ length: 8 }, (_, i) => ({ at: new Date().toISOString(), text: `problem ${i}` }));
    const { window } = await open(() => state({ problems }));
    const rows = [...window.document.querySelectorAll("#problems li")].map((li) => li.firstChild.textContent);
    expect(rows).toEqual(["problem 0", "problem 1", "problem 2", "problem 3", "problem 4", "3 older not shown"]);
  });

  it("extension_popup_decision_uses_answer: shows a held turn in full and sends the operator's answer", async () => {
    const text = "word ".repeat(100).trim();
    const approvals = [{ id: "a1", provider: "chatgpt", conversation: "c1", text }];
    const { window, sent } = await open((message) => (message.type === "polylogue.popup" ? state({ approvals }) : { ok: true }));
    const row = window.document.querySelector("#approvals li");
    expect(row.querySelector("p.meta").textContent).toBe(text);
    expect(row.querySelector("span").textContent).toBe("ChatGPT: send into conversation c1");
    const [allow, deny] = row.querySelectorAll("button");
    expect([allow.textContent, deny.textContent]).toEqual(["Allow", "Deny"]);
    deny.click();
    await until(() => sent.some((m) => m.type === "polylogue.answer"), "the answer");
    expect(sent.find((m) => m.type === "polylogue.answer")).toEqual({ type: "polylogue.answer", id: "a1", value: "deny" });
    allow.click();
    await until(() => sent.filter((m) => m.type === "polylogue.answer").length === 2, "the second answer");
    expect(sent.filter((m) => m.type === "polylogue.answer")[1]).toMatchObject({ value: "allow" });
  });

  it("keeps Capture available and says why when the capture request fails", async () => {
    const { window, text } = await open((message) => {
      if (message.type === "polylogue.capture") throw new Error("Extension context invalidated.");
      return state();
    });
    window.document.getElementById("capture").click();
    await until(() => text("page-state") === "The extension could not check this page", "the failure");
    expect(text("page-detail")).toBe("Extension context invalidated.");
    expect(window.document.getElementById("capture").disabled).toBe(false);
  });
});
