import { JSDOM } from "jsdom";
import { afterEach, describe, expect, it, vi } from "vitest";

import {
  classifyBrowserActionFailure,
  executeChatGptBrowserActionInPage,
} from "../src/actions/chatgpt.js";

function installPage(url = "https://chatgpt.com/") {
  const dom = new JSDOM(`<!doctype html><body>
    <button aria-pressed="true">Chat</button>
    <button aria-pressed="false">Work</button>
    <button class="__composer-pill">Pro</button>
    <div role="menuitem" data-has-submenu>GPT-5.6 Sol</div>
    <div role="menuitemradio" aria-checked="true">GPT-5.6 Sol</div>
    <div role="menuitemradio" aria-checked="true">Pro</div>
    <div id="prompt-textarea"></div>
    <button class="composer-submit-button-color">Send</button>
  </body>`, { url });
  globalThis.window = dom.window;
  globalThis.document = dom.window.document;
  globalThis.location = dom.window.location;
  globalThis.Event = dom.window.Event;
  globalThis.InputEvent = dom.window.InputEvent;
  globalThis.MouseEvent = dom.window.MouseEvent;
  document.execCommand = vi.fn((_command, _ui, value) => {
    document.querySelector("#prompt-textarea").textContent = value;
    return true;
  });
  return dom;
}

function action(patch = {}) {
  return {
    action_id: "action-1",
    receiver_id: "rx-browser-action-test",
    provider: "chatgpt",
    operation: "conversation.create",
    target: { conversation_id: "new", conversation_url: null, project_ref: null },
    text: "Implement the targeted change and report substantive output.",
    presentation: {
      surface: "chat",
      model_slug: "gpt-5-6-pro",
      model_label: "GPT-5.6 Sol",
      effort_label: "Pro",
    },
    submit_policy: "stage_only",
    ...patch,
  };
}

function responseJson(body, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    headers: { get: vi.fn(() => null) },
    json: vi.fn(async () => body),
  };
}

describe("ChatGPT browser action adapter", () => {
  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
    for (const name of ["window", "document", "location", "Event", "InputEvent", "MouseEvent", "fetch"]) {
      delete globalThis[name];
    }
  });

  it("classifies rate, safety, auth, capability, and drift outcomes", () => {
    expect(classifyBrowserActionFailure(
      new Error("temporarily limited access to your conversations to protect your data"),
    )).toMatchObject({ outcome: "provider_warning" });
    expect(classifyBrowserActionFailure(new Error("HTTP 429 too many requests"), 45)).toMatchObject({
      outcome: "rate_limited",
      retry_after_seconds: 45,
    });
    expect(classifyBrowserActionFailure(new Error("access to conversations temporarily blocked for safety")))
      .toMatchObject({ outcome: "safety_locked" });
    expect(classifyBrowserActionFailure(new Error("Cloudflare challenge forbidden")))
      .toMatchObject({ outcome: "auth_challenge" });
    expect(classifyBrowserActionFailure(new Error("model mismatch before submit")))
      .toMatchObject({ outcome: "capability_mismatch" });
    expect(classifyBrowserActionFailure(new Error("composer selector disappeared")))
      .toMatchObject({ outcome: "provider_drift" });
  });

  it("stages a verified Chat / GPT-5.6 Sol / Pro draft without submitting", async () => {
    installPage();
    const tool = document.createElement("span");
    tool.dataset.inlineSelectionPill = "";
    tool.textContent = "Create image";
    document.querySelector("#prompt-textarea").appendChild(tool);
    const result = await executeChatGptBrowserActionInPage(action(), []);

    expect(result).toMatchObject({
      ok: true,
      outcome: "drafted",
      provider_turn_id: null,
      observed_surface: "Chat",
      observed_model: "GPT-5.6 Sol",
      observed_effort: "Pro",
      provider_evidence: { composer_verified: true },
    });
    expect(document.querySelector("#prompt-textarea [data-inline-selection-pill]")).toBeNull();
  });

  it("selects Chat from the provider radio controls before composing", async () => {
    installPage();
    const [chat, work] = [...document.querySelectorAll("button")]
      .filter((button) => ["Chat", "Work"].includes(button.textContent));
    chat.removeAttribute("aria-pressed");
    chat.setAttribute("role", "radio");
    chat.setAttribute("aria-checked", "false");
    chat.setAttribute("data-state", "off");
    work.removeAttribute("aria-pressed");
    work.setAttribute("role", "radio");
    work.setAttribute("aria-checked", "true");
    work.setAttribute("data-state", "on");
    chat.addEventListener("click", () => {
      chat.setAttribute("aria-checked", "true");
      chat.setAttribute("data-state", "on");
      work.setAttribute("aria-checked", "false");
      work.setAttribute("data-state", "off");
    });

    const result = await executeChatGptBrowserActionInPage(action(), []);

    expect(result).toMatchObject({ ok: true, outcome: "drafted", observed_surface: "Chat" });
    expect(chat.getAttribute("aria-checked")).toBe("true");
    expect(work.getAttribute("aria-checked")).toBe("false");
  });

  it("returns the exact new user-turn receipt for a reply", async () => {
    installPage("https://chatgpt.com/c/conversation-1");
    let reads = 0;
    globalThis.fetch = vi.fn(async () => {
      reads += 1;
      const mapping = {
        old: {
          message: {
            id: "old-user-turn",
            author: { role: "user" },
            content: { parts: ["Earlier turn"] },
          },
        },
      };
      if (reads > 1) {
        mapping.submitted = {
          message: {
            id: "new-user-turn",
            author: { role: "user" },
            content: { parts: ["Implement the targeted change and report substantive output."] },
            status: "finished_successfully",
          },
        };
      }
      return responseJson({
        mapping,
        current_node: "assistant-running",
        update_time: 1784160000,
        gizmo_id: null,
      });
    });

    const result = await executeChatGptBrowserActionInPage(action({
      operation: "conversation.reply",
      target: { conversation_id: "conversation-1", conversation_url: null, project_ref: null },
      submit_policy: "submit_once",
    }), []);

    expect(result).toMatchObject({
      ok: true,
      outcome: "submitted",
      provider_conversation_id: "conversation-1",
      provider_turn_id: "new-user-turn",
      provider_evidence: {
        current_node: "assistant-running",
        user_turn_status: "finished_successfully",
      },
    });
    expect(reads).toBeGreaterThanOrEqual(2);
  });

  it("uses an exact DOM message id when authenticated conversation reads are inaccessible", async () => {
    installPage("https://chatgpt.com/c/conversation-1");
    globalThis.fetch = vi.fn(async () => responseJson({
      detail: { code: "conversation_inaccessible", can_retry: false },
    }, 404));
    document.querySelector("button.composer-submit-button-color").addEventListener("click", () => {
      const message = document.createElement("div");
      message.dataset.messageAuthorRole = "user";
      message.dataset.messageId = "new-dom-user-turn";
      message.textContent = "Implement the targeted change and report substantive output.";
      document.body.appendChild(message);
    });

    const result = await executeChatGptBrowserActionInPage(action({
      operation: "conversation.reply",
      target: { conversation_id: "conversation-1", conversation_url: null, project_ref: null },
      submit_policy: "submit_once",
    }), []);

    expect(result).toMatchObject({
      ok: true,
      provider_turn_id: "new-dom-user-turn",
      provider_evidence: {
        receipt_source: "provider_dom",
        conversation_read_status: 404,
      },
    });
  });

  it("fails closed before submit when the requested project does not match", async () => {
    installPage("https://chatgpt.com/c/conversation-1");
    const result = await executeChatGptBrowserActionInPage(action({
      operation: "conversation.reply",
      target: {
        conversation_id: "conversation-1",
        conversation_url: null,
        project_ref: "g-p-required",
      },
      submit_policy: "submit_once",
    }), []);
    expect(result).toMatchObject({
      ok: false,
      detail: "project mismatch before compose",
      submission_may_have_occurred: false,
    });
  });

  it("survives MAIN-world serialization without module bindings", async () => {
    installPage();
    const serialized = (0, eval)(`(${executeChatGptBrowserActionInPage.toString()})`);
    await expect(serialized(action({ provider: "claude" }), [])).resolves.toMatchObject({
      ok: false,
      detail: "unsupported provider or surface target mismatch",
      submission_may_have_occurred: false,
    });
  });
});
