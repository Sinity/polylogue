import { JSDOM } from "jsdom";
import { afterEach, describe, expect, it, vi } from "vitest";

import {
  classifyLaunchFailure,
  executeChatGptLaunchInPage,
  inspectChatGptLaunchPage,
} from "../src/launch/chatgpt_launch.js";

function installPage(body, url = "https://chatgpt.com/c/conversation-1") {
  const dom = new JSDOM(`<!doctype html><body>${body}</body>`, { url });
  globalThis.window = dom.window;
  globalThis.document = dom.window.document;
  globalThis.location = dom.window.location;
  globalThis.Event = dom.window.Event;
  globalThis.InputEvent = dom.window.InputEvent;
  globalThis.MouseEvent = dom.window.MouseEvent;
  return dom;
}

describe("ChatGPT Sol Pro launch adapter", () => {
  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
    delete globalThis.window;
    delete globalThis.document;
    delete globalThis.location;
    delete globalThis.Event;
    delete globalThis.InputEvent;
    delete globalThis.MouseEvent;
  });

  it("classifies provider throttles separately from safety and auth locks", () => {
    expect(classifyLaunchFailure(new Error("HTTP 429 too many requests"), 300)).toMatchObject({
      outcome: "rate_limited",
      retry_after_seconds: 300,
    });
    expect(classifyLaunchFailure(new Error("Access to conversations temporarily blocked for safety"))).toMatchObject({
      outcome: "safety_locked",
    });
    expect(classifyLaunchFailure(new Error("Cloudflare challenge forbidden"))).toMatchObject({
      outcome: "auth_challenge",
    });
  });

  it("fails before touching the page when a job is not ordinary Chat GPT-5.6 Sol Pro", async () => {
    installPage('<div id="prompt-textarea"></div>', "https://chatgpt.com/");
    await expect(executeChatGptLaunchInPage({
      mode: "work",
      model_slug: "gpt-5-6-pro",
      model_label: "GPT-5.6 Sol",
      effort_label: "Pro",
    }, [])).resolves.toMatchObject({
      ok: false,
      detail: "protocol_job_target_mismatch",
      submission_may_have_occurred: false,
    });
  });

  it("survives chrome.scripting MAIN-world serialization without module bindings", async () => {
    installPage('<div id="prompt-textarea"></div>', "https://chatgpt.com/");
    const serialized = (0, eval)(`(${executeChatGptLaunchInPage.toString()})`);
    await expect(serialized({
      mode: "work",
      model_slug: "gpt-5-6-pro",
      model_label: "GPT-5.6 Sol",
      effort_label: "Pro",
    }, [])).resolves.toMatchObject({
      ok: false,
      detail: "protocol_job_target_mismatch",
      submission_may_have_occurred: false,
    });
  });

  it("fails closed when the selected model disappears before submit", async () => {
    vi.useFakeTimers();
    installPage(`
      <button aria-pressed="true">Chat</button>
      <button aria-pressed="false">Work</button>
      <button class="__composer-pill">Pro</button>
      <div role="menuitem" data-has-submenu>GPT-5.6 Sol</div>
      <div role="menuitemradio" aria-checked="true">GPT-5.6 Sol</div>
      <div role="menuitemradio" aria-checked="true">Pro</div>
      <div id="prompt-textarea"></div>
      <button class="composer-submit-button-color">Send</button>
    `, "https://chatgpt.com/");
    document.execCommand = vi.fn((_command, _ui, value) => {
      document.querySelector("#prompt-textarea").textContent = value;
      return true;
    });
    const original = document.querySelectorAll.bind(document);
    let checkedReads = 0;
    vi.spyOn(document, "querySelectorAll").mockImplementation((selector) => {
      if (selector === '[role="menuitemradio"][aria-checked="true"]') {
        checkedReads += 1;
        if (checkedReads >= 3) {
          return [original(selector)[1]];
        }
      }
      return original(selector);
    });

    const launch = executeChatGptLaunchInPage({
      prompt: "Produce a durable result.",
      mode: "chat",
      model_slug: "gpt-5-6-pro",
      model_label: "GPT-5.6 Sol",
      effort_label: "Pro",
    }, []);
    await vi.advanceTimersByTimeAsync(5_100);
    await expect(launch).resolves.toMatchObject({
      ok: false,
      detail: "final_model_selection_timeout",
      submission_may_have_occurred: false,
    });
  });

  it("waits for both final selector radios before submitting", async () => {
    installPage(`
      <button aria-pressed="true">Chat</button>
      <button aria-pressed="false">Work</button>
      <button class="__composer-pill">Pro</button>
      <div role="menuitem" data-has-submenu>GPT-5.6 Sol</div>
      <div role="menuitemradio" aria-checked="true">GPT-5.6 Sol</div>
      <div role="menuitemradio" aria-checked="true">Pro</div>
      <div id="prompt-textarea"></div>
      <button class="composer-submit-button-color">Send</button>
    `, "https://chatgpt.com/");
    document.execCommand = vi.fn((_command, _ui, value) => {
      document.querySelector("#prompt-textarea").textContent = value.replaceAll("\n", "\n\n\n");
      return true;
    });
    document.querySelector(".composer-submit-button-color").addEventListener("click", () => {
      globalThis.window.history.pushState({}, "", "/c/conversation-new");
    });
    const original = document.querySelectorAll.bind(document);
    let checkedReads = 0;
    vi.spyOn(document, "querySelectorAll").mockImplementation((selector) => {
      if (selector === '[role="menuitemradio"][aria-checked="true"]') {
        checkedReads += 1;
        if (checkedReads === 3) return [original(selector)[0]];
      }
      return original(selector);
    });

    await expect(executeChatGptLaunchInPage({
      prompt: "# Mission: Produce a durable result.\n\nExplain the work directly and attach the handoff.",
      mode: "chat",
      model_slug: "gpt-5-6-pro",
      model_label: "GPT-5.6 Sol",
      effort_label: "Pro",
    }, [])).resolves.toMatchObject({
      ok: true,
      conversation_id: "conversation-new",
      preflight: { mode: "Chat", model: "GPT-5.6 Sol", effort: "Pro" },
    });
    expect(checkedReads).toBeGreaterThanOrEqual(4);
  });

  it("only recognizes a required ZIP produced in an assistant turn", () => {
    installPage(`
      <section data-testid="conversation-turn-1">
        <div data-message-author-role="user">
          Require polylogue-sol-pro-launch-handoff.zip in the answer.
        </div>
      </section>
      <section data-testid="conversation-turn-2"><div>Still working</div></section>
    `);
    expect(inspectChatGptLaunchPage()).toMatchObject({
      assistant_turns: 1,
      handoff_name: null,
    });

    document.querySelector('[data-testid="conversation-turn-2"]').insertAdjacentHTML(
      "beforeend",
      '<a href="https://chatgpt.com/backend-api/files/result">polylogue-sol-pro-launch-handoff.zip</a>',
    );
    expect(inspectChatGptLaunchPage()).toMatchObject({
      assistant_turns: 1,
      handoff_name: "polylogue-sol-pro-launch-handoff.zip",
      handoff_href: "https://chatgpt.com/backend-api/files/result",
    });
  });

  it("recognizes the job-specific readable handoff filename", () => {
    const filename = "polylogue-extension-mission-control-sol-next-extension-handoff.zip";
    installPage(`
      <section data-testid="conversation-turn-1">
        <a href="sandbox:/mnt/data/${filename}">${filename}</a>
      </section>
    `);

    expect(inspectChatGptLaunchPage(filename)).toMatchObject({
      assistant_turns: 1,
      handoff_name: filename,
    });
  });

  it("detects a provider safety lock while monitoring", () => {
    installPage('<section data-testid="conversation-turn-2">Access to conversations temporarily blocked</section>');
    expect(inspectChatGptLaunchPage()).toMatchObject({ safety_lock: true });
  });

  it("keeps ordinary rate limits distinct from provider safety locks", () => {
    installPage('<section data-testid="conversation-turn-2">Too many requests; rate limit reached</section>');
    expect(inspectChatGptLaunchPage()).toMatchObject({ soft_warning: false, rate_limited: true, safety_lock: false });
  });

  it("classifies the conversation-access anti-abuse banner as a soft warning", () => {
    installPage(`<section data-testid="conversation-turn-2">
      Too many requests. You’re making requests too quickly.
      We’ve temporarily limited access to your conversations to protect your data.
      Please wait a few minutes before trying again.
    </section>`);
    expect(inspectChatGptLaunchPage()).toMatchObject({
      soft_warning: true,
      rate_limited: false,
      safety_lock: false,
    });
  });

  it("treats the current Pro stop-answering control as a busy run", () => {
    installPage('<button data-testid="stop-button" aria-label="Stop answering"></button>');
    expect(inspectChatGptLaunchPage()).toMatchObject({ busy: true });
  });

});
