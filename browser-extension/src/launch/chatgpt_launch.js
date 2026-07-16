export function classifyLaunchFailure(value, retryAfterSeconds = null) {
  const text = String(value?.message || value || "").toLowerCase();
  if (/too many requests|rate.?limit|http_?429/.test(text)) {
    return { outcome: "rate_limited", retry_after_seconds: retryAfterSeconds, detail: String(value) };
  }
  if (/safety|temporarily blocked|unusual activity|access to conversations/.test(text)) {
    return { outcome: "safety_locked", retry_after_seconds: retryAfterSeconds, detail: String(value) };
  }
  if (/challenge|captcha|cloudflare|unauthorized|forbidden|http_?401|http_?403|sign.?in/.test(text)) {
    return { outcome: "auth_challenge", retry_after_seconds: null, detail: String(value) };
  }
  if (/model|gpt-5\.6|sol|effort|work selected|chat selected|preflight|protocol|selector|handoff|manifest|checksum|archive|zip/.test(text)) {
    return { outcome: "protocol_mismatch", retry_after_seconds: null, detail: String(value) };
  }
  return { outcome: "network_error", retry_after_seconds: retryAfterSeconds, detail: String(value) };
}

export async function executeChatGptLaunchInPage(job, attachments) {
  // chrome.scripting serializes this function into MAIN world without module
  // lexical bindings. Keep every submit-boundary constant inside the body.
  const chatGptHost = "chatgpt.com";
  const requiredModel = "GPT-5.6 Sol";
  const requiredEffort = "Pro";
  const sleep = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));
  const textOf = (node) => String(node?.innerText || node?.textContent || "").trim();
  const normalizedText = (value) => String(value || "").replace(/\s+/g, " ").trim();
  const waitFor = async (predicate, timeoutMs, label) => {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
      const result = predicate();
      if (result) return result;
      await sleep(100);
    }
    throw new Error(`${label}_timeout`);
  };
  const pointerClick = (node) => {
    if (!node) throw new Error("protocol_selector_missing");
    const init = { bubbles: true, cancelable: true, composed: true, pointerId: 1, pointerType: "mouse", isPrimary: true };
    if (typeof PointerEvent === "function") {
      node.dispatchEvent(new PointerEvent("pointerdown", init));
      node.dispatchEvent(new PointerEvent("pointerup", init));
    }
    node.dispatchEvent(new MouseEvent("mousedown", init));
    node.dispatchEvent(new MouseEvent("mouseup", init));
    node.click();
  };
  const selectedMode = () => {
    const buttons = [...document.querySelectorAll("button")];
    const chat = buttons.find((button) => textOf(button) === "Chat");
    const work = buttons.find((button) => textOf(button) === "Work");
    const selected = (button) => button?.getAttribute("aria-pressed") === "true"
      || button?.getAttribute("data-state") === "active"
      || String(button?.className || "").includes("text-token-text-primary");
    return { chat: selected(chat), work: selected(work) };
  };
  const checkedModelOptions = () => [...document.querySelectorAll('[role="menuitemradio"][aria-checked="true"]')]
    .map((node) => textOf(node));

  let submissionMayHaveOccurred = false;
  try {
  if (location.hostname !== chatGptHost && !location.hostname.endsWith(`.${chatGptHost}`)) {
    throw new Error("protocol_wrong_host");
  }
  if (job.mode !== "chat" || job.model_slug !== "gpt-5-6-pro" || job.model_label !== requiredModel || job.effort_label !== requiredEffort) {
    throw new Error("protocol_job_target_mismatch");
  }
  if (/^\/c\//.test(location.pathname)) throw new Error("protocol_new_chat_required");

  await waitFor(() => document.querySelector("#prompt-textarea"), 30_000, "composer");
  const mode = selectedMode();
  if (!mode.chat || mode.work) throw new Error(`preflight_mode_mismatch:chat=${mode.chat}:work=${mode.work}`);

  const pill = await waitFor(
    () => [...document.querySelectorAll("button")]
      .find((button) => button.classList.contains("__composer-pill") && textOf(button) === requiredEffort),
    10_000,
    "effort_pill",
  );
  pointerClick(pill);
  const modelMenu = await waitFor(
    () => [...document.querySelectorAll('[role="menuitem"][data-has-submenu]')]
      .find((node) => textOf(node) === requiredModel),
    5_000,
    "model_menu",
  );
  pointerClick(modelMenu);
  await waitFor(() => checkedModelOptions().includes(requiredModel), 5_000, "model_submenu");
  const checked = checkedModelOptions();
  if (!checked.includes(requiredModel) || !checked.includes(requiredEffort)) {
    throw new Error(`preflight_model_mismatch:${checked.join("|")}`);
  }
  pointerClick(pill);

  const fileInput = document.querySelector('input[type="file"]:not([accept="image/*"])');
  if (attachments.length && !fileInput) throw new Error("protocol_file_input_missing");
  if (attachments.length) {
    const transfer = new DataTransfer();
    for (const item of attachments) {
      const binary = atob(item.content_base64);
      const bytes = new Uint8Array(binary.length);
      for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
      transfer.items.add(new File([bytes], item.name, { type: item.mime_type }));
    }
    fileInput.files = transfer.files;
    fileInput.dispatchEvent(new Event("change", { bubbles: true, composed: true }));
    for (const item of attachments) {
      await waitFor(() => document.body.innerText.includes(item.name), 120_000, `upload_${item.name}`);
    }
  }

  const composer = document.querySelector("#prompt-textarea");
  composer.focus();
  composer.replaceChildren();
  composer.dispatchEvent(new InputEvent("input", { bubbles: true, cancelable: true, inputType: "deleteContent" }));
  if (!document.execCommand("insertText", false, job.prompt)) throw new Error("protocol_prompt_insert_failed");
  const expectedPromptPrefix = normalizedText(job.prompt).slice(0, 120);
  if (!normalizedText(textOf(composer)).startsWith(expectedPromptPrefix)) {
    throw new Error("protocol_prompt_verification_failed");
  }

  // Re-read the mode after upload/composer work: React can rerender the mode
  // switch independently, and this is the final fail-closed submit boundary.
  const finalMode = selectedMode();
  const finalPill = [...document.querySelectorAll("button")]
    .find((button) => button.classList.contains("__composer-pill") && textOf(button) === requiredEffort);
  if (finalPill) pointerClick(finalPill);
  const finalModelMenu = finalPill
    ? await waitFor(
      () => [...document.querySelectorAll('[role="menuitem"][data-has-submenu]')]
        .find((node) => textOf(node) === requiredModel),
      5_000,
      "final_model_menu",
    )
    : null;
  if (finalModelMenu) pointerClick(finalModelMenu);
  const finalChecked = finalModelMenu
    ? await waitFor(() => {
      const values = checkedModelOptions();
      return values.includes(requiredModel) && values.includes(requiredEffort)
        ? values
        : null;
    }, 5_000, "final_model_selection")
    : [];
  if (finalPill) pointerClick(finalPill);
  if (
    !finalMode.chat
    || finalMode.work
    || !finalPill
    || !finalChecked.includes(requiredModel)
    || !finalChecked.includes(requiredEffort)
  ) {
    throw new Error("preflight_changed_before_submit");
  }
  const send = await waitFor(
    () => [...document.querySelectorAll("button")]
      .find((button) => button.classList.contains("composer-submit-button-color") && !button.disabled),
    30_000,
    "send_button",
  );
  submissionMayHaveOccurred = true;
  pointerClick(send);
  const navigationDeadline = Date.now() + 30_000;
  while (!/^\/c\//.test(location.pathname) && Date.now() < navigationDeadline) {
    const pageText = textOf(document.body);
    if (/too many requests|rate.?limit/i.test(pageText)) throw new Error("too many requests during submit");
    if (/temporarily blocked|unusual activity|access to conversations/i.test(pageText)) {
      throw new Error("access to conversations temporarily blocked for safety");
    }
    await sleep(100);
  }
  if (!/^\/c\//.test(location.pathname)) throw new Error("protocol_conversation_navigation_timeout");
  const conversationId = location.pathname.split("/").filter(Boolean)[1] || null;
  return {
    ok: true,
    phase: "submitted",
    conversation_id: conversationId,
    conversation_url: location.href,
    preflight: { mode: "Chat", model: requiredModel, effort: requiredEffort },
  };
  } catch (error) {
    return {
      ok: false,
      detail: String(error?.message || error),
      submission_may_have_occurred: submissionMayHaveOccurred,
    };
  }
}

export function inspectChatGptLaunchPage() {
  const textOf = (node) => String(node?.innerText || node?.textContent || "");
  const text = textOf(document.body);
  const softWarning = /temporarily limited access to your conversations to protect your data/i.test(text);
  const busy = [...document.querySelectorAll("button")].some((button) =>
    /pro thinking|stop generating|stop responding|stop answering/i.test(
      button.getAttribute("aria-label") || textOf(button),
    ));
  const conversationId = location.pathname.split("/").filter(Boolean)[1] || null;
  const handoffName = "polylogue-sol-pro-launch-handoff.zip";
  const turns = [...document.querySelectorAll('section[data-testid^="conversation-turn-"]')];
  const assistantTurns = turns.filter((turn) => !turn.querySelector('[data-message-author-role="user"]'));
  const handoffNode = assistantTurns
    .flatMap((turn) => [...turn.querySelectorAll("a, button")])
    .find((node) => [textOf(node), node.getAttribute("aria-label"), node.getAttribute("download"), node.getAttribute("href")]
      .filter(Boolean)
      .some((value) => String(value).includes(handoffName)));
  return {
    busy,
    conversation_id: conversationId,
    conversation_url: location.href,
    assistant_turns: assistantTurns.length,
    handoff_name: handoffNode ? handoffName : null,
    handoff_href: handoffNode?.href || null,
    soft_warning: softWarning,
    rate_limited: !softWarning && /too many requests|rate.?limit/i.test(text),
    safety_lock: !softWarning && /temporarily blocked|unusual activity|access to conversations/i.test(text),
  };
}
