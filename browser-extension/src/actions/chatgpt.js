export function classifyBrowserActionFailure(value, retryAfterSeconds = null) {
  const detail = String(value?.message || value || "browser_action_failed");
  const text = detail.toLowerCase();
  if (/temporarily limited access.*protect your data|provider soft warning/.test(text)) {
    return { outcome: "provider_warning", retry_after_seconds: retryAfterSeconds, detail };
  }
  if (/too many requests|rate.?limit|http_?429/.test(text)) {
    return { outcome: "rate_limited", retry_after_seconds: retryAfterSeconds, detail };
  }
  if (/safety|temporarily blocked|unusual activity|access to conversations/.test(text)) {
    return { outcome: "safety_locked", retry_after_seconds: retryAfterSeconds, detail };
  }
  if (/challenge|captcha|cloudflare|unauthorized|forbidden|http_?401|http_?403|sign.?in/.test(text)) {
    return { outcome: "auth_challenge", retry_after_seconds: null, detail };
  }
  if (/unsupported|target mismatch|work selected|chat selected|model mismatch|effort|project mismatch|capability_mismatch|active_composer_tool/.test(text)) {
    return { outcome: "capability_mismatch", retry_after_seconds: null, detail };
  }
  if (/protocol|selector|composer|file input|upload|provider response|turn receipt/.test(text)) {
    return { outcome: "provider_drift", retry_after_seconds: null, detail };
  }
  return { outcome: "network_error", retry_after_seconds: retryAfterSeconds, detail };
}

export async function executeChatGptBrowserActionInPage(action, attachments) {
  // chrome.scripting serializes this function into MAIN world. All helpers and
  // constants therefore intentionally live inside the function body.
  const sleep = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));
  const textOf = (node) => String(node?.innerText || node?.textContent || "").trim();
  const normalizedText = (value) => String(value || "").replace(/\s+/g, " ").trim();
  const expectedText = normalizedText(action.text);
  const expectedModel = action.presentation.model_label;
  const expectedEffort = action.presentation.effort_label;
  const expectedProject = String(action.target.project_ref || "").match(/g-p-[A-Za-z0-9]+/)?.[0] || null;
  const waitFor = async (predicate, timeoutMs, label) => {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
      const result = await predicate();
      if (result) return result;
      await sleep(100);
    }
    throw new Error(`${label}_timeout`);
  };
  const pointerClick = (node) => {
    if (!node) throw new Error("protocol_selector_missing");
    const init = {
      bubbles: true,
      cancelable: true,
      composed: true,
      pointerId: 1,
      pointerType: "mouse",
      isPrimary: true,
    };
    if (typeof PointerEvent === "function") {
      node.dispatchEvent(new PointerEvent("pointerdown", init));
      node.dispatchEvent(new PointerEvent("pointerup", init));
    }
    node.dispatchEvent(new MouseEvent("mousedown", init));
    node.dispatchEvent(new MouseEvent("mouseup", init));
    node.click();
  };
  const modeButtons = () => {
    const buttons = [...document.querySelectorAll("button")];
    const chat = buttons.find((button) => textOf(button) === "Chat");
    const work = buttons.find((button) => textOf(button) === "Work");
    const selected = (button) => button?.getAttribute("aria-pressed") === "true"
      || button?.getAttribute("aria-checked") === "true"
      || ["active", "on", "checked"].includes(button?.getAttribute("data-state"))
      || String(button?.className || "").includes("text-token-text-primary");
    return { chat, work, chatSelected: selected(chat), workSelected: selected(work) };
  };
  const selectedMode = () => {
    const mode = modeButtons();
    return { chat: mode.chatSelected, work: mode.workSelected };
  };
  const checkedModelOptions = () => [...document.querySelectorAll('[role="menuitemradio"][aria-checked="true"]')]
    .map((node) => textOf(node));
  const conversationIdFromLocation = () => {
    const parts = location.pathname.split("/").filter(Boolean);
    const marker = parts.indexOf("c");
    return marker >= 0 ? parts[marker + 1] || null : null;
  };
  const providerConversation = async (conversationId, allowUnavailable = false) => {
    const response = await fetch(`/backend-api/conversation/${encodeURIComponent(conversationId)}`, {
      credentials: "include",
    });
    if (allowUnavailable && [401, 403, 404].includes(response.status)) {
      return { conversation: null, status: response.status };
    }
    if (!response.ok) {
      const retryAfter = response.headers.get("Retry-After");
      const error = new Error(`provider response http_${response.status}${retryAfter ? ` retry_after_${retryAfter}` : ""}`);
      error.retryAfterSeconds = Number.parseInt(retryAfter || "", 10) || null;
      throw error;
    }
    return { conversation: await response.json(), status: response.status };
  };
  const userMessages = (conversation) => Object.values(conversation?.mapping || {})
    .map((node) => node?.message)
    .filter((message) => message?.author?.role === "user");
  const messageText = (message) => normalizedText(
    (message?.content?.parts || []).filter((part) => typeof part === "string").join("\n"),
  );
  const domUserMessages = () => [...document.querySelectorAll('[data-message-author-role="user"][data-message-id]')]
    .map((node) => ({ id: node.getAttribute("data-message-id"), text: normalizedText(textOf(node)) }))
    .filter((message) => message.id);

  let submissionMayHaveOccurred = false;
  try {
    if (location.hostname !== "chatgpt.com" && !location.hostname.endsWith(".chatgpt.com")) {
      throw new Error("protocol_wrong_host");
    }
    if (action.provider !== "chatgpt" || action.presentation.surface !== "chat") {
      throw new Error("unsupported provider or surface target mismatch");
    }
    const locationConversationId = conversationIdFromLocation();
    if (action.operation === "conversation.create" && locationConversationId) {
      throw new Error("protocol_new_conversation_target_required");
    }
    if (
      action.operation === "conversation.reply"
      && locationConversationId !== action.target.conversation_id
    ) {
      throw new Error("protocol_reply_target_mismatch");
    }
    if (expectedProject && !location.pathname.includes(expectedProject)) {
      throw new Error("project mismatch before compose");
    }

    let baselineUserIds = new Set();
    let providerBaselineComplete = action.operation !== "conversation.reply";
    if (action.operation === "conversation.reply") {
      const baselineRead = await providerConversation(action.target.conversation_id, true);
      const baseline = baselineRead.conversation;
      providerBaselineComplete = Boolean(baseline);
      baselineUserIds = baseline
        ? new Set(userMessages(baseline).map((message) => message.id))
        : new Set(domUserMessages().map((message) => message.id));
      if (expectedProject && baseline && baseline.gizmo_id !== expectedProject) {
        throw new Error("project mismatch in provider response");
      }
    }

    await waitFor(() => document.querySelector("#prompt-textarea"), 30_000, "composer");
    const initialMode = await waitFor(() => {
      const mode = modeButtons();
      return mode.chat && mode.work ? mode : null;
    }, 10_000, "surface_controls");
    if (!initialMode.chatSelected || initialMode.workSelected) {
      pointerClick(initialMode.chat);
    }
    const mode = await waitFor(() => {
      const observed = selectedMode();
      return observed.chat && !observed.work ? observed : null;
    }, 5_000, "chat_surface_selection");
    if (!mode.chat || mode.work) throw new Error(`chat selected mismatch:chat=${mode.chat}:work=${mode.work}`);

    const pill = await waitFor(
      () => [...document.querySelectorAll("button")]
        .find((button) => button.classList.contains("__composer-pill") && textOf(button) === expectedEffort),
      10_000,
      "effort_pill",
    );
    pointerClick(pill);
    const modelMenu = await waitFor(
      () => [...document.querySelectorAll('[role="menuitem"][data-has-submenu]')]
        .find((node) => textOf(node) === expectedModel),
      5_000,
      "model_menu",
    );
    pointerClick(modelMenu);
    await waitFor(() => checkedModelOptions().includes(expectedModel), 5_000, "model_submenu");
    const checked = checkedModelOptions();
    if (!checked.includes(expectedModel) || !checked.includes(expectedEffort)) {
      throw new Error(`model mismatch:${checked.join("|")}`);
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
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(composer);
    selection.removeAllRanges();
    selection.addRange(range);
    if (!document.execCommand("insertText", false, action.text)) throw new Error("protocol_prompt_insert_failed");
    await sleep(100);
    if (composer.querySelector("[data-inline-selection-pill]")) {
      throw new Error("capability_mismatch:active_composer_tool_remains");
    }
    if (normalizedText(textOf(composer)) !== expectedText) {
      throw new Error("protocol_prompt_verification_failed");
    }

    const finalMode = selectedMode();
    const finalPill = [...document.querySelectorAll("button")]
      .find((button) => button.classList.contains("__composer-pill") && textOf(button) === expectedEffort);
    if (finalPill) pointerClick(finalPill);
    const finalModelMenu = finalPill
      ? await waitFor(
        () => [...document.querySelectorAll('[role="menuitem"][data-has-submenu]')]
          .find((node) => textOf(node) === expectedModel),
        5_000,
        "final_model_menu",
      )
      : null;
    if (finalModelMenu) pointerClick(finalModelMenu);
    const finalChecked = finalModelMenu
      ? await waitFor(() => {
        const values = checkedModelOptions();
        return values.includes(expectedModel) && values.includes(expectedEffort) ? values : null;
      }, 5_000, "final_model_selection")
      : [];
    if (finalPill) pointerClick(finalPill);
    if (!finalMode.chat || finalMode.work || !finalPill || !finalChecked.includes(expectedModel)) {
      throw new Error("model mismatch before submit");
    }

    if (action.submit_policy === "stage_only") {
      return {
        ok: true,
        outcome: "drafted",
        provider_conversation_id: locationConversationId,
        provider_conversation_url: location.href,
        provider_turn_id: null,
        observed_surface: "Chat",
        observed_model: expectedModel,
        observed_effort: expectedEffort,
        observed_project_ref: expectedProject,
        provider_evidence: { composer_verified: true, attachment_count: attachments.length },
      };
    }

    const send = await waitFor(
      () => [...document.querySelectorAll("button")]
        .find((button) => button.classList.contains("composer-submit-button-color") && !button.disabled),
      30_000,
      "send_button",
    );
    submissionMayHaveOccurred = true;
    pointerClick(send);
    const conversationId = await waitFor(conversationIdFromLocation, 30_000, "conversation_navigation");
    const receipt = await waitFor(async () => {
      const providerRead = await providerConversation(conversationId, true);
      const conversation = providerRead.conversation;
      if (conversation && providerBaselineComplete) {
        const submitted = userMessages(conversation)
          .find((message) => !baselineUserIds.has(message.id) && messageText(message) === expectedText);
        if (!submitted) return null;
        return { conversation, submitted, source: "provider_api", readStatus: providerRead.status };
      }
      const submitted = domUserMessages()
        .find((message) => !baselineUserIds.has(message.id) && message.text === expectedText);
      if (!submitted) return null;
      return { conversation: null, submitted, source: "provider_dom", readStatus: providerRead.status };
    }, 30_000, "provider_turn_receipt");
    if (expectedProject && receipt.conversation && receipt.conversation.gizmo_id !== expectedProject) {
      throw new Error("project mismatch after submit");
    }
    return {
      ok: true,
      outcome: "submitted",
      provider_conversation_id: conversationId,
      provider_conversation_url: location.href,
      provider_turn_id: receipt.submitted.id,
      observed_surface: "Chat",
      observed_model: expectedModel,
      observed_effort: expectedEffort,
      observed_project_ref: receipt.conversation?.gizmo_id || expectedProject,
      provider_evidence: {
        receipt_source: receipt.source,
        conversation_read_status: receipt.readStatus,
        current_node: receipt.conversation?.current_node || null,
        conversation_update_time: receipt.conversation?.update_time || null,
        user_turn_status: receipt.submitted.status || null,
        attachment_count: attachments.length,
      },
    };
  } catch (error) {
    return {
      ok: false,
      detail: String(error?.message || error),
      submission_may_have_occurred: submissionMayHaveOccurred,
      retry_after_seconds: error?.retryAfterSeconds || null,
    };
  }
}
