// The popup: what Polylogue holds of this page, what waits, and the few
// choices the operator makes here. Everything shown comes from the
// background's `polylogue.popup` answer.
(function () {
  const words = globalThis.polylogueWording;
  const $ = (id) => document.getElementById(id);
  let tabId = null;

  function setDot(id, tone) {
    $(id).className = `dot ${tone}`;
  }

  function item(...lines) {
    const li = document.createElement("li");
    lines.forEach((text, i) => {
      const p = document.createElement(i ? "p" : "span");
      if (i) p.className = "meta";
      p.textContent = text;
      li.append(p);
    });
    return li;
  }

  function showPolylogue(state) {
    const { status } = state;
    let tone = "ok";
    let label = "Running";
    let detail = "";
    if (status.host === "missing") {
      tone = "bad";
      label = "Not installed for this browser";
      detail = `Run: polylogue install-agent chrome --extension-id ${chrome.runtime.id}`;
    } else if (status.host !== "ok") {
      tone = "bad";
      label = "Polylogue did not answer";
    } else if (status.daemon !== "running") {
      tone = "warn";
      label = "The daemon is not running";
      detail = "Captures are stored and read once it starts: polylogue daemon";
    } else if (status.chat_actions) {
      detail = "Chat actions are on.";
    }
    setDot("polylogue-dot", tone);
    $("polylogue-state").textContent = label;
    $("polylogue-detail").textContent = detail;
    $("waiting").hidden = state.waiting === 0;
    $("waiting").textContent = `${state.waiting} capture${state.waiting === 1 ? "" : "s"} waiting to be stored.`;
  }

  function showPage(page) {
    const said = words.conversation(page);
    setDot("page-dot", said.tone);
    $("page-state").textContent = said.label;
    $("page-detail").textContent = said.detail;
    $("capture").disabled = !page?.conversation;
    $("reader").disabled = page?.state !== "archived";
  }

  function showApprovals(actions) {
    $("approvals-section").hidden = actions.length === 0;
    $("approvals").replaceChildren(...actions.map((action) => {
      const li = item(
        `${words.providerName(action.provider)}: send into conversation ${action.conversation}`,
        action.text.length > 200 ? `${action.text.slice(0, 200)}…` : action.text,
      );
      const buttons = document.createElement("div");
      buttons.className = "buttons";
      for (const [label, approve] of [["Approve", true], ["Decline", false]]) {
        const button = document.createElement("button");
        button.textContent = label;
        if (approve) button.className = "primary";
        button.addEventListener("click", async () => {
          button.disabled = true;
          await chrome.runtime.sendMessage({ type: "polylogue.decide", id: action.id, approve });
          await refresh();
        });
        buttons.append(button);
      }
      li.append(buttons);
      return li;
    }));
  }

  function showBackfill(jobs) {
    $("backfill-section").hidden = jobs.length === 0;
    $("backfill").replaceChildren(...jobs.map((job) => item(
      `${words.providerName(job.provider)}: ${job.status}`,
      `${job.listed} listed, ${job.captured} captured, ${job.unchanged} already in the archive, ${job.failed} failed`
        + (job.detail ? `. ${job.detail}` : ""),
    )));
  }

  function showProblems(problems) {
    $("problems-section").hidden = problems.length === 0;
    $("problems").replaceChildren(...problems.slice(0, 5).map((problem) => item(problem.text, words.ago(problem.at))));
  }

  async function refresh() {
    const state = await chrome.runtime.sendMessage({ type: "polylogue.popup", tabId });
    if (!state || state.error) {
      $("polylogue-state").textContent = `The extension could not answer: ${state?.error || "no answer"}`;
      return;
    }
    showPolylogue(state);
    showPage(state.page);
    showApprovals(state.approvals);
    showBackfill(state.backfill);
    showProblems(state.problems);
    $("automatic").checked = state.settings.automatic;
    $("panel").checked = state.settings.panel;
    $("version").textContent = `Extension ${state.version}`;
  }

  $("capture").addEventListener("click", async () => {
    $("capture").disabled = true;
    $("page-state").textContent = "Capturing…";
    showPage(await chrome.runtime.sendMessage({ type: "polylogue.capture", tabId }));
    await refresh();
  });
  $("reader").addEventListener("click", async () => {
    const answer = await chrome.runtime.sendMessage({ type: "polylogue.reader" });
    if (answer?.error) $("page-detail").textContent = "The reader is not running; start it with `polylogue web`.";
  });
  for (const id of ["automatic", "panel"]) {
    $(id).addEventListener("change", async () => {
      await chrome.runtime.sendMessage({ type: "polylogue.settings", [id]: $(id).checked });
    });
  }

  chrome.tabs.query({ active: true, currentWindow: true }).then(([tab]) => {
    tabId = tab?.id ?? null;
    return refresh();
  });
})();
