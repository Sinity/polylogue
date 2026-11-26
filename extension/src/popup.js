// The popup: what Polylogue holds of this page, what waits, and the few
// choices the operator makes here. Everything shown comes from the
// background's `polylogue.popup` answer.
(function () {
  const words = globalThis.polylogueWording;
  const $ = (id) => document.getElementById(id);
  const PROBLEMS_SHOWN = 5;
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
      detail = `Run: polylogue install chrome --extension-id ${chrome.runtime.id}`;
    } else if (status.host !== "ok") {
      tone = "bad";
      label = "Polylogue did not answer";
    } else if (status.daemon !== "running") {
      tone = "warn";
      label = "The daemon did not answer";
      detail = "Captures are still stored. It is stopped (start it with `polylogue daemon`) or busy.";
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
    // A failed request leaves Capture available, so the operator can retry.
    $("capture").disabled = !page?.conversation && !page?.error;
    $("reader").disabled = !globalThis.polylogueCaptureState.inArchive(said.state);
  }

  function showApprovals(actions) {
    $("approvals-section").hidden = actions.length === 0;
    $("approvals").replaceChildren(...actions.map((action) => {
      // The turn is shown whole: it is posted under the operator's account.
      const what = action.conversation ? `send into conversation ${action.conversation}` : "submit a new conversation";
      const li = item(`${words.providerName(action.provider)}: ${what}`, action.text);
      const buttons = document.createElement("div");
      buttons.className = "buttons";
      // The operator's answer, as `polylogue SESSION answer allow|deny` gives it.
      for (const [label, value] of [["Allow", "allow"], ["Deny", "deny"]]) {
        const button = document.createElement("button");
        button.textContent = label;
        if (value === "allow") button.className = "primary";
        button.addEventListener("click", async () => {
          button.disabled = true;
          await chrome.runtime.sendMessage({ type: "polylogue.answer", id: action.id, value });
          await refresh();
        });
        buttons.append(button);
      }
      li.append(buttons);
      return li;
    }));
  }

  function showImportJobs(jobs) {
    $("import_job-section").hidden = jobs.length === 0;
    $("import_job").replaceChildren(...jobs.map((job) => item(
      `${words.providerName(job.provider)}: ${job.state}`,
      `${job.listed} listed, ${job.captured} captured, ${job.unchanged} already in the archive, ${job.failed} failed`
        + (job.detail ? `. ${job.detail}` : ""),
    )));
  }

  function showProblems(problems) {
    $("problems-section").hidden = problems.length === 0;
    const shown = problems.slice(0, PROBLEMS_SHOWN).map((problem) => item(problem.text, words.ago(problem.at)));
    if (problems.length > PROBLEMS_SHOWN) shown.push(item(`${problems.length - PROBLEMS_SHOWN} older not shown`));
    $("problems").replaceChildren(...shown);
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
    showImportJobs(state.importJob);
    showProblems(state.problems);
    $("automatic").checked = state.settings.automatic;
    $("panel").checked = state.settings.panel;
    $("version").textContent = `Extension ${state.version}`;
  }

  $("capture").addEventListener("click", async () => {
    $("capture").disabled = true;
    $("page-state").textContent = "Capturing…";
    let answer;
    try {
      answer = await chrome.runtime.sendMessage({ type: "polylogue.capture", tabId });
    } catch (error) {
      answer = { error: String(error?.message || error) };
    }
    showPage(answer);
    // A refresh would replace the failure with the page's state.
    if (!answer?.error) await refresh();
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
