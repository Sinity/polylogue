const field = document.getElementById("facet-field");
const value = document.getElementById("facet-value");
const suggestions = document.getElementById("facet-suggestions");
const status = document.getElementById("completion-status");
let timer;
let controller;
let version = 0;
function complete() {
  clearTimeout(timer);
  controller?.abort();
  const current = ++version;
  suggestions.replaceChildren();
  timer = setTimeout(async () => {
    controller = new AbortController();
    try {
      const prefix = field.value + ":";
      const response = await fetch("/api/v1/complete?word=" + encodeURIComponent(prefix + value.value), { credentials: "same-origin", signal: controller.signal });
      if (!response.ok) throw new Error("Suggestions unavailable (" + response.status + ")");
      const data = await response.json();
      if (current !== version) return;
      for (const entry of data) {
        const option = document.createElement("option");
        option.value = entry.value.startsWith(prefix) ? entry.value.slice(prefix.length) : entry.value;
        // The completion API quotes values for query text; facet values are literals.
        if (option.value.startsWith('"')) {
          option.value = option.value.slice(1, -1).replace(/\\(.)/gs, "$1");
        }
        option.label = entry.help;
        suggestions.append(option);
      }
      status.textContent = "Completion values are suggestions, not counts.";
    } catch (error) { if (error.name !== "AbortError" && current === version) status.textContent = error.message; }
  }, 200);
}
value?.addEventListener("input", complete);
field?.addEventListener("change", complete);

const form = document.getElementById("save-search");
let attempt;
let writing = false;
form?.addEventListener("submit", async event => {
  event.preventDefault();
  if (writing) return;
  const operation = event.submitter.value;
  const name = form.elements.name.value;
  const query = form.dataset.query;
  const intended = JSON.stringify({ operation, name, query });
  if (attempt?.intended !== intended) attempt = { intended, request_id: crypto.randomUUID() };
  const output = document.getElementById("save-status");
  writing = true;
  try {
    const response = await fetch("/api/v1/" + operation, {
      method: "POST", credentials: "same-origin", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ request_id: attempt.request_id, name: name || undefined, query }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.message ?? "Write failed");
    output.textContent = (operation === "save" ? "Query saved: " : "Results kept: ") + JSON.stringify(data);
    attempt = undefined;
  } catch (error) { output.textContent = error.message + ". Submit again to retry the same request."; }
  finally { writing = false; }
});
