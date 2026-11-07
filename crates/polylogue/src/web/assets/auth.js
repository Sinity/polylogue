// Product sign-in exchange. The shared shell owns reader interactions.
export async function authenticate({ document = globalThis.document, location = globalThis.location, history = globalThis.history, fetch = globalThis.fetch } = {}) {
  const auth = document?.getElementById("auth");
  if (!auth || location.pathname !== "/auth") return;
  const token = location.hash.slice(1);
  history.replaceState(null, "", "/auth");
  if (!token) return;
  try {
    const response = await fetch("/auth", { method: "POST", headers: { "Content-Type": "application/x-www-form-urlencoded" }, body: "token=" + encodeURIComponent(token), credentials: "same-origin" });
    if (response.ok) location.replace("/");
    else auth.textContent = "The access token is stale. Run polylogue web again.";
  } catch {
    auth.textContent = "Sign-in could not reach the daemon. Run polylogue web again.";
  }
}
if (typeof document !== "undefined") void authenticate();
