// Product sign-in exchange. The shared shell owns reader interactions.
export async function authenticate({ document = globalThis.document, location = globalThis.location, history = globalThis.history, fetch = globalThis.fetch, sessionStorage = globalThis.sessionStorage } = {}) {
  const auth = document?.getElementById("auth");
  if (!auth || location.pathname !== "/auth") return;
  const requested = new URLSearchParams(location.search || "").get("next");
  const localPage = value => value?.startsWith("/") && !value.startsWith("//") && !value.includes("\\") && !/[\r\n]/.test(value);
  if (localPage(requested)) sessionStorage?.setItem("polylogue-page", requested);
  const saved = sessionStorage?.getItem("polylogue-page");
  const next = localPage(saved) ? saved : "/";
  const token = location.hash.slice(1);
  history.replaceState(null, "", "/auth");
  if (!token) return;
  try {
    const response = await fetch("/auth", { method: "POST", headers: { "Content-Type": "application/x-www-form-urlencoded" }, body: "token=" + encodeURIComponent(token), credentials: "same-origin" });
    if (response.ok) {
      sessionStorage?.removeItem("polylogue-page");
      location.replace(next);
    }
    else auth.textContent = "The access token is stale. The daemon may have restarted. Run polylogue web again to sign in and return to your page.";
  } catch {
    auth.textContent = "Sign-in could not reach the daemon. Run polylogue web again.";
  }
}
if (typeof document !== "undefined") void authenticate();
