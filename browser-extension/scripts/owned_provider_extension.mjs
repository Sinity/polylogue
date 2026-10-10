#!/usr/bin/env node
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { createProofExtension, verifyCandidateResources } from "./proof_extension.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const digest = bytes => createHash("sha256").update(bytes).digest("hex");
const BOOTSTRAP = `import { createBackgroundAdapters } from "./src/background/adapters.js";
import { startBackgroundRuntime } from "./src/background/runtime.js";
import { createOwnedProviderBrowser } from "./owned_provider_browser.mjs";
globalThis.__polylogueOwnedProviderProofReady = (async () => {
const scope = await (await fetch(chrome.runtime.getURL("owned-scope.json"))).json();
const owner = await createOwnedProviderBrowser(chrome, scope.targets);
await chrome.storage.local.set({
  receiverBaseUrl: scope.receiverUrl,
  polylogueReceiverPairing: { receiver_id: scope.receiverId, api_schema: "polylogue-browser-capture/v1",
    endpoint: scope.receiverUrl, state: "online", dev_override: true },
  polylogueAmbientSettings: { enabled: true, automatic_capture_enabled: true, disabled_sites: {} },
});
startBackgroundRuntime(createBackgroundAdapters(owner.browser));
globalThis.__polylogueOwnedProviderProof = owner;
return true;
})();
`;
const generatedFiles = ["owned_provider_browser.mjs", "proof_bootstrap.mjs", "owned-scope.json"];
function invalid() { return new Error("proof_owned_provider_binding_invalid"); }
function validateScope(scope, bound) {
  const endpoint = new URL(scope.receiverUrl);
  if (endpoint.protocol !== "http:" || endpoint.hostname !== "127.0.0.1" || endpoint.origin !== scope.receiverUrl
      || typeof scope.receiverId !== "string" || !scope.receiverId || !Array.isArray(scope.targets) || !scope.targets.length) throw invalid();
  const seen = new Set(); const windows = new Set();
  for (const target of scope.targets) {
    const url = new URL(target.url);
    const host = target.name === "chatgpt" ? "chatgpt.com" : target.name === "claude" ? "claude.ai" : null;
    const route = target.name === "chatgpt" ? "c" : "chat";
    if (!host || seen.has(target.name) || url.origin !== `https://${host}` || url.search || url.hash
        || typeof target.nativeId !== "string" || !/^[A-Za-z0-9_-]+$/.test(target.nativeId)
        || url.pathname !== `/${route}/${target.nativeId}` || url.toString() !== target.url
        || (bound ? !Number.isInteger(target.windowId) || target.windowId < 0 || windows.has(target.windowId) : target.windowId !== null)) throw invalid();
    seen.add(target.name); windows.add(target.windowId);
  }
}
function manifestFor(candidate, key, scope) {
  return { manifest_version: 3, name: "Polylogue owned provider runtime proof", version: candidate.version, key,
    permissions: candidate.permissions.filter(permission => permission !== "activeTab"),
    host_permissions: [...new Set([`${scope.receiverUrl}/*`, ...scope.targets.map(target => `${new URL(target.url).origin}/*`),
      ...(scope.targets.some(target => target.name === "claude") ? ["https://*.claudeusercontent.com/*"] : [])])],
    background: { service_worker: "proof_bootstrap.mjs", type: "module" } };
}
function recordGenerated(extensionRoot, binding) {
  return { ...binding, generated_resources: generatedFiles.map(file => [file, digest(readFileSync(path.join(extensionRoot, file)))]) };
}
export function createOwnedProviderProofExtension({ destination, hostName, receiverUrl, receiverId, targets, sourceRoot = ROOT }) {
  const scope = { receiverUrl, receiverId, targets: targets.map(target => ({ ...target, windowId: null })) };
  validateScope(scope, false);
  const base = createProofExtension({ destination, hostName, sourceRoot });
  const oldManifest = JSON.parse(readFileSync(path.join(destination, "manifest.json"), "utf8"));
  const candidate = JSON.parse(readFileSync(path.join(sourceRoot, "manifest.json"), "utf8"));
  writeFileSync(path.join(destination, "manifest.json"), `${JSON.stringify(manifestFor(candidate, oldManifest.key, scope), null, 2)}\n`);
  writeFileSync(path.join(destination, "owned_provider_browser.mjs"), readFileSync(path.join(ROOT, "scripts/owned_provider_browser.mjs")));
  writeFileSync(path.join(destination, "proof_bootstrap.mjs"), BOOTSTRAP);
  writeFileSync(path.join(destination, "owned-scope.json"), `${JSON.stringify(scope)}\n`);
  const binding = recordGenerated(destination, { ...base, kind: "owned-provider-runtime", owned_targets_bound: false,
    substitutions: { ...base.substitutions,
      manifest: "fresh key; production worker through owned-tab browser authority; no static content scripts or action",
      currentWindow: "first declared owned proof window", bootstrap: "receiver configured before production worker registration; automatic capture enabled" } });
  writeFileSync(path.join(destination, "proof-binding.json"), `${JSON.stringify(binding, null, 2)}\n`);
  return binding;
}
export function verifyOwnedProviderProofExtension(extensionRoot, { bound = true, sourceRoot = ROOT } = {}) {
  const binding = JSON.parse(readFileSync(path.join(extensionRoot, "proof-binding.json"), "utf8"));
  const scope = JSON.parse(readFileSync(path.join(extensionRoot, "owned-scope.json"), "utf8"));
  validateScope(scope, bound);
  const manifest = JSON.parse(readFileSync(path.join(extensionRoot, "manifest.json"), "utf8"));
  const candidate = JSON.parse(readFileSync(path.join(sourceRoot, "manifest.json"), "utf8"));
  const key = Buffer.from(manifest.key, "base64");
  const id = [...digest(key).slice(0, 32)].map(char => String.fromCharCode(97 + parseInt(char, 16))).join("");
  if (binding.kind !== "owned-provider-runtime" || binding.owned_targets_bound !== bound
      || id !== binding.extension_id || digest(key) !== binding.key_sha256
      || !/^com\.polylogue\.browser_capture\.proof_[a-f0-9]{32}$/.test(binding.host_name)
      || JSON.stringify(manifest) !== JSON.stringify(manifestFor(candidate, manifest.key, scope))
      || !readFileSync(path.join(extensionRoot, "proof_bootstrap.mjs")).equals(Buffer.from(BOOTSTRAP))
      || !readFileSync(path.join(extensionRoot, "owned_provider_browser.mjs")).equals(readFileSync(path.join(ROOT, "scripts/owned_provider_browser.mjs")))
      || JSON.stringify(recordGenerated(extensionRoot, binding).generated_resources) !== JSON.stringify(binding.generated_resources)) throw invalid();
  verifyCandidateResources(extensionRoot, binding, sourceRoot);
  return binding;
}
export function bindOwnedProviderTargets(extensionRoot, targets) {
  const binding = verifyOwnedProviderProofExtension(extensionRoot, { bound: false });
  const scope = JSON.parse(readFileSync(path.join(extensionRoot, "owned-scope.json"), "utf8"));
  if (targets.length !== scope.targets.length || targets.some((target, index) => target.url !== scope.targets[index].url)) throw invalid();
  const bound = { ...scope, targets: scope.targets.map((target, index) => ({ ...target, windowId: targets[index].windowId })) };
  validateScope(bound, true);
  writeFileSync(path.join(extensionRoot, "owned-scope.json"), `${JSON.stringify(bound)}\n`);
  writeFileSync(path.join(extensionRoot, "proof-binding.json"), `${JSON.stringify(recordGenerated(extensionRoot,
    { ...binding, owned_targets_bound: true }), null, 2)}\n`);
  return verifyOwnedProviderProofExtension(extensionRoot);
}
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const args = process.argv.slice(2);
  if (args.length !== 6 || args[0] !== "--destination" || args[2] !== "--host" || args[4] !== "--scope") throw invalid();
  const scope = JSON.parse(readFileSync(args[5], "utf8"));
  process.stdout.write(`${JSON.stringify(createOwnedProviderProofExtension({ destination: path.resolve(args[1]), hostName: args[3], ...scope }))}\n`);
}
