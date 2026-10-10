// @vitest-environment node
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, expect, it } from "vitest";
import { bindOwnedProviderTargets, createOwnedProviderProofExtension, verifyOwnedProviderProofExtension } from "../scripts/owned_provider_extension.mjs";

const owned = [];
afterEach(() => { for (const root of owned.splice(0)) rmSync(root, { recursive: true, force: true }); });
function build() {
  const root = mkdtempSync(path.join(tmpdir(), "polylogue-owned-provider-artifact-")); owned.push(root);
  const destination = path.join(root, "extension");
  const binding = createOwnedProviderProofExtension({ destination, hostName: `com.polylogue.browser_capture.proof_${"b".repeat(32)}`,
    receiverUrl: "http://127.0.0.1:18765", receiverId: "neutral-receiver", targets: [
      { name: "chatgpt", url: "https://chatgpt.com/c/neutral-first", nativeId: "neutral-first" },
      { name: "claude", url: "https://claude.ai/chat/neutral-second", nativeId: "neutral-second" },
    ] });
  return { destination, binding };
}
function targets() { return [{ windowId: 21, url: "https://chatgpt.com/c/neutral-first" }, { windowId: 22, url: "https://claude.ai/chat/neutral-second" }]; }

it("binds fresh runtime identity before loading, with no static scripts or operator popup", () => {
  const { destination, binding } = build();
  expect(verifyOwnedProviderProofExtension(destination, { bound: false })).toEqual(binding);
  expect(() => verifyOwnedProviderProofExtension(destination)).toThrow("proof_owned_provider_binding_invalid");
  const bound = bindOwnedProviderTargets(destination, targets());
  expect(bound.extension_id).toBe(binding.extension_id); expect(bound.key_sha256).toBe(binding.key_sha256);
  expect(bound.owned_targets_bound).toBe(true);
  const manifest = JSON.parse(readFileSync(path.join(destination, "manifest.json"), "utf8"));
  expect(manifest.content_scripts).toBeUndefined(); expect(manifest.action).toBeUndefined();
  expect(manifest.background).toEqual({ service_worker: "proof_bootstrap.mjs", type: "module" });
  expect(manifest.host_permissions).toEqual(["http://127.0.0.1:18765/*", "https://chatgpt.com/*", "https://claude.ai/*", "https://*.claudeusercontent.com/*"]);
  expect(JSON.parse(readFileSync(path.join(destination, "owned-scope.json"), "utf8")).targets).toEqual([
    { ...targets()[0], name: "chatgpt", nativeId: "neutral-first" }, { ...targets()[1], name: "claude", nativeId: "neutral-second" },
  ]);
  expect(() => bindOwnedProviderTargets(destination, targets())).toThrow("proof_owned_provider_binding_invalid");
});

it("refuses changed targets and physical window aliasing before binding", () => {
  const { destination } = build();
  expect(() => bindOwnedProviderTargets(destination, [{ ...targets()[0], url: "https://chatgpt.com/c/other" }, targets()[1]])).toThrow("proof_owned_provider_binding_invalid");
  expect(() => bindOwnedProviderTargets(destination, [targets()[0], { ...targets()[1], windowId: 21 }])).toThrow("proof_owned_provider_binding_invalid");
  expect(verifyOwnedProviderProofExtension(destination, { bound: false }).owned_targets_bound).toBe(false);
});

it("refuses extra static provider registration or replaced authority code", () => {
  for (const file of ["manifest.json", "owned_provider_browser.mjs", "proof_bootstrap.mjs", "owned-scope.json"]) {
    const { destination } = build(); bindOwnedProviderTargets(destination, targets());
    const target = path.join(destination, file);
    if (file === "manifest.json") {
      const manifest = JSON.parse(readFileSync(target, "utf8")); manifest.content_scripts = [{ matches: ["https://chatgpt.com/*"], js: ["src/content/chatgpt.js"] }];
      writeFileSync(target, JSON.stringify(manifest));
    } else writeFileSync(target, `${readFileSync(target, "utf8")}\n `);
    expect(() => verifyOwnedProviderProofExtension(destination)).toThrow("proof_owned_provider_binding_invalid");
  }
});
