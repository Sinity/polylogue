#!/usr/bin/env node
import { createHash, generateKeyPairSync } from "node:crypto";
import { cpSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const HOST_LITERAL = 'const NATIVE_HOST = "com.polylogue.browser_capture";';
const NATIVE_FILE = "src/background/native_fetch.js";
const sha256 = value => createHash("sha256").update(value).digest("hex");

function resources(root) {
  return readdirSync(path.join(root, "src"), { recursive: true, withFileTypes: true })
    .filter(entry => entry.isFile()).map(entry => path.relative(root, path.join(entry.parentPath, entry.name))).sort();
}

export function createProofExtension({ destination, hostName, sourceRoot = ROOT }) {
  if (!/^com\.polylogue\.browser_capture\.proof_[a-f0-9]{32}$/.test(hostName)) throw new Error("proof_native_host_invalid");
  // An existing directory may belong to another extension. Never overwrite it.
  mkdirSync(destination);
  cpSync(path.join(sourceRoot, "src"), path.join(destination, "src"), { recursive: true });
  const candidateManifest = JSON.parse(readFileSync(path.join(sourceRoot, "manifest.json"), "utf8"));
  const { publicKey } = generateKeyPairSync("rsa", { modulusLength: 2048, publicKeyEncoding: { type: "spki", format: "der" } });
  const manifest = { manifest_version: 3, name: "Polylogue isolated native transport proof", version: candidateManifest.version,
    key: publicKey.toString("base64"), permissions: ["nativeMessaging"] };
  writeFileSync(path.join(destination, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`);
  const nativePath = path.join(destination, NATIVE_FILE);
  const native = readFileSync(nativePath, "utf8");
  if (native.split(HOST_LITERAL).length !== 2) throw new Error("proof_native_host_binding_missing");
  writeFileSync(nativePath, native.replace(HOST_LITERAL, `const NATIVE_HOST = ${JSON.stringify(hostName)};`));
  writeFileSync(path.join(destination, "proof.html"), '<!doctype html><meta charset="utf-8"><title>Polylogue isolated transport proof</title>');
  cpSync(path.join(ROOT, "scripts/proof_transport.mjs"), path.join(destination, "proof_transport.mjs"));
  const files = resources(sourceRoot);
  const candidateRows = files.map(file => [file, sha256(readFileSync(path.join(sourceRoot, file)))]);
  const proofRows = files.map(file => [file, sha256(readFileSync(path.join(destination, file)))]);
  const extensionId = [...sha256(publicKey).slice(0, 32)].map(digit => String.fromCharCode(97 + parseInt(digit, 16))).join("");
  const binding = { extension_id: extensionId, host_name: hostName, version: manifest.version, key_sha256: sha256(publicKey),
    candidate_bundle_sha256: sha256(JSON.stringify(candidateRows)), proof_bundle_sha256: sha256(JSON.stringify(proofRows)),
    resources: proofRows, candidate_resources: candidateRows, substitutions: { [NATIVE_FILE]: { from: "com.polylogue.browser_capture", to: hostName },
      manifest: "isolated nativeMessaging page; fresh public key; no background, content scripts, action, or host permissions" } };
  writeFileSync(path.join(destination, "proof-binding.json"), `${JSON.stringify(binding, null, 2)}\n`);
  return binding;
}

export function verifyProofExtension(extensionRoot, sourceRoot = ROOT) {
  const binding = JSON.parse(readFileSync(path.join(extensionRoot, "proof-binding.json"), "utf8"));
  const manifest = JSON.parse(readFileSync(path.join(extensionRoot, "manifest.json"), "utf8"));
  const candidateManifest = JSON.parse(readFileSync(path.join(sourceRoot, "manifest.json"), "utf8"));
  const key = Buffer.from(manifest.key, "base64");
  const id = [...sha256(key).slice(0, 32)].map(digit => String.fromCharCode(97 + parseInt(digit, 16))).join("");
  if (manifest.background || manifest.content_scripts || manifest.action || manifest.host_permissions
      || JSON.stringify(manifest.permissions) !== '["nativeMessaging"]'
      || sha256(key) !== binding.key_sha256 || id !== binding.extension_id || manifest.version !== candidateManifest.version
      || !/^com\.polylogue\.browser_capture\.proof_[a-f0-9]{32}$/.test(binding.host_name)) throw new Error("proof_extension_binding_invalid");
  return verifyCandidateResources(extensionRoot, binding, sourceRoot);
}

export function verifyCandidateResources(extensionRoot, binding, sourceRoot = ROOT) {
  if (JSON.stringify(resources(sourceRoot)) !== JSON.stringify(resources(extensionRoot))) throw new Error("proof_extension_resource_mismatch");
  for (const file of resources(sourceRoot)) {
    const expected = readFileSync(path.join(sourceRoot, file));
    let actual = readFileSync(path.join(extensionRoot, file));
    if (file === NATIVE_FILE) actual = Buffer.from(actual.toString("utf8").replace(`const NATIVE_HOST = ${JSON.stringify(binding.host_name)};`, HOST_LITERAL));
    if (!actual.equals(expected)) throw new Error("proof_extension_resource_mismatch");
  }
  if (!readFileSync(path.join(extensionRoot, "proof_transport.mjs")).equals(readFileSync(path.join(ROOT, "scripts/proof_transport.mjs")))) throw new Error("proof_extension_resource_mismatch");
  const candidateRows = resources(sourceRoot).map(file => [file, sha256(readFileSync(path.join(sourceRoot, file)))]);
  const proofRows = resources(extensionRoot).map(file => [file, sha256(readFileSync(path.join(extensionRoot, file)))]);
  if (JSON.stringify(candidateRows) !== JSON.stringify(binding.candidate_resources) || JSON.stringify(proofRows) !== JSON.stringify(binding.resources)
      || sha256(JSON.stringify(candidateRows)) !== binding.candidate_bundle_sha256 || sha256(JSON.stringify(proofRows)) !== binding.proof_bundle_sha256) throw new Error("proof_extension_binding_invalid");
  return binding;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const args = process.argv.slice(2);
  if (args.length !== 4 || args[0] !== "--destination" || args[2] !== "--host") throw new Error("usage: proof_extension.mjs --destination DIR --host NAME");
  process.stdout.write(`${JSON.stringify(createProofExtension({ destination: path.resolve(args[1]), hostName: args[3] }))}\n`);
}
