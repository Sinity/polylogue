import { readFileSync, writeFileSync } from "node:fs";
const root = new URL("../", import.meta.url);
const dependency = JSON.parse(readFileSync(new URL("node_modules/js-sha256/package.json", root), "utf8"));
const declared = JSON.parse(readFileSync(new URL("package.json", root), "utf8")).dependencies["js-sha256"];
if (dependency.version !== declared) throw new Error("incremental_sha256_dependency_version_mismatch");
const source = readFileSync(new URL("node_modules/js-sha256/src/sha256.js", root), "utf8");
writeFileSync(new URL("src/vendor/sha256.js", root), `// Generated from locked js-sha256 ${declared}. See scripts/sync-sha256.mjs.\nconst module = { exports: {} };\nconst process = undefined;\nconst window = undefined;\nconst self = globalThis;\n${source}\nexport const createSha256 = module.exports.create;\n`);
writeFileSync(new URL("src/vendor/js-sha256-LICENSE.txt", root), readFileSync(new URL("node_modules/js-sha256/LICENSE.txt", root)));
// The tokenizer is an ordinary locked browser ESM dependency, copied with
// its relative module graph and license; no eval or runtime package loader.
const tokenizer = JSON.parse(readFileSync(new URL("node_modules/@streamparser/json/package.json", root), "utf8"));
const tokenizerDeclared = JSON.parse(readFileSync(new URL("package.json", root), "utf8")).dependencies["@streamparser/json"];
if (tokenizer.version !== tokenizerDeclared) throw new Error("json_tokenizer_dependency_version_mismatch");
const { cpSync, mkdirSync, readdirSync } = await import("node:fs");
const target = new URL("src/vendor/streamparser-json/", root);
mkdirSync(target, { recursive: true });
cpSync(new URL("node_modules/@streamparser/json/dist/mjs/", root), target, { recursive: true, filter: (source) => source.endsWith(".js") || !source.includes(".") });
writeFileSync(new URL("LICENSE", target), readFileSync(new URL("node_modules/@streamparser/json/LICENSE", root)));

// Keep generated JavaScript byte-reproducible while removing upstream trailing
// whitespace rejected by the repository diff checks and references to source maps
// that are not shipped. Runtime tokens are unchanged.
function normalizeGeneratedWhitespace(directory) {
  for (const entry of readdirSync(directory, { withFileTypes: true })) {
    const path = new URL(entry.name + (entry.isDirectory() ? "/" : ""), directory);
    if (entry.isDirectory()) normalizeGeneratedWhitespace(path);
    else if (entry.name.endsWith(".js")) writeFileSync(path, readFileSync(path, "utf8").replace(/^[ \t]*\/\/# sourceMappingURL=.*(?:\n|$)/gm, "").replace(/[ \t]+$/gm, ""));
  }
}
normalizeGeneratedWhitespace(target);
