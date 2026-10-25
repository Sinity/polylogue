function compareJsonKeys(left, right) {
  // Checkpoint encoding orders Unicode scalar values, whereas JavaScript
  // string comparison orders UTF-16 code units. Keep supplementary keys equal
  // across checkpoint exports.
  let leftIndex = 0; let rightIndex = 0;
  while (leftIndex < left.length && rightIndex < right.length) {
    const leftPoint = left.codePointAt(leftIndex); const rightPoint = right.codePointAt(rightIndex);
    if (leftPoint !== rightPoint) return leftPoint - rightPoint;
    leftIndex += leftPoint > 0xFFFF ? 2 : 1; rightIndex += rightPoint > 0xFFFF ? 2 : 1;
  }
  return leftIndex < left.length ? 1 : rightIndex < right.length ? -1 : 0;
}

export function sortedJson(value) {
  if (value === null || typeof value === "boolean") return JSON.stringify(value);
  if (typeof value === "string") return JSON.stringify(value.normalize("NFC"));
  if (typeof value === "number") {
    if (!Number.isSafeInteger(value)) throw new Error("capture_job_unsafe_number");
    return String(value);
  }
  if (Array.isArray(value)) return `[${value.map(sortedJson).join(",")}]`;
  if (!value || typeof value !== "object") throw new Error("capture_job_unsupported_json");
  const entries = Object.keys(value)
    .map((key) => [key.normalize("NFC"), value[key]])
    .sort(([left], [right]) => compareJsonKeys(left, right));
  if (entries.some(([key], index) => index > 0 && entries[index - 1][0] === key)) {
    throw new Error("capture_job_key_collision");
  }
  return `{${entries.map(([key, entry]) => `${JSON.stringify(key)}:${sortedJson(entry)}`).join(",")}}`;
}

async function hmac(token, message) {
  const key = await crypto.subtle.importKey("raw", new TextEncoder().encode(token), { name: "HMAC", hash: "SHA-256" }, false, ["sign"]);
  const bytes = new Uint8Array(await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(message)));
  return btoa(String.fromCharCode(...bytes)).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

export async function deriveAccountScope(scopeNamespace, provider, accountHandle) {
  if (!/^[a-z][a-z0-9_.-]{0,63}$/.test(provider) || !String(accountHandle).trim()) throw new Error("capture_job_invalid_scope_input");
  return `h1:${await hmac(scopeNamespace, `polylogue:account-scope:v1\0${provider}\0${String(accountHandle).normalize("NFKC").trim()}`)}`;
}
