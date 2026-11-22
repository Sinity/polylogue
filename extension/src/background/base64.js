// Bytes to and from base64 text, as message frames carry them.

/** `bytes` (a Uint8Array) as base64. */
export function encode(bytes) {
  let text = "";
  // In pieces: one call cannot take a whole file's bytes as arguments.
  for (let at = 0; at < bytes.length; at += 0x8000) text += String.fromCharCode(...bytes.subarray(at, at + 0x8000));
  return btoa(text);
}

/** The bytes of base64 `data`. */
export function decode(data) {
  return Uint8Array.from(atob(data), (c) => c.charCodeAt(0));
}
