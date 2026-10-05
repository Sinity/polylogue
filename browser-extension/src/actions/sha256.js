// Incremental SHA-256, FIPS 180-4 section 6.2.2. Retains one block only.
const primes = [];
for (let candidate = 2; primes.length < 64; candidate += 1) {
  if (primes.every((prime) => candidate % prime !== 0)) primes.push(candidate);
}
const constants = primes.map((prime) => Math.floor((Math.cbrt(prime) % 1) * 0x100000000) >>> 0);
const initial = primes.slice(0, 8).map((prime) => Math.floor((Math.sqrt(prime) % 1) * 0x100000000) >>> 0);
const rotate = (value, count) => (value >>> count) | (value << (32 - count));

export class AttachmentSha256 {
  constructor() {
    this.state = initial.slice();
    this.block = new Uint8Array(64);
    this.words = new Uint32Array(64);
    this.used = 0;
    this.size = 0;
    this.finished = false;
  }

  compress(bytes, offset = 0) {
    const words = this.words;
    for (let i = 0; i < 16; i += 1) {
      const at = offset + i * 4;
      words[i] = (bytes[at] << 24) | (bytes[at + 1] << 16) | (bytes[at + 2] << 8) | bytes[at + 3];
    }
    for (let i = 16; i < 64; i += 1) {
      const x = words[i - 15], y = words[i - 2];
      words[i] = words[i - 16] + (rotate(x, 7) ^ rotate(x, 18) ^ (x >>> 3))
        + words[i - 7] + (rotate(y, 17) ^ rotate(y, 19) ^ (y >>> 10));
    }
    let [a, b, c, d, e, f, g, h] = this.state;
    for (let i = 0; i < 64; i += 1) {
      const t1 = (h + (rotate(e, 6) ^ rotate(e, 11) ^ rotate(e, 25))
        + ((e & f) ^ (~e & g)) + constants[i] + words[i]) >>> 0;
      const t2 = ((rotate(a, 2) ^ rotate(a, 13) ^ rotate(a, 22))
        + ((a & b) ^ (a & c) ^ (b & c))) >>> 0;
      [h, g, f, e, d, c, b, a] = [g, f, e, (d + t1) >>> 0, c, b, a, (t1 + t2) >>> 0];
    }
    [a, b, c, d, e, f, g, h].forEach((value, i) => { this.state[i] = (this.state[i] + value) >>> 0; });
  }

  update(bytes) {
    if (this.finished) throw new Error("protocol_attachment_hash_finished");
    this.size += bytes.byteLength;
    if (!Number.isSafeInteger(this.size)) throw new Error("protocol_attachment_size_unrepresentable");
    let offset = 0;
    if (this.used) {
      const count = Math.min(64 - this.used, bytes.byteLength);
      this.block.set(bytes.subarray(0, count), this.used);
      this.used += count;
      offset += count;
      if (this.used === 64) { this.compress(this.block); this.used = 0; }
    }
    while (offset + 64 <= bytes.byteLength) { this.compress(bytes, offset); offset += 64; }
    if (offset < bytes.byteLength) {
      this.block.set(bytes.subarray(offset), this.used);
      this.used += bytes.byteLength - offset;
    }
  }

  digestHex() {
    if (this.finished) throw new Error("protocol_attachment_hash_finished");
    this.finished = true;
    this.block[this.used++] = 0x80;
    this.block.fill(0, this.used);
    if (this.used > 56) { this.compress(this.block); this.block.fill(0); }
    const view = new DataView(this.block.buffer);
    view.setUint32(56, Math.floor(this.size / 0x20000000));
    view.setUint32(60, (this.size % 0x20000000) * 8);
    this.compress(this.block);
    return this.state.map((value) => value.toString(16).padStart(8, "0")).join("");
  }
}
