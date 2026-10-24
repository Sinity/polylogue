import { createSha256 } from "../vendor/sha256.js";

export class AttachmentSha256 {
  constructor() {
    this.hash = createSha256();
    this.size = 0;
    this.finished = false;
  }

  update(bytes) {
    if (this.finished) throw new Error("protocol_attachment_hash_finished");
    this.size += bytes.byteLength;
    if (!Number.isSafeInteger(this.size)) throw new Error("protocol_attachment_size_unrepresentable");
    this.hash.update(bytes);
  }

  digestHex() {
    if (this.finished) throw new Error("protocol_attachment_hash_finished");
    this.finished = true;
    return this.hash.hex();
  }
}
