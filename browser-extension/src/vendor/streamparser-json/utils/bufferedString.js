/**
 * The accumulators that the tokenizer gathers strings and numbers into while
 * their bytes arrive.
 *
 * @module
 */
/**
 * A {@linkcode StringBuilder} that accumulates the token as a JavaScript
 * string. This is the default: it's the fastest option for the small strings
 * and numbers that dominate real JSON.
 */
export class NonBufferedString {
    constructor() {
        // fatal: true makes invalid byte sequences (e.g. a lead byte followed by a
        // non-continuation byte) throw instead of silently decoding to U+FFFD.
        this.decoder = new TextDecoder("utf-8", { fatal: true });
        // Pieces appended since the last toString(), not yet folded into `string`.
        this.pending = [];
        this.string = "";
        this.byteLength = 0;
    }
    appendChar(char) {
        this.pending.push(String.fromCharCode(char));
        this.byteLength += 1;
    }
    appendBuf(buf, start = 0, end = buf.length) {
        this.pending.push(this.decoder.decode(buf.subarray(start, end)));
        this.byteLength += end - start;
    }
    appendCharCode(code) {
        this.pending.push(String.fromCharCode(code));
    }
    reset() {
        this.pending = [];
        this.string = "";
        this.byteLength = 0;
    }
    // Folds only the pieces appended since the last call into `string`, so
    // repeated calls (one per chunk when emitting partial tokens) stay linear
    // overall instead of re-joining the whole accumulated string every time.
    toString() {
        if (this.pending.length > 0) {
            this.string += this.pending.join("");
            this.pending = [];
        }
        return this.string;
    }
}
/**
 * A {@linkcode StringBuilder} that accumulates the token's bytes into a
 * fixed-size `Uint8Array` and only decodes them once the buffer is full.
 *
 * Enabled through the tokenizer's `stringBufferSize`/`numberBufferSize`
 * options. It avoids V8's over-allocation on repeated string concatenation,
 * which is what makes very large strings and numbers exhaust memory, at the
 * cost of an encoding/decoding round trip that isn't worth it for small values.
 */
export class BufferedString {
    /**
     * @param bufferSize The size, in bytes, of the buffer to accumulate into.
     */
    constructor(bufferSize) {
        // fatal: true makes invalid byte sequences (e.g. a lead byte followed by a
        // non-continuation byte) throw instead of silently decoding to U+FFFD.
        this.decoder = new TextDecoder("utf-8", { fatal: true });
        this.bufferOffset = 0;
        this.string = "";
        this.byteLength = 0;
        this.buffer = new Uint8Array(bufferSize);
    }
    appendChar(char) {
        if (this.bufferOffset >= this.buffer.length)
            this.flushStringBuffer();
        this.buffer[this.bufferOffset++] = char;
        this.byteLength += 1;
    }
    appendBuf(buf, start = 0, end = buf.length) {
        const size = end - start;
        if (this.bufferOffset + size > this.buffer.length)
            this.flushStringBuffer();
        if (size > this.buffer.length) {
            // Span larger than the working buffer: decode it straight into the
            // string instead of copying it in (buffer.set would overflow). Safe
            // because callers only append complete-character spans -- the tokenizer
            // never splits a multi-byte char across appendBuf calls -- so decoding
            // this span on its own can't cut through the middle of a character.
            this.string += this.decoder.decode(buf.subarray(start, end));
            this.byteLength += size;
            return;
        }
        this.buffer.set(buf.subarray(start, end), this.bufferOffset);
        this.bufferOffset += size;
        this.byteLength += size;
    }
    appendCharCode(code) {
        this.flushStringBuffer();
        this.string += String.fromCharCode(code);
    }
    flushStringBuffer() {
        this.string += this.decoder.decode(this.buffer.subarray(0, this.bufferOffset));
        this.bufferOffset = 0;
    }
    reset() {
        this.string = "";
        this.bufferOffset = 0;
        this.byteLength = 0;
    }
    toString() {
        this.flushStringBuffer();
        return this.string;
    }
}
