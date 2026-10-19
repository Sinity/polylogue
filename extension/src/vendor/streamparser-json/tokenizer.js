/**
 * A JSON-compliant tokenizer that turns a utf-8 stream into JSON tokens.
 *
 * @example
 * ```ts
 * import Tokenizer from "@streamparser/json/tokenizer.js";
 *
 * const tokenizer = new Tokenizer();
 * tokenizer.onToken = ({ token, value, offset }) => {
 *   // process the token
 * };
 *
 * tokenizer.write('{ "test": ["a"] }');
 * ```
 *
 * @module
 */
import { BufferedString, NonBufferedString, } from "./utils/bufferedString.js";
import TokenType from "./utils/types/tokenType.js";
import { escapedSequences } from "./utils/utf-8.js";
// Tokenizer States
var TokenizerStates;
(function (TokenizerStates) {
    TokenizerStates[TokenizerStates["START"] = 0] = "START";
    TokenizerStates[TokenizerStates["ENDED"] = 1] = "ENDED";
    TokenizerStates[TokenizerStates["ERROR"] = 2] = "ERROR";
    TokenizerStates[TokenizerStates["TRUE1"] = 3] = "TRUE1";
    TokenizerStates[TokenizerStates["TRUE2"] = 4] = "TRUE2";
    TokenizerStates[TokenizerStates["TRUE3"] = 5] = "TRUE3";
    TokenizerStates[TokenizerStates["FALSE1"] = 6] = "FALSE1";
    TokenizerStates[TokenizerStates["FALSE2"] = 7] = "FALSE2";
    TokenizerStates[TokenizerStates["FALSE3"] = 8] = "FALSE3";
    TokenizerStates[TokenizerStates["FALSE4"] = 9] = "FALSE4";
    TokenizerStates[TokenizerStates["NULL1"] = 10] = "NULL1";
    TokenizerStates[TokenizerStates["NULL2"] = 11] = "NULL2";
    TokenizerStates[TokenizerStates["NULL3"] = 12] = "NULL3";
    TokenizerStates[TokenizerStates["STRING_DEFAULT"] = 13] = "STRING_DEFAULT";
    TokenizerStates[TokenizerStates["STRING_AFTER_BACKSLASH"] = 14] = "STRING_AFTER_BACKSLASH";
    TokenizerStates[TokenizerStates["STRING_UNICODE_DIGIT_1"] = 15] = "STRING_UNICODE_DIGIT_1";
    TokenizerStates[TokenizerStates["STRING_UNICODE_DIGIT_2"] = 16] = "STRING_UNICODE_DIGIT_2";
    TokenizerStates[TokenizerStates["STRING_UNICODE_DIGIT_3"] = 17] = "STRING_UNICODE_DIGIT_3";
    TokenizerStates[TokenizerStates["STRING_UNICODE_DIGIT_4"] = 18] = "STRING_UNICODE_DIGIT_4";
    TokenizerStates[TokenizerStates["STRING_INCOMPLETE_CHAR"] = 19] = "STRING_INCOMPLETE_CHAR";
    TokenizerStates[TokenizerStates["NUMBER_AFTER_INITIAL_MINUS"] = 20] = "NUMBER_AFTER_INITIAL_MINUS";
    TokenizerStates[TokenizerStates["NUMBER_AFTER_INITIAL_ZERO"] = 21] = "NUMBER_AFTER_INITIAL_ZERO";
    TokenizerStates[TokenizerStates["NUMBER_AFTER_INITIAL_NON_ZERO"] = 22] = "NUMBER_AFTER_INITIAL_NON_ZERO";
    TokenizerStates[TokenizerStates["NUMBER_AFTER_FULL_STOP"] = 23] = "NUMBER_AFTER_FULL_STOP";
    TokenizerStates[TokenizerStates["NUMBER_AFTER_DECIMAL"] = 24] = "NUMBER_AFTER_DECIMAL";
    TokenizerStates[TokenizerStates["NUMBER_AFTER_E"] = 25] = "NUMBER_AFTER_E";
    TokenizerStates[TokenizerStates["NUMBER_AFTER_E_AND_SIGN"] = 26] = "NUMBER_AFTER_E_AND_SIGN";
    TokenizerStates[TokenizerStates["NUMBER_AFTER_E_AND_DIGIT"] = 27] = "NUMBER_AFTER_E_AND_DIGIT";
    TokenizerStates[TokenizerStates["SEPARATOR"] = 28] = "SEPARATOR";
    TokenizerStates[TokenizerStates["BOM_OR_START"] = 29] = "BOM_OR_START";
    TokenizerStates[TokenizerStates["BOM"] = 30] = "BOM";
})(TokenizerStates || (TokenizerStates = {}));
function TokenizerStateToString(tokenizerState) {
    return [
        "START",
        "ENDED",
        "ERROR",
        "TRUE1",
        "TRUE2",
        "TRUE3",
        "FALSE1",
        "FALSE2",
        "FALSE3",
        "FALSE4",
        "NULL1",
        "NULL2",
        "NULL3",
        "STRING_DEFAULT",
        "STRING_AFTER_BACKSLASH",
        "STRING_UNICODE_DIGIT_1",
        "STRING_UNICODE_DIGIT_2",
        "STRING_UNICODE_DIGIT_3",
        "STRING_UNICODE_DIGIT_4",
        "STRING_INCOMPLETE_CHAR",
        "NUMBER_AFTER_INITIAL_MINUS",
        "NUMBER_AFTER_INITIAL_ZERO",
        "NUMBER_AFTER_INITIAL_NON_ZERO",
        "NUMBER_AFTER_FULL_STOP",
        "NUMBER_AFTER_DECIMAL",
        "NUMBER_AFTER_E",
        "NUMBER_AFTER_E_AND_SIGN",
        "NUMBER_AFTER_E_AND_DIGIT",
        "SEPARATOR",
        "BOM_OR_START",
        "BOM",
    ][tokenizerState];
}
const defaultOpts = {
    stringBufferSize: 0,
    numberBufferSize: 0,
    separator: undefined,
    emitPartialTokens: false,
};
/** The error thrown when the tokenizer is misconfigured or hits invalid JSON. */
export class TokenizerError extends Error {
    /**
     * @param message What went wrong.
     */
    constructor(message) {
        super(message);
        // Typescript is broken. This is a workaround
        Object.setPrototypeOf(this, TokenizerError.prototype);
    }
}
// A non-integer buffer size (e.g. 0.5) silently truncates when passed to
// `new Uint8Array(size)` (0.5 becomes a *zero-length* buffer) instead of
// throwing, so every appended byte gets silently dropped rather than
// buffered -- corrupting the parsed value instead of failing loudly.
function validateBufferSize(name, size) {
    if (size === undefined)
        return;
    if (!Number.isInteger(size) || size < 0) {
        throw new TokenizerError(`Invalid "${name}": ${size}. Expected a non-negative integer.`);
    }
}
// Byte length of the UTF-8 character starting with `leadByte`. Invalid or
// continuation lead bytes fall through to 3/4 here and are rejected later by
// the fatal TextDecoder when the bytes are actually decoded.
function utf8SequenceLength(leadByte) {
    if (leadByte >= 194 && leadByte <= 223)
        return 2;
    if (leadByte <= 239)
        return 3;
    return 4;
}
// Index just past the last COMPLETE multi-byte character of the run starting at
// `start`. Stops at the first ASCII byte, or at a character whose bytes would
// run past the end of the buffer (a boundary split the caller carries over).
function multiByteRunEnd(buffer, start) {
    let j = start;
    while (j < buffer.length && buffer[j] >= 128) {
        const seqLength = utf8SequenceLength(buffer[j]);
        if (j + seqLength > buffer.length)
            break; // split across the chunk boundary
        j += seqLength;
    }
    return j;
}
/**
 * A JSON-compliant tokenizer that turns a utf-8 stream into JSON tokens.
 *
 * Data is pushed in with {@linkcode Tokenizer.write} and the resulting tokens
 * come back through the {@linkcode Tokenizer.onToken} callback, which the user
 * is expected to override. Feed the tokens to a `TokenParser` to get JSON
 * values back, or use a `JSONParser`, which chains both.
 *
 * @example
 * ```ts
 * import Tokenizer from "@streamparser/json/tokenizer.js";
 *
 * const tokenizer = new Tokenizer({ separator: "\n" });
 * tokenizer.onToken = ({ token, value, offset }) => {
 *   // process the token
 * };
 * tokenizer.onError = (err) => console.error(err);
 *
 * tokenizer.write('{ "test": ["a"] }');
 * tokenizer.end();
 * ```
 */
export default class Tokenizer {
    /**
     * @param opts How to tokenize. See {@linkcode TokenizerOptions}.
     */
    constructor(opts) {
        this.state = 29 /* TokenizerStates.BOM_OR_START */;
        this.bomIndex = 0;
        this.separatorIndex = 0;
        this.escapedCharsByteLength = 0;
        this.bytes_remaining = 0; // number of bytes remaining in multi byte utf8 char to read after split boundary
        this.bytes_in_sequence = 0; // bytes in multi byte utf8 char to read
        this.char_split_buffer = new Uint8Array(4); // for rebuilding chars split before boundary is reached
        this.encoder = new TextEncoder();
        this.offset = -1;
        this.streamByteLength = 0; // Total bytes consumed across all write() calls before the current one
        opts = Object.assign(Object.assign({}, defaultOpts), opts);
        validateBufferSize("stringBufferSize", opts.stringBufferSize);
        validateBufferSize("numberBufferSize", opts.numberBufferSize);
        this.emitPartialTokens = opts.emitPartialTokens === true;
        this.bufferedString =
            opts.stringBufferSize && opts.stringBufferSize > 4
                ? new BufferedString(opts.stringBufferSize)
                : new NonBufferedString();
        this.bufferedNumber =
            opts.numberBufferSize && opts.numberBufferSize > 0
                ? new BufferedString(opts.numberBufferSize)
                : new NonBufferedString();
        this.separator = opts.separator;
        this.separatorBytes = opts.separator
            ? this.encoder.encode(opts.separator)
            : undefined;
    }
    /** Whether the tokenizer is ended, and thus no longer accepting data. */
    get isEnded() {
        return this.state === 1 /* TokenizerStates.ENDED */;
    }
    // Appends the code unit decoded from one \uXXXX escape, matching
    // JSON.parse's handling of surrogates: a valid high/low surrogate pair
    // combines into one character; an unpaired high or low surrogate is kept
    // as a raw UTF-16 code unit rather than replaced or dropped (JS strings
    // are free to contain lone surrogates; only encoding them as UTF-8 bytes
    // is lossy, which is why appendCharCode -- not the encoder -- is used for
    // them).
    appendUnicodeCodeUnit(intVal) {
        if (this.highSurrogate !== undefined) {
            if (intVal >= 0xdc00 && intVal <= 0xdfff) {
                // <56320,57343> - valid low surrogate: combine with the pending
                // high surrogate into a single character.
                const unicodeString = String.fromCharCode(this.highSurrogate, intVal);
                const unicodeBuffer = this.encoder.encode(unicodeString);
                this.bufferedString.appendBuf(unicodeBuffer);
                // len(\u0000)=6 minus the fact you're appending len(buf)
                this.escapedCharsByteLength += 6 - unicodeBuffer.byteLength;
                this.highSurrogate = undefined;
                return;
            }
            // Not a matching low surrogate: the pending high surrogate stands on
            // its own, and intVal is processed independently below.
            this.flushPendingHighSurrogate();
        }
        if (intVal >= 0xd800 && intVal <= 0xdbff) {
            // <55296,56319> - high surrogate: defer until we know whether a
            // matching low surrogate follows.
            this.highSurrogate = intVal;
            this.escapedCharsByteLength += 6;
            return;
        }
        if (intVal >= 0xdc00 && intVal <= 0xdfff) {
            // <56320,57343> - lone low surrogate with no preceding high
            // surrogate: keep as a raw code unit.
            this.bufferedString.appendCharCode(intVal);
            this.escapedCharsByteLength += 6;
            return;
        }
        const unicodeString = String.fromCharCode(intVal);
        const unicodeBuffer = this.encoder.encode(unicodeString);
        this.bufferedString.appendBuf(unicodeBuffer);
        // len(\u0000)=6 minus the fact you're appending len(buf)
        this.escapedCharsByteLength += 6 - unicodeBuffer.byteLength;
    }
    flushPendingHighSurrogate() {
        if (this.highSurrogate !== undefined) {
            this.bufferedString.appendCharCode(this.highSurrogate);
            this.highSurrogate = undefined;
        }
    }
    // Stash the leading bytes of a multi-byte character split across the chunk
    // boundary; STRING_INCOMPLETE_CHAR completes it from the next chunk.
    startIncompleteChar(buffer, start) {
        this.bytes_in_sequence = utf8SequenceLength(buffer[start]);
        this.bytes_remaining = start + this.bytes_in_sequence - buffer.length;
        this.char_split_buffer.set(buffer.subarray(start));
        this.state = 19 /* TokenizerStates.STRING_INCOMPLETE_CHAR */;
    }
    /**
     * Pushes the next chunk of the JSON stream into the tokenizer.
     *
     * Tokenizing happens synchronously, so every token in `input` is emitted
     * through {@linkcode Tokenizer.onToken} before this returns. A chunk may end
     * anywhere, including in the middle of a multi-byte character; the rest of it
     * is picked up from the next chunk.
     *
     * @param input The chunk to tokenize: a string, a `TypedArray`, or any
     * iterable of utf-8 byte values.
     * @throws {TokenizerError} If the data is not valid JSON and no
     * {@linkcode Tokenizer.onError} callback has been set.
     */
    write(input) {
        try {
            let buffer;
            if (input instanceof Uint8Array) {
                buffer = input;
            }
            else if (typeof input === "string") {
                if (this.pendingStringSurrogate !== undefined) {
                    input = this.pendingStringSurrogate + input;
                    this.pendingStringSurrogate = undefined;
                }
                const lastCharCode = input.charCodeAt(input.length - 1);
                if (lastCharCode >= 0xd800 && lastCharCode <= 0xdbff) {
                    // Lone high surrogate at the very end of this chunk: hold it back
                    // instead of encoding it (and corrupting it into U+FFFD) alone,
                    // in case the next chunk supplies its matching low surrogate.
                    this.pendingStringSurrogate = input[input.length - 1];
                    input = input.slice(0, -1);
                }
                buffer = this.encoder.encode(input);
            }
            else if (ArrayBuffer.isView(input)) {
                buffer = new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
            }
            else if (input !== null &&
                typeof input === "object" &&
                typeof input[Symbol.iterator] === "function") {
                // Any Iterable<number>, not just literal Arrays (e.g. Set, Map
                // values(), a generator) -- matching the public write() signature,
                // which already types `input` as Iterable<number> | string.
                buffer = Uint8Array.from(input);
            }
            else {
                throw new TypeError("Unexpected type. The `write` function only accepts Iterables (e.g. Arrays, Sets, Generators), TypedArrays and Strings.");
            }
            for (let i = 0; i < buffer.length; i += 1) {
                const n = buffer[i]; // get current byte from buffer
                switch (this.state) {
                    // @ts-expect-error fall through case
                    case 29 /* TokenizerStates.BOM_OR_START */:
                        if (n === 0xef) {
                            this.bom = [0xef, 0xbb, 0xbf];
                            this.bomIndex += 1;
                            this.state = 30 /* TokenizerStates.BOM */;
                            continue;
                        }
                        if (input instanceof Uint16Array) {
                            if (n === 0xfe) {
                                this.bom = [0xfe, 0xff];
                                this.bomIndex += 1;
                                this.state = 30 /* TokenizerStates.BOM */;
                                continue;
                            }
                            if (n === 0xff) {
                                this.bom = [0xff, 0xfe];
                                this.bomIndex += 1;
                                this.state = 30 /* TokenizerStates.BOM */;
                                continue;
                            }
                        }
                        if (input instanceof Uint32Array) {
                            if (n === 0x00) {
                                this.bom = [0x00, 0x00, 0xfe, 0xff];
                                this.bomIndex += 1;
                                this.state = 30 /* TokenizerStates.BOM */;
                                continue;
                            }
                            if (n === 0xff) {
                                this.bom = [0xff, 0xfe, 0x00, 0x00];
                                this.bomIndex += 1;
                                this.state = 30 /* TokenizerStates.BOM */;
                                continue;
                            }
                        }
                    case 0 /* TokenizerStates.START */:
                        this.offset += 1;
                        if (this.separatorBytes && n === this.separatorBytes[0]) {
                            if (this.separatorBytes.length === 1) {
                                this.state = 0 /* TokenizerStates.START */;
                                this.onToken({
                                    token: TokenType.SEPARATOR,
                                    value: this.separator,
                                    offset: this.offset + this.separatorBytes.length - 1,
                                });
                                continue;
                            }
                            this.state = 28 /* TokenizerStates.SEPARATOR */;
                            continue;
                        }
                        if (n === 32 /* charset.SPACE */ ||
                            n === 10 /* charset.NEWLINE */ ||
                            n === 13 /* charset.CARRIAGE_RETURN */ ||
                            n === 9 /* charset.TAB */) {
                            // whitespace
                            continue;
                        }
                        if (n === 123 /* charset.LEFT_CURLY_BRACKET */) {
                            this.onToken({
                                token: TokenType.LEFT_BRACE,
                                value: "{",
                                offset: this.offset,
                            });
                            continue;
                        }
                        if (n === 125 /* charset.RIGHT_CURLY_BRACKET */) {
                            this.onToken({
                                token: TokenType.RIGHT_BRACE,
                                value: "}",
                                offset: this.offset,
                            });
                            continue;
                        }
                        if (n === 91 /* charset.LEFT_SQUARE_BRACKET */) {
                            this.onToken({
                                token: TokenType.LEFT_BRACKET,
                                value: "[",
                                offset: this.offset,
                            });
                            continue;
                        }
                        if (n === 93 /* charset.RIGHT_SQUARE_BRACKET */) {
                            this.onToken({
                                token: TokenType.RIGHT_BRACKET,
                                value: "]",
                                offset: this.offset,
                            });
                            continue;
                        }
                        if (n === 58 /* charset.COLON */) {
                            this.onToken({
                                token: TokenType.COLON,
                                value: ":",
                                offset: this.offset,
                            });
                            continue;
                        }
                        if (n === 44 /* charset.COMMA */) {
                            this.onToken({
                                token: TokenType.COMMA,
                                value: ",",
                                offset: this.offset,
                            });
                            continue;
                        }
                        if (n === 116 /* charset.LATIN_SMALL_LETTER_T */) {
                            this.state = 3 /* TokenizerStates.TRUE1 */;
                            continue;
                        }
                        if (n === 102 /* charset.LATIN_SMALL_LETTER_F */) {
                            this.state = 6 /* TokenizerStates.FALSE1 */;
                            continue;
                        }
                        if (n === 110 /* charset.LATIN_SMALL_LETTER_N */) {
                            this.state = 10 /* TokenizerStates.NULL1 */;
                            continue;
                        }
                        if (n === 34 /* charset.QUOTATION_MARK */) {
                            this.bufferedString.reset();
                            this.escapedCharsByteLength = 0;
                            this.state = 13 /* TokenizerStates.STRING_DEFAULT */;
                            continue;
                        }
                        if (n >= 49 /* charset.DIGIT_ONE */ && n <= 57 /* charset.DIGIT_NINE */) {
                            this.bufferedNumber.reset();
                            this.bufferedNumber.appendChar(n);
                            this.state = 22 /* TokenizerStates.NUMBER_AFTER_INITIAL_NON_ZERO */;
                            continue;
                        }
                        if (n === 48 /* charset.DIGIT_ZERO */) {
                            this.bufferedNumber.reset();
                            this.bufferedNumber.appendChar(n);
                            this.state = 21 /* TokenizerStates.NUMBER_AFTER_INITIAL_ZERO */;
                            continue;
                        }
                        if (n === 45 /* charset.HYPHEN_MINUS */) {
                            this.bufferedNumber.reset();
                            this.bufferedNumber.appendChar(n);
                            this.state = 20 /* TokenizerStates.NUMBER_AFTER_INITIAL_MINUS */;
                            continue;
                        }
                        break;
                    // STRING
                    case 13 /* TokenizerStates.STRING_DEFAULT */:
                        if (n === 34 /* charset.QUOTATION_MARK */) {
                            this.flushPendingHighSurrogate();
                            const string = this.bufferedString.toString();
                            this.state = 0 /* TokenizerStates.START */;
                            this.onToken({
                                token: TokenType.STRING,
                                value: string,
                                offset: this.offset,
                            });
                            this.offset +=
                                this.escapedCharsByteLength +
                                    this.bufferedString.byteLength +
                                    1;
                            continue;
                        }
                        if (n === 92 /* charset.REVERSE_SOLIDUS */) {
                            this.state = 14 /* TokenizerStates.STRING_AFTER_BACKSLASH */;
                            continue;
                        }
                        if (n >= 128) {
                            this.flushPendingHighSurrogate();
                            // Decode the whole run of complete multi-byte characters in one
                            // TextDecoder call, rather than one character at a time -- much
                            // faster for multi-byte text (CJK/emoji). ASCII stays on the
                            // per-character appendChar path below.
                            const runEnd = multiByteRunEnd(buffer, i);
                            if (runEnd > i) {
                                this.bufferedString.appendBuf(buffer, i, runEnd);
                                i = runEnd - 1; // the for-loop's i += 1 lands on runEnd
                            }
                            // A character straddling the chunk boundary is carried over to
                            // the next chunk via STRING_INCOMPLETE_CHAR.
                            if (runEnd < buffer.length && buffer[runEnd] >= 128) {
                                this.startIncompleteChar(buffer, runEnd);
                                i = buffer.length - 1;
                            }
                            continue;
                        }
                        if (n >= 32 /* charset.SPACE */) {
                            this.flushPendingHighSurrogate();
                            let j = i;
                            while (j < buffer.length) {
                                const b = buffer[j];
                                if (b < 32 /* charset.SPACE */ ||
                                    b >= 128 ||
                                    b === 34 /* charset.QUOTATION_MARK */ ||
                                    b === 92 /* charset.REVERSE_SOLIDUS */)
                                    break;
                                j += 1;
                            }
                            // appendBuf is one TextDecoder call: worth it only once the run is
                            // long enough to amortize that fixed cost. Short strings (keys,
                            // ids) dominate real JSON, so append those char-by-char instead --
                            // always-appendBuf regresses key/record-heavy JSON ~12%.
                            if (j - i >= 16) {
                                this.bufferedString.appendBuf(buffer, i, j);
                            }
                            else {
                                for (let k = i; k < j; k += 1)
                                    this.bufferedString.appendChar(buffer[k]);
                            }
                            i = j - 1;
                            continue;
                        }
                        break;
                    case 19 /* TokenizerStates.STRING_INCOMPLETE_CHAR */: {
                        // check for carry over of a multi byte char split between data chunks
                        // & fill temp buffer it with start of this data chunk up to the boundary limit set in the last iteration
                        // The rest of the sequence might still not be complete if this chunk is smaller
                        // than the number of bytes still missing (e.g. one byte at a time), so only
                        // consume what's actually available and keep waiting otherwise.
                        const available = Math.min(this.bytes_remaining, buffer.length - i);
                        this.char_split_buffer.set(buffer.subarray(i, i + available), this.bytes_in_sequence - this.bytes_remaining);
                        this.bytes_remaining -= available;
                        if (this.bytes_remaining > 0) {
                            i = buffer.length - 1;
                            continue;
                        }
                        this.bufferedString.appendBuf(this.char_split_buffer, 0, this.bytes_in_sequence);
                        i += available - 1;
                        this.state = 13 /* TokenizerStates.STRING_DEFAULT */;
                        continue;
                    }
                    case 14 /* TokenizerStates.STRING_AFTER_BACKSLASH */: {
                        const controlChar = escapedSequences[n];
                        if (controlChar) {
                            this.flushPendingHighSurrogate();
                            this.bufferedString.appendChar(controlChar);
                            this.escapedCharsByteLength += 1; // len(\")=2 minus the fact you're appending len(controlChar)=1
                            this.state = 13 /* TokenizerStates.STRING_DEFAULT */;
                            continue;
                        }
                        if (n === 117 /* charset.LATIN_SMALL_LETTER_U */) {
                            this.unicode = "";
                            this.state = 15 /* TokenizerStates.STRING_UNICODE_DIGIT_1 */;
                            continue;
                        }
                        break;
                    }
                    case 15 /* TokenizerStates.STRING_UNICODE_DIGIT_1 */:
                    case 16 /* TokenizerStates.STRING_UNICODE_DIGIT_2 */:
                    case 17 /* TokenizerStates.STRING_UNICODE_DIGIT_3 */:
                        if ((n >= 48 /* charset.DIGIT_ZERO */ && n <= 57 /* charset.DIGIT_NINE */) ||
                            (n >= 65 /* charset.LATIN_CAPITAL_LETTER_A */ &&
                                n <= 70 /* charset.LATIN_CAPITAL_LETTER_F */) ||
                            (n >= 97 /* charset.LATIN_SMALL_LETTER_A */ &&
                                n <= 102 /* charset.LATIN_SMALL_LETTER_F */)) {
                            this.unicode += String.fromCharCode(n);
                            this.state += 1;
                            continue;
                        }
                        break;
                    case 18 /* TokenizerStates.STRING_UNICODE_DIGIT_4 */:
                        if ((n >= 48 /* charset.DIGIT_ZERO */ && n <= 57 /* charset.DIGIT_NINE */) ||
                            (n >= 65 /* charset.LATIN_CAPITAL_LETTER_A */ &&
                                n <= 70 /* charset.LATIN_CAPITAL_LETTER_F */) ||
                            (n >= 97 /* charset.LATIN_SMALL_LETTER_A */ &&
                                n <= 102 /* charset.LATIN_SMALL_LETTER_F */)) {
                            const intVal = parseInt(this.unicode + String.fromCharCode(n), 16);
                            this.appendUnicodeCodeUnit(intVal);
                            this.state = 13 /* TokenizerStates.STRING_DEFAULT */;
                            continue;
                        }
                        break;
                    // Number
                    case 20 /* TokenizerStates.NUMBER_AFTER_INITIAL_MINUS */:
                        if (n === 48 /* charset.DIGIT_ZERO */) {
                            this.bufferedNumber.appendChar(n);
                            this.state = 21 /* TokenizerStates.NUMBER_AFTER_INITIAL_ZERO */;
                            continue;
                        }
                        if (n >= 49 /* charset.DIGIT_ONE */ && n <= 57 /* charset.DIGIT_NINE */) {
                            this.bufferedNumber.appendChar(n);
                            this.state = 22 /* TokenizerStates.NUMBER_AFTER_INITIAL_NON_ZERO */;
                            continue;
                        }
                        break;
                    case 21 /* TokenizerStates.NUMBER_AFTER_INITIAL_ZERO */:
                        if (n === 46 /* charset.FULL_STOP */) {
                            this.bufferedNumber.appendChar(n);
                            this.state = 23 /* TokenizerStates.NUMBER_AFTER_FULL_STOP */;
                            continue;
                        }
                        if (n === 101 /* charset.LATIN_SMALL_LETTER_E */ ||
                            n === 69 /* charset.LATIN_CAPITAL_LETTER_E */) {
                            this.bufferedNumber.appendChar(n);
                            this.state = 25 /* TokenizerStates.NUMBER_AFTER_E */;
                            continue;
                        }
                        i -= 1;
                        this.state = 0 /* TokenizerStates.START */;
                        this.emitNumber();
                        continue;
                    case 22 /* TokenizerStates.NUMBER_AFTER_INITIAL_NON_ZERO */:
                        if (n >= 48 /* charset.DIGIT_ZERO */ && n <= 57 /* charset.DIGIT_NINE */) {
                            this.bufferedNumber.appendChar(n);
                            continue;
                        }
                        if (n === 46 /* charset.FULL_STOP */) {
                            this.bufferedNumber.appendChar(n);
                            this.state = 23 /* TokenizerStates.NUMBER_AFTER_FULL_STOP */;
                            continue;
                        }
                        if (n === 101 /* charset.LATIN_SMALL_LETTER_E */ ||
                            n === 69 /* charset.LATIN_CAPITAL_LETTER_E */) {
                            this.bufferedNumber.appendChar(n);
                            this.state = 25 /* TokenizerStates.NUMBER_AFTER_E */;
                            continue;
                        }
                        i -= 1;
                        this.state = 0 /* TokenizerStates.START */;
                        this.emitNumber();
                        continue;
                    case 23 /* TokenizerStates.NUMBER_AFTER_FULL_STOP */:
                        if (n >= 48 /* charset.DIGIT_ZERO */ && n <= 57 /* charset.DIGIT_NINE */) {
                            this.bufferedNumber.appendChar(n);
                            this.state = 24 /* TokenizerStates.NUMBER_AFTER_DECIMAL */;
                            continue;
                        }
                        break;
                    case 24 /* TokenizerStates.NUMBER_AFTER_DECIMAL */:
                        if (n >= 48 /* charset.DIGIT_ZERO */ && n <= 57 /* charset.DIGIT_NINE */) {
                            this.bufferedNumber.appendChar(n);
                            continue;
                        }
                        if (n === 101 /* charset.LATIN_SMALL_LETTER_E */ ||
                            n === 69 /* charset.LATIN_CAPITAL_LETTER_E */) {
                            this.bufferedNumber.appendChar(n);
                            this.state = 25 /* TokenizerStates.NUMBER_AFTER_E */;
                            continue;
                        }
                        i -= 1;
                        this.state = 0 /* TokenizerStates.START */;
                        this.emitNumber();
                        continue;
                    // @ts-expect-error fall through case
                    case 25 /* TokenizerStates.NUMBER_AFTER_E */:
                        if (n === 43 /* charset.PLUS_SIGN */ || n === 45 /* charset.HYPHEN_MINUS */) {
                            this.bufferedNumber.appendChar(n);
                            this.state = 26 /* TokenizerStates.NUMBER_AFTER_E_AND_SIGN */;
                            continue;
                        }
                    case 26 /* TokenizerStates.NUMBER_AFTER_E_AND_SIGN */:
                        if (n >= 48 /* charset.DIGIT_ZERO */ && n <= 57 /* charset.DIGIT_NINE */) {
                            this.bufferedNumber.appendChar(n);
                            this.state = 27 /* TokenizerStates.NUMBER_AFTER_E_AND_DIGIT */;
                            continue;
                        }
                        break;
                    case 27 /* TokenizerStates.NUMBER_AFTER_E_AND_DIGIT */:
                        if (n >= 48 /* charset.DIGIT_ZERO */ && n <= 57 /* charset.DIGIT_NINE */) {
                            this.bufferedNumber.appendChar(n);
                            continue;
                        }
                        i -= 1;
                        this.state = 0 /* TokenizerStates.START */;
                        this.emitNumber();
                        continue;
                    // TRUE
                    case 3 /* TokenizerStates.TRUE1 */:
                        if (n === 114 /* charset.LATIN_SMALL_LETTER_R */) {
                            this.state = 4 /* TokenizerStates.TRUE2 */;
                            continue;
                        }
                        break;
                    case 4 /* TokenizerStates.TRUE2 */:
                        if (n === 117 /* charset.LATIN_SMALL_LETTER_U */) {
                            this.state = 5 /* TokenizerStates.TRUE3 */;
                            continue;
                        }
                        break;
                    case 5 /* TokenizerStates.TRUE3 */:
                        if (n === 101 /* charset.LATIN_SMALL_LETTER_E */) {
                            this.state = 0 /* TokenizerStates.START */;
                            this.onToken({
                                token: TokenType.TRUE,
                                value: true,
                                offset: this.offset,
                            });
                            this.offset += 3;
                            continue;
                        }
                        break;
                    // FALSE
                    case 6 /* TokenizerStates.FALSE1 */:
                        if (n === 97 /* charset.LATIN_SMALL_LETTER_A */) {
                            this.state = 7 /* TokenizerStates.FALSE2 */;
                            continue;
                        }
                        break;
                    case 7 /* TokenizerStates.FALSE2 */:
                        if (n === 108 /* charset.LATIN_SMALL_LETTER_L */) {
                            this.state = 8 /* TokenizerStates.FALSE3 */;
                            continue;
                        }
                        break;
                    case 8 /* TokenizerStates.FALSE3 */:
                        if (n === 115 /* charset.LATIN_SMALL_LETTER_S */) {
                            this.state = 9 /* TokenizerStates.FALSE4 */;
                            continue;
                        }
                        break;
                    case 9 /* TokenizerStates.FALSE4 */:
                        if (n === 101 /* charset.LATIN_SMALL_LETTER_E */) {
                            this.state = 0 /* TokenizerStates.START */;
                            this.onToken({
                                token: TokenType.FALSE,
                                value: false,
                                offset: this.offset,
                            });
                            this.offset += 4;
                            continue;
                        }
                        break;
                    // NULL
                    case 10 /* TokenizerStates.NULL1 */:
                        if (n === 117 /* charset.LATIN_SMALL_LETTER_U */) {
                            this.state = 11 /* TokenizerStates.NULL2 */;
                            continue;
                        }
                        break;
                    case 11 /* TokenizerStates.NULL2 */:
                        if (n === 108 /* charset.LATIN_SMALL_LETTER_L */) {
                            this.state = 12 /* TokenizerStates.NULL3 */;
                            continue;
                        }
                        break;
                    case 12 /* TokenizerStates.NULL3 */:
                        if (n === 108 /* charset.LATIN_SMALL_LETTER_L */) {
                            this.state = 0 /* TokenizerStates.START */;
                            this.onToken({
                                token: TokenType.NULL,
                                value: null,
                                offset: this.offset,
                            });
                            this.offset += 3;
                            continue;
                        }
                        break;
                    case 28 /* TokenizerStates.SEPARATOR */:
                        this.separatorIndex += 1;
                        if (!this.separatorBytes ||
                            n !== this.separatorBytes[this.separatorIndex]) {
                            break;
                        }
                        if (this.separatorIndex === this.separatorBytes.length - 1) {
                            this.state = 0 /* TokenizerStates.START */;
                            this.onToken({
                                token: TokenType.SEPARATOR,
                                value: this.separator,
                                offset: this.offset + this.separatorIndex,
                            });
                            this.separatorIndex = 0;
                        }
                        continue;
                    // BOM support
                    case 30 /* TokenizerStates.BOM */:
                        if (n === this.bom[this.bomIndex]) {
                            if (this.bomIndex === this.bom.length - 1) {
                                this.state = 0 /* TokenizerStates.START */;
                                this.bom = undefined;
                                this.bomIndex = 0;
                                continue;
                            }
                            this.bomIndex += 1;
                            continue;
                        }
                        break;
                    case 1 /* TokenizerStates.ENDED */:
                        if (n === 32 /* charset.SPACE */ ||
                            n === 10 /* charset.NEWLINE */ ||
                            n === 13 /* charset.CARRIAGE_RETURN */ ||
                            n === 9 /* charset.TAB */) {
                            // whitespace
                            continue;
                        }
                }
                throw new TokenizerError(`Unexpected "${String.fromCharCode(n)}" at chunk position "${i}" (absolute position "${this.streamByteLength + i}") in state ${TokenizerStateToString(this.state)}`);
            }
            this.streamByteLength += buffer.length;
            if (this.emitPartialTokens) {
                switch (this.state) {
                    case 3 /* TokenizerStates.TRUE1 */:
                    case 4 /* TokenizerStates.TRUE2 */:
                    case 5 /* TokenizerStates.TRUE3 */:
                        this.onToken({
                            token: TokenType.TRUE,
                            value: true,
                            offset: this.offset,
                            partial: true,
                        });
                        break;
                    case 6 /* TokenizerStates.FALSE1 */:
                    case 7 /* TokenizerStates.FALSE2 */:
                    case 8 /* TokenizerStates.FALSE3 */:
                    case 9 /* TokenizerStates.FALSE4 */:
                        this.onToken({
                            token: TokenType.FALSE,
                            value: false,
                            offset: this.offset,
                            partial: true,
                        });
                        break;
                    case 10 /* TokenizerStates.NULL1 */:
                    case 11 /* TokenizerStates.NULL2 */:
                    case 12 /* TokenizerStates.NULL3 */:
                        this.onToken({
                            token: TokenType.NULL,
                            value: null,
                            offset: this.offset,
                            partial: true,
                        });
                        break;
                    case 13 /* TokenizerStates.STRING_DEFAULT */: {
                        const string = this.bufferedString.toString();
                        this.onToken({
                            token: TokenType.STRING,
                            value: string,
                            offset: this.offset,
                            partial: true,
                        });
                        break;
                    }
                    case 21 /* TokenizerStates.NUMBER_AFTER_INITIAL_ZERO */:
                    case 22 /* TokenizerStates.NUMBER_AFTER_INITIAL_NON_ZERO */:
                    case 24 /* TokenizerStates.NUMBER_AFTER_DECIMAL */:
                    case 27 /* TokenizerStates.NUMBER_AFTER_E_AND_DIGIT */:
                        try {
                            this.onToken({
                                token: TokenType.NUMBER,
                                value: this.parseNumber(this.bufferedNumber.toString()),
                                offset: this.offset,
                                partial: true,
                            });
                        }
                        catch (_a) {
                            // Number couldn't be parsed. Do nothing.
                        }
                }
            }
        }
        catch (err) {
            this.error(err);
        }
    }
    emitNumber() {
        this.onToken({
            token: TokenType.NUMBER,
            value: this.parseNumber(this.bufferedNumber.toString()),
            offset: this.offset,
        });
        this.offset += this.bufferedNumber.byteLength - 1;
    }
    /**
     * Turns the characters of a JSON number into a JavaScript value.
     *
     * Equivalent to `Number(numberStr)`. Override it to handle numbers that a
     * JavaScript number can't represent, for example by keeping them as strings.
     *
     * @param numberStr The number, as it appeared in the JSON stream.
     * @returns The parsed number.
     */
    parseNumber(numberStr) {
        return Number(numberStr);
    }
    /**
     * Puts the tokenizer in an error state and reports `err` through
     * {@linkcode Tokenizer.onError}. The tokenizer can't be used afterwards.
     *
     * @param err What went wrong.
     */
    error(err) {
        if (this.state !== 1 /* TokenizerStates.ENDED */) {
            this.state = 2 /* TokenizerStates.ERROR */;
        }
        this.onError(err);
    }
    /**
     * Signals that the stream is over, flushing any number that was still being
     * tokenized and then ending the tokenizer, which can't be used afterwards.
     *
     * @throws {TokenizerError} If the stream ended in the middle of a token and no
     * {@linkcode Tokenizer.onError} callback has been set.
     */
    end() {
        switch (this.state) {
            case 21 /* TokenizerStates.NUMBER_AFTER_INITIAL_ZERO */:
            case 22 /* TokenizerStates.NUMBER_AFTER_INITIAL_NON_ZERO */:
            case 24 /* TokenizerStates.NUMBER_AFTER_DECIMAL */:
            case 27 /* TokenizerStates.NUMBER_AFTER_E_AND_DIGIT */:
                this.state = 1 /* TokenizerStates.ENDED */;
                this.emitNumber();
                this.onEnd();
                break;
            case 29 /* TokenizerStates.BOM_OR_START */:
            case 0 /* TokenizerStates.START */:
            case 2 /* TokenizerStates.ERROR */:
                this.state = 1 /* TokenizerStates.ENDED */;
                this.onEnd();
                break;
            default:
                this.error(new TokenizerError(`Tokenizer ended in the middle of a token (state: ${TokenizerStateToString(this.state)}). Either not all the data was received or the data was invalid.`));
        }
    }
    /**
     * Called with every token found in the stream. Override it to consume them;
     * by default it throws.
     *
     * @param parsedToken The token and where it was found.
     */
    // biome-ignore lint/correctness/noUnusedFunctionParameters: override point; the parameter is part of the public signature
    onToken(parsedToken) {
        // Override me
        throw new TokenizerError('Can\'t emit tokens before the "onToken" callback has been set up.');
    }
    /**
     * Called when the data can't be tokenized. Override it to handle errors
     * asynchronously; by default it throws, so the error surfaces out of the
     * {@linkcode Tokenizer.write} or {@linkcode Tokenizer.end} call that caused it.
     *
     * @param err What went wrong.
     */
    onError(err) {
        // Override me
        throw err;
    }
    /** Called once the tokenizer has ended. Override it to react to that; by default it does nothing. */
    onEnd() {
        // Override me
    }
}
