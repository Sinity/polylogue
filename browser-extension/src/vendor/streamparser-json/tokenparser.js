/**
 * A parser that assembles the tokens emitted by the tokenizer into JSON values.
 *
 * @example
 * ```ts
 * import Tokenizer from "@streamparser/json/tokenizer.js";
 * import TokenParser from "@streamparser/json/tokenparser.js";
 *
 * const tokenizer = new Tokenizer();
 * const tokenParser = new TokenParser({ paths: ["$.*"] });
 * tokenizer.onToken = tokenParser.write.bind(tokenParser);
 * tokenParser.onValue = ({ value }) => {
 *   // process the value
 * };
 *
 * tokenizer.write('{ "test": ["a"] }');
 * ```
 *
 * @module
 */
import TokenType from "./utils/types/tokenType.js";
// Parser States
var TokenParserState;
(function (TokenParserState) {
    TokenParserState[TokenParserState["VALUE"] = 0] = "VALUE";
    TokenParserState[TokenParserState["KEY"] = 1] = "KEY";
    TokenParserState[TokenParserState["COLON"] = 2] = "COLON";
    TokenParserState[TokenParserState["COMMA"] = 3] = "COMMA";
    TokenParserState[TokenParserState["ENDED"] = 4] = "ENDED";
    TokenParserState[TokenParserState["ERROR"] = 5] = "ERROR";
    TokenParserState[TokenParserState["SEPARATOR"] = 6] = "SEPARATOR";
})(TokenParserState || (TokenParserState = {}));
function TokenParserStateToString(state) {
    return ["VALUE", "KEY", "COLON", "COMMA", "ENDED", "ERROR", "SEPARATOR"][state];
}
// Plain bracket assignment invokes the inherited `Object.prototype.__proto__`
// setter for that one key name, letting a "__proto__" key in the input alter
// obj's actual prototype instead of becoming a property of obj. Only that key
// needs the safe (but slower) Object.defineProperty path -- every other key,
// i.e. the overwhelming majority, keeps the fast, JIT-friendly assignment.
function setProperty(obj, key, value) {
    if (key === "__proto__") {
        Object.defineProperty(obj, key, {
            value,
            writable: true,
            enumerable: true,
            configurable: true,
        });
        return;
    }
    obj[key] = value;
}
const defaultOpts = {
    paths: undefined,
    keepStack: true,
    separator: undefined,
    emitPartialValues: false,
};
/** The error thrown when the token parser is misconfigured or gets an unexpected token. */
export class TokenParserError extends Error {
    /**
     * @param message What went wrong.
     */
    constructor(message) {
        super(message);
        // Typescript is broken. This is a workaround
        Object.setPrototypeOf(this, TokenParserError.prototype);
    }
}
/**
 * A parser that assembles the tokens emitted by a tokenizer into JSON values.
 *
 * Tokens are pushed in with {@linkcode TokenParser.write} and the resulting
 * values come back through the {@linkcode TokenParser.onValue} callback, which
 * the user is expected to override. Values are emitted innermost first, as soon
 * as each one is complete, and can be narrowed down to the ones of interest with
 * the `paths` option.
 *
 * @example
 * ```ts
 * import Tokenizer from "@streamparser/json/tokenizer.js";
 * import TokenParser from "@streamparser/json/tokenparser.js";
 *
 * const tokenizer = new Tokenizer();
 * const tokenParser = new TokenParser();
 * tokenizer.onToken = tokenParser.write.bind(tokenParser);
 * tokenParser.onValue = ({ value, key, parent, stack }) => {
 *   // process the value
 * };
 *
 * tokenizer.write('{ "test": ["a"] }');
 * // onValue is called 3 times: "a", ["a"] and { test: ["a"] }
 * ```
 */
export default class TokenParser {
    /**
     * @param opts What to emit and how. See {@linkcode TokenParserOptions}.
     * @throws {TokenParserError} If any of the configured `paths` is not a valid selector.
     */
    constructor(opts) {
        this.state = 0 /* TokenParserState.VALUE */;
        this.mode = undefined;
        this.key = undefined;
        this.value = undefined;
        this.stack = [];
        // Tracked explicitly rather than inferred from `value` because keepStack:false
        // deletes emitted properties from `value`, so Object.keys(value).length can no
        // longer be trusted to tell an empty object from one whose members were purged.
        this.memberCount = 0;
        opts = Object.assign(Object.assign({}, defaultOpts), opts);
        if (opts.paths) {
            const root = { children: new Map(), terminal: false };
            // A match-everything selector makes the whole set match everything, which
            // we represent by leaving the trie undefined (as with no paths at all).
            let matchEverything = false;
            for (const path of opts.paths) {
                if (path === undefined || path === "$*") {
                    matchEverything = true;
                    continue;
                }
                if (!path.startsWith("$"))
                    throw new TokenParserError(`Invalid selector "${path}". Should start with "$".`);
                const segments = path.split(".").slice(1);
                if (segments.includes(""))
                    throw new TokenParserError(`Invalid selector "${path}". ".." syntax not supported.`);
                let node = root;
                for (const segment of segments) {
                    let child = node.children.get(segment);
                    if (!child) {
                        child = { children: new Map(), terminal: false };
                        node.children.set(segment, child);
                    }
                    node = child;
                }
                node.terminal = true;
            }
            if (!matchEverything)
                this.selectorTrie = root;
        }
        this.keepStack = opts.keepStack || false;
        this.separator = opts.separator;
        if (!opts.emitPartialValues) {
            this.emitPartial = () => { };
        }
    }
    shouldEmit() {
        if (!this.selectorTrie)
            return true;
        return this.matchesSelector(this.selectorTrie, 0);
    }
    // Depth-first walk of the selector trie down the current value's key path:
    //   [stack[1].key, ..., stack[n-1].key, this.key]   (n = stack.length)
    // A value matches iff some branch reaches a terminal node at the exact depth.
    // Recursion (rather than an explicit frontier) keeps the common single-path
    // walk allocation-free and short-circuits on the first match, like the old
    // rescan did, while collapsing its O(number of selectors) cost to O(depth).
    matchesSelector(node, level) {
        const keyCount = this.stack.length;
        if (level === keyCount)
            return node.terminal;
        const key = level < keyCount - 1 ? this.stack[level + 1].key : this.key;
        // "*" matches any key; try it first since it needs no key lookup.
        const wildcard = node.children.get("*");
        if (wildcard && this.matchesSelector(wildcard, level + 1))
            return true;
        // Then a literal match. Resolving the key to a string allocates for numeric
        // array indices, so skip it unless there's a literal child to match.
        const hasLiteralChild = node.children.size > (wildcard ? 1 : 0);
        if (hasLiteralChild) {
            const segment = key === null || key === void 0 ? void 0 : key.toString();
            if (segment !== undefined) {
                const child = node.children.get(segment);
                if (child && this.matchesSelector(child, level + 1))
                    return true;
            }
        }
        return false;
    }
    push() {
        this.stack.push({
            key: this.key,
            value: this.value,
            mode: this.mode,
            emit: this.shouldEmit(),
            memberCount: this.memberCount,
        });
    }
    pop() {
        const value = this.value;
        // biome-ignore lint/suspicious/noImplicitAnyLet: assigned via the destructuring assignment below
        let emit;
        ({
            key: this.key,
            value: this.value,
            mode: this.mode,
            emit,
            memberCount: this.memberCount,
        } = this.stack.pop());
        this.state =
            this.mode !== undefined ? 3 /* TokenParserState.COMMA */ : 0 /* TokenParserState.VALUE */;
        this.emit(value, emit);
    }
    emit(value, emit) {
        if (!this.keepStack &&
            this.value &&
            this.stack.every((item) => !item.emit)) {
            if (Array.isArray(this.value)) {
                // Shrinking `.length` drops the slot, unlike `delete`, which only leaves a hole
                this.value.length -= 1;
            }
            else {
                delete this.value[this.key];
            }
        }
        if (emit) {
            this.onValue({
                value: value,
                key: this.key,
                parent: this.value,
                stack: this.stack,
            });
        }
        if (this.stack.length === 0) {
            if (this.separator) {
                this.state = 6 /* TokenParserState.SEPARATOR */;
            }
            else if (this.separator === undefined) {
                this.end();
            }
            // else if separator === '', expect next JSON object.
        }
    }
    emitPartial(value) {
        if (!this.shouldEmit())
            return;
        if (this.state === 1 /* TokenParserState.KEY */) {
            this.onValue({
                value: undefined,
                key: value,
                parent: this.value,
                stack: this.stack,
                partial: true,
            });
            return;
        }
        this.onValue({
            value: value,
            key: this.key,
            parent: this.value,
            stack: this.stack,
            partial: true,
        });
    }
    /** Whether the token parser is ended, and thus no longer accepting tokens. */
    get isEnded() {
        return this.state === 4 /* TokenParserState.ENDED */;
    }
    /**
     * Pushes the next token into the parser.
     *
     * Parsing happens synchronously, so every value that the token completes is
     * emitted through {@linkcode TokenParser.onValue} before this returns.
     *
     * @param parsedTokenInfo The token to process, as emitted by a tokenizer.
     * @throws {TokenParserError} If the token can't appear at this point of the
     * JSON document and no {@linkcode TokenParser.onError} callback has been set.
     */
    write({ token, value, partial, }) {
        try {
            if (partial) {
                if (this.state !== 0 /* TokenParserState.VALUE */ &&
                    this.state !== 1 /* TokenParserState.KEY */) {
                    throw new TokenParserError(`Unexpected partial ${TokenType[token]} (${JSON.stringify(value)}) in state ${TokenParserStateToString(this.state)}`);
                }
                this.emitPartial(value);
                return;
            }
            if (this.state === 0 /* TokenParserState.VALUE */) {
                if (token === TokenType.STRING ||
                    token === TokenType.NUMBER ||
                    token === TokenType.TRUE ||
                    token === TokenType.FALSE ||
                    token === TokenType.NULL) {
                    if (this.mode === 0 /* TokenParserMode.OBJECT */) {
                        setProperty(this.value, this.key, value);
                        this.state = 3 /* TokenParserState.COMMA */;
                        this.memberCount++;
                    }
                    else if (this.mode === 1 /* TokenParserMode.ARRAY */) {
                        this.value.push(value);
                        this.state = 3 /* TokenParserState.COMMA */;
                        this.memberCount++;
                    }
                    this.emit(value, this.shouldEmit());
                    return;
                }
                if (token === TokenType.LEFT_BRACE) {
                    this.memberCount++;
                    this.push();
                    if (this.mode === 0 /* TokenParserMode.OBJECT */) {
                        const val = {};
                        setProperty(this.value, this.key, val);
                        this.value = val;
                    }
                    else if (this.mode === 1 /* TokenParserMode.ARRAY */) {
                        const val = {};
                        this.value.push(val);
                        this.value = val;
                    }
                    else {
                        this.value = {};
                    }
                    this.mode = 0 /* TokenParserMode.OBJECT */;
                    this.state = 1 /* TokenParserState.KEY */;
                    this.key = undefined;
                    this.memberCount = 0;
                    this.emitPartial();
                    return;
                }
                if (token === TokenType.LEFT_BRACKET) {
                    this.memberCount++;
                    this.push();
                    if (this.mode === 0 /* TokenParserMode.OBJECT */) {
                        const val = [];
                        setProperty(this.value, this.key, val);
                        this.value = val;
                    }
                    else if (this.mode === 1 /* TokenParserMode.ARRAY */) {
                        const val = [];
                        this.value.push(val);
                        this.value = val;
                    }
                    else {
                        this.value = [];
                    }
                    this.mode = 1 /* TokenParserMode.ARRAY */;
                    this.state = 0 /* TokenParserState.VALUE */;
                    this.key = 0;
                    this.memberCount = 0;
                    this.emitPartial();
                    return;
                }
                if (this.mode === 1 /* TokenParserMode.ARRAY */ &&
                    token === TokenType.RIGHT_BRACKET &&
                    this.memberCount === 0) {
                    this.pop();
                    return;
                }
            }
            if (this.state === 1 /* TokenParserState.KEY */) {
                if (token === TokenType.STRING) {
                    this.key = value;
                    this.state = 2 /* TokenParserState.COLON */;
                    this.emitPartial();
                    return;
                }
                if (token === TokenType.RIGHT_BRACE && this.memberCount === 0) {
                    this.pop();
                    return;
                }
            }
            if (this.state === 2 /* TokenParserState.COLON */) {
                if (token === TokenType.COLON) {
                    this.state = 0 /* TokenParserState.VALUE */;
                    return;
                }
            }
            if (this.state === 3 /* TokenParserState.COMMA */) {
                if (token === TokenType.COMMA) {
                    if (this.mode === 1 /* TokenParserMode.ARRAY */) {
                        this.state = 0 /* TokenParserState.VALUE */;
                        this.key += 1;
                        return;
                    }
                    /* istanbul ignore else */
                    if (this.mode === 0 /* TokenParserMode.OBJECT */) {
                        this.state = 1 /* TokenParserState.KEY */;
                        return;
                    }
                }
                if ((token === TokenType.RIGHT_BRACE &&
                    this.mode === 0 /* TokenParserMode.OBJECT */) ||
                    (token === TokenType.RIGHT_BRACKET &&
                        this.mode === 1 /* TokenParserMode.ARRAY */)) {
                    this.pop();
                    return;
                }
            }
            if (this.state === 6 /* TokenParserState.SEPARATOR */) {
                if (token === TokenType.SEPARATOR && value === this.separator) {
                    this.state = 0 /* TokenParserState.VALUE */;
                    return;
                }
            }
            // Edge case in which the separator is just whitespace and it's found in the middle of the JSON
            if (token === TokenType.SEPARATOR &&
                this.state !== 6 /* TokenParserState.SEPARATOR */ &&
                Array.from(value)
                    .map((n) => n.charCodeAt(0))
                    .every((n) => n === 32 /* charset.SPACE */ ||
                    n === 10 /* charset.NEWLINE */ ||
                    n === 13 /* charset.CARRIAGE_RETURN */ ||
                    n === 9 /* charset.TAB */)) {
                // whitespace
                return;
            }
            throw new TokenParserError(`Unexpected ${TokenType[token]} (${JSON.stringify(value)}) in state ${TokenParserStateToString(this.state)}`);
        }
        catch (err) {
            this.error(err);
        }
    }
    /**
     * Puts the token parser in an error state and reports `err` through
     * {@linkcode TokenParser.onError}. The parser can't be used afterwards.
     *
     * @param err What went wrong.
     */
    error(err) {
        if (this.state !== 4 /* TokenParserState.ENDED */) {
            this.state = 5 /* TokenParserState.ERROR */;
        }
        this.onError(err);
    }
    /**
     * Signals that there are no more tokens, ending the token parser, which can't
     * be used afterwards.
     *
     * @throws {Error} If the JSON document was left half-parsed and no
     * {@linkcode TokenParser.onError} callback has been set.
     */
    end() {
        if ((this.state !== 0 /* TokenParserState.VALUE */ &&
            this.state !== 6 /* TokenParserState.SEPARATOR */) ||
            this.stack.length > 0) {
            this.error(new Error(`Parser ended in mid-parsing (state: ${TokenParserStateToString(this.state)}). Either not all the data was received or the data was invalid.`));
        }
        else {
            this.state = 4 /* TokenParserState.ENDED */;
            this.onEnd();
        }
    }
    /**
     * Called with every value that matches the configured `paths`. Override it to
     * consume them; by default it throws.
     *
     * @param parsedElementInfo The value and where it was found. Its `parent` and
     * `stack` are live references into the parser's in-progress structures, so use
     * `cloneParsedElementInfo` to snapshot them if they need to outlive the call.
     */
    // biome-ignore lint/correctness/noUnusedFunctionParameters: override point; the parameter is part of the public signature
    onValue(parsedElementInfo) {
        // Override me
        throw new TokenParserError('Can\'t emit data before the "onValue" callback has been set up.');
    }
    /**
     * Called when the tokens don't add up to valid JSON. Override it to handle
     * errors asynchronously; by default it throws, so the error surfaces out of the
     * {@linkcode TokenParser.write} or {@linkcode TokenParser.end} call that caused it.
     *
     * @param err What went wrong.
     */
    onError(err) {
        // Override me
        throw err;
    }
    /** Called once the token parser has ended. Override it to react to that; by default it does nothing. */
    onEnd() {
        // Override me
    }
}
