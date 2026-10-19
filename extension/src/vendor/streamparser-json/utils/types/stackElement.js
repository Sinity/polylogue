/**
 * The shape of the token parser's stack, i.e. the chain of containers that a
 * parsed value is nested in.
 *
 * @module
 */
/** Whether the container being parsed is a JSON object or a JSON array. */
export var TokenParserMode;
(function (TokenParserMode) {
    /** The container is a JSON object, so its members are keyed by property name. */
    TokenParserMode[TokenParserMode["OBJECT"] = 0] = "OBJECT";
    /** The container is a JSON array, so its members are keyed by index. */
    TokenParserMode[TokenParserMode["ARRAY"] = 1] = "ARRAY";
})(TokenParserMode || (TokenParserMode = {}));
