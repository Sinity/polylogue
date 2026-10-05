/**
 * The JSON token types emitted by the tokenizer.
 *
 * @module
 */
/** The type of a JSON token, as reported by the tokenizer's `onToken` callback. */
var TokenType;
(function (TokenType) {
    /** `{` */
    TokenType[TokenType["LEFT_BRACE"] = 0] = "LEFT_BRACE";
    /** `}` */
    TokenType[TokenType["RIGHT_BRACE"] = 1] = "RIGHT_BRACE";
    /** `[` */
    TokenType[TokenType["LEFT_BRACKET"] = 2] = "LEFT_BRACKET";
    /** `]` */
    TokenType[TokenType["RIGHT_BRACKET"] = 3] = "RIGHT_BRACKET";
    /** `:` */
    TokenType[TokenType["COLON"] = 4] = "COLON";
    /** `,` */
    TokenType[TokenType["COMMA"] = 5] = "COMMA";
    /** `true` */
    TokenType[TokenType["TRUE"] = 6] = "TRUE";
    /** `false` */
    TokenType[TokenType["FALSE"] = 7] = "FALSE";
    /** `null` */
    TokenType[TokenType["NULL"] = 8] = "NULL";
    /** A string, with all its escape sequences already resolved. */
    TokenType[TokenType["STRING"] = 9] = "STRING";
    /** A number, already parsed into a JavaScript number. */
    TokenType[TokenType["NUMBER"] = 10] = "NUMBER";
    /** The configured separator between consecutive JSON documents. */
    TokenType[TokenType["SEPARATOR"] = 11] = "SEPARATOR";
})(TokenType || (TokenType = {}));
export default TokenType;
