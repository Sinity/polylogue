export {
/** A full JSON parser: a tokenizer and a token parser wired to each other. */
default as JSONParser, } from "./jsonparser.js";
export {
/** A JSON-compliant tokenizer that turns a utf-8 stream into JSON tokens. */
default as Tokenizer,
/** The error thrown when the tokenizer is misconfigured or hits invalid JSON. */
TokenizerError, } from "./tokenizer.js";
export {
/** A parser that assembles the tokens emitted by a tokenizer into JSON values. */
default as TokenParser,
/** The error thrown when the token parser is misconfigured or gets an unexpected token. */
TokenParserError, } from "./tokenparser.js";
/** The types of the JSON values that the parser produces. */
export * as JsonTypes from "./utils/types/jsonTypes.js";
export {
/** Whether the container being parsed is a JSON object or a JSON array. */
TokenParserMode, } from "./utils/types/stackElement.js";
export {
/** The type of a JSON token, as reported by the tokenizer's `onToken` callback. */
default as TokenType, } from "./utils/types/tokenType.js";
/** The utf-8 byte values that the tokenizer matches the incoming stream against. */
export * as utf8 from "./utils/utf-8.js";
