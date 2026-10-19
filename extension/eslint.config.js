import js from "@eslint/js";

export default [
  js.configs.recommended,
  {
    files: ["src/**/*.js"],
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: "script",
      globals: {
        window: "readonly",
        document: "readonly",
        chrome: "readonly",
        MutationObserver: "readonly",
        setTimeout: "readonly",
        clearTimeout: "readonly",
        console: "readonly",
        URL: "readonly",
        URLSearchParams: "readonly",
        fetch: "readonly",
        TextEncoder: "readonly",
        AbortController: "readonly",
        structuredClone: "readonly",
        btoa: "readonly",
        atob: "readonly",
        location: "readonly",
        PointerEvent: "readonly",
        MouseEvent: "readonly",
        KeyboardEvent: "readonly",
        DataTransfer: "readonly",
        File: "readonly",
        Blob: "readonly",
        Headers: "readonly",
        Response: "readonly",
        ReadableStream: "readonly",
        Event: "readonly",
        InputEvent: "readonly",
        crypto: "readonly",
        navigator: "readonly",
      },
    },
    rules: {
      "no-unused-vars": ["warn", { argsIgnorePattern: "^_" }],
      "no-undef": "error",
    },
  },
  {
    files: ["src/background.js", "src/background/**/*.js", "src/actions/**/*.js", "src/backfill/**/*.js", "src/capture/**/*.js", "src/vendor/**/*.js"],
    languageOptions: {
      sourceType: "module",
    },
  },
  {
    files: ["src/vendor/sha256.js"],
    languageOptions: {
      // The locked UMD source names these loaders in guarded inactive branches.
      globals: { global: "readonly", define: "readonly", require: "readonly" },
    },
  },
  {
    files: ["src/vendor/**/*.js"],
    languageOptions: { globals: { TextDecoder: "readonly" } },
  },
  {
    files: ["src/vendor/streamparser-json/tokenizer.js"],
    // Preserve the locked compiler output's intentional tokenizer fallthrough.
    // Vendor-byte parity is checked by the build and streaming parser laws.
    rules: { "no-fallthrough": "off" },
  },
  {
    files: ["tests/**/*.js"],
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: "module",
      globals: {
        describe: "readonly",
        it: "readonly",
        expect: "readonly",
        vi: "readonly",
        beforeEach: "readonly",
        afterEach: "readonly",
        URL: "readonly",
        URLSearchParams: "readonly",
        document: "readonly",
        structuredClone: "readonly",
        chrome: "readonly",
        TextEncoder: "readonly",
        btoa: "readonly",
        setTimeout: "readonly",
      },
    },
  },
];
