import js from "@eslint/js";
import globals from "globals";

export default [
  js.configs.recommended,
  {
    // Page scripts and the popup are classic scripts in a browser page.
    files: ["src/**/*.js"],
    languageOptions: {
      ecmaVersion: 2023,
      sourceType: "script",
      globals: { ...globals.browser, chrome: "readonly" },
    },
    rules: {
      "no-unused-vars": ["error", { argsIgnorePattern: "^_", varsIgnorePattern: "^_" }],
    },
  },
  {
    // The service worker and the code it runs in pages are modules.
    files: ["src/background.js", "src/background/**/*.js", "src/actions/**/*.js"],
    languageOptions: { sourceType: "module" },
  },
  {
    files: ["tests/**/*.js", "scripts/**/*.mjs", "*.config.js"],
    languageOptions: {
      ecmaVersion: 2023,
      sourceType: "module",
      globals: { ...globals.node, ...globals.browser },
    },
  },
];
