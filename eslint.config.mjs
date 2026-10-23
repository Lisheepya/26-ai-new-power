import js from "@eslint/js";
import globals from "globals";
import sonarjs from "eslint-plugin-sonarjs";

const sourceFiles = [
  "docs/**/*.js",
  "scripts/**/*.{js,cjs,mjs}",
  "tests/**/*.{js,cjs,mjs}",
];

export default [
  {
    ignores: ["node_modules/**"],
  },
  {
    ...js.configs.recommended,
    files: sourceFiles,
  },
  {
    ...sonarjs.configs.recommended,
    files: sourceFiles,
  },
  {
    files: sourceFiles,
    languageOptions: {
      ecmaVersion: "latest",
    },
    rules: {
      complexity: ["error", 10],
      "max-depth": ["error", 3],
      "max-lines-per-function": ["error", 50],
      "sonarjs/cognitive-complexity": ["error", 15],
    },
  },
  {
    files: ["docs/**/*.js"],
    languageOptions: {
      sourceType: "script",
      globals: globals.browser,
    },
  },
  {
    files: ["scripts/**/*.{js,mjs}", "tests/**/*.{js,mjs}"],
    languageOptions: {
      sourceType: "module",
      globals: globals.node,
    },
  },
  {
    files: ["scripts/**/*.cjs", "tests/**/*.cjs"],
    languageOptions: {
      sourceType: "commonjs",
      globals: globals.node,
    },
  },
  {
    files: ["docs/app.js"],
    languageOptions: {
      globals: {
        module: "readonly",
      },
    },
  },
];
