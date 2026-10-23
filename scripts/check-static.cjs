const { spawnSync } = require("node:child_process");
const { existsSync } = require("node:fs");
const { resolve } = require("node:path");

const isWindows = process.platform === "win32";
const checks = [
  {
    name: "ESLint + SonarJS",
    cli: "eslint/bin/eslint.js",
    args: ["docs", "scripts", "tests"],
  },
  {
    name: "jscpd duplication rate",
    cli: "jscpd/bin/jscpd",
    args: [
      "docs",
      "scripts",
      "tests",
      "--format",
      "javascript",
      "--formats-exts",
      "javascript:js,cjs,mjs",
      "--min-lines",
      "5",
      "--min-tokens",
      "50",
      "--reporters",
      "console",
      "--noTips",
      "--no-colors",
    ],
  },
  {
    name: "Knip dead-code analysis",
    cli: "knip/bin/knip.js",
    args: ["--no-progress", "--reporter", "compact"],
  },
];

let failed = false;

for (const check of checks) {
  const cli = resolve("node_modules", check.cli);

  console.log(`\n=== ${check.name} ===`);

  if (!existsSync(cli)) {
    console.error(`Missing local CLI: ${cli}. Run pnpm install first.`);
    failed = true;
    continue;
  }

  const result = spawnSync(process.execPath, [cli, ...check.args], {
    cwd: process.cwd(),
    encoding: "utf8",
    windowsHide: isWindows,
    maxBuffer: 10 * 1024 * 1024,
  });

  if (result.stdout) process.stdout.write(result.stdout);
  if (result.stderr) process.stderr.write(result.stderr);

  if (result.error) {
    console.error(result.error.message);
    failed = true;
    continue;
  }

  const status = result.status ?? 1;
  if (check.name === "Knip dead-code analysis" && status === 0 && !result.stdout && !result.stderr) {
    console.log("[Knip] 0 findings.");
  }
  console.log(`[${check.name}] exit code: ${status}`);
  failed ||= status !== 0;
}

console.log(`\nStatic checks ${failed ? "found issues or failed to run" : "completed successfully"}.`);
process.exitCode = failed ? 1 : 0;
