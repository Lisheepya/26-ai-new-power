const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const { predictAi4i } = require("../docs/app.js");

const root = path.resolve(__dirname, "..");
const results = JSON.parse(fs.readFileSync(path.join(root, "docs/results.json"), "utf8"));
const golden = JSON.parse(fs.readFileSync(path.join(__dirname, "golden.json"), "utf8"));

function readSource(relativePath) {
  return fs.readFileSync(path.join(root, relativePath), "utf8");
}

function findTableRow(report, headerMarker, rowLabel) {
  const lines = report.split(/\r?\n/);
  const headerIndex = lines.findIndex((line) => line.startsWith("|") && line.includes(headerMarker));
  assert.notEqual(headerIndex, -1, `report table header not found: ${headerMarker}`);
  const row = lines.slice(headerIndex + 2).find((line) => line.startsWith(`| ${rowLabel} |`));
  assert.ok(row, `report row not found: ${rowLabel}`);
  return row.split("|").slice(1, -1).map((cell) => cell.trim());
}

function parseSavedParameters() {
  const source = readSource("docs/app.js");
  const match = source.match(/^const TRAINED_PARAMETERS = (\{.*\});$/m);
  assert.ok(match, "saved frontend model parameters must be present in docs/app.js");
  return JSON.parse(match[1]);
}

test("AI4I model inputs exclude all target and failure-mode labels", () => {
  const features = results.ai4i.metadata.feature_names;
  const excluded = new Set(["Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]);
  const leaked = features.filter((feature) => excluded.has(feature.split("__").at(-1)));

  assert.deepEqual(leaked, [], `AI4I feature list contains excluded columns: ${leaked.join(", ")}`);
});

test("AI4I MLP detects failures and predicts both classes", () => {
  const model = results.ai4i.models.mlp;
  const { threshold, recall } = model.metrics;
  const predictions = model.evaluation_data.positive_scores.map((score) => Number(score >= threshold));

  assert.ok(recall > 0, `AI4I MLP failure recall must be > 0; got ${recall}`);
  assert.deepEqual(
    [...new Set(predictions)].sort(),
    [0, 1],
    "AI4I MLP predictions must include both non-failure and failure classes",
  );
});

test("published result numbers agree with source artifacts and both research reports", () => {
  for (const [resultKey, sourceKey] of [
    ["iris", "iris_metrics"],
    ["ai4i", "ai4i_metrics"],
    ["ai4i_eda", "ai4i_eda"],
  ]) {
    const source = JSON.parse(readSource(results.source_files[sourceKey]));
    assert.deepEqual(results[resultKey], source, `docs/results.json ${resultKey} numbers differ from ${sourceKey}`);
  }

  const ai4iReport = readSource(results.source_files.ai4i_report);
  for (const modelName of ["logistic_regression", "mlp"]) {
    const metric = results.ai4i.models[modelName].metrics;
    const label = modelName === "mlp" ? "MLP" : "Logistic Regression";
    const actual = findTableRow(ai4iReport, "Accuracy", label);
    const expected = [
      label,
      `${(metric.accuracy * 100).toFixed(2)}%`,
      `${(metric.precision * 100).toFixed(2)}%`,
      `${(metric.recall * 100).toFixed(2)}%`,
      `${(metric.f1 * 100).toFixed(2)}%`,
      metric.roc_auc.toFixed(3),
      metric.pr_auc.toFixed(3),
    ];
    assert.deepEqual(actual, expected, `AI4I ${label} report metrics must match docs/results.json`);

    const matrix = metric.confusion_matrix;
    const confusion = findTableRow(ai4iReport, "TN", label);
    assert.deepEqual(
      confusion,
      [label, String(matrix[0][0]), String(matrix[0][1]), String(matrix[1][0]), String(matrix[1][1])],
      `AI4I ${label} report confusion counts must match docs/results.json`,
    );
  }

  const irisReport = readSource(results.source_files.iris_report);
  const iris = results.iris;
  assert.ok(irisReport.includes(`**${iris.accuracy.toFixed(3)}**`), "Iris report accuracy must match docs/results.json");
  assert.ok(irisReport.includes(`（${iris.test_samples} 筆）`), "Iris report test size must match docs/results.json");
  assert.ok(irisReport.includes(`${iris.iterations} 次迭代`), "Iris report iteration count must match docs/results.json");
  const irisMatrix = findTableRow(irisReport, "真實分類", "setosa");
  assert.deepEqual(irisMatrix, ["setosa", ...iris.confusion_matrix[0].map(String)]);
});

test("saved preprocessing statistics reload and reproduce Python AI4I predictions", () => {
  const saved = parseSavedParameters().ai4i;
  const reference = golden.ai4i;
  const mean = saved.scaler.mean;
  const scale = saved.scaler.scale;

  assert.equal(saved.features[0], "Type");
  assert.deepEqual(saved.features.slice(1), reference.training.numeric_feature_order);
  assert.deepEqual(mean, reference.training.scaler_mean, "saved StandardScaler mean must match Python training");
  assert.deepEqual(scale, reference.training.scaler_scale, "saved StandardScaler std must match Python training");
  assert.ok(mean.every(Number.isFinite), "saved scaler mean values must be finite");
  assert.ok(scale.every((value) => Number.isFinite(value) && value > 0), "saved scaler std values must be finite and positive");

  for (const example of reference.examples) {
    const probability = predictAi4i(example.input);
    assert.ok(Math.abs(probability - example.probabilities[1]) < 1e-6);
    assert.equal(Number(probability >= 0.5), example.className);
  }
});
