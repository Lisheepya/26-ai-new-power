const assert = require("node:assert/strict");
const test = require("node:test");
const golden = require("./golden.json");
const { predictAi4i, predictIris } = require("../docs/app.js");

test("Iris browser inference matches Python-generated golden predictions", () => {
  assert.equal(golden.dataset, "Iris");
  assert.equal(golden.examples.length, 3);
  assert.deepEqual(golden.training.feature_order, [
    "sepal_length_cm",
    "sepal_width_cm",
    "petal_length_cm",
    "petal_width_cm",
  ]);

  for (const example of golden.examples) {
    const result = predictIris(example.input);
    assert.equal(result.className, example.className);
    assert.deepEqual(result.classes, golden.classes);
    result.probabilities.forEach((probability, index) => {
      assert.ok(Math.abs(probability - example.probabilities[index]) < 1e-6);
    });
  }
});

test("AI4I browser inference matches the trained Python MLP", () => {
  const normalExample = {
    type: "L",
    airTemperature: 300.7,
    processTemperature: 312.0,
    rotationalSpeed: 1550,
    torque: 32.4,
    toolWear: 5,
  };
  const failureExample = {
    type: "L",
    airTemperature: 300.8,
    processTemperature: 309.9,
    rotationalSpeed: 1312,
    torque: 65.3,
    toolWear: 192,
  };

  assert.ok(Math.abs(predictAi4i(normalExample) - 1.9501775971860003e-6) < 1e-9);
  assert.ok(Math.abs(predictAi4i(failureExample) - 0.9278141414534714) < 1e-9);
});

test("browser inference rejects invalid input", () => {
  assert.throws(() => predictIris([5.1, 3.5, NaN, 0.2]), /finite/);
  assert.throws(() => predictAi4i({ type: "X", airTemperature: 300, processTemperature: 310, rotationalSpeed: 1500, torque: 40, toolWear: 10 }), /Type/);
});
