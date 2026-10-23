const assert = require("node:assert/strict");
const test = require("node:test");
const { predictAi4i, predictIris } = require("../docs/app.js");

test("Iris browser inference matches the trained Python MLP", () => {
  const examples = [
    [[5.1, 3.5, 1.4, 0.2], "Iris-setosa", [0.9999989312512698, 6.198978745402083e-7, 4.4885085562717255e-7]],
    [[6.0, 2.9, 4.5, 1.5], "Iris-versicolor", [2.2175299738702352e-7, 0.999999707203564, 7.10434386091155e-8]],
    [[6.5, 3.0, 5.2, 2.0], "Iris-virginica", [1.4750762171091174e-6, 3.064320746530598e-5, 0.9999678817163176]],
  ];

  for (const [measurements, expectedClass, expectedProbabilities] of examples) {
    const result = predictIris(measurements);
    assert.equal(result.className, expectedClass);
    result.probabilities.forEach((probability, index) => {
      assert.ok(Math.abs(probability - expectedProbabilities[index]) < 1e-9);
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
