const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");

const appPath = path.join(__dirname, "../docs/app.js");
const fixedResults = JSON.parse(fs.readFileSync(path.join(__dirname, "fixtures/docs-results.json"), "utf8"));

class FakeElement {
  constructor() {
    this.children = [];
    this.listeners = {};
    this.attributes = {};
    this.fields = {};
    this.style = {};
    this.hidden = false;
    this.className = "";
    this._textContent = "";
    this.classList = {
      toggle: (className, enabled) => {
        const classes = new Set(this.className.split(/\s+/).filter(Boolean));
        if (enabled) classes.add(className);
        else classes.delete(className);
        this.className = [...classes].join(" ");
      },
    };
  }

  get textContent() {
    return this._textContent + this.children.map((child) => child.textContent).join("");
  }

  set textContent(value) {
    this._textContent = String(value);
    this.children = [];
  }

  append(...children) {
    this.children.push(...children);
  }

  replaceChildren(...children) {
    this._textContent = "";
    this.children = [...children];
  }

  setAttribute(name, value) {
    this.attributes[name] = String(value);
  }

  addEventListener(name, listener) {
    this.listeners[name] = listener;
  }
}

function createPage(fetchImplementation) {
  const ids = [
    "results-container", "reports-links", "figures-gallery", "results-status",
    "iris-form", "ai4i-form", "show-iris", "show-ai4i", "result", "result-content",
    "form-heading",
  ];
  const elements = Object.fromEntries(ids.map((id) => [id, new FakeElement()]));
  const document = {
    getElementById: (id) => elements[id],
    createElement: () => new FakeElement(),
  };
  const globals = {
    document,
    fetch: fetchImplementation,
    FormData: class {
      constructor(form) {
        this.form = form;
      }

      get(name) {
        return this.form.fields[name];
      }
    },
  };
  const previousDescriptors = Object.fromEntries(
    Object.keys(globals).map((name) => [name, Object.getOwnPropertyDescriptor(globalThis, name)]),
  );

  Object.entries(globals).forEach(([name, value]) => {
    Object.defineProperty(globalThis, name, { configurable: true, writable: true, value });
  });
  delete require.cache[require.resolve(appPath)];
  require(appPath);

  return {
    elements,
    cleanup() {
      delete require.cache[require.resolve(appPath)];
      Object.entries(previousDescriptors).forEach(([name, descriptor]) => {
        if (descriptor) Object.defineProperty(globalThis, name, descriptor);
        else delete globalThis[name];
      });
    },
  };
}

async function flushPagePromises() {
  await new Promise((resolve) => setImmediate(resolve));
}

test("homepage renders accuracy, precision, and recall from fixed results.json", async () => {
  const homepage = fs.readFileSync(path.join(__dirname, "../docs/index.html"), "utf8");
  assert.match(homepage, /<script src="app\.js" defer><\/script>/);
  assert.match(homepage, /id="results-container"/);
  const page = createPage(async () => ({
    ok: true,
    json: async () => fixedResults,
  }));
  try {
    await flushPagePromises();

    const results = page.elements["results-container"].textContent;
    for (const metric of ["Accuracy", "Precision", "Recall"]) {
      assert.ok(results.includes(metric), `homepage should display ${metric}`);
    }
    for (const value of ["0.91", "0.82", "0.73"]) {
      assert.ok(results.includes(value), `homepage should display fixed metric value ${value}`);
    }
  } finally {
    page.cleanup();
  }
});

test("known Iris form input displays the predicted class name", async () => {
  const page = createPage(async () => ({
    ok: true,
    json: async () => fixedResults,
  }));
  const form = page.elements["iris-form"];
  form.fields = {
    sepalLength: "5.1",
    sepalWidth: "3.5",
    petalLength: "1.4",
    petalWidth: "0.2",
  };
  try {
    form.listeners.submit({ preventDefault() {} });

    assert.equal(page.elements.result.hidden, false);
    assert.equal(page.elements["result-content"].children[0].textContent, "setosa");
  } finally {
    page.cleanup();
  }
});

test("results fetch failure displays an error message", async () => {
  const page = createPage(async () => {
    throw new Error("network unavailable");
  });

  try {
    await flushPagePromises();

    assert.match(page.elements["results-status"].textContent, /無法讀取 results\.json/);
  } finally {
    page.cleanup();
  }
});
