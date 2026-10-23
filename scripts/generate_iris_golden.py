"""Generate browser inference fixtures from the trained Python Iris pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from iris_analysis import FEATURE_COLUMNS, _train_and_evaluate, load_dataset  # noqa: E402


FIXED_INPUTS = [
    [5.1, 3.5, 1.4, 0.2],
    [6.0, 2.9, 4.5, 1.5],
    [6.5, 3.0, 5.2, 2.0],
]


def _build_golden() -> dict[str, Any]:
    model, metrics = _train_and_evaluate(load_dataset())
    classes = [str(label) for label in model.classes_]
    examples = []
    for values in FIXED_INPUTS:
        sample = pd.DataFrame([values], columns=FEATURE_COLUMNS, dtype=float)
        probabilities = model.predict_proba(sample)[0]
        prediction = model.predict(sample)[0]
        examples.append(
            {
                "input": values,
                "className": str(prediction),
                "probabilities": [float(value) for value in probabilities],
            }
        )

    return {
        "dataset": "Iris",
        "model": "StandardScaler + MLPClassifier",
        "training": {
            "random_state": metrics["random_state"],
            "train_samples": metrics["train_samples"],
            "iterations": metrics["iterations"],
            "feature_order": FEATURE_COLUMNS,
        },
        "classes": classes,
        "examples": examples,
    }


def main() -> None:
    destination = ROOT_DIR / "tests" / "golden.json"
    golden = _build_golden()
    destination.write_text(
        json.dumps(golden, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(golden['examples'])} Python predictions to {destination}")


if __name__ == "__main__":
    main()
