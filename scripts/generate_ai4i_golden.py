"""Add fixed AI4I MLP predictions from the Python training pipeline to golden.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from ai4i_data import load_ai4i_dataset  # noqa: E402
from ai4i_preprocessing import (  # noqa: E402
    FEATURE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
    prepare_ai4i_data,
)


FIXED_INPUTS = [
    {
        "type": "L",
        "airTemperature": 300.7,
        "processTemperature": 312.0,
        "rotationalSpeed": 1550,
        "torque": 32.4,
        "toolWear": 5,
    },
    {
        "type": "L",
        "airTemperature": 300.8,
        "processTemperature": 309.9,
        "rotationalSpeed": 1312,
        "torque": 65.3,
        "toolWear": 192,
    },
    {
        "type": "M",
        "airTemperature": 299.9,
        "processTemperature": 310.2,
        "rotationalSpeed": 1450,
        "torque": 45.0,
        "toolWear": 120,
    },
]


def _to_model_frame(example: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Type": example["type"],
                "Air temperature [K]": example["airTemperature"],
                "Process temperature [K]": example["processTemperature"],
                "Rotational speed [rpm]": example["rotationalSpeed"],
                "Torque [Nm]": example["torque"],
                "Tool wear [min]": example["toolWear"],
            }
        ],
        columns=FEATURE_COLUMNS,
    )


def _build_ai4i_golden() -> dict[str, Any]:
    prepared = prepare_ai4i_data(load_ai4i_dataset(), test_size=0.2, random_state=42)
    model = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
    )
    model.fit(prepared.X_train, prepared.y_train)

    examples = []
    for values in FIXED_INPUTS:
        transformed = prepared.preprocessor.transform(_to_model_frame(values))
        probabilities = model.predict_proba(transformed)[0]
        examples.append(
            {
                "input": values,
                "className": int(model.classes_[int(np.argmax(probabilities))]),
                "probabilities": [float(value) for value in probabilities],
            }
        )

    scaler = prepared.preprocessor.named_transformers_["numeric"]
    return {
        "dataset": "AI4I 2020",
        "model": "OneHotEncoder + StandardScaler + MLPClassifier",
        "training": {
            "random_state": 42,
            "train_samples": int(len(prepared.y_train)),
            "iterations": int(model.n_iter_),
            "numeric_feature_order": list(NUMERIC_FEATURE_COLUMNS),
            "scaler_mean": [float(value) for value in scaler.mean_],
            "scaler_scale": [float(value) for value in scaler.scale_],
        },
        "classes": [int(value) for value in model.classes_],
        "examples": examples,
    }


def main() -> None:
    destination = ROOT_DIR / "tests" / "golden.json"
    golden = json.loads(destination.read_text(encoding="utf-8"))
    golden["ai4i"] = _build_ai4i_golden()
    destination.write_text(
        json.dumps(golden, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(golden['ai4i']['examples'])} Python AI4I predictions to {destination}")


if __name__ == "__main__":
    main()
