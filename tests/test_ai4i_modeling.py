"""Contract tests for AI4I binary model evaluation and training."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import ai4i_modeling
from ai4i_modeling import evaluate_binary_predictions, run_ai4i_modeling
from ai4i_preprocessing import prepare_ai4i_data


def make_dataset(rows: int = 150) -> pd.DataFrame:
    """Build a small deterministic AI4I-shaped data frame for fast training."""
    rng = np.random.default_rng(2026)
    target = np.tile(np.array([0] * 4 + [1]), rows // 5 + 1)[:rows]
    rng.shuffle(target)
    return pd.DataFrame(
        {
            "UDI": np.arange(rows),
            "Product ID": [f"M{i:05d}" for i in range(rows)],
            "Type": np.resize(np.array(["L", "M", "H"]), rows),
            "Air temperature [K]": rng.normal(300 + target * 1.5, 2, rows),
            "Process temperature [K]": rng.normal(310 + target * 1.0, 2, rows),
            "Rotational speed [rpm]": rng.normal(1500 - target * 40, 80, rows),
            "Torque [Nm]": rng.normal(40 + target * 4, 8, rows),
            "Tool wear [min]": rng.integers(0, 250, rows),
            "Machine failure": target,
            "TWF": np.zeros(rows, dtype=int),
            "HDF": np.zeros(rows, dtype=int),
            "PWF": np.zeros(rows, dtype=int),
            "OSF": np.zeros(rows, dtype=int),
            "RNF": np.zeros(rows, dtype=int),
        }
    )


class BinaryPredictionMetricsTests(unittest.TestCase):
    def test_reports_thresholded_metrics_and_probability_curves(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_scores = np.array([0.1, 0.7, 0.4, 0.8, 0.3, 0.6])
        y_pred = (y_scores >= 0.5).astype(int)

        result = evaluate_binary_predictions(y_true, y_scores)

        self.assertEqual(result["positive_class"], 1)
        self.assertEqual(result["threshold"], 0.5)
        self.assertAlmostEqual(result["accuracy"], accuracy_score(y_true, y_pred))
        self.assertAlmostEqual(result["precision"], precision_score(y_true, y_pred))
        self.assertAlmostEqual(result["recall"], recall_score(y_true, y_pred))
        self.assertAlmostEqual(result["f1"], f1_score(y_true, y_pred))
        self.assertAlmostEqual(result["roc_auc"], roc_auc_score(y_true, y_scores))
        self.assertAlmostEqual(
            result["pr_auc"], average_precision_score(y_true, y_scores)
        )

    def test_confusion_matrix_keeps_binary_order_when_predictions_omit_positive(self):
        y_true = np.array([0, 1, 0, 1])
        y_scores = np.array([0.1, 0.2, 0.3, 0.4])

        result = evaluate_binary_predictions(y_true, y_scores, threshold=0.99)

        self.assertEqual(result["confusion_labels"], [0, 1])
        self.assertEqual(result["confusion_matrix"], [[2, 0], [2, 0]])
        self.assertEqual(result["threshold"], 0.99)


class AI4IModelingTests(unittest.TestCase):
    def test_models_share_reproducible_stratified_test_split_and_report_metadata(self):
        data = make_dataset()

        first = run_ai4i_modeling(data, test_size=0.2, random_state=19)
        second = run_ai4i_modeling(data, test_size=0.2, random_state=19)

        metadata = first["metadata"]
        self.assertEqual(metadata["split"]["test_size"], 0.2)
        self.assertEqual(metadata["split"]["train_samples"], 120)
        self.assertEqual(metadata["split"]["test_samples"], 30)
        self.assertTrue(metadata["split"]["stratified"])
        self.assertEqual(metadata["random_state"], 19)
        self.assertEqual(metadata["positive_class"], 1)
        self.assertEqual(len(first["dataset"]["data_sha256"]), 64)
        self.assertEqual(first["metadata"], second["metadata"])
        self.assertEqual(
            first["models"]["logistic_regression"]["model_params"]["penalty"], "l2"
        )

        self.assertEqual(set(first["models"]), {"logistic_regression", "mlp"})
        self.assertEqual(first["models"], second["models"])
        for model_name, model_result in first["models"].items():
            with self.subTest(model=model_name):
                self.assertIn("model_params", model_result)
                self.assertTrue(model_result["model_params"])
                metrics = model_result["metrics"]
                for key in (
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "roc_auc",
                    "pr_auc",
                ):
                    self.assertIn(key, metrics)
                    self.assertTrue(np.isfinite(metrics[key]))
                self.assertEqual(metrics["threshold"], 0.5)
                self.assertEqual(metrics["positive_class"], 1)
                self.assertEqual(metrics["confusion_labels"], [0, 1])
                self.assertEqual(sum(map(sum, metrics["confusion_matrix"])), 30)
                evaluation_data = model_result["evaluation_data"]
                self.assertEqual(len(evaluation_data["y_true"]), 30)
                self.assertEqual(len(evaluation_data["positive_scores"]), 30)
                self.assertTrue(set(evaluation_data["y_true"]).issubset({0, 1}))
                self.assertTrue(
                    np.isfinite(evaluation_data["positive_scores"]).all()
                )
                self.assertTrue(
                    ((np.asarray(evaluation_data["positive_scores"]) >= 0)
                    & (np.asarray(evaluation_data["positive_scores"]) <= 1)).all()
                )

        mlp = first["models"]["mlp"]
        self.assertTrue(mlp["training"]["loss_curve"])
        importance = mlp["permutation_importance"]
        self.assertEqual(importance["scoring"], "average_precision")
        self.assertEqual(
            importance["feature_names"],
            [
                "Type",
                "Air temperature [K]",
                "Process temperature [K]",
                "Rotational speed [rpm]",
                "Torque [Nm]",
                "Tool wear [min]",
            ],
        )
        self.assertEqual(importance["type_grouping"], "all Type one-hot columns permuted together")
        self.assertEqual(
            len(importance["importances_mean"]), len(importance["feature_names"])
        )
        self.assertEqual(
            len(importance["importances_std"]), len(importance["feature_names"])
        )
        self.assertTrue(np.isfinite(importance["importances_mean"]).all())
        self.assertTrue(np.isfinite(importance["importances_std"]).all())

    def test_modeling_records_independent_udi_ordered_split_metadata(self):
        data = make_dataset(150)
        data["UDI"] = np.random.default_rng(41).permutation(np.arange(1, 151))
        expected_order = np.argsort(data["UDI"].to_numpy(), kind="stable")

        report = run_ai4i_modeling(
            data,
            test_size=0.2,
            random_state=19,
            split_strategy="udi_ordered",
        )

        split = report["metadata"]["split"]
        self.assertEqual(split["strategy"], "udi_ordered")
        self.assertEqual(split["method"], "UDI-ordered row-block sensitivity split")
        self.assertFalse(split["stratified"])
        self.assertEqual(split["random_state"], None)
        self.assertEqual(split["train_samples"], 120)
        self.assertEqual(split["test_samples"], 30)
        np.testing.assert_array_equal(split["train_row_positions"], expected_order[:120])
        np.testing.assert_array_equal(split["test_row_positions"], expected_order[120:])
        self.assertNotIn("UDI", report["metadata"]["feature_names"])
        self.assertEqual(
            split["test_class_counts"],
            {
                str(key): int(value)
                for key, value in data.iloc[expected_order[120:]]["Machine failure"]
                .value_counts()
                .items()
            },
        )

    def test_estimators_fit_only_the_same_training_partition(self):
        data = make_dataset()
        prepared = prepare_ai4i_data(data, test_size=0.2, random_state=23)

        class RecordingEstimator:
            instances = []

            def __init__(self, **params):
                self.params = params
                self.__class__.instances.append(self)

            def fit(self, X, y):
                self.fit_X = np.array(X, copy=True)
                self.fit_y = np.array(y, copy=True)
                self.classes_ = np.array([0, 1])
                self.n_iter_ = 1
                return self

            def predict_proba(self, X):
                self.predict_X = np.array(X, copy=True)
                return np.tile(np.array([[0.8, 0.2]]), (len(X), 1))

            def get_params(self, deep=False):
                return dict(self.params)

        RecordingEstimator.instances = []
        def fake_permutation_importance(model, prepared, **kwargs):
            feature_names = [
                "Type",
                "Air temperature [K]",
                "Process temperature [K]",
                "Rotational speed [rpm]",
                "Torque [Nm]",
                "Tool wear [min]",
            ]
            return {
                "scoring": "average_precision",
                "n_repeats": 10,
                "random_state": kwargs["random_state"],
                "evaluation_partition": "held-out test set; post-hoc explanation only",
                "type_grouping": "all Type one-hot columns permuted together",
                "feature_names": feature_names,
                "importances_mean": [0.0] * len(feature_names),
                "importances_std": [0.0] * len(feature_names),
            }

        with (
            patch.object(ai4i_modeling, "LogisticRegression", RecordingEstimator),
            patch.object(ai4i_modeling, "MLPClassifier", RecordingEstimator),
            patch.object(
                ai4i_modeling,
                "_grouped_permutation_importance",
                fake_permutation_importance,
            ),
        ):
            run_ai4i_modeling(data, test_size=0.2, random_state=23)

        self.assertEqual(len(RecordingEstimator.instances), 2)
        for estimator in RecordingEstimator.instances:
            np.testing.assert_array_equal(estimator.fit_X, prepared.X_train)
            np.testing.assert_array_equal(estimator.fit_y, prepared.y_train.to_numpy())
            np.testing.assert_array_equal(estimator.predict_X, prepared.X_test)
            self.assertEqual(len(estimator.fit_X), 120)
            self.assertEqual(len(estimator.predict_X), 30)

    def test_writes_machine_readable_result_json_with_contract_keys(self):
        data = make_dataset()
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            output_path = Path(directory) / "model_results.json"

            result = run_ai4i_modeling(
                data, test_size=0.2, random_state=7, output_path=output_path
            )

            self.assertTrue(output_path.is_file())
            with output_path.open(encoding="utf-8") as stream:
                serialized = json.load(stream)
            self.assertEqual(serialized, result)
            self.assertIn("metadata", serialized)
            self.assertIn("models", serialized)
            for package in ("python", "numpy", "pandas", "scikit_learn"):
                self.assertIn(package, serialized["metadata"]["environment"])
            self.assertEqual(
                set(serialized["models"]), {"logistic_regression", "mlp"}
            )
            self.assertEqual(serialized["metadata"]["random_state"], 7)
            self.assertIn("average precision", serialized["metadata"]["metric_definitions"]["pr_auc"].lower())
            self.assertIn("rows=true", serialized["metadata"]["confusion_matrix_layout"])


if __name__ == "__main__":
    unittest.main()
