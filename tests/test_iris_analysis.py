import json
import tempfile
import unittest
from pathlib import Path

from iris_analysis import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    inspect_dataset,
    load_dataset,
    project_pca_3d,
    run_analysis,
    split_dataset,
)


class IrisAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_dataset()

    def test_dataset_has_expected_schema_and_classes(self):
        self.assertEqual(list(self.data.columns), [*FEATURE_COLUMNS, TARGET_COLUMN])
        self.assertEqual(len(self.data), 150)
        self.assertEqual(set(self.data[TARGET_COLUMN]), {
            "Iris-setosa",
            "Iris-versicolor",
            "Iris-virginica",
        })

    def test_exploration_checks_counts_quality_and_class_balance(self):
        summary = inspect_dataset(self.data)
        self.assertEqual(summary["rows"], 150)
        self.assertEqual(summary["feature_count"], 4)
        self.assertEqual(summary["missing_values"], 0)
        self.assertEqual(summary["duplicate_rows"], 3)
        self.assertEqual(summary["class_counts"], {
            "Iris-setosa": 50,
            "Iris-versicolor": 50,
            "Iris-virginica": 50,
        })

    def test_split_is_stratified_and_disjoint(self):
        X_train, X_test, y_train, y_test = split_dataset(self.data)
        self.assertEqual(len(X_test), 30)
        self.assertEqual(len(X_train), 120)
        self.assertEqual(y_test.value_counts().to_dict(), {
            "Iris-setosa": 10,
            "Iris-versicolor": 10,
            "Iris-virginica": 10,
        })
        self.assertEqual(set(X_train.index).intersection(X_test.index), set())
        self.assertEqual(set(FEATURE_COLUMNS), set(X_train.columns))

    def test_pca_returns_three_finite_components_and_variance(self):
        coordinates, explained_variance = project_pca_3d(self.data[FEATURE_COLUMNS])
        self.assertEqual(coordinates.shape, (150, 3))
        self.assertEqual(explained_variance.shape, (3,))
        self.assertTrue((coordinates == coordinates).all())
        self.assertTrue((explained_variance >= 0).all())
        self.assertLessEqual(float(explained_variance.sum()), 1.0)

    def test_end_to_end_run_writes_artifacts_and_evaluation(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            output = run_analysis(output_dir=Path(directory))
            for filename in (
                "pca_3d.png",
                "training_loss.png",
                "confusion_matrix.png",
                "classification_metrics.png",
                "metrics.json",
                "小論文.md",
            ):
                self.assertTrue((Path(directory) / filename).is_file(), filename)
            metrics = json.loads((Path(directory) / "metrics.json").read_text(encoding="utf-8"))
            self.assertGreaterEqual(metrics["accuracy"], 0.8)
            self.assertEqual(metrics["test_samples"], 30)
            self.assertEqual(output["metrics"]["accuracy"], metrics["accuracy"])


if __name__ == "__main__":
    unittest.main()
