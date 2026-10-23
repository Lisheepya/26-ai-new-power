"""Contract tests for visual artifacts connected to AI4I model reports."""

import tempfile
import unittest
import copy
from pathlib import Path

import numpy as np
import pandas as pd

from ai4i_modeling import run_ai4i_modeling
from ai4i_visualization import create_ai4i_visualizations, _pca_legend_labels


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
EXPECTED_VISUALS = {
    "pca_3d",
    "confusion_matrices",
    "precision_recall_curve",
    "roc_curve",
    "mlp_loss_curve",
    "mlp_permutation_importance",
}


def make_dataset(rows: int = 100) -> pd.DataFrame:
    """Make a deterministic, small AI4I-shaped fixture for end-to-end plotting."""
    rng = np.random.default_rng(1776)
    target = np.tile(np.array([0] * 4 + [1]), rows // 5 + 1)[:rows]
    rng.shuffle(target)
    return pd.DataFrame(
        {
            "UDI": np.arange(rows),
            "Product ID": [f"M{i:05d}" for i in range(rows)],
            "Type": np.resize(np.array(["L", "M", "H"]), rows),
            "Air temperature [K]": rng.normal(300 + target, 2, rows),
            "Process temperature [K]": rng.normal(310 + target, 2, rows),
            "Rotational speed [rpm]": rng.normal(1500 - target * 30, 75, rows),
            "Torque [Nm]": rng.normal(40 + target * 3, 8, rows),
            "Tool wear [min]": rng.integers(0, 250, rows),
            "Machine failure": target,
            "TWF": np.zeros(rows, dtype=int),
            "HDF": np.zeros(rows, dtype=int),
            "PWF": np.zeros(rows, dtype=int),
            "OSF": np.zeros(rows, dtype=int),
            "RNF": np.zeros(rows, dtype=int),
        }
    )


class AI4IVisualizationTests(unittest.TestCase):
    def test_creates_six_pngs_from_a_modeling_report(self):
        data = make_dataset()
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            report = run_ai4i_modeling(
                data,
                test_size=0.2,
                random_state=29,
                output_path=root / "model_comparison.json",
            )
            output_dir = root / "visualizations"

            paths = create_ai4i_visualizations(data, report, output_dir)

            self.assertEqual(set(paths), EXPECTED_VISUALS)
            for name, path_value in paths.items():
                with self.subTest(visualization=name):
                    path = Path(path_value)
                    self.assertTrue(path.is_file(), f"missing {name}: {path}")
                    self.assertEqual(path.parent.resolve(), output_dir.resolve())
                    with path.open("rb") as image_file:
                        self.assertEqual(image_file.read(8), PNG_SIGNATURE)
                    self.assertGreater(path.stat().st_size, 100)

            test_count = report["metadata"]["split"]["test_samples"]
            for model_result in report["models"].values():
                evaluation_data = model_result["evaluation_data"]
                self.assertEqual(len(evaluation_data["y_true"]), test_count)
                self.assertEqual(
                    len(evaluation_data["positive_scores"]), test_count
                )
                self.assertEqual(
                    model_result["metrics"]["confusion_labels"], [0, 1]
                )

            mlp = report["models"]["mlp"]
            self.assertTrue(mlp["training"]["loss_curve"])
            self.assertTrue(mlp["permutation_importance"]["feature_names"])

            changed_sensor_data = data.copy()
            changed_sensor_data.loc[0, "Torque [Nm]"] += 0.01
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                create_ai4i_visualizations(
                    changed_sensor_data,
                    report,
                    output_dir,
                )

            stale_metrics = copy.deepcopy(report)
            stale_metrics["models"]["mlp"]["metrics"]["recall"] = 0.0
            with self.assertRaisesRegex(ValueError, "do not match"):
                create_ai4i_visualizations(data, stale_metrics, output_dir)

    def test_pca_legend_discloses_failure_counts_and_prevalence(self):
        legend, prevalence = _pca_legend_labels(
            train_y=np.array([0] * 8 + [1] * 2),
            test_y=np.array([0] * 3 + [1]),
        )

        self.assertEqual(legend["no_failure_train"], "No failure — train (n=8)")
        self.assertEqual(legend["failure_train"], "Failure — train (n=2)")
        self.assertEqual(legend["no_failure_test"], "No failure — test (n=3)")
        self.assertEqual(legend["failure_test"], "Failure — test (n=1)")
        self.assertEqual(prevalence, "Failure prevalence: 21.4% (3/14)")


if __name__ == "__main__":
    unittest.main()
