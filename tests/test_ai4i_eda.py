import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ai4i_data import EXPECTED_COLUMNS
from ai4i_eda import (
    create_ai4i_eda_artifacts,
    summarize_ai4i_dataset,
)


def make_dataset() -> pd.DataFrame:
    """Small, deliberately varied sample for deterministic EDA tests."""
    return pd.DataFrame(
        {
            "UDI": pd.Series([1, 2, 3, 4], dtype="int64"),
            "Product ID": ["M001", "L002", "H003", "M004"],
            "Type": ["M", "L", "H", "M"],
            "Air temperature [K]": [298.0, 300.0, 302.0, 304.0],
            "Process temperature [K]": [308.0, 310.0, 312.0, 314.0],
            "Rotational speed [rpm]": pd.Series([1000, 1200, 1400, 1600], dtype="int64"),
            "Torque [Nm]": [20.0, 30.0, 40.0, 50.0],
            "Tool wear [min]": pd.Series([0, 10, 20, 30], dtype="int64"),
            "Machine failure": pd.Series([0, 1, 0, 1], dtype="int64"),
            "TWF": pd.Series([0, 1, 0, 0], dtype="int64"),
            "HDF": pd.Series([0, 0, 0, 1], dtype="int64"),
            "PWF": pd.Series([0, 0, 0, 0], dtype="int64"),
            "OSF": pd.Series([0, 0, 0, 1], dtype="int64"),
            "RNF": pd.Series([0, 0, 0, 0], dtype="int64"),
        }
    )


class AI4IEDATests(unittest.TestCase):
    def test_summary_reports_schema_quality_and_descriptive_statistics(self):
        data = make_dataset()
        data.loc[0, "Torque [Nm]"] = float("nan")
        data.loc[3] = data.loc[2]

        report = summarize_ai4i_dataset(data)

        self.assertEqual(report["rows"], 4)
        self.assertEqual(report["column_count"], 14)
        self.assertEqual(report["columns"], EXPECTED_COLUMNS)
        self.assertIn(report["column_types"]["Type"], {"object", "str"})
        self.assertEqual(report["missing_values"], 1)
        self.assertEqual(report["missing_by_column"]["Torque [Nm]"], 1)
        self.assertEqual(report["duplicate_rows"], 1)
        self.assertEqual(report["descriptive_statistics"]["Rotational speed [rpm]"]["mean"], 1250)
        self.assertEqual(report["sensor_distributions"]["Torque [Nm]"]["count"], 3)

    def test_summary_counts_machine_failure_and_all_failure_modes(self):
        report = summarize_ai4i_dataset(make_dataset())

        self.assertEqual(report["label_counts"]["Machine failure"], {"0": 2, "1": 2})
        self.assertEqual(report["label_counts"]["TWF"], {"0": 3, "1": 1})
        self.assertEqual(report["label_counts"]["HDF"], {"0": 3, "1": 1})
        self.assertEqual(report["label_counts"]["PWF"], {"0": 4, "1": 0})
        self.assertEqual(report["label_counts"]["OSF"], {"0": 3, "1": 1})
        self.assertEqual(report["label_counts"]["RNF"], {"0": 4, "1": 0})

    def test_summary_reports_type_failure_rates_with_counts(self):
        report = summarize_ai4i_dataset(make_dataset())

        self.assertEqual(report["type_failure_rates"]["M"]["total"], 2)
        self.assertEqual(report["type_failure_rates"]["M"]["failures"], 1)
        self.assertEqual(report["type_failure_rates"]["M"]["failure_rate"], 0.5)
        self.assertEqual(report["type_failure_rates"]["L"]["failure_rate"], 1.0)
        self.assertEqual(report["type_failure_rates"]["H"]["failure_rate"], 0.0)

    def test_summary_reports_sensor_correlations_and_per_type_sensor_statistics(self):
        report = summarize_ai4i_dataset(make_dataset())

        self.assertAlmostEqual(
            report["sensor_correlation_matrix"]["Air temperature [K]"]["Torque [Nm]"],
            1.0,
        )
        self.assertEqual(
            report["sensor_distributions_by_type"]["H"]["Air temperature [K]"]["count"],
            1,
        )
        self.assertEqual(
            report["sensor_distributions_by_type"]["H"]["Air temperature [K]"]["mean"],
            302,
        )

    def test_summary_notes_explain_synthetic_data_and_failure_imbalance(self):
        data = make_dataset()
        data["Machine failure"] = pd.Series([0, 0, 0, 1], dtype="int64")

        report = summarize_ai4i_dataset(data)

        self.assertEqual(report["machine_failure_balance"]["positive_count"], 1)
        self.assertEqual(report["machine_failure_balance"]["negative_count"], 3)
        self.assertEqual(report["machine_failure_balance"]["positive_prevalence_percent"], 25.0)
        self.assertEqual(report["label_prevalence"]["Machine failure"]["positive_count"], 1)
        self.assertEqual(
            report["label_prevalence"]["Machine failure"]["positive_prevalence_percent"],
            25.0,
        )
        notes = " ".join(report["summary_notes"]).casefold()
        self.assertIn("synthetic", notes)
        self.assertIn("25.00%", notes)
        self.assertIn("imbalance", notes)
        self.assertIn("overlap", notes)

    def test_artifact_generation_writes_machine_readable_report_and_charts(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp_dir:
            output_dir = Path(temp_dir)

            artifacts = create_ai4i_eda_artifacts(make_dataset(), output_dir)

            report_path = output_dir / "summary.json"
            self.assertTrue(report_path.is_file())
            self.assertEqual(json.loads(report_path.read_text(encoding="utf-8"))["rows"], 4)
            self.assertEqual(set(artifacts["charts"]), {
                "label_counts",
                "type_failure_rates",
                "sensor_distributions_by_type",
                "sensor_correlation_heatmap",
            })
            for chart_path in artifacts["charts"].values():
                self.assertTrue(Path(chart_path).is_file())
                self.assertGreater(Path(chart_path).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
