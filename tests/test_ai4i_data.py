import json
import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from ai4i_data import (
    EXPECTED_COLUMNS,
    download_ai4i_dataset,
    inspect_ai4i_dataset,
    load_ai4i_dataset,
    write_validation_report,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]


def make_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "UDI": pd.Series([1, 2], dtype="int64"),
            "Product ID": ["M14860", "L47181"],
            "Type": ["M", "L"],
            "Air temperature [K]": [298.1, 298.2],
            "Process temperature [K]": [308.6, 308.7],
            "Rotational speed [rpm]": pd.Series([1551, 1408], dtype="int64"),
            "Torque [Nm]": [42.8, 46.3],
            "Tool wear [min]": pd.Series([0, 3], dtype="int64"),
            "Machine failure": pd.Series([0, 0], dtype="int64"),
            "TWF": pd.Series([0, 0], dtype="int64"),
            "HDF": pd.Series([0, 0], dtype="int64"),
            "PWF": pd.Series([0, 0], dtype="int64"),
            "OSF": pd.Series([0, 0], dtype="int64"),
            "RNF": pd.Series([0, 0], dtype="int64"),
        }
    )


def make_zip(csv_bytes: bytes) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("ai4i2020.csv", csv_bytes)
    return buffer.getvalue()


class AI4IDataTests(unittest.TestCase):
    def test_inspection_reports_expected_schema_types_and_quality(self):
        report = inspect_ai4i_dataset(make_dataset(), expected_rows=2)

        self.assertEqual(report["rows"], 2)
        self.assertEqual(report["columns"], EXPECTED_COLUMNS)
        self.assertEqual(report["missing_values"], 0)
        self.assertEqual(report["duplicate_rows"], 0)
        self.assertEqual(report["column_types"]["UDI"], "int64")
        self.assertTrue(report["valid"])

    def test_inspection_reports_missing_values_and_duplicate_rows(self):
        data = make_dataset()
        data.loc[0, "Torque [Nm]"] = float("nan")
        data.loc[1] = data.loc[0]

        report = inspect_ai4i_dataset(data, expected_rows=2)

        self.assertEqual(report["missing_values"], 2)
        self.assertEqual(report["missing_by_column"]["Torque [Nm]"], 2)
        self.assertEqual(report["duplicate_rows"], 1)
        self.assertFalse(report["valid"])

    def test_inspection_rejects_unexpected_column_names_or_order(self):
        data = make_dataset().rename(columns={"RNF": "random_failure"})

        with self.assertRaisesRegex(ValueError, "columns"):
            inspect_ai4i_dataset(data, expected_rows=2)

    def test_inspection_rejects_unexpected_row_count(self):
        with self.assertRaisesRegex(ValueError, "row count"):
            inspect_ai4i_dataset(make_dataset(), expected_rows=10000)

    def test_inspection_rejects_invalid_column_types(self):
        data = make_dataset()
        data["UDI"] = data["UDI"].astype("string")

        with self.assertRaisesRegex(ValueError, "UDI"):
            inspect_ai4i_dataset(data, expected_rows=2)

    def test_load_dataset_reads_and_validates_csv(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_DIR) as directory:
            path = Path(directory) / "ai4i2020.csv"
            make_dataset().to_csv(path, index=False)

            loaded = load_ai4i_dataset(path, expected_rows=2)

        self.assertEqual(list(loaded.columns), EXPECTED_COLUMNS)
        self.assertEqual(len(loaded), 2)

    def test_download_extracts_official_csv_and_validates_before_saving(self):
        csv_bytes = make_dataset().to_csv(index=False).encode("utf-8")
        response = io.BytesIO(make_zip(csv_bytes))

        with tempfile.TemporaryDirectory(dir=PROJECT_DIR) as directory:
            destination = Path(directory) / "data" / "ai4i2020.csv"
            with patch("ai4i_data.urlopen", return_value=response):
                report = download_ai4i_dataset(
                    destination,
                    url="https://example.test/ai4i.zip",
                    expected_rows=2,
                )

            self.assertTrue(destination.is_file())
            saved = pd.read_csv(destination)

        self.assertEqual(report["rows"], 2)
        self.assertEqual(list(saved.columns), EXPECTED_COLUMNS)

    def test_validation_report_is_written_as_utf8_json(self):
        report = inspect_ai4i_dataset(make_dataset(), expected_rows=2)
        with tempfile.TemporaryDirectory(dir=PROJECT_DIR) as directory:
            path = Path(directory) / "validation.json"

            write_validation_report(report, path)

            loaded = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(loaded["rows"], 2)
        self.assertEqual(loaded["missing_values"], 0)


if __name__ == "__main__":
    unittest.main()
