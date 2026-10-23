"""Download and validate the UCI AI4I 2020 predictive-maintenance dataset."""

from __future__ import annotations

import argparse
import io
import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_integer_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)


ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "data" / "ai4i2020.csv"
VALIDATION_REPORT_PATH = ROOT_DIR / "outputs" / "ai4i_data_validation.json"
DATASET_PAGE_URL = (
    "https://archive.ics.uci.edu/dataset/601/"
    "ai4i+2020+predictive+maintenance+dataset"
)
DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/601/"
    "ai4i+2020+predictive+maintenance+dataset.zip"
)
EXPECTED_ROWS = 10_000
EXPECTED_COLUMNS = [
    "UDI",
    "Product ID",
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]
STRING_COLUMNS = {"Product ID", "Type"}
INTEGER_COLUMNS = {
    "UDI",
    "Rotational speed [rpm]",
    "Tool wear [min]",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
}
NUMERIC_COLUMNS = set(EXPECTED_COLUMNS) - STRING_COLUMNS


def inspect_ai4i_dataset(
    data: pd.DataFrame,
    *,
    expected_rows: int | None = EXPECTED_ROWS,
) -> dict[str, Any]:
    """Validate the AI4I schema and report row, dtype, missing, and duplicate counts."""
    if list(data.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            "Unexpected AI4I columns or order. "
            f"Expected {EXPECTED_COLUMNS!r}; got {list(data.columns)!r}."
        )
    if expected_rows is not None and len(data) != expected_rows:
        raise ValueError(f"Unexpected row count: expected {expected_rows}, got {len(data)}.")

    for column in STRING_COLUMNS:
        dtype = data[column].dtype
        if not (is_string_dtype(dtype) or is_object_dtype(dtype)):
            raise ValueError(f"Column {column!r} must contain string values; got {dtype}.")
    for column in NUMERIC_COLUMNS:
        dtype = data[column].dtype
        if not is_numeric_dtype(dtype) or is_bool_dtype(dtype):
            raise ValueError(f"Column {column!r} must be numeric; got {dtype}.")
    for column in INTEGER_COLUMNS:
        dtype = data[column].dtype
        if not is_integer_dtype(dtype) or is_bool_dtype(dtype):
            raise ValueError(f"Column {column!r} must be integer; got {dtype}.")

    missing_by_column = {
        str(column): int(count) for column, count in data.isna().sum().items()
    }
    missing_values = int(sum(missing_by_column.values()))
    duplicate_rows = int(data.duplicated().sum())
    return {
        "dataset": "AI4I 2020 Predictive Maintenance Dataset",
        "source": DATASET_PAGE_URL,
        "license": "CC BY 4.0",
        "rows": int(len(data)),
        "expected_rows": expected_rows,
        "columns": list(data.columns),
        "column_count": int(len(data.columns)),
        "column_types": {str(column): str(dtype) for column, dtype in data.dtypes.items()},
        "missing_values": missing_values,
        "missing_by_column": missing_by_column,
        "duplicate_rows": duplicate_rows,
        "valid": missing_values == 0 and duplicate_rows == 0,
    }


def load_ai4i_dataset(
    path: Path | str = DATA_PATH,
    *,
    expected_rows: int | None = EXPECTED_ROWS,
) -> pd.DataFrame:
    """Load a local CSV and enforce its schema, expected size, and column types."""
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"AI4I data file not found: {dataset_path}")
    data = pd.read_csv(dataset_path)
    inspect_ai4i_dataset(data, expected_rows=expected_rows)
    return data


def _extract_csv(payload: bytes) -> bytes:
    """Accept either a plain CSV response or the UCI ZIP download."""
    buffer = io.BytesIO(payload)
    if not zipfile.is_zipfile(buffer):
        return payload

    with zipfile.ZipFile(buffer) as archive:
        matches = [
            name
            for name in archive.namelist()
            if Path(name).name.casefold() == "ai4i2020.csv"
        ]
        if len(matches) != 1:
            raise ValueError(
                "UCI archive must contain exactly one ai4i2020.csv; "
                f"found {len(matches)}."
            )
        return archive.read(matches[0])


def download_ai4i_dataset(
    destination: Path | str = DATA_PATH,
    *,
    url: str = DATASET_URL,
    expected_rows: int | None = EXPECTED_ROWS,
) -> dict[str, Any]:
    """Download the official UCI CSV/ZIP, validate it, and save it atomically."""
    request = Request(url, headers={"User-Agent": "ai4i-data-audit/1.0"})
    with urlopen(request, timeout=60) as response:
        csv_bytes = _extract_csv(response.read())

    data = pd.read_csv(io.BytesIO(csv_bytes))
    report = inspect_ai4i_dataset(data, expected_rows=expected_rows)
    if not report["valid"]:
        raise ValueError(
            "Downloaded dataset failed quality checks: "
            f"{report['missing_values']} missing values and "
            f"{report['duplicate_rows']} duplicate rows."
        )

    target_path = Path(destination)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=target_path.parent, prefix=f".{target_path.name}.", delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(csv_bytes)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        temporary_path.replace(target_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return report


def write_validation_report(
    report: dict[str, Any],
    path: Path | str = VALIDATION_REPORT_PATH,
) -> Path:
    """Write the validation summary as UTF-8 JSON and return its path."""
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report_path


def run_validation(
    data_path: Path | str = DATA_PATH,
    report_path: Path | str = VALIDATION_REPORT_PATH,
    *,
    expected_rows: int | None = EXPECTED_ROWS,
) -> dict[str, Any]:
    """Validate the local dataset, save the audit report, and return its contents."""
    data = load_ai4i_dataset(data_path, expected_rows=expected_rows)
    report = inspect_ai4i_dataset(data, expected_rows=expected_rows)
    write_validation_report(report, report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download and validate the official UCI archive before auditing.",
    )
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="Path for the CSV data file")
    parser.add_argument(
        "--report", type=Path, default=VALIDATION_REPORT_PATH, help="Path for the JSON audit report"
    )
    args = parser.parse_args()

    if args.download:
        download_report = download_ai4i_dataset(args.data)
        print(
            f"Downloaded {download_report['rows']} rows and "
            f"{download_report['column_count']} columns to {args.data}"
        )
    report = run_validation(args.data, args.report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Validation report written to: {args.report}")
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
