import unittest

import numpy as np
import pandas as pd

from ai4i_preprocessing import (
    EXCLUDED_COLUMNS,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    prepare_ai4i_data,
)


def make_dataset(rows: int = 250) -> pd.DataFrame:
    """Return a deterministic AI4I-shaped frame with a nontrivial failure rate."""
    rng = np.random.default_rng(123)
    failures = np.array([0] * (rows - rows // 10) + [1] * (rows // 10))
    rng.shuffle(failures)
    return pd.DataFrame(
        {
            "UDI": np.arange(rows),
            "Product ID": [f"M{i:05d}" for i in range(rows)],
            "Type": np.resize(np.array(["L", "M", "H"]), rows),
            "Air temperature [K]": rng.normal(300, 3, rows),
            "Process temperature [K]": rng.normal(310, 3, rows),
            "Rotational speed [rpm]": rng.normal(1500, 100, rows),
            "Torque [Nm]": rng.normal(40, 10, rows),
            "Tool wear [min]": rng.integers(0, 250, rows),
            "Machine failure": failures,
            "TWF": rng.integers(0, 2, rows),
            "HDF": rng.integers(0, 2, rows),
            "PWF": rng.integers(0, 2, rows),
            "OSF": rng.integers(0, 2, rows),
            "RNF": rng.integers(0, 2, rows),
        }
    )


class AI4IPreprocessingTests(unittest.TestCase):
    def test_only_type_and_five_sensors_are_model_features(self):
        data = make_dataset()

        prepared = prepare_ai4i_data(data)

        expected = (
            "Type",
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]",
        )
        self.assertEqual(FEATURE_COLUMNS, expected)
        self.assertEqual(TARGET_COLUMN, "Machine failure")
        self.assertEqual(
            set(EXCLUDED_COLUMNS),
            {"UDI", "Product ID", "Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"},
        )
        self.assertEqual(tuple(prepared.X_train_raw.columns), expected)
        self.assertEqual(tuple(prepared.X_test_raw.columns), expected)
        self.assertEqual(set(prepared.y_train.unique()), {0, 1})
        self.assertEqual(set(prepared.y_test.unique()), {0, 1})
        self.assertEqual(prepared.X_train.shape[1], len(prepared.feature_names))
        self.assertEqual(prepared.X_test.shape[1], len(prepared.feature_names))
        self.assertEqual(prepared.y_train.name, "Machine failure")
        self.assertEqual(prepared.y_test.name, "Machine failure")

    def test_split_is_stratified_and_reproducible(self):
        data = make_dataset(500)

        first = prepare_ai4i_data(data, test_size=0.2, random_state=19)
        second = prepare_ai4i_data(data, test_size=0.2, random_state=19)

        self.assertEqual(len(first.X_test_raw), 100)
        self.assertAlmostEqual(first.y_train.mean(), data[TARGET_COLUMN].mean(), delta=0.01)
        self.assertAlmostEqual(first.y_test.mean(), data[TARGET_COLUMN].mean(), delta=0.01)
        pd.testing.assert_frame_equal(first.X_train_raw, second.X_train_raw)
        pd.testing.assert_frame_equal(first.X_test_raw, second.X_test_raw)
        pd.testing.assert_series_equal(first.y_train, second.y_train)
        pd.testing.assert_series_equal(first.y_test, second.y_test)
        self.assertEqual(first.split_strategy, "stratified_random")
        self.assertEqual(set(first.train_row_positions) & set(first.test_row_positions), set())
        self.assertEqual(
            set(first.train_row_positions) | set(first.test_row_positions),
            set(range(len(data))),
        )

    def test_udi_ordered_split_uses_disjoint_sorted_blocks_and_train_only_fit(self):
        data = make_dataset(500)
        data["UDI"] = np.random.default_rng(20261023).permutation(np.arange(1, 501))
        expected_order = np.argsort(data["UDI"].to_numpy(), kind="stable")

        prepared = prepare_ai4i_data(
            data,
            test_size=0.2,
            random_state=19,
            split_strategy="udi_ordered",
        )

        expected_train = expected_order[:400]
        expected_test = expected_order[400:]
        np.testing.assert_array_equal(prepared.train_row_positions, expected_train)
        np.testing.assert_array_equal(prepared.test_row_positions, expected_test)
        self.assertEqual(prepared.split_strategy, "udi_ordered")
        self.assertEqual(len(prepared.X_train_raw), 400)
        self.assertEqual(len(prepared.X_test_raw), 100)
        self.assertLess(
            data.iloc[list(prepared.train_row_positions)]["UDI"].max(),
            data.iloc[list(prepared.test_row_positions)]["UDI"].min(),
        )
        self.assertEqual(
            int(prepared.y_train.sum()),
            int(data.iloc[expected_train][TARGET_COLUMN].sum()),
        )
        self.assertEqual(
            int(prepared.y_test.sum()),
            int(data.iloc[expected_test][TARGET_COLUMN].sum()),
        )

        numeric_columns = list(
            ("Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]")
        )
        scaler = prepared.preprocessor.named_transformers_["numeric"]
        np.testing.assert_allclose(
            scaler.mean_, prepared.X_train_raw[numeric_columns].mean().to_numpy()
        )

    def test_udi_ordered_split_rejects_duplicate_identifiers(self):
        data = make_dataset(100)
        data.loc[1, "UDI"] = data.loc[0, "UDI"]

        with self.assertRaisesRegex(ValueError, "UDI.*unique"):
            prepare_ai4i_data(data, split_strategy="udi_ordered")

    def test_unknown_split_strategy_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "split_strategy"):
            prepare_ai4i_data(make_dataset(), split_strategy="unknown")

    def test_scaler_and_encoder_are_fitted_on_training_rows_only(self):
        data = make_dataset()
        prepared = prepare_ai4i_data(data, test_size=0.25, random_state=7)

        numeric_scaler = prepared.preprocessor.named_transformers_["numeric"]
        type_encoder = prepared.preprocessor.named_transformers_["type"]
        numeric_columns = [
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]",
        ]
        expected_train_means = prepared.X_train_raw[numeric_columns].mean().to_numpy()
        full_data_means = data[numeric_columns].mean().to_numpy()

        np.testing.assert_allclose(numeric_scaler.mean_, expected_train_means)
        self.assertFalse(np.allclose(numeric_scaler.mean_, full_data_means))
        np.testing.assert_array_equal(
            type_encoder.categories_[0],
            np.sort(prepared.X_train_raw["Type"].unique()),
        )

    def test_missing_required_columns_are_rejected(self):
        data = make_dataset().drop(columns=["Torque [Nm]"])

        with self.assertRaisesRegex(ValueError, r"Torque \[Nm\]"):
            prepare_ai4i_data(data)

    def test_non_binary_target_is_rejected(self):
        data = make_dataset()
        data.loc[data.index[0], TARGET_COLUMN] = 2

        with self.assertRaisesRegex(ValueError, "binary"):
            prepare_ai4i_data(data)


if __name__ == "__main__":
    unittest.main()
