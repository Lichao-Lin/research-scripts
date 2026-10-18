import io
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forecasting import FEATURE_COLUMNS, prepare_forecast_data, rolling_forecast


def sample_data(count=40):
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "Trddt": pd.bdate_range("2026-01-01", periods=count),
        "CSSD1": rng.normal(size=count),
        "Dretwdeq": np.arange(count, dtype=float) / 1000,
    })


class ForecastTests(unittest.TestCase):
    def test_targets_follow_original_dates_after_sorting(self):
        source = sample_data().sample(frac=1, random_state=2)
        original = source.copy(deep=True)
        prepared = prepare_forecast_data(source)
        ordered = original.sort_values("Trddt").reset_index(drop=True)
        for row in prepared.itertuples():
            position = ordered.index[ordered.Trddt == row.Trddt][0]
            self.assertEqual(row.Target_date, ordered.iloc[position + 1].Trddt)
            self.assertEqual(row.Return_next, ordered.iloc[position + 1].Dretwdeq)
        pd.testing.assert_frame_equal(source, original)

    def test_missing_features_do_not_skip_target_dates(self):
        source = sample_data()
        source.loc[20, "CSSD1"] = np.nan
        prepared = prepare_forecast_data(source)
        row = prepared.loc[prepared.Trddt == source.loc[19, "Trddt"]].iloc[0]
        self.assertEqual(row.Target_date, source.loc[20, "Trddt"])
        self.assertEqual(row.Return_next, source.loc[20, "Dretwdeq"])

    def test_unrelated_missing_column_does_not_remove_observations(self):
        source = sample_data()
        source["unused"] = np.nan
        self.assertEqual(len(prepare_forecast_data(source)), len(source) - 10)

    def test_missing_target_does_not_shift_to_later_return(self):
        source = sample_data()
        source.loc[20, "Dretwdeq"] = np.nan
        prepared = prepare_forecast_data(source)
        self.assertNotIn(source.loc[19, "Trddt"], prepared.Trddt.tolist())

    def test_duplicate_dates_are_rejected(self):
        source = sample_data()
        source.loc[20, "Trddt"] = source.loc[19, "Trddt"]
        with self.assertRaisesRegex(ValueError, "unique"):
            prepare_forecast_data(source)

    def test_missing_required_column_is_reported(self):
        with self.assertRaisesRegex(ValueError, "Dretwdeq"):
            prepare_forecast_data(sample_data().drop(columns="Dretwdeq"))

    def test_rolling_output_preserves_values_and_training_order(self):
        prepared = prepare_forecast_data(sample_data())
        original = prepared.copy(deep=True)
        calls = []

        class RecordingModel:
            feature_importances_ = np.ones(len(FEATURE_COLUMNS)) / len(FEATURE_COLUMNS)

            def fit(self, x, y):
                calls.append((x.index.tolist(), y.index.tolist()))
                self.prediction = float(y.mean())

            def predict(self, x):
                self.test_index = x.index[0]
                self.assert_training_end = calls[-1][0][-1] < self.test_index
                if not self.assert_training_end:
                    raise AssertionError("Training includes forecast row")
                return [self.prediction]

        output, _, importance = rolling_forecast(prepared, 8, RecordingModel)
        pd.testing.assert_frame_equal(prepared, original)
        pd.testing.assert_frame_equal(output.drop(columns="Return_pred"), prepared.iloc[8:])
        self.assertEqual(len(output), len(prepared) - 8)
        self.assertEqual(importance.shape, (len(output), len(FEATURE_COLUMNS)))
        for position, (x_indices, y_indices) in enumerate(calls):
            self.assertEqual(x_indices, list(range(position, position + 8)))
            self.assertEqual(y_indices, x_indices)
            self.assertAlmostEqual(output.Return_pred.iloc[position], prepared.Return_next.iloc[position:position + 8].mean())
        csv = io.StringIO()
        output.to_csv(csv, index=False)
        csv.seek(0)
        restored = pd.read_csv(csv, parse_dates=["Trddt", "Target_date"])
        pd.testing.assert_frame_equal(
            restored[["Trddt", "Dretwdeq", "Target_date", "Return_next", "Return_pred"]],
            output[["Trddt", "Dretwdeq", "Target_date", "Return_next", "Return_pred"]].reset_index(drop=True),
        )

    def test_short_data_and_invalid_window_are_rejected(self):
        prepared = prepare_forecast_data(sample_data())
        for size in [0, -1, 1.5, True]:
            with self.subTest(size=size), self.assertRaisesRegex(ValueError, "positive integer"):
                rolling_forecast(prepared, size)
        for length in [0, 7, 8]:
            with self.subTest(length=length), self.assertRaisesRegex(ValueError, "valid observations"):
                rolling_forecast(prepared.iloc[:length], 8)

    def test_exactly_one_out_of_sample_forecast(self):
        from xgboost import XGBRegressor

        prepared = prepare_forecast_data(sample_data(23))
        output, model, _ = rolling_forecast(prepared, len(prepared) - 1)
        self.assertIsInstance(model, XGBRegressor)
        self.assertEqual(model.get_params()["n_estimators"], 100)
        self.assertEqual(len(output), 1)
        self.assertTrue(np.isfinite(output.Return_pred.iloc[0]))
        self.assertEqual(output.Target_date.iloc[0], sample_data(23).Trddt.iloc[-1])

    def test_training_labels_are_known_at_forecast_time_even_with_gaps(self):
        source = sample_data(50)
        source.loc[25, "CSSD1"] = np.nan
        prepared = prepare_forecast_data(source)
        case = self

        class AvailabilityModel:
            feature_importances_ = np.ones(len(FEATURE_COLUMNS)) / len(FEATURE_COLUMNS)

            def fit(self, x, y):
                self.label_dates = prepared.loc[y.index, "Target_date"]

            def predict(self, x):
                forecast_time = prepared.loc[x.index[0], "Trddt"]
                case.assertTrue((self.label_dates <= forecast_time).all())
                return [0.0]

        output, _, _ = rolling_forecast(prepared, 8, AvailabilityModel)
        self.assertGreater(len(output), 1)

    def test_nonfinite_predictions_are_rejected(self):
        class InvalidModel:
            def fit(self, x, y):
                pass

            def predict(self, x):
                return [np.nan]

        prepared = prepare_forecast_data(sample_data())
        with self.assertRaisesRegex(ValueError, "non-finite"):
            rolling_forecast(prepared, 8, InvalidModel)


if __name__ == "__main__":
    unittest.main()
