"""Prepare next-observation targets and preserve observed returns in forecasts."""

from numbers import Integral

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "CSSD1", "CSSD1_lag1", "CSSD1_lag2", "CSSD1_lag5",
    "CSSD1_ma5", "CSSD1_ma10",
]


def prepare_forecast_data(source):
    """Create targets before filtering, assuming one row per trading date."""
    required = {"Trddt", "CSSD1", "Dretwdeq"}
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    data = source.copy()
    data["Trddt"] = pd.to_datetime(data["Trddt"], errors="raise")
    if data["Trddt"].isna().any() or data["Trddt"].duplicated().any():
        raise ValueError("Trddt must contain unique, non-missing trading dates")
    data = data.sort_values("Trddt").reset_index(drop=True)
    for column in ["CSSD1", "Dretwdeq"]:
        data[column] = pd.to_numeric(data[column], errors="raise")

    # Bind each target to the next original observation, not the next valid feature row.
    data["Target_date"] = data["Trddt"].shift(-1)
    data["Return_next"] = data["Dretwdeq"].shift(-1)
    for lag in [1, 2, 5]:
        data[f"CSSD1_lag{lag}"] = data["CSSD1"].shift(lag)
    for window in [5, 10]:
        data[f"CSSD1_ma{window}"] = data["CSSD1"].rolling(window).mean()

    numeric = FEATURE_COLUMNS + ["Return_next"]
    data[numeric] = data[numeric].replace([np.inf, -np.inf], np.nan)
    return data.dropna(subset=numeric + ["Target_date"]).reset_index(drop=True)


def rolling_forecast(data, window_size=240, model_factory=None):
    """Train on preceding valid observations and forecast at each row's close."""
    if isinstance(window_size, bool) or not isinstance(window_size, Integral) or window_size < 1:
        raise ValueError("window_size must be a positive integer")
    if len(data) <= window_size:
        raise ValueError(
            f"Need more than {window_size} valid observations; got {len(data)}"
        )
    if model_factory is None:
        from xgboost import XGBRegressor

        def model_factory():
            return XGBRegressor(
                n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
            )

    features = data[FEATURE_COLUMNS]
    targets = data["Return_next"]
    predictions, importances = [], []
    for end in range(window_size, len(data)):
        model = model_factory()
        model.fit(features.iloc[end - window_size:end], targets.iloc[end - window_size:end])
        prediction = float(model.predict(features.iloc[end:end + 1])[0])
        if not np.isfinite(prediction):
            raise ValueError("Model produced a non-finite prediction")
        predictions.append(prediction)
        importances.append(model.feature_importances_)

    output = data.iloc[window_size:].copy()
    output["Return_pred"] = predictions
    importance_frame = pd.DataFrame(importances, index=output.index, columns=FEATURE_COLUMNS)
    return output, model, importance_frame
