"""Identical features for historical training and recursive prediction."""
import numpy as np
import pandas as pd

LAGS = (1, 2, 3, 6, 12, 24)
WINDOWS = (3, 6, 12)


def feature_row(history, date):
    if len(history) < 24:
        raise ValueError("Need at least 24 previous months.")
    if history.index[-1] + pd.offsets.MonthBegin(1) != date:
        raise ValueError("History must end exactly one month before the prediction date.")
    row = {"month_sin": np.sin(2 * np.pi * date.month / 12),
           "month_cos": np.cos(2 * np.pi * date.month / 12),
           "days_in_month": date.days_in_month}
    row.update({f"lag_{lag}": float(history.iloc[-lag]) for lag in LAGS})
    row.update({f"rolling_mean_{window}": float(history.iloc[-window:].mean()) for window in WINDOWS})
    return row


def training_matrix(y):
    rows = [feature_row(y.iloc[:i], y.index[i]) for i in range(24, len(y))]
    return pd.DataFrame(rows, index=y.index[24:]), y.iloc[24:]


def recursive_predict(model, history, horizon):
    history = history.copy()
    predictions = []
    for _ in range(horizon):
        date = history.index[-1] + pd.offsets.MonthBegin(1)
        x = pd.DataFrame([feature_row(history, date)])
        prediction = max(0.0, float(model.predict(x)[0]))
        predictions.append(prediction)
        # Future steps use previous predictions, never held-out actuals.
        history.loc[date] = prediction
    return np.asarray(predictions)
