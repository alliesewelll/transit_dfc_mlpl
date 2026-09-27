import warnings
import numpy as np
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from .features import training_matrix, recursive_predict

MODELS = ("seasonal_naive", "random_forest", "gradient_boosting", "sarima")


def fit_model(name, y):
    if name == "seasonal_naive":
        return None
    if name == "sarima":
        # Scale to millions for numerical conditioning; invert after prediction.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = SARIMAX(y / 1_000_000, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12),
                            enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=200)
        if not model.mle_retvals.get("converged", False):
            raise RuntimeError("SARIMA optimization did not converge.")
        model.fit_warning_messages = [str(w.message) for w in caught]
        return model
    if name == "random_forest":
        model = RandomForestRegressor(n_estimators=200, min_samples_leaf=3, max_features=0.8, random_state=42, n_jobs=1)
    elif name == "gradient_boosting":
        model = HistGradientBoostingRegressor(max_iter=150, max_leaf_nodes=7, learning_rate=0.05,
                                             l2_regularization=1.0, early_stopping=False, random_state=42)
    else:
        raise ValueError(f"Unknown model: {name}")
    x, target = training_matrix(y)
    return model.fit(x, target)


def predict(name, model, y, horizon):
    if not 1 <= horizon <= 24:
        raise ValueError("Horizon must be between 1 and 24 months.")
    if name == "seasonal_naive":
        values = np.resize(y.iloc[-12:].to_numpy(), horizon)
    elif name == "sarima":
        values = np.asarray(model.forecast(horizon)) * 1_000_000
    else:
        values = recursive_predict(model, y, horizon)
    if not np.isfinite(values).all():
        raise ValueError("Model produced a non-finite forecast.")
    return np.maximum(values, 0)


def metrics(actual, predicted):
    errors = np.asarray(actual) - np.asarray(predicted)
    denominator = np.abs(actual).sum()
    return {"mae": float(np.abs(errors).mean()), "rmse": float(np.sqrt((errors ** 2).mean())),
            "wape_percent": float(100 * np.abs(errors).sum() / denominator) if denominator else None}
