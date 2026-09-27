"""Run from the repository root: python -m src.train --data 'data/raw/file.xlsx'."""
import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from .data import load_monthly, validate_target
from .modeling import MODELS, fit_model, predict, metrics

ROOT = Path(__file__).resolve().parents[1]


def train(data_path, ntd_id="00001", mode="MB", tos="DO", horizon=12, start=None, output_root=ROOT):
    if not 1 <= horizon <= 24:
        raise ValueError("Horizon must be between 1 and 24 months.")
    monthly, meta = load_monthly(data_path, ntd_id, mode, tos)
    if start:
        monthly = monthly.loc[pd.Timestamp(start):]
    y = monthly["ridership"]
    validate_target(y)
    # Three disjoint validation blocks precede a completely untouched test block.
    if len(y) - 4 * horizon < 48:
        raise ValueError("Not enough history: need at least 48 + 4*horizon months.")
    validation_rows, validation_predictions, failures = [], [], {}
    for name in MODELS:
        print(f"Validating {name}...", flush=True)
        model_rows, model_predictions = [], []
        try:
            for fold in range(3):
                cut = len(y) - (4 - fold) * horizon
                history, actual = y.iloc[:cut], y.iloc[cut:cut + horizon]
                fitted = fit_model(name, history)
                forecast = predict(name, fitted, history, horizon)
                model_rows.append({"model": name, "fold": fold + 1, "train_end": str(history.index[-1].date()),
                                   **metrics(actual, forecast)})
                model_predictions.extend({"model": name, "fold": fold + 1, "date": str(date.date()),
                                          "actual": float(value), "prediction": float(pred)}
                                         for date, value, pred in zip(actual.index, actual, forecast))
            validation_rows.extend(model_rows)
            validation_predictions.extend(model_predictions)
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            failures[name] = str(exc)
            print(f"Excluded {name}: {exc}", flush=True)
    if not validation_rows:
        raise RuntimeError(f"All models failed: {failures}")
    folds = pd.DataFrame(validation_rows)
    ranking = folds.groupby("model")[["mae", "rmse", "wape_percent"]].mean().sort_values(["mae", "rmse"])
    winner = str(ranking.index[0])
    # Test once after selection. Never select a model by this test score.
    history, actual = y.iloc[:-horizon], y.iloc[-horizon:]
    test_model = fit_model(winner, history)
    test_forecast = predict(winner, test_model, history, horizon)
    test = pd.DataFrame({"actual": actual, "prediction": test_forecast,
                         "seasonal_naive": predict("seasonal_naive", None, history, horizon),
                         "upt_listed_estimate": monthly.loc[actual.index, "upt_listed_estimate"]})
    production_model = fit_model(winner, y)
    future_dates = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    future = pd.DataFrame({"prediction": predict(winner, production_model, y, horizon)}, index=future_dates)
    future.index.name = "date"
    meta.update({"selected_model": winner, "horizon": horizon, "rows": len(y),
                 "history_start": str(y.index[0].date()), "history_end": str(y.index[-1].date()),
                 "test_start": str(actual.index[0].date()), "test_end": str(actual.index[-1].date()),
                 "test_metrics": metrics(actual, test_forecast),
                 "test_baseline_metrics": metrics(actual, test["seasonal_naive"]),
                 "excluded_models": failures, "upt_listed_estimate_months": int(monthly["upt_listed_estimate"].sum()),
                 "source_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
                 "created_utc": datetime.now(timezone.utc).isoformat(),
                 "versions": {"python": platform.python_version(), "pandas": pd.__version__, "sklearn": sklearn.__version__},
                 "fit_warnings": getattr(production_model, "fit_warning_messages", []),
                 "evaluation": "Fixed-origin recursive forecasts; 3 validation blocks; final test excluded from selection.",
                 "limitations": "Revised NTD release; not a real-time vintage backtest. Estimate flags are release-specific. Point forecasts only. Forecast begins after latest observed month, not today's date."})
    for folder in ("data/processed", "models", "reports"):
        (output_root / folder).mkdir(parents=True, exist_ok=True)
    monthly.to_csv(output_root / "data/processed/monthly.csv")
    folds.to_csv(output_root / "reports/validation_folds.csv", index=False)
    ranking.to_csv(output_root / "reports/model_comparison.csv")
    pd.DataFrame(validation_predictions).to_csv(output_root / "reports/validation_predictions.csv", index=False)
    test.to_csv(output_root / "reports/test_predictions.csv", index_label="date")
    future.to_csv(output_root / "reports/forecast.csv")
    (output_root / "reports/metadata.json").write_text(json.dumps(meta, indent=2, allow_nan=False))
    joblib.dump({"model": production_model, "name": winner, "history": y, "metadata": meta}, output_root / "models/forecast.joblib")
    print(ranking.to_string())
    print(f"Selected: {winner}; final test MAE: {meta['test_metrics']['mae']:,.0f} trips")
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--ntd-id", default="00001")
    parser.add_argument("--mode", default="MB")
    parser.add_argument("--tos", default="DO")
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--start", help="Optional first modeling month, e.g. 2010-01-01")
    args = parser.parse_args()
    train(args.data, args.ntd_id, args.mode, args.tos, args.horizon, args.start)


if __name__ == "__main__":
    main()
