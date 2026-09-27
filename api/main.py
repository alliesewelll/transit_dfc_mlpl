import json
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

ROOT = Path(__file__).resolve().parents[1]
app = FastAPI(title="Transit Forecast", version="0.1.0")


def records(path):
    frame = pd.read_csv(path)
    return json.loads(frame.to_json(orient="records"))


@app.get("/")
def dashboard():
    return FileResponse(ROOT / "api/dashboard.html")


@app.get("/health")
def health():
    return {"status": "ok", "trained": (ROOT / "reports/metadata.json").exists()}


@app.get("/results")
def results():
    try:
        return {"metadata": json.loads((ROOT / "reports/metadata.json").read_text()),
                "history": records(ROOT / "data/processed/monthly.csv"),
                "comparison": records(ROOT / "reports/model_comparison.csv"),
                "test": records(ROOT / "reports/test_predictions.csv"),
                "forecast": records(ROOT / "reports/forecast.csv")}
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Run python -m src.train --data PATH first.")


@app.get("/forecast")
def forecast():
    """Return the saved forecast, with its actual historical cutoff and horizon."""
    data = results()
    return {"model": data["metadata"]["selected_model"],
            "history_end": data["metadata"]["history_end"],
            "horizon": data["metadata"]["horizon"], "forecast": data["forecast"]}


@app.get("/download/{name}")
def download(name: str):
    allowed = {"forecast", "model_comparison", "test_predictions", "validation_folds", "validation_predictions"}
    if name not in allowed:
        raise HTTPException(status_code=404, detail="Unknown report")
    path = ROOT / "reports" / f"{name}.csv"
    if not path.exists():
        raise HTTPException(status_code=503, detail="Train the models first.")
    return FileResponse(path, media_type="text/csv", filename=path.name)
