"""DS422 Smart Room Occupancy Detection - FastAPI backend.
Loads smart_room_occupancy_model.pkl (a sklearn Pipeline: median imputer + Random Forest,
exported from Lab_Project__232_35_189.ipynb) and serves live predictions.
IMPORTANT: the .pkl was created with scikit-learn 1.7.2 (the Colab version that reproduces
the notebook's results exactly). Use scikit-learn==1.7.2 - see requirements.txt.
Run:  uvicorn app:app --host 127.0.0.1 --port 8002
"""
import json
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

HERE = Path(__file__).resolve().parent
STATE = {"model": None, "metrics": None, "presets": None, "startup_error": None}
LABEL_NAMES = {0: "Vacancy", 1: "Stationary Human Presence", 3: "Other Activity / Motion"}

try:
    STATE["model"] = joblib.load(HERE / "smart_room_occupancy_model.pkl")
    STATE["metrics"] = json.loads((HERE / "metrics.json").read_text())
    STATE["presets"] = json.loads((HERE / "presets.json").read_text())
except Exception as exc:
    STATE["startup_error"] = f"{type(exc).__name__}: {exc}"

app = FastAPI(title="Smart Room Occupancy Prediction API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class PredictRequest(BaseModel):
    features: dict

@app.get("/health")
def health():
    return {"status": "ok" if STATE["model"] is not None else "error",
            "model_loaded": STATE["model"] is not None,
            "model_file": "smart_room_occupancy_model.pkl",
            "sklearn_note": "pipeline exported with scikit-learn 1.7.2",
            "startup_error": STATE["startup_error"]}

@app.get("/metrics")
def metrics():
    return STATE["metrics"]

@app.get("/presets")
def presets():
    return STATE["presets"]

@app.post("/predict")
def predict(req: PredictRequest):
    if STATE["model"] is None:
        return {"error": "Model not loaded", "detail": STATE["startup_error"]}
    order = STATE["metrics"]["feature_order"]
    missing = [c for c in order if c not in req.features]
    if missing:
        return {"error": f"Missing features: {missing[:5]}... ({len(missing)} missing)"}
    row = pd.DataFrame([{c: float(req.features[c]) for c in order}], columns=order)
    model = STATE["model"]
    pred = int(model.predict(row)[0])
    proba = model.predict_proba(row)[0]
    classes = [int(c) for c in model.classes_]
    probabilities = [{"label": c, "status": LABEL_NAMES[c], "probability": round(float(p), 4)} for c, p in zip(classes, proba)]
    pir_vals = [float(req.features[f"PIR_{i}"]) for i in range(1, 56)]
    mean_pir = sum(pir_vals) / len(pir_vals)
    temp = float(req.features["Temperature_F"])
    if pred == 3:
        reason = ("The model sees the 'Other Activity / Motion' pattern: PIR readings are far above the "
                  "empty-room level and the temperature is recorded as 0 F - in this dataset every "
                  "Other Activity row has Temperature_F = 0, which is why temperature is the top feature.")
    elif pred == 1:
        reason = ("The model sees a person present but mostly still: temperature is in the normal office "
                  "range and the PIR pattern matches the Stationary Human examples (driven most by "
                  "PIR_4, PIR_47 and PIR_48).")
    else:
        reason = ("The model sees the empty-room pattern: temperature is in the normal office range and "
                  "the PIR sensors sit at their flat background level, like the Vacancy examples.")
    return {"prediction": pred, "status": LABEL_NAMES[pred],
            "confidence": round(float(max(proba)), 4), "probabilities": probabilities,
            "mean_pir": round(mean_pir, 1), "temperature_f": temp,
            "reason": reason, "model": "Random Forest pipeline (smart_room_occupancy_model.pkl)"}

# Serve the frontend (repo root) from the same service, so one Render URL runs
# both the page and the API. Mounted last, after the API routes above.
from fastapi.staticfiles import StaticFiles
FRONTEND_DIR = HERE.parent
if (FRONTEND_DIR / "index.html").exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
