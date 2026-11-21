# DS422 — Smart Room Occupancy Detection: 3D Website + Live Model Backend

A sensor-ceiling themed single-page website for the DS422 (ML-I) lab project of
Kaniz Fatema Ema (232-35-189), with a **live prediction** powered by the exported
Random Forest pipeline (`smart_room_occupancy_model.pkl`) through a FastAPI backend.

## Project structure
```
occupancy_3d_website/
├── index.html   # 8 sections: Hero → Live Prediction → Performance → Confusion
│                #   Matrix → Feature Importance → Dataset → Pipeline → Conclusion
├── styles.css   # Warm control-room theme, responsive
├── app.js       # Sensor grid, presets, sliders, predict flow (never fakes a result)
├── bg3d.js      # Three.js sensor-field background (graceful fallback)
├── data.js      # Verified metrics, top features and the 3 real preset rows
└── backend/
    ├── app.py                        # FastAPI: GET /health /metrics /presets, POST /predict
    ├── smart_room_occupancy_model.pkl# The exported pipeline (imputer + Random Forest)
    ├── metrics.json                  # Verified results (all 4 models match the notebook exactly)
    ├── presets.json                  # One real, correctly-predicted test row per class
    ├── train_and_verify.py           # Re-trains from the dataset CSV and re-verifies everything
    └── requirements.txt              # NOTE: scikit-learn pinned to 1.7.2 (see below)
```

## Run it (two terminals)
**Terminal 1 — backend:**
```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --host 127.0.0.1 --port 8002
```
**Terminal 2 — frontend:**
```bash
cd occupancy_3d_website
python3 -m http.server 5502
# open http://127.0.0.1:5502
```
The predictor calls `http://127.0.0.1:8002` by default; change the "Backend URL"
field on the page if needed. If the backend is down, the page shows a clear
error — it never fakes a result.

## ⚠ Version note (important)
The notebook was run in Colab with **scikit-learn 1.7.2**, and Random Forest
results differ slightly under other versions. The exported .pkl was re-trained
and verified under 1.7.2, reproducing the notebook **exactly**:
all four models (RF 99.41% / 98.88% / 98.72% / 98.80%), the confusion matrix
[[1247, 2, 1], [5, 161, 1], [0, 0, 114]], and the first test sample (0 → 0).
Keep `scikit-learn==1.7.2` from requirements.txt when running the backend.

## Viva rehearsal
Use the three preset buttons — each loads a real test row:
- **Vacancy** → Vacancy · **Stationary Human** → Stationary Human Presence ·
  **Other Activity** → Other Activity / Motion (watch the sensor ceiling light up).
