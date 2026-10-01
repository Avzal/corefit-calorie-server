"""Tiny FastAPI wrapper around CalorieCLIP — the only job of this file is
"take a photo, return an estimated calorie count for the whole plate".
No database, no auth — Nutrition.jsx calls this directly from the browser.
"""
import io
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from calorie_clip import CalorieCLIP

MODEL_DIR = Path(__file__).parent
app = FastAPI(title="Corefit calorie estimator")

# Wide open for local dev. For a real deployment, replace "*" with your app's
# actual origin (e.g. "https://yourapp.example") so random sites can't call it.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

model = None


@app.on_event("startup")
def load_model():
    global model
    model = CalorieCLIP.from_pretrained(str(MODEL_DIR))


@app.get("/health")
def health():
    return {"ok": model is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(503, "Model still loading")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Upload an image file")

    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not read that image")

    calories = model.predict(image)
    calories = max(0, round(calories))
    return {
        "calories": calories,
        "note": "Estimate for the whole plate — not per-item.",
    }
