# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import numpy as np
import joblib

app = FastAPI(
    title="ML Model API",
    description="API for ML model inference (Iris)",
    version="1.0"
)

# Load the saved model on startup (required)
@app.on_event("startup")
def load_model():
    global model, target_names
    bundle = joblib.load("model.pkl")
    model = bundle["model"]
    target_names = list(bundle["target_names"])

# Input validation with Pydantic (required)
class PredictionInput(BaseModel):
    sepal_length: float = Field(..., gt=0)
    sepal_width: float = Field(..., gt=0)
    petal_length: float = Field(..., gt=0)
    petal_width: float = Field(..., gt=0)

class PredictionOutput(BaseModel):
    prediction: str
    confidence: Optional[float] = None

# 1) Health check endpoint
@app.get("/")
def health_check():
    return {"status": "healthy", "message": "ML Model API is running"}

# 2) Predict endpoint (main endpoint required)
@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    try:
        features = np.array([[
            input_data.sepal_length,
            input_data.sepal_width,
            input_data.petal_length,
            input_data.petal_width
        ]])
        probs = model.predict_proba(features)[0]
        idx = int(np.argmax(probs))
        label = target_names[idx]
        return PredictionOutput(prediction=label, confidence=float(probs[idx]))
    except Exception as e:
        # Proper error handling (required)
        raise HTTPException(status_code=400, detail=str(e))

# 3) Model info endpoint
@app.get("/model-info")
def model_info():
    return {
        "model_type": "LogisticRegression with StandardScaler",
        "problem_type": "classification",
        "features": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
        "classes": target_names
    }
