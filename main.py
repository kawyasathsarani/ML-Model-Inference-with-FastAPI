from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# Create FastAPI app
app = FastAPI(title="ML Model API", description="API for Iris Classification")

# Load saved model
model = joblib.load("model.pkl")

# Species names (from iris dataset)
species = ["setosa", "versicolor", "virginica"]

# Input schema for prediction
from pydantic import BaseModel, Field

class PredictionInput(BaseModel):
    sepal_length: float
    inline_size: float = Field(..., alias="inline-size")  # valid Python name with alias
    petal_length: float
    inline_size: float = Field(..., alias="inline-size")

 


# Output schema
class PredictionOutput(BaseModel):
    prediction: str

# Health check endpoint
@app.get("/")
def health_check():
    return {"status": "healthy", "message": "ML Model API is running"}

# Prediction endpoint
@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    try:
        # Convert input to numpy array
        features = np.array([[input_data.sepal_length, input_data.sepal_width,
                              input_data.petal_length, input_data.petal_width]])
        
        # Make prediction
        pred = model.predict(features)[0]
        
        # Return result
        return {"prediction": species[pred]}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Model info endpoint
@app.get("/model-info")
def model_info():
    return {
        "model_type": "Logistic Regression",
        "problem_type": "Classification",
        "features": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
        "classes": species
    }