from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from typing import List

# Load your trained model
model = joblib.load("model.pkl")

app = FastAPI(title="ML Model API", description="API for ML model inference")

# Define input schema
class PredictionInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class PredictionOutput(BaseModel):
    prediction:  str
    confidence: float = None

@app.get("/")
def plant_check():
    return {"status": "Flower_Type", "message": "ML Model API is running"}

@app.post("/predict", response_model=PredictionOutput)

def predict(input_data: PredictionInput):
    features = np.array([[input_data.sepal_length, input_data.sepal_width, input_data.petal_length, input_data.petal_width]])
    try:
         
        # Convert input to model format
         features = np.array([[ 
            input_data.sepal_length, 
            input_data.sepal_width,
            input_data.petal_length, 
            input_data.petal_width
        ]])


      # Make prediction
         prediction = model.predict(features)[0]   
         probabilities = model.predict_proba(features)[0]  
         confidence = float(np.max(probabilities))

         target_mapping = {0: "setosa", 1: "versicolor", 2: "virginica"}

      # Return prediction
         return PredictionOutput(
            prediction=target_mapping[int(prediction)],
             confidence=confidence
            )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
@app.get("/model-info")
def model_info():
    return {
    "model_type": "KNN ",
    "problem_type": "classification",
    "features": ["sepal_length", "sepal_width", "petal_length", "petal_width"]
  }