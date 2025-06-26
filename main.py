from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
from typing import List, Optional
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Heart Disease Prediction API",
    description="A machine learning API for predicting heart disease based on patient features",
    version="1.0.0"
)

# Load the trained model
try:
    model = joblib.load("heart_disease_model.pkl")
except FileNotFoundError:
    print("Warning: Model file not found. Please ensure 'heart_disease_model.pkl' is in the current directory.")
    model = None

# Define the input data model
class HeartDiseaseInput(BaseModel):
    age: int
    sex: int  # 0 = female, 1 = male
    cp: int  # chest pain type (0-3)
    trestbps: int  # resting blood pressure
    chol: int  # serum cholesterol
    fbs: int  # fasting blood sugar > 120 mg/dl (0 = false, 1 = true)
    restecg: int  # resting electrocardiographic results (0-2)
    thalach: int  # maximum heart rate achieved
    exang: int  # exercise induced angina (0 = no, 1 = yes)
    oldpeak: float  # ST depression induced by exercise relative to rest
    slope: int  # slope of peak exercise ST segment (0-2)
    ca: int  # number of major vessels colored by fluoroscopy (0-4)
    thal: int  # thalassemia (0-3)

# Define the prediction response model
class HeartDiseasePrediction(BaseModel):
    prediction: int  # 0 = no heart disease, 1 = heart disease
    probability: float
    confidence: str

# Define model info response
class ModelInfo(BaseModel):
    model_type: str
    accuracy: float
    features: List[str]
    description: str

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Heart Disease Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Make heart disease prediction",
            "/model-info": "GET - Get model information",
            "/health": "GET - Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": pd.Timestamp.now().isoformat()
    }

@app.get("/model-info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the trained model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfo(
        model_type="Logistic Regression",
        accuracy=0.8852,  # From your notebook results
        features=[
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"
        ],
        description="Logistic Regression model trained on UCI Heart Disease dataset with 88.52% accuracy"
    )

@app.post("/predict", response_model=HeartDiseasePrediction)
async def predict_heart_disease(input_data: HeartDiseaseInput):
    """Predict heart disease based on patient features"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data.dict()])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]  # Probability of heart disease
        
        # Determine confidence level
        if probability > 0.8:
            confidence = "High"
        elif probability > 0.6:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        return HeartDiseasePrediction(
            prediction=int(prediction),
            probability=float(probability),
            confidence=confidence
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict-batch")
async def predict_batch(input_data: List[HeartDiseaseInput]):
    """Predict heart disease for multiple patients"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert list of inputs to DataFrame
        input_list = [data.dict() for data in input_data]
        input_df = pd.DataFrame(input_list)
        
        # Make predictions
        predictions = model.predict(input_df)
        probabilities = model.predict_proba(input_df)[:, 1]
        
        # Format results
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            confidence = "High" if prob > 0.8 else "Medium" if prob > 0.6 else "Low"
            results.append({
                "patient_id": i + 1,
                "prediction": int(pred),
                "probability": float(prob),
                "confidence": confidence
            })
        
        return {
            "predictions": results,
            "total_patients": len(results)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 