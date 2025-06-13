from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import mlflow.sklearn
import numpy as np
from typing import Dict, List, Any
import os

app = FastAPI(title="Iris Classifier API", version="1.0.0")

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class PredictionResult(BaseModel):
    prediction: str
    probabilities: Dict[str, float]
    model_version: str

class LoadModelRequest(BaseModel):
    run_id: str

# Cache for the loaded model
_model = None
_model_version = None

def load_latest_model():
    """Load the latest model from MLflow."""
    global _model, _model_version
    
    if _model is None:
        try:
            # Connect to MLflow server
            mlflow.set_tracking_uri("http://mlflow:5000")
            
            # Get the latest run with 'logreg_demo' in the run name
            runs = mlflow.search_runs(
                experiment_ids=["0"],  # Default experiment ID
                filter_string="attributes.run_name LIKE '%logreg_demo%'"
            )
            
            if runs.empty:
                raise ValueError("No trained model found in MLflow")
                
            latest_run = runs.iloc[0]
            model_uri = f"runs:/{latest_run.run_id}/model"
            
            # Load the model
            _model = mlflow.sklearn.load_model(model_uri)
            _model_version = latest_run.run_id
            print(f"Loaded model version: {_model_version}")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            _model = None
            _model_version = None
    
    return _model, _model_version

@app.get("/health", tags=["health"])
async def health() -> Dict[str, str]:
    """Simple health check endpoint."""
    return {"status": "ok"}

# Example root endpoint
@app.get("/")
async def root() -> Dict[str, str]:
    return {"message": "Welcome to the MLOps Framework API"}

@app.post("/predict", response_model=PredictionResult)
async def predict(features: IrisFeatures):
    """Predict the iris flower type from input features."""
    model, model_version = load_latest_model()
    
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train a model first."
        )
    
    # Prepare input features
    input_data = np.array([
        [features.sepal_length, features.sepal_width, 
         features.petal_length, features.petal_width]
    ])
    
    try:
        # Get prediction and probabilities
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        
        # Map class indices to class names
        class_names = ['setosa', 'versicolor', 'virginica']
        predicted_class = class_names[int(prediction)]
        
        # Format probabilities
        prob_dict = {
            class_name: float(prob) 
            for class_name, prob in zip(class_names, probabilities)
        }
        
        return {
            "prediction": predicted_class,
            "probabilities": prob_dict,
            "model_version": model_version
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

# -------------------------
# New endpoint: /load-model
# -------------------------

@app.post("/load-model", tags=["model"])
async def load_model_endpoint(payload: LoadModelRequest):
    """Load a specific MLflow run's model into memory.

    Request body:
    {
        "run_id": "<mlflow_run_id>"
    }
    """
    global _model, _model_version

    try:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns_test"))
        model_uri = f"runs:/{payload.run_id}/model"
        _model = mlflow.sklearn.load_model(model_uri)
        _model_version = payload.run_id
        return {"status": "loaded", "model_version": _model_version}
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Failed to load model: {str(e)}")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("serving.main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)