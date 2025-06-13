from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import numpy as np
from typing import Dict, List, Any, Optional
import os

EXPERIMENT_NAME = "iris-demo"

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

class RegisterModelRequest(BaseModel):
    run_id: str
    model_name: str = EXPERIMENT_NAME

class PromoteRequest(BaseModel):
    stage: str  # e.g., "Staging" or "Production"

class TrainRequest(BaseModel):
    model_name: str
    C: float = 1.0
    max_iter: int = 200

# Cache for the loaded model
_model = None
_model_version = None

def load_latest_model():
    """Load the latest model from MLflow."""
    global _model, _model_version
    
    if _model is None:
        try:
            # Connect to MLflow tracking URI (env or default)
            mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
            mlflow.set_experiment(EXPERIMENT_NAME)
            
            # Get the latest run with 'logreg_demo' in the run name
            runs = mlflow.search_runs(
                filter_string="attributes.run_name LIKE '%logreg_demo%'",
                order_by=["start_time DESC"],
                max_results=1,
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
    global _model, _model_version

    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train a model first."
        )
    
    model, model_version = _model, _model_version
    
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
        # Always try local store first based on tracking URI if it uses file:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "")
        local_root = tracking_uri.replace("file:", "") if tracking_uri.startswith("file:") else ""

        if local_root:
            potential_local = f"{local_root}/{EXPERIMENT_NAME}/{payload.run_id}/artifacts/model"
            model_uri = potential_local if os.path.exists(potential_local) else None
        else:
            model_uri = None

        if not model_uri:
            mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
            model_uri = f"runs:/{payload.run_id}/model"
        _model = mlflow.sklearn.load_model(model_uri)
        _model_version = payload.run_id
        return {"status": "loaded", "model_version": _model_version}
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Failed to load model: {str(e)}")

@app.post("/register-model", tags=["model"])
async def register_model(payload: RegisterModelRequest):
    """Register an MLflow run as a new model version in the Model Registry."""
    client = MlflowClient()
    model_uri = f"runs:/{payload.run_id}/model"
    try:
        mv = client.create_model_version(
            name=payload.model_name,
            source=model_uri,
            run_id=payload.run_id,
        )
        return {"model_name": mv.name, "version": mv.version, "status": mv.status}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Registration failed: {str(e)}")

@app.post("/model/{model_name}/{version}/promote", tags=["model"])
async def promote_model(model_name: str, version: str, payload: PromoteRequest):
    """Transition a model version to a new stage (e.g., Staging, Production)."""
    client = MlflowClient()
    try:
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=payload.stage,
            archive_existing_versions=False,
        )
        return {"model_name": model_name, "version": version, "new_stage": payload.stage}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/model/{model_name}", tags=["model"])
async def delete_model(model_name: str):
    """Delete a registered model and all its versions."""
    client = MlflowClient()
    try:
        client.delete_registered_model(model_name)
        return {"status": "deleted", "model_name": model_name}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/model/{model_name}/{version}", tags=["model"])
async def delete_model_version(model_name: str, version: str):
    """Delete a specific model version from the registry."""
    client = MlflowClient()
    try:
        client.delete_model_version(name=model_name, version=version)
        return {"status": "deleted", "model_name": model_name, "version": version}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ---------- Train endpoint ----------
@app.post("/train", tags=["mlflow"])
async def train(payload: TrainRequest, background_tasks: BackgroundTasks):
    """Trigger a training run using train_demo and log to MLflow.
    Runs in background to avoid blocking request."""
    try:
        from mlops_framework.train import train_demo
    except ModuleNotFoundError:
        from src.mlops_framework.train import train_demo

    def _run_train():
        try:
            mlflow.set_tag("model_name", payload.model_name)
            result = train_demo(C=payload.C, max_iter=payload.max_iter)
            print("Training completed", result)
        except Exception as exc:
            print("Training failed", exc)

    background_tasks.add_task(_run_train)
    return {"status": "training_started", "model_name": payload.model_name}

# ---------- Synchronous Train endpoint ----------
@app.post("/train-sync", tags=["mlflow"])
async def train_sync(payload: TrainRequest):
    """Train synchronously and return run info (run_id, experiment, default model name, metrics)."""
    try:
        from mlops_framework.train import train_demo
    except ModuleNotFoundError:
        from src.mlops_framework.train import train_demo

    mlflow.set_tag("model_name", payload.model_name)
    result = train_demo(C=payload.C, max_iter=payload.max_iter)
    run_id = result.get("run_id")
    metrics = result.get("metrics", {})
    return {
        "run_id": run_id,
        "experiment": EXPERIMENT_NAME,
        "model_name": payload.model_name,
        "metrics": metrics,
    }

# ----------- MLflow query endpoints -----------

@app.get("/experiments", tags=["mlflow"])
async def list_experiments():
    """Return all experiments id & name."""
    client = MlflowClient()
    experiments = client.search_experiments(max_results=10000)
    return [
        {"experiment_id": exp.experiment_id, "name": exp.name}
        for exp in experiments
    ]

@app.get("/experiments/{experiment_name}/runs", tags=["mlflow"])
async def list_runs(experiment_name: str, max_results: int = 20):
    """List recent runs of an experiment by name."""
    client = MlflowClient()
    # Resolve experiment id
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=max_results,
    )
    return [
        {
            "run_id": r.info.run_id,
            "status": r.info.status,
            "start_time": r.info.start_time,
            "run_name": r.data.tags.get("mlflow.runName", "")
        }
        for r in runs
    ]

@app.get("/models", tags=["model"])
async def list_registered_models():
    """List all registered models."""
    client = MlflowClient()
    models = client.search_registered_models()
    return [
        {
            "name": m.name,
            "latest_versions": [
                {"version": v.version, "stage": v.current_stage, "run_id": v.run_id}
                for v in m.latest_versions
            ] if m.latest_versions else []
        }
        for m in models
    ]

@app.get("/models/{model_name}/versions", tags=["model"])
async def list_model_versions(model_name: str):
    """List versions of a given model."""
    client = MlflowClient()
    versions = client.search_model_versions(filter_string=f"name='{model_name}'")
    if not versions:
        raise HTTPException(status_code=404, detail="Model not found")
    return [
        {
            "version": v.version,
            "stage": v.current_stage,
            "status": v.status,
            "run_id": v.run_id,
            "creation_time": v.creation_timestamp,
        }
        for v in versions
    ]

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("serving.main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)