from fastapi import FastAPI
from typing import Dict
import os

app = FastAPI(title="MLOps Framework API", version="0.1.0")

@app.get("/health", tags=["health"])
async def health() -> Dict[str, str]:
    """Simple health check endpoint."""
    return {"status": "ok"}

# Example root endpoint
@app.get("/")
async def root() -> Dict[str, str]:
    return {"message": "Welcome to the MLOps Framework API"}

# Future endpoints (model prediction, registry, etc.) will be added here.

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("serving.main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)