import pytest
import os
from fastapi.testclient import TestClient
from fastapi import status
import mlflow

# 設定測試環境變數
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5001"  # 或使用 "sqlite:///test_mlruns.db" 進行本地測試

@pytest.fixture(scope="session")
def test_client():
    # 只在測試時動態導入，避免與主程式衝突
    from serving.main import app
    with TestClient(app) as client:
        yield client

@pytest.fixture(scope="session")
def trained_model():
    """訓練一個測試用模型並返回 run_id"""
    from src.mlops_framework.train import train_demo
    metrics = train_demo()
    run_id = mlflow.active_run().info.run_id
    mlflow.end_run()
    return run_id, metrics
