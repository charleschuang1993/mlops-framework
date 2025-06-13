import pytest
import os

# 設定測試環境變數（必須在匯入 mlflow 之前）
os.environ["MLFLOW_TRACKING_URI"] = "file:./mlruns_test"  # 本地路徑避免 /mlflow 只讀錯誤

from fastapi.testclient import TestClient
from fastapi import status
import mlflow

# 只在測試時動態導入，避免與主程式衝突
@pytest.fixture(scope="session")
def test_client():
    from serving.main import app
    with TestClient(app) as client:
        yield client

@pytest.fixture(scope="session")
def trained_model():
    """訓練一個測試用模型並返回 run_id"""
    from src.mlops_framework.train import train_demo
    metrics = train_demo()
    run = mlflow.last_active_run()
    run_id = run.info.run_id if run else ""
    return run_id, metrics
