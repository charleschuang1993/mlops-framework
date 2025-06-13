import pytest
import os
import tempfile

# 為每次 test session 建立獨立的 MLflow file store
_MLRUNS_DIR = tempfile.mkdtemp(prefix="mlruns_test_")
# 設定測試環境變數（必須在匯入 mlflow 之前）
os.environ["MLFLOW_TRACKING_URI"] = f"file:{_MLRUNS_DIR}"

from fastapi.testclient import TestClient
from fastapi import status
import mlflow

# 建立 Default 實驗 (experiment_id=0) 以避免首次使用空目錄時找不到
mlflow.set_tracking_uri(f"file:{_MLRUNS_DIR}")
EXPERIMENT_NAME = "iris-demo"
mlflow.set_experiment(EXPERIMENT_NAME)

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
    result = train_demo(mlflow_tracking_uri=f"file:{_MLRUNS_DIR}")
    return result["run_id"], result["metrics"]
