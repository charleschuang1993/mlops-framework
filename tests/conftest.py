import pytest
import os
import tempfile
import sys
from pathlib import Path

# Add project root to sys.path for CI runners where PYTHONPATH is not set
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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

# --- 測試結束自動清理 MLflow 暫存目錄 ---
import atexit
import shutil

def _cleanup():
    if os.path.exists(_MLRUNS_DIR):
        shutil.rmtree(_MLRUNS_DIR, ignore_errors=True)
        print(f"Cleaned up temporary MLflow directory: {_MLRUNS_DIR}")

atexit.register(_cleanup)
