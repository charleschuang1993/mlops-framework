import pytest
import numpy as np
from fastapi import status

def test_train_demo_accuracy(trained_model):
    """測試訓練結果的準確率是否合理"""
    _, metrics = trained_model
    assert metrics['accuracy'] > 0.9, "模型準確率應高於 90%"

class TestPredictionAPI:
    def test_predict_without_model(self, test_client):
        """測試模型未載入時的回應"""
        # 注意：此測試假設 API 啟動時未載入模型
        response = test_client.post(
            "/predict",
            json={
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        )
        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert "Model not loaded" in response.json()["detail"]

    def test_predict_success(self, test_client, trained_model):
        """測試預測端點是否正常運作"""
        run_id, _ = trained_model
        
        # 觸發模型載入（替換為你的實際載入端點）
        response = test_client.post("/load-model", json={"run_id": run_id})
        assert response.status_code == status.HTTP_200_OK

        # 測試預測
        test_data = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        response = test_client.post("/predict", json=test_data)
        
        assert response.status_code == status.HTTP_200_OK
        result = response.json()
        assert "prediction" in result
        assert "probabilities" in result
        assert result["prediction"] in ["setosa", "versicolor", "virginica"]
        assert np.isclose(sum(result["probabilities"].values()), 1.0, atol=1e-6), "機率總和應為 1"

    def test_invalid_input(self, test_client):
        """測試無效輸入的錯誤處理"""
        response = test_client.post(
            "/predict",
            json={"invalid": "data"}  # 缺少必要欄位
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

# 執行測試: pytest -v tests/
