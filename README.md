# MLOps 模型服務框架

![Test Status](https://github.com/yourusername/mlops-framework/actions/workflows/ci-cd.yml/badge.svg)
[![codecov](https://codecov.io/gh/yourusername/mlops-framework/graph/badge.svg?token=YOUR_CODECOV_TOKEN)](https://codecov.io/gh/yourusername/mlops-framework)

一個基於 MLflow 和 FastAPI 的 MLOps 框架，用於訓練、追蹤和部署機器學習模型。

## 功能特點

- 🚀 使用 MLflow 追蹤實驗和模型版本控制
- 🎯 支援即時推論 API (FastAPI)
- 📦 容器化部署 (Docker + Docker Compose)
- 🔄 CI/CD 自動化測試與部署
- 🧪 完整的單元測試與整合測試

## 專案結構

```
mlops-framework/
├── .github/workflows/        # GitHub Actions 工作流程
│   └── ci-cd.yml            # CI/CD 設定
├── configs/                  # Hydra 設定檔
│   ├── config.yaml           # 主設定檔
│   └── ...
├── data/                    # 資料目錄 (DVC 追蹤)
├── serving/                 # API 服務程式碼
│   ├── Dockerfile           # API 服務容器設定
│   └── main.py              # FastAPI 應用程式
├── src/                     # 原始碼
│   ├── mlops_framework/     # 核心套件
│   │   ├── __init__.py
│   │   ├── data.py         # 資料處理
│   │   ├── pipeline.py     # 管線邏輯
│   │   └── train.py        # 訓練邏輯
│   └── cli.py              # 命令列介面
├── tests/                   # 測試程式碼
│   ├── conftest.py         # pytest 設定
│   └── test_pipeline.py    # 單元測試
├── .gitignore
├── docker-compose.yml       # 服務編排
├── Dockerfile.mlflow        # MLflow 服務容器設定
├── pyproject.toml          # Python 專案設定
└── README.md
├── pyproject.toml              # 專案定義與依賴
└── README.md                   # 說明文件
```
```

## Quickstart

1. Build and start all services (Postgres, MLflow, FastAPI):

```bash
docker compose up -d --build
```

2. Open the UIs:

* MLflow Tracking Server: <http://localhost:5000>
* FastAPI docs (Swagger): <http://localhost:8000/docs>

3. Stop services:

```bash
docker compose down
```

## Development tips

On your development laptop, create a virtual environment and install dependencies:

```bash
poetry install  # or pip install -r requirements.txt if you export one
uvicorn serving.main:app --reload --port 8000 --host 0.0.0.0
```

Set the environment variable so the API connects to your desktop MLflow server:

```bash
export MLFLOW_TRACKING_URI=http://<desktop-ip>:5000