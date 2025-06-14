# MLOps 模型服務框架 / MLOps Model Serving Framework

> **雙語說明 | Bilingual Documentation**

---

## 簡介（中文）

這是一個以 **FastAPI + MLflow** 為核心、搭配 **Docker Compose** 及 **GitHub Actions CI/CD** 的端到端 MLOps 框架，協助團隊快速完成：

1. 🏗 **模型生命週期管理** – 從資料處理、訓練、註冊、載入到線上推論，一條龍自動化。
2. 🗂 **實驗追蹤與版本控制** – 透過 MLflow 實驗與 Model Registry，精準記錄參數、指標與 Artifact。
3. 🔌 **RESTful 服務** – 使用 FastAPI 提供非同步 `/train → /train-status → /register-model → /load-model → /predict` 流程。
4. ♻️ **可重現與可擴充** – 容器化部署、乾淨的專案結構、完整單元與整合測試，方便 CI/CD 及多環境落地。

### 特色亮點

| 功能 | 說明 |
|------|------|
| 🚀 非同步訓練 | `/train` 端點將訓練任務丟入背景執行，API 即刻回應不阻塞 |
| 🔍 進度查詢 | `/train-status` 依 `model_name` 標籤回傳最近一次 run 狀態 |
| 🏷 自動註冊 | `/register-model` 偵測不到 Registry 時，自動建立 Registered Model 再創建版本 |
| 📦 快速載入 | `/load-model` 以 `run_id` 載入模型到記憶體，隨即可供 `/predict` 呼叫 |
| 🛠 CI/CD | GitHub Actions 針對測試、lint、Docker Build 自動化；可擴充為自動部署 |

---

## Overview (English)

This repository provides an **end-to-end MLOps scaffold** powered by **FastAPI** and **MLflow**, packaged with **Docker Compose** and wired into **GitHub Actions**.

Key goals:

1. 🏗 **Full model life-cycle** – data → train → register → load → infer via REST endpoints.
2. 🗂 **Experiment tracking & versioning** – MLflow captures parameters, metrics, and artifacts; Model Registry governs promotion flow.
3. 🔌 **Async REST services** – non-blocking training endpoint with status polling for smooth UX.
4. ♻️ **Reproducible & extensible** – containerised, test-driven, ready for CI/CD pipelines and multi-env deployment.

### Highlights

| Feature | Details |
|---------|---------|
| 🚀 Async training | `/train` schedules background tasks to keep the API responsive |
| 🔍 Progress check | `/train-status` returns the latest run state filtered by `model_name` tag |
| 🏷 Auto registration | `/register-model` creates a Registered Model on-the-fly when absent |
| 📦 Instant load | `/load-model` retrieves a model by `run_id` into memory for immediate `/predict` |
| 🛠 CI/CD ready | GitHub Actions for tests, lint, Docker build; extendable to deploy steps |

---

## Quickstart

```bash
# 1. Build & launch Postgres, MLflow & FastAPI
$ docker compose up -d --build

# 2. Open UIs
MLflow : http://localhost:5000
docs  : http://localhost:8000/docs

# 3. Tear down
$ docker compose down -v
```

### Example Workflow

```bash
# Trigger async training
curl -X POST http://localhost:8000/train -H "Content-Type: application/json" -d '{"model_name":"iris-demo"}'

# Poll status
curl http://localhost:8000/train-status?model_name=iris-demo

# Register trained run
curl -X POST http://localhost:8000/register-model -H "Content-Type: application/json" \
     -d '{"run_id":"<RUN_ID>", "model_name":"iris-demo"}'

# Load model into memory
curl -X POST http://localhost:8000/load-model -H "Content-Type: application/json" \
     -d '{"run_id":"<RUN_ID>"}'

# Predict
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
     -d '{"data":[[5.1,3.5,1.4,0.2]]}'
```

---

## Project Layout

```
mlops-framework/
├── serving/          # FastAPI service (Dockerfile, main.py)
├── src/              # Core library (data, train, pipeline, utils)
├── tests/            # Unit & integration tests (pytest)
├── docker-compose.yml
├── .github/workflows/
└── README.md
```

---

## Roadmap

- [ ] JWT/OAuth2 authentication
- [ ] Promotion API (`/promote`) for stage transitions
- [ ] Kubernetes manifests & Helm chart

---

## License

MIT © 2025 atgenomix


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