# mlops-framework
```text
mlops-framework/
├── configs/
│   ├── config.yaml           # Hydra 主設定檔
│   ├── data/
│   │   └── iris.yaml
│   ├── model/
│   │   └── logreg.yaml
│   └── trainer/
│       └── default.yaml
├── data/                       # DVC 指標檔會放在這裡
├── src/
│   ├── mlops_framework/
│   │   ├── __init__.py
│   │   ├── data.py           # 資料處理邏輯
│   │   ├── pipeline.py       # 核心管線邏輯
│   │   └── train.py          # 訓練與評估邏輯
│   └── cli.py                  # Typer + Hydra 入口
├── serving/
│   ├── main.py
│   └── Dockerfile
├── tests/
│   └── test_pipeline.py
├── .gitignore
├── docker-compose.yml
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