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