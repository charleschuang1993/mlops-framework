import typer
import rich
from pathlib import Path

from mlops_framework import pipeline, train

app = typer.Typer(help="CLI entry for MLOps framework demo")


@app.command()
def run_pipeline(config_path: Path | None = typer.Option(None, help="Path to YAML/JSON config")):
    """Run the end-to-end demo pipeline."""
    # For now we ignore config_path; future versions can parse it.
    metrics = pipeline.run()
    rich.print({"metrics": metrics})


@app.command()
def train_demo(C: float = 1.0, max_iter: int = 200):
    """Train logistic regression demo and log to MLflow."""
    metrics = train.train_demo(C=C, max_iter=max_iter)
    rich.print({"metrics": metrics})


if __name__ == "__main__":
    app()