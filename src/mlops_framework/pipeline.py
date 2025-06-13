from __future__ import annotations

"""High-level pipeline orchestration helpers."""

import logging
from typing import Any, Dict

from .train import train_demo

logger = logging.getLogger(__name__)

__all__ = ["run"]


def run(config: Dict[str, Any] | None = None) -> Dict[str, float]:
    """Run the default demo pipeline.

    Currently this is just train_demo but in the future could chain multiple
    steps (data versioning, feature engineering, model registry, etc.).
    """
    config = config or {}
    logger.info("Starting demo pipeline with config=%s", config)

    metrics = train_demo(**config)

    logger.info("Pipeline finished. Metrics: %s", metrics)
    return metrics
