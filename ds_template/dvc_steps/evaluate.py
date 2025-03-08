"""Evaluate the model's predictions."""

import argparse
from pathlib import Path

import polars as pl

from ds_template.config import MAIN_DIR, logger
from ds_template.data.data_set import TargetDataSet


def mse(predictions: pl.LazyFrame, target: pl.LazyFrame) -> float:
    """Calculate the mean squared error."""
    return float(
        pl.concat([predictions.rename({"target": "pred"}), target], how="horizontal")
        .with_columns((pl.col("pred") - pl.col("target")).pow(2).alias("squared_error"))
        .select(pl.col("squared_error").mean().pow(0.5))
        .collect()
        .item()
    )


def evaluate(predictions: TargetDataSet, target: TargetDataSet) -> pl.LazyFrame:
    """Evaluate the model's predictions."""
    return pl.LazyFrame(
        {
            "dataset": ["train", "val", "test"],
            "mse": [
                mse(predictions.train_y, target.train_y),
                mse(predictions.val_y, target.val_y),
                mse(predictions.test_y, target.test_y),
            ],
        }
    )


def main(predictions_dir: Path, target_dir: Path, metrics_path: Path) -> None:
    """Evaluate the model's predictions."""
    full_predictions_dir = MAIN_DIR / predictions_dir
    predictions = TargetDataSet.load(full_predictions_dir)

    full_target_dir = MAIN_DIR / target_dir
    target = TargetDataSet.load(full_target_dir)

    metrics = evaluate(predictions, target)

    full_metrics_path = MAIN_DIR / metrics_path
    full_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.sink_csv(full_metrics_path)
    logger.info(f"Metrics saved to {full_metrics_path}")


def cli() -> None:
    """Evaluate the model's predictions."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_dir", type=Path, required=True)
    parser.add_argument("--target_dir", type=Path, required=True)
    parser.add_argument("--metrics_path", type=Path, required=True)

    args = parser.parse_args()

    main(args.predictions_dir, args.target_dir, args.metrics_path)


if __name__ == "__main__":
    cli()
