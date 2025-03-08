"""Train model."""

import argparse
import pickle
from pathlib import Path

import polars as pl

from ds_template.config import MAIN_DIR
from ds_template.data.data_set import FeatureDataSet, TargetDataSet
from ds_template.models.constant_model import ConstantModel


def read_lf(*args: Path) -> pl.LazyFrame:
    """Read lazy frames from paths."""
    return pl.concat(
        (pl.scan_parquet(MAIN_DIR / path) for path in args),
        how="horizontal",
        rechunk=True,
        parallel=True,
    )


def main(
    feature_path: Path,
    data_path: Path,
    output_path: Path,
    model_path: Path,
    constant: float,
) -> None:
    """Train model."""
    full_feature_path = MAIN_DIR / feature_path
    feature_data_set: FeatureDataSet = FeatureDataSet.load(full_feature_path)

    full_data_path = MAIN_DIR / data_path
    target_data_set: TargetDataSet = TargetDataSet.load(full_data_path)

    model = ConstantModel(constant)
    model.fit(feature_data_set.train_X, target_data_set.train_y)
    pred_data_set = TargetDataSet(
        train_y=model.predict(feature_data_set.train_X),
        val_y=model.predict(feature_data_set.val_X),
        test_y=model.predict(feature_data_set.test_X),
    )

    full_output_path = MAIN_DIR / output_path
    pred_data_set.dump(full_output_path)

    full_model_path = MAIN_DIR / model_path
    full_model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(full_model_path, "wb") as f:
        pickle.dump(model, f)


def cli() -> None:
    """Train model."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-path", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--constant", type=float, required=True)
    args = parser.parse_args()

    main(
        args.feature_path,
        args.data_dir,
        args.output_dir,
        args.model_path,
        args.constant,
    )


if __name__ == "__main__":
    cli()
