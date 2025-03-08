"""Inference step."""

import argparse
import pickle
from dataclasses import asdict
from pathlib import Path

import polars as pl
from polars.testing import assert_frame_equal

from ds_template.config import MAIN_DIR
from ds_template.data.data_set import FeatureDataSet, TargetDataSet
from ds_template.inference.inference import Inference


def main(
    train_data_dir: Path,
    train_preds_path: Path,
    inference_data_path: Path,
    feature_path: Path,
    model_path: Path,
    output_path: Path,
    inference_path: Path,
) -> None:
    """Main function."""
    full_train_data_dir = MAIN_DIR / train_data_dir
    train_feature_data_set = FeatureDataSet.load(full_train_data_dir)

    full_train_preds_path = MAIN_DIR / train_preds_path
    train_target_data_set = TargetDataSet.load(full_train_preds_path)

    full_inference_data_path = MAIN_DIR / inference_data_path
    inference_lf = pl.scan_csv(full_inference_data_path)

    with feature_path.open("rb") as f:
        feature = pickle.load(f)

    with model_path.open("rb") as f:
        model = pickle.load(f)

    inference = Inference(feature, model)
    for X, y in zip(asdict(train_feature_data_set).values(), asdict(train_target_data_set).values()):
        assert_frame_equal(inference.inference(X), y)

    inference_lf = inference.inference(inference_lf)
    full_output_path = MAIN_DIR / output_path
    full_output_path.parent.mkdir(parents=True, exist_ok=True)
    inference_lf.sink_parquet(full_output_path)

    full_inference_path = MAIN_DIR / inference_path
    full_inference_path.parent.mkdir(parents=True, exist_ok=True)
    with full_inference_path.open("wb") as f:
        pickle.dump(inference, f)


def cli() -> None:
    """CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_dir", type=Path, required=True)
    parser.add_argument("--train_preds_path", type=Path, required=True)
    parser.add_argument("--inference_data_path", type=Path, required=True)
    parser.add_argument("--feature_path", type=Path, required=True)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--inference_path", type=Path, required=True)
    args = parser.parse_args()
    main(**vars(args))


if __name__ == "__main__":
    cli()
