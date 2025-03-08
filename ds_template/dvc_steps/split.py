#!/usr/bin/env python3

"""Split data into train, validation and test sets."""

import argparse
from pathlib import Path

import polars as pl

from ds_template.config import MAIN_DIR, logger
from ds_template.data.data_set import FeatureDataSet, TargetDataSet


def split(lf: pl.LazyFrame) -> tuple[FeatureDataSet, TargetDataSet]:
    """Split data into train, validation and test sets."""
    return FeatureDataSet(
        train_X=lf.slice(0, 6),
        val_X=lf.slice(6, 2),
        test_X=lf.slice(8, None),
    ), TargetDataSet(
        train_y=lf.slice(0, 6).select("target"),
        val_y=lf.slice(6, 2).select("target"),
        test_y=lf.slice(8, None).select("target"),
    )


def main(input_path: Path, output_dir: Path) -> None:
    """Split data into train, validation and test sets."""
    input_lf = pl.scan_csv(MAIN_DIR / input_path)
    logger.info(f"Loaded data from {input_path} and split it into train, validation and test sets")

    feature_data_set, target_data_set = split(input_lf)

    full_output_path = MAIN_DIR / output_dir
    full_output_path.mkdir(parents=True, exist_ok=True)
    feature_data_set.dump(full_output_path)
    target_data_set.dump(full_output_path)
    logger.info(f"Saved data to {full_output_path}")


def cli() -> None:
    """CLI for loading and cleaning data."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    main(args.input_path, args.output_dir)


if __name__ == "__main__":
    cli()
