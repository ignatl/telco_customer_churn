"""Feature engineering."""

import argparse
import pickle
from pathlib import Path

from ds_template.config import MAIN_DIR, logger
from ds_template.data.data_set import FeatureDataSet
from ds_template.features.mock_feature import MockFeature


def featurize(feature_data_set: FeatureDataSet, column_name: str) -> tuple[FeatureDataSet, MockFeature]:
    """Feature engineering."""
    logger.info(f"Column name: {column_name}")
    mock_feature = MockFeature(column_name)

    mock_feature.fit(feature_data_set.train_X)
    result_data_set = FeatureDataSet(
        train_X=mock_feature.transform(feature_data_set.train_X),
        val_X=mock_feature.transform(feature_data_set.val_X),
        test_X=mock_feature.transform(feature_data_set.test_X),
    )
    return result_data_set, mock_feature


def main(input_dir: Path, output_dir: Path, obj_path: Path, column_name: str) -> None:
    """Feature engineering."""
    full_input_dir = MAIN_DIR / input_dir
    feature_data_set: FeatureDataSet = FeatureDataSet.load(full_input_dir)
    logger.info(f"Loaded data from {input_dir}")

    logger.info("Feature engineering...")
    featured_sets, mock_feature = featurize(feature_data_set, column_name)

    full_output_dir = MAIN_DIR / output_dir
    featured_sets.dump(full_output_dir)
    logger.info(f"Saved data to {output_dir}")

    full_obj_path = MAIN_DIR / obj_path
    full_obj_path.parent.mkdir(parents=True, exist_ok=True)
    with open(full_obj_path, "wb") as f:
        pickle.dump(mock_feature, f)
    logger.info(f"Saved object to {obj_path}")


def cli() -> None:
    """Feature engineering."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--obj_path", type=Path, required=True)
    parser.add_argument("--column_name", type=str, required=True)
    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.obj_path, args.column_name)


if __name__ == "__main__":
    cli()
