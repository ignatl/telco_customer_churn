"""Data set class which represents the splitted data."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypeVar

import polars as pl

T = TypeVar("T", bound="DataSetMixIn")


@dataclass
class DataSetMixIn:
    """Mixin class for the data set."""

    @classmethod
    def load(cls: type[T], path: Path) -> T:
        """Load the data set from the given path."""
        path_dict: dict[str, pl.LazyFrame] = {}
        for set_name in cls.__dataclass_fields__.keys():
            set_path = path / f"{set_name}.parquet"
            if not set_path.exists():
                raise FileNotFoundError(f"File {set_path} does not exist.")
            set_lf = pl.scan_parquet(set_path)
            path_dict[set_name] = set_lf
        return cls(**path_dict)

    def dump(self, path: Path) -> None:
        """Dump the data set to the given path."""
        path.mkdir(parents=True, exist_ok=True)
        for set_name, set_lf in asdict(self).items():
            set_path = path / f"{set_name}.parquet"
            set_lf.sink_parquet(set_path)


@dataclass
class FeatureDataSet(DataSetMixIn):
    """Data set class which represents the splitted data."""

    train_X: pl.LazyFrame
    val_X: pl.LazyFrame
    test_X: pl.LazyFrame


@dataclass
class TargetDataSet(DataSetMixIn):
    """Data set class which represents the splitted data."""

    train_y: pl.LazyFrame
    val_y: pl.LazyFrame
    test_y: pl.LazyFrame
