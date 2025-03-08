"""Mock feature."""

import polars as pl

from ds_template.features.feature import Feature


class MockFeature(Feature):
    """Mock feature."""

    def __init__(self, column_name: str):
        """Initialize the feature."""
        self.column_name = column_name

    def fit(self, X: pl.LazyFrame, y: pl.LazyFrame | None = None) -> "MockFeature":
        """Fit the feature."""
        return self

    def transform(self, X: pl.LazyFrame) -> pl.LazyFrame:
        """Transform the feature."""
        return X.select(pl.col(self.column_name).str.len_chars().alias("fake_feature"))
