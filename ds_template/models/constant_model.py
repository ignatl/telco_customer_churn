"""Constant model."""

import polars as pl

from ds_template.models.trainable import Trainable


class ConstantModel(Trainable):
    """Constant model."""

    def __init__(self, constant: float):
        """Initialize the model."""
        self.constant = constant

    def fit(self, X: pl.LazyFrame, y: pl.LazyFrame | None = None) -> "ConstantModel":
        """Fit the model."""
        return self

    def predict(self, X: pl.LazyFrame) -> pl.LazyFrame:
        """Predict the model."""
        len_x = X.select(pl.len()).collect().item()
        return pl.LazyFrame({"target": [self.constant] * len_x})
