"""Inference module. Develope it from scratch for every project."""

import polars as pl

from ds_template.features.feature import Feature
from ds_template.models.trainable import Trainable


class Inference:
    """Inference class."""

    def __init__(self, feature: Feature, model: Trainable):
        """Initialize the inference class."""
        self.feature = feature
        self.model = model

    def inference(self, X: pl.LazyFrame) -> pl.LazyFrame:
        """Inference the target."""
        feature_X = self.feature.transform(X)
        return self.model.predict(feature_X)
