"""Trainable abstract base class."""

import abc

import polars as pl


class Trainable(metaclass=abc.ABCMeta):
    """Trainable abstract base class."""

    @abc.abstractmethod
    def fit(self, X: pl.LazyFrame, y: pl.LazyFrame | None = None) -> "Trainable":
        """Fit the model."""
        ...

    @abc.abstractmethod
    def predict(self, X: pl.LazyFrame) -> pl.LazyFrame:
        """Predict the model."""
        ...

    def fit_predict(self, X: pl.LazyFrame, y: pl.LazyFrame | None = None) -> pl.LazyFrame:
        """Fit and predict the model."""
        return self.fit(X, y).predict(X)
