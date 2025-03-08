"""Feature abstract base class."""

import abc

import polars as pl


class Feature(metaclass=abc.ABCMeta):
    """Feature abstract base class."""

    @abc.abstractmethod
    def fit(self, X: pl.LazyFrame, y: pl.LazyFrame | None = None) -> "Feature":
        """Fit the feature."""
        ...

    @abc.abstractmethod
    def transform(self, X: pl.LazyFrame) -> pl.LazyFrame:
        """Transform the feature."""
        ...

    def fit_transform(self, X: pl.LazyFrame, y: pl.LazyFrame | None = None) -> pl.LazyFrame:
        """Fit and transform the feature."""
        return self.fit(X, y).transform(X)
