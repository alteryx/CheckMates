import numpy as np
import pandas as pd
from scipy.stats import yeojohnson

data = {
    "Column1": np.random.normal(0, 1, 1000),  # Normally distributed data
    "Column2": np.random.exponential(1, 1000),  # Right-skewed data
    "Column3": 1 / (np.random.gamma(2, 2, 1000)),  # Left-skewed data
}

X = pd.DataFrame(data)

_cols_to_normalize = "Column2"


def transform(self, X, _cols_to_normalize):
    """Transforms input by normalizing distribution.

    Args:
        X (pd.DataFrame): Data to transform.
        y (pd.Series, optional): Target Data

    Returns:
        pd.DataFrame: Transformed X
    """
    # If there are no columns to normalize, return early
    if not _cols_to_normalize:
        return self

    # Only select the skewed column to normalize
    x_t = X[_cols_to_normalize]
    X_t = X

    # Transform the data
    X_t[_cols_to_normalize] = yeojohnson(x_t)

    # Reinit woodwork
    X_t.ww.init()


transform(X, _cols_to_normalize, None)
