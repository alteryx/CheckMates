"""Utility methods for EvalML pipelines."""
from typing import Union

import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

from checkmates.data_checks import DataCheckActionCode
from checkmates.pipelines.components import (  # noqa: F401
    DropColumns,
    DropRowsTransformer,
    PerColumnImputer,
    TargetImputer,
    TimeSeriesImputer,
    TimeSeriesRegularizer,
)
from checkmates.pipelines.training_validation_split import TrainingValidationSplit
from checkmates.pipelines.transformers import SimpleNormalizer
from checkmates.problem_types import is_classification, is_regression, is_time_series
from checkmates.utils import infer_feature_types


def _make_component_list_from_actions(actions):
    """Creates a list of components from the input DataCheckAction list.

    Args:
        actions (list(DataCheckAction)): List of DataCheckAction objects used to create list of components

    Returns:
        list(ComponentBase): List of components used to address the input actions
    """
    components = []
    cols_to_drop = []
    indices_to_drop = []
    cols_to_normalize = []


    for action in actions:
        if action.action_code == DataCheckActionCode.REGULARIZE_AND_IMPUTE_DATASET:
            metadata = action.metadata
            parameters = metadata.get("parameters", {})
            components.extend(
                [
                    TimeSeriesRegularizer(
                        time_index=parameters.get("time_index", None),
                        frequency_payload=parameters["frequency_payload"],
                    ),
                    TimeSeriesImputer(),
                ],
            )
        elif action.action_code == DataCheckActionCode.DROP_COL:
            cols_to_drop.extend(action.metadata["columns"])
        elif action.action_code == DataCheckActionCode.TRANSFORM_FEATURES:
            cols_to_normalize.extend(action.metadata["columns"])
        elif action.action_code == DataCheckActionCode.IMPUTE_COL:
            metadata = action.metadata
            parameters = metadata.get("parameters", {})
            if metadata["is_target"]:
                components.append(
                    TargetImputer(impute_strategy=parameters["impute_strategy"]),
                )
            else:
                impute_strategies = parameters["impute_strategies"]
                components.append(PerColumnImputer(impute_strategies=impute_strategies))
        elif action.action_code == DataCheckActionCode.DROP_ROWS:
            indices_to_drop.extend(action.metadata["rows"])
    if cols_to_drop:
        cols_to_drop = sorted(set(cols_to_drop))
        components.append(DropColumns(columns=cols_to_drop))
    if indices_to_drop:
        indices_to_drop = sorted(set(indices_to_drop))
        components.append(DropRowsTransformer(indices_to_drop=indices_to_drop))
    if cols_to_normalize:
        cols_to_normalize = sorted(set(cols_to_normalize))
        components.append(SimpleNormalizer(columns=cols_to_normalize))

    return components


def split_data(
    X,
    y,
    problem_type,
    problem_configuration=None,
    test_size=None,
    random_seed=0,
):
    """Split data into train and test sets.

    Args:
        X (pd.DataFrame or np.ndarray): data of shape [n_samples, n_features]
        y (pd.Series, or np.ndarray): target data of length [n_samples]
        problem_type (str or ProblemTypes): type of supervised learning problem. see evalml.problem_types.problemtype.all_problem_types for a full list.
        problem_configuration (dict): Additional parameters needed to configure the search. For example,
            in time series problems, values should be passed in for the time_index, gap, and max_delay variables.
        test_size (float): What percentage of data points should be included in the test set. Defaults to 0.2 (20%) for non-timeseries problems and 0.1
            (10%) for timeseries problems.
        random_seed (int): Seed for the random number generator. Defaults to 0.

    Returns:
        pd.DataFrame, pd.DataFrame, pd.Series, pd.Series: Feature and target data each split into train and test sets.

    Examples:
        >>> X = pd.DataFrame([1, 2, 3, 4, 5, 6], columns=["First"])
        >>> y = pd.Series([8, 9, 10, 11, 12, 13])
        ...
        >>> X_train, X_validation, y_train, y_validation = split_data(X, y, "regression", random_seed=42)
        >>> X_train
           First
        5      6
        2      3
        4      5
        3      4
        >>> X_validation
           First
        0      1
        1      2
        >>> y_train
        5    13
        2    10
        4    12
        3    11
        dtype: int64
        >>> y_validation
        0    8
        1    9
        dtype: int64
    """
    X = infer_feature_types(X)
    y = infer_feature_types(y)

    data_splitter = None
    if is_time_series(problem_type):
        if test_size is None:
            test_size = 0.1
            if (
                problem_configuration is not None
                and "forecast_horizon" in problem_configuration
            ):
                fh_pct = problem_configuration["forecast_horizon"] / len(X)
                test_size = max(test_size, fh_pct)
        data_splitter = TrainingValidationSplit(
            test_size=test_size,
            shuffle=False,
            stratify=None,
            random_seed=random_seed,
        )
    else:
        if test_size is None:
            test_size = 0.2
        if is_regression(problem_type):
            data_splitter = ShuffleSplit(
                n_splits=1,
                test_size=test_size,
                random_state=random_seed,
            )
        elif is_classification(problem_type):
            data_splitter = StratifiedShuffleSplit(
                n_splits=1,
                test_size=test_size,
                random_state=random_seed,
            )

    train, test = next(data_splitter.split(X, y))

    X_train = X.ww.iloc[train]
    X_test = X.ww.iloc[test]
    y_train = y.ww.iloc[train]
    y_test = y.ww.iloc[test]

    return X_train, X_test, y_train, y_test


def drop_infinity(
    data: Union[pd.DataFrame, pd.Series],
) -> Union[pd.DataFrame, pd.Series]:
    """Removes infinity values."""
    ww = data.ww._schema is not None
    replace = data.ww.replace if ww else data.replace
    return replace([np.inf, -np.inf], np.nan)
