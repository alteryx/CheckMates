import pandas as pd
import pytest
import woodwork as ww
from sklearn import datasets
from woodwork import logical_types as ww_logical_types

from checkmates.objectives.utils import get_core_objectives
from checkmates.problem_types import ProblemTypes


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "skip_offline: mark test to be skipped if offline (https://oss.alteryx.com cannot be reached)",
    )
    config.addinivalue_line(
        "markers",
        "skip_during_conda: mark test to be skipped if running during conda build",
    )


@pytest.fixture
def dummy_data_check_name():
    return "dummy_data_check_name"


@pytest.fixture
def get_test_data_with_or_without_primary_key():
    def _get_test_data_with_primary_key(input_type, has_primary_key):
        X = None
        if input_type == "Integer":
            X_dict = {
                "col_1_id": [0, 1, 2, 3],
                "col_2": [2, 3, 4, 5],
                "col_3_id": [1, 1, 2, 3],
                "col_5": [0, 0, 1, 2],
            }
            if not has_primary_key:
                X_dict["col_1_id"] = [1, 1, 2, 3]
            X = pd.DataFrame.from_dict(X_dict)

        elif input_type == "IntegerNullable":
            X_dict = {
                "col_1_id": pd.Series([0, 1, 2, 3], dtype="Int64"),
                "col_2": pd.Series([2, 3, 4, 5], dtype="Int64"),
                "col_3_id": pd.Series([1, 1, 2, 3], dtype="Int64"),
                "col_5": pd.Series([0, 0, 1, 2], dtype="Int64"),
            }
            if not has_primary_key:
                X_dict["col_1_id"] = pd.Series([1, 1, 2, 3], dtype="Int64")
            X = pd.DataFrame.from_dict(X_dict)

        elif input_type == "Double":
            X_dict = {
                "col_1_id": [0.0, 1.0, 2.0, 3.0],
                "col_2": [2, 3, 4, 5],
                "col_3_id": [1, 1, 2, 3],
                "col_5": [0, 0, 1, 2],
            }
            if not has_primary_key:
                X_dict["col_1_id"] = [1.0, 1.0, 2.0, 3.0]
            X = pd.DataFrame.from_dict(X_dict)

        elif input_type == "Unknown":
            X_dict = {
                "col_1_id": ["a", "b", "c", "d"],
                "col_2": ["w", "x", "y", "z"],
                "col_3_id": [
                    "123456789012345",
                    "234567890123456",
                    "3456789012345678",
                    "45678901234567",
                ],
                "col_5": ["0", "0", "1", "2"],
            }
            if not has_primary_key:
                X_dict["col_1_id"] = ["b", "b", "c", "d"]
            X = pd.DataFrame.from_dict(X_dict)

        elif input_type == "Categorical":
            X_dict = {
                "col_1_id": ["a", "b", "c", "d"],
                "col_2": ["w", "x", "y", "z"],
                "col_3_id": [
                    "123456789012345",
                    "234567890123456",
                    "3456789012345678",
                    "45678901234567",
                ],
                "col_5": ["0", "0", "1", "2"],
            }
            if not has_primary_key:
                X_dict["col_1_id"] = ["b", "b", "c", "d"]
            X = pd.DataFrame.from_dict(X_dict)
            X.ww.init(
                logical_types={
                    "col_1_id": "categorical",
                    "col_2": "categorical",
                    "col_5": "categorical",
                },
            )
        return X

    return _get_test_data_with_primary_key


@pytest.fixture
def X_y_binary():
    X, y = datasets.make_classification(
        n_samples=100,
        n_features=20,
        n_informative=2,
        n_redundant=2,
        random_state=0,
    )
    X = pd.DataFrame(X)
    X.ww.init(logical_types={col: "double" for col in X.columns})
    y = ww.init_series(pd.Series(y), logical_type="integer")
    return X, y


@pytest.fixture
def X_y_regression():
    X, y = datasets.make_regression(
        n_samples=100,
        n_features=20,
        n_informative=3,
        random_state=0,
    )
    X = pd.DataFrame(X)
    X.ww.init(logical_types={col: "double" for col in X.columns})
    y = ww.init_series(pd.Series(y), logical_type="double")
    return X, y


@pytest.fixture
def time_series_core_objectives():
    return get_core_objectives(ProblemTypes.TIME_SERIES_REGRESSION)


@pytest.fixture
def make_data_type():
    """Helper function to convert numpy or pandas input to the appropriate type for tests."""

    def _make_data_type(data_type, data, nullable=False):
        if data_type == "li":
            if isinstance(data, pd.DataFrame):
                data = data.to_numpy()
            data = data.tolist()
            return data
        if data_type != "np":
            if len(data.shape) == 1:
                data = pd.Series(data)
            else:
                data = pd.DataFrame(data)
        if data_type == "ww":
            if len(data.shape) == 1:
                data = ww.init_series(data)
                if nullable and isinstance(
                    data.ww.logical_type,
                    ww_logical_types.Integer,
                ):
                    data = ww.init_series(
                        data,
                        logical_type=ww_logical_types.IntegerNullable,
                    )
                elif nullable and isinstance(
                    data.ww.logical_type,
                    ww_logical_types.Boolean,
                ):
                    data = ww.init_series(
                        data,
                        logical_type=ww_logical_types.BooleanNullable,
                    )
            else:
                data.ww.init()
        return data

    return _make_data_type
