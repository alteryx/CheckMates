"""Woodwork utility methods."""
import numpy as np
import pandas as pd
import woodwork as ww

numeric_and_boolean_ww = [
    ww.logical_types.Integer.type_string,
    ww.logical_types.Double.type_string,
    ww.logical_types.Boolean.type_string,
    ww.logical_types.Age.type_string,
    ww.logical_types.AgeFractional.type_string,
    ww.logical_types.IntegerNullable.type_string,
    ww.logical_types.BooleanNullable.type_string,
    ww.logical_types.AgeNullable.type_string,
]


def _numpy_to_pandas(array):
    if len(array.shape) == 1:
        data = pd.Series(array)
    else:
        data = pd.DataFrame(array)
    return data


def _list_to_pandas(list):
    return _numpy_to_pandas(np.array(list))


def infer_feature_types(data, feature_types=None):
    """Create a Woodwork structure from the given list, pandas, or numpy input, with specified types for columns. If a column's type is not specified, it will be inferred by Woodwork.

    Args:
        data (pd.DataFrame, pd.Series): Input data to convert to a Woodwork data structure.
        feature_types (string, ww.logical_type obj, dict, optional): If data is a 2D structure, feature_types must be a dictionary
            mapping column names to the type of data represented in the column. If data is a 1D structure, then feature_types must be
            a Woodwork logical type or a string representing a Woodwork logical type ("Double", "Integer", "Boolean", "Categorical", "Datetime", "NaturalLanguage")

    Returns:
        A Woodwork data structure where the data type of each column was either specified or inferred.

    Raises:
        ValueError: If there is a mismatch between the dataframe and the woodwork schema.
    """
    if isinstance(data, list):
        data = _list_to_pandas(data)
    elif isinstance(data, np.ndarray):
        data = _numpy_to_pandas(data)

    if data.ww.schema is not None:
        if isinstance(data, pd.DataFrame) and not ww.is_schema_valid(
            data,
            data.ww.schema,
        ):
            ww_error = ww.get_invalid_schema_message(data, data.ww.schema)
            if "dtype mismatch" in ww_error:
                ww_error = (
                    "Dataframe types are not consistent with logical types. This usually happens "
                    "when a data transformation does not go through the ww accessor. Call df.ww.init() to "
                    f"get rid of this message. This is a more detailed message about the mismatch: {ww_error}"
                )
            else:
                ww_error = f"{ww_error}. Please initialize ww with df.ww.init() to get rid of this message."
            raise ValueError(ww_error)
        return data

    if isinstance(data, pd.Series):
        if all(data.isna()):
            data = data.replace(pd.NA, np.nan)
            feature_types = "Double"
        return ww.init_series(data, logical_type=feature_types)
    else:
        ww_data = data.copy()
        ww_data.ww.init(logical_types=feature_types)
        return ww_data
