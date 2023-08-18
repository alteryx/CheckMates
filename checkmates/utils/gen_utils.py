"""General utility methods."""
import logging
from collections import namedtuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def _get_subclasses(base_class):
    """Gets all of the leaf nodes in the hiearchy tree for a given base class.

    Args:
        base_class (abc.ABCMeta): Class to find all of the children for.

    Returns:
        subclasses (list): List of all children that are not base classes.
    """
    classes_to_check = base_class.__subclasses__()
    subclasses = []

    while classes_to_check:
        subclass = classes_to_check.pop()
        children = subclass.__subclasses__()

        if children:
            classes_to_check.extend(children)
        else:
            subclasses.append(subclass)

    return subclasses


class classproperty:
    """Allows function to be accessed as a class level property.

    Example:
    .. code-block::

        class DataCheckActionCode(Enum):
        "Enum for data check action code".

        DROP_COL = "drop_col"
        "Action code for dropping a column."

        DROP_ROWS = "drop_rows"
        "Action code for dropping rows."

        IMPUTE_COL = "impute_col"
        "Action code for imputing a column."

        TRANSFORM_TARGET = "transform_target"
        "Action code for transforming the target data."

        REGULARIZE_AND_IMPUTE_DATASET = "regularize_and_impute_dataset"
        "Action code for regularizing and imputing all features and target time series data."

        SET_FIRST_COL_ID = "set_first_col_id"
        "Action code for setting the first column as an id column."

        @classproperty
        def _all_values(cls):
            return {code.value.upper(): code for code in list(cls)}

        def __str__(self):
            "String representation of the DataCheckActionCode enum."
            datacheck_action_code_dict = {
                DataCheckActionCode.DROP_COL.name: "drop_col",
                DataCheckActionCode.DROP_ROWS.name: "drop_rows",
                DataCheckActionCode.IMPUTE_COL.name: "impute_col",
                DataCheckActionCode.TRANSFORM_TARGET.name: "transform_target",
                DataCheckActionCode.REGULARIZE_AND_IMPUTE_DATASET.name: "regularize_and_impute_dataset",
                DataCheckActionCode.SET_FIRST_COL_ID.name: "set_first_col_id",
            }
            return datacheck_action_code_dict[self.name]
    """

    def __init__(self, func):
        self.func = func

    def __get__(self, _, klass):
        """Get property value."""
        return self.func(klass)


def contains_all_ts_parameters(problem_configuration):
    """Validates that the problem configuration contains all required keys.

    Args:
        problem_configuration (dict): Problem configuration.

    Returns:
        bool, str: True if the configuration contains all parameters. If False, msg is a non-empty
            string with error message.
    """
    required_parameters = {"time_index", "gap", "max_delay", "forecast_horizon"}
    msg = ""
    if (
        not problem_configuration
        or not all(p in problem_configuration for p in required_parameters)
        or problem_configuration["time_index"] is None
    ):
        msg = (
            "problem_configuration must be a dict containing values for at least the time_index, gap, max_delay, "
            f"and forecast_horizon parameters, and time_index cannot be None. Received {problem_configuration}."
        )
    return not (msg), msg


_validation_result = namedtuple(
    "TSParameterValidationResult",
    ("is_valid", "msg", "smallest_split_size", "max_window_size", "n_obs", "n_splits"),
)


def are_ts_parameters_valid_for_split(
    gap,
    max_delay,
    forecast_horizon,
    n_obs,
    n_splits,
):
    """Validates the time series parameters in problem_configuration are compatible with split sizes.

    Args:
        gap (int): gap value.
        max_delay (int): max_delay value.
        forecast_horizon (int): forecast_horizon value.
        n_obs (int): Number of observations in the dataset.
        n_splits (int): Number of cross validation splits.

    Returns:
        TsParameterValidationResult - named tuple with four fields
            is_valid (bool): True if parameters are valid.
            msg (str): Contains error message to display. Empty if is_valid.
            smallest_split_size (int): Smallest split size given n_obs and n_splits.
            max_window_size (int): Max window size given gap, max_delay, forecast_horizon.
    """
    eval_size = forecast_horizon * n_splits
    train_size = n_obs - eval_size
    window_size = gap + max_delay + forecast_horizon
    msg = ""
    if train_size <= window_size:
        msg = (
            f"Since the data has {n_obs} observations, n_splits={n_splits}, and a forecast horizon of {forecast_horizon}, "
            f"the smallest split would have {train_size} observations. "
            f"Since {gap + max_delay + forecast_horizon} (gap + max_delay + forecast_horizon) >= {train_size}, "
            "then at least one of the splits would be empty by the time it reaches the pipeline. "
            "Please use a smaller number of splits, reduce one or more these parameters, or collect more data."
        )
    return _validation_result(not msg, msg, train_size, window_size, n_obs, n_splits)

def safe_repr(value):
    """Convert the given value into a string that can safely be used for repr.

    Args:
        value: The item to convert

    Returns:
        String representation of the value
    """
    if isinstance(value, float):
        if pd.isna(value):
            return "np.nan"
        if np.isinf(value):
            return f"float('{repr(value)}')"
    return repr(value)