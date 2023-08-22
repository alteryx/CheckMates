"""Utility methods for CheckMates objectives."""
from typing import Optional

import pandas as pd

from checkmates import objectives
from checkmates.exceptions import ObjectiveCreationError, ObjectiveNotFoundError
from checkmates.objectives.objective_base import ObjectiveBase
from checkmates.problem_types import ProblemTypes, handle_problem_types
from checkmates.utils.gen_utils import _get_subclasses
from checkmates.utils.logger import get_logger

logger = get_logger(__file__)


def get_non_core_objectives():
    """Get non-core objective classes.

    Non-core objectives are objectives that are domain-specific. Users typically need to configure these objectives
    before using them in AutoMLSearch.

    Returns:
        List of ObjectiveBase classes
    """
    return [
        objectives.MeanSquaredLogError,
        objectives.RootMeanSquaredLogError,
    ]


def get_all_objective_names():
    """Get a list of the names of all objectives.

    Returns:
        list (str): Objective names
    """
    all_objectives_dict = _all_objectives_dict()
    return list(all_objectives_dict.keys())


def _all_objectives_dict():
    all_objectives = _get_subclasses(ObjectiveBase)
    objectives_dict = {}
    for objective in all_objectives:
        if "checkmates.objectives" not in objective.__module__:
            continue
        objectives_dict[objective.name.lower()] = objective
    return objectives_dict


def get_objective(objective, return_instance=False, **kwargs):
    """Returns the Objective class corresponding to a given objective name.

    Args:
        objective (str or ObjectiveBase): Name or instance of the objective class.
        return_instance (bool): Whether to return an instance of the objective. This only applies if objective
            is of type str. Note that the instance will be initialized with default arguments.
        kwargs (Any): Any keyword arguments to pass into the objective. Only used when return_instance=True.

    Returns:
        ObjectiveBase if the parameter objective is of type ObjectiveBase. If objective is instead a valid
        objective name, function will return the class corresponding to that name. If return_instance is True,
        an instance of that objective will be returned.

    Raises:
        TypeError: If objective is None.
        TypeError: If objective is not a string and not an instance of ObjectiveBase.
        ObjectiveNotFoundError: If input objective is not a valid objective.
        ObjectiveCreationError: If objective cannot be created properly.
    """
    if objective is None:
        raise TypeError("Objective parameter cannot be NoneType")
    if isinstance(objective, ObjectiveBase):
        return objective
    all_objectives_dict = _all_objectives_dict()
    if not isinstance(objective, str):
        raise TypeError(
            "If parameter objective is not a string, it must be an instance of ObjectiveBase!",
        )
    if objective.lower() not in all_objectives_dict:
        raise ObjectiveNotFoundError(
            f"{objective} is not a valid Objective! "
            "Use checkmates.objectives.get_all_objective_names() "
            "to get a list of all valid objective names. ",
        )

    objective_class = all_objectives_dict[objective.lower()]

    if return_instance:
        try:
            return objective_class(**kwargs)
        except TypeError as e:
            raise ObjectiveCreationError(
                f"In get_objective, cannot pass in return_instance=True for {objective} because {str(e)}",
            )

    return objective_class


def get_problem_type(
    input_problem_type: Optional[str],
    target_data: pd.Series,
) -> ProblemTypes:
    """Helper function to determine if classification problem is binary or multiclass dependent on target variable values."""
    if not input_problem_type:
        raise ValueError("problem type is required")
    if input_problem_type.lower() == "classification":
        values: pd.Series = target_data.value_counts()
        if values.size == 2:
            return ProblemTypes.BINARY
        elif values.size > 2:
            return ProblemTypes.MULTICLASS
        else:
            message: str = "The target field contains less than two unique values. It cannot be used for modeling."
            logger.error(message, exc_info=True)
            raise ValueError(message)

    if input_problem_type.lower() == "regression":
        return ProblemTypes.REGRESSION

    if input_problem_type.lower() == "time series regression":
        return ProblemTypes.TIME_SERIES_REGRESSION

    message = f"Unexpected problem type provided in configuration: {input_problem_type}"
    logger.error(message, exc_info=True)
    raise ValueError(message)


def get_default_primary_search_objective(problem_type):
    """Get the default primary search objective for a problem type.

    Args:
        problem_type (str or ProblemType): Problem type of interest.

    Returns:
        ObjectiveBase: primary objective instance for the problem type.
    """
    problem_type = handle_problem_types(problem_type)
    objective_name = {
        "binary": "Log Loss Binary",
        "multiclass": "Log Loss Multiclass",
        "regression": "R2",
        "time series regression": "MedianAE",
        "time series binary": "Log Loss Binary",
        "time series multiclass": "Log Loss Multiclass",
    }[problem_type.value]
    return get_objective(objective_name, return_instance=True)


def get_core_objectives(problem_type):
    """Returns all core objective instances associated with the given problem type.

    Core objectives are designed to work out-of-the-box for any dataset.

    Args:
        problem_type (str/ProblemTypes): Type of problem

    Returns:
        List of ObjectiveBase instances

    Examples:
        >>> for objective in get_core_objectives("regression"):
        ...     print(objective.name)
        ExpVariance
        MaxError
        MedianAE
        MSE
        MAE
        R2
        Root Mean Squared Error
        >>> for objective in get_core_objectives("binary"):
        ...     print(objective.name)
        MCC Binary
        Log Loss Binary
        Gini
        AUC
        Precision
        F1
        Balanced Accuracy Binary
        Accuracy Binary
    """
    problem_type = handle_problem_types(problem_type)
    all_objectives_dict = _all_objectives_dict()
    objectives = [
        obj()
        for obj in all_objectives_dict.values()
        if obj.is_defined_for_problem_type(problem_type)
        and obj not in get_non_core_objectives()
    ]
    return objectives
