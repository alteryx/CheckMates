"""The supported types of machine learning problems."""
from datachecks.problem_types.problem_types import ProblemTypes
from datachecks.problem_types.utils import (
    handle_problem_types,
    detect_problem_type,
    is_regression,
    is_binary,
    is_multiclass,
    is_classification,
    is_time_series,
)
