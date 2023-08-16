"""Base class for all multiclass classification objectives."""
from checkmates.objectives.objective_base import ObjectiveBase
from checkmates.problem_types import ProblemTypes


class MulticlassClassificationObjective(ObjectiveBase):
    """Base class for all multiclass classification objectives."""

    problem_types = [ProblemTypes.MULTICLASS, ProblemTypes.TIME_SERIES_MULTICLASS]
    """[ProblemTypes.MULTICLASS, ProblemTypes.TIME_SERIES_MULTICLASS]"""