"""Standard machine learning objective functions."""
import numpy as np
import pandas as pd
from sklearn import metrics

from checkmates.objectives.regression_objective import RegressionObjective
from checkmates.utils import classproperty
from checkmates.objectives.binary_classification_objective import BinaryClassificationObjective
from checkmates.objectives.multiclass_classification_objective import MulticlassClassificationObjective


class LogLossBinary(BinaryClassificationObjective):
    """Log Loss for binary classification.

    Example:
        >>> y_true = pd.Series([0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1])
        >>> y_pred = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        >>> np.testing.assert_almost_equal(LogLossBinary().objective_function(y_true, y_pred), 19.6601745)
    """

    name = "Log Loss Binary"
    greater_is_better = False
    score_needs_proba = True
    perfect_score = 0.0
    is_bounded_like_percentage = False  # Range [0, Inf)
    expected_range = [0, 1]

    def objective_function(
        self,
        y_true,
        y_predicted,
        y_train=None,
        X=None,
        sample_weight=None,
    ):
        """Objective function for log loss for binary classification."""
        return metrics.log_loss(y_true, y_predicted, sample_weight=sample_weight)

class LogLossMulticlass(MulticlassClassificationObjective):
    """Log Loss for multiclass classification.

    Example:
        >>> y_true = [0, 1, 2, 0, 2, 1]
        >>> y_pred = [[0.7, 0.2, 0.1],
        ...           [0.3, 0.5, 0.2],
        ...           [0.1, 0.3, 0.6],
        ...           [0.9, 0.1, 0.0],
        ...           [0.3, 0.1, 0.6],
        ...           [0.5, 0.5, 0.0]]
        >>> np.testing.assert_almost_equal(LogLossMulticlass().objective_function(y_true, y_pred), 0.4783301)
    """

    name = "Log Loss Multiclass"
    greater_is_better = False
    score_needs_proba = True
    perfect_score = 0.0
    is_bounded_like_percentage = False  # Range [0, Inf)
    expected_range = [0, 1]

    def objective_function(
        self,
        y_true,
        y_predicted,
        y_train=None,
        X=None,
        sample_weight=None,
    ):
        """Objective function for log loss for multiclass classification."""
        return metrics.log_loss(y_true, y_predicted, sample_weight=sample_weight)

class R2(RegressionObjective):
    """Coefficient of determination for regression.

    Example:
        >>> y_true = pd.Series([1.5, 2, 3, 1, 0.5, 1, 2.5, 2.5, 1, 0.5, 2])
        >>> y_pred = pd.Series([1.5, 2.5, 2, 1, 0.5, 1, 3, 2.25, 0.75, 0.25, 1.75])
        >>> np.testing.assert_almost_equal(R2().objective_function(y_true, y_pred), 0.7638036)
    """

    name = "R2"
    greater_is_better = True
    score_needs_proba = False
    perfect_score = 1
    is_bounded_like_percentage = False  # Range (-Inf, 1]
    expected_range = [-1, 1]

    def objective_function(
        self,
        y_true,
        y_predicted,
        y_train=None,
        X=None,
        sample_weight=None,
    ):
        """Objective function for coefficient of determination for regression."""
        return metrics.r2_score(y_true, y_predicted, sample_weight=sample_weight)

class MedianAE(RegressionObjective):
    """Median absolute error for regression.

    Example:
        >>> y_true = pd.Series([1.5, 2, 3, 1, 0.5, 1, 2.5, 2.5, 1, 0.5, 2])
        >>> y_pred = pd.Series([1.5, 2.5, 2, 1, 0.5, 1, 3, 2.25, 0.75, 0.25, 1.75])
        >>> np.testing.assert_almost_equal(MedianAE().objective_function(y_true, y_pred), 0.25)
    """

    name = "MedianAE"
    greater_is_better = False
    score_needs_proba = False
    perfect_score = 0.0
    is_bounded_like_percentage = False  # Range [0, Inf)
    expected_range = [0, float("inf")]

    def objective_function(
        self,
        y_true,
        y_predicted,
        y_train=None,
        X=None,
        sample_weight=None,
    ):
        """Objective function for median absolute error for regression."""
        return metrics.median_absolute_error(
            y_true,
            y_predicted,
            sample_weight=sample_weight,
        )



class RootMeanSquaredLogError(RegressionObjective):
    """Root mean squared log error for regression.

    Only valid for nonnegative inputs. Otherwise, will throw a ValueError.

    Example:
        >>> y_true = pd.Series([1.5, 2, 3, 1, 0.5, 1, 2.5, 2.5, 1, 0.5, 2])
        >>> y_pred = pd.Series([1.5, 2.5, 2, 1, 0.5, 1, 3, 2.25, 0.75, 0.25, 1.75])
        >>> np.testing.assert_almost_equal(RootMeanSquaredLogError().objective_function(y_true, y_pred), 0.13090204)
    """

    name = "Root Mean Squared Log Error"
    greater_is_better = False
    score_needs_proba = False
    perfect_score = 0.0
    is_bounded_like_percentage = False  # Range [0, Inf)
    expected_range = [0, float("inf")]

    def objective_function(
        self,
        y_true,
        y_predicted,
        y_train=None,
        X=None,
        sample_weight=None,
    ):
        """Objective function for root mean squared log error for regression."""

        def rmsle(y_true, y_pred):
            return np.sqrt(
                metrics.mean_squared_log_error(
                    y_true,
                    y_pred,
                    sample_weight=sample_weight,
                ),
            )

        # Multiseries time series regression
        if isinstance(y_true, pd.DataFrame):
            raw_rmsles = []
            for i in range(len(y_true.columns)):
                y_true_i = y_true.iloc[:, i]
                y_predicted_i = y_predicted.iloc[:, i]
                raw_rmsles.append(rmsle(y_true_i, y_predicted_i))
            return np.mean(raw_rmsles)

        # All univariate regression
        return rmsle(y_true, y_predicted)

    @classproperty
    def positive_only(self):
        """If True, this objective is only valid for positive data."""
        return True


class MeanSquaredLogError(RegressionObjective):
    """Mean squared log error for regression.

    Only valid for nonnegative inputs. Otherwise, will throw a ValueError.

    Example:
        >>> y_true = pd.Series([1.5, 2, 3, 1, 0.5, 1, 2.5, 2.5, 1, 0.5, 2])
        >>> y_pred = pd.Series([1.5, 2.5, 2, 1, 0.5, 1, 3, 2.25, 0.75, 0.25, 1.75])
        >>> np.testing.assert_almost_equal(MeanSquaredLogError().objective_function(y_true, y_pred), 0.0171353)
    """

    name = "Mean Squared Log Error"
    greater_is_better = False
    score_needs_proba = False
    perfect_score = 0.0
    is_bounded_like_percentage = False  # Range [0, Inf)
    expected_range = [0, float("inf")]

    def objective_function(
        self,
        y_true,
        y_predicted,
        y_train=None,
        X=None,
        sample_weight=None,
    ):
        """Objective function for mean squared log error for regression."""
        return metrics.mean_squared_log_error(
            y_true,
            y_predicted,
            sample_weight=sample_weight,
        )

    @classproperty
    def positive_only(self):
        """If True, this objective is only valid for positive data."""
        return True
