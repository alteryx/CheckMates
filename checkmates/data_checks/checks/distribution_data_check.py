"""Data check that checks if the target data contains certain distributions that may need to be transformed prior training to improve model performance."""
import diptest
import numpy as np
import woodwork as ww

from checkmates.data_checks import (
    DataCheck,
    DataCheckActionCode,
    DataCheckActionOption,
    DataCheckError,
    DataCheckMessageCode,
    DataCheckWarning,
)
from checkmates.utils import infer_feature_types


class DistributionDataCheck(DataCheck):
    """Check if the overall data contains certain distributions that may need to be transformed prior training to improve model performance. Uses the skew test and yeojohnson transformation."""

    def validate(self, X, y):
        """Check if the overall data has a skewed or bimodal distribution.

        Args:
            X (pd.DataFrame, np.ndarray): Overall data to check for skewed or bimodal distributions.
            y (pd.Series, np.ndarray): Target data to check for underlying distributions.

        Returns:
            dict (DataCheckError): List with DataCheckErrors if certain distributions are found in the overall data.

        Examples:
            >>> import pandas as pd

            Features and target data that exhibit a skewed distribution will raise a warning for the user to transform the data.

            >>> X = [5, 7, 8, 9, 10, 11, 12, 15, 20]
            >>> data_check = DistributionDataCheck()
            >>> assert data_check.validate(X, y) == [
            ...     {
            ...         "message": "Data may have a skewed distribution.",
            ...         "data_check_name": "DistributionDataCheck",
            ...         "level": "warning",
            ...         "code": "SKEWED_DISTRIBUTION",
            ...         "details": {"distribution type": "positive skew", "Skew Value": 0.7939, "Bimodal Coefficient": 1.0,},
            ...         "action_options": [
            ...             {
            ...                 "code": "TRANSFORM_TARGET",
            ...                 "data_check_name": "DistributionDataCheck",
            ...                 "parameters": {},
            ...                 "metadata": {
                                    "is_skew": True,
                                    "transformation_strategy": "yeojohnson",
            ...                 }
            ...             }
            ...         ]
            ...     }
            ... ]
            ...
            >>> X = pd.Series([1, 1, 1, 2, 2, 3, 4, 4, 5, 5, 5])
            >>> assert target_check.validate(X, y) == []
            ...
            ...
            >>> X = pd.Series(pd.date_range("1/1/21", periods=10))
            >>> assert target_check.validate(X, y) == [
            ...     {
            ...         "message": "Target is unsupported datetime type. Valid Woodwork logical types include: integer, double, age, age_fractional",
            ...         "data_check_name": "DistributionDataCheck",
            ...         "level": "error",
            ...         "details": {"columns": None, "rows": None, "unsupported_type": "datetime"},
            ...         "code": "TARGET_UNSUPPORTED_TYPE",
            ...         "action_options": []
            ...     }
            ... ]
        """
        messages = []

        if y is None:
            messages.append(
                DataCheckError(
                    message="Data is None",
                    data_check_name=self.name,
                    message_code=DataCheckMessageCode.TARGET_IS_NONE,
                    details={},
                ).to_dict(),
            )
            return messages

        y = infer_feature_types(y)
        allowed_types = [
            ww.logical_types.Integer.type_string,
            ww.logical_types.Double.type_string,
            ww.logical_types.Age.type_string,
            ww.logical_types.AgeFractional.type_string,
        ]
        is_supported_type = y.ww.logical_type.type_string in allowed_types

        if not is_supported_type:
            messages.append(
                DataCheckError(
                    message="Target is unsupported {} type. Valid Woodwork logical types include: {}".format(
                        y.ww.logical_type.type_string,
                        ", ".join([ltype for ltype in allowed_types]),
                    ),
                    data_check_name=self.name,
                    message_code=DataCheckMessageCode.TARGET_UNSUPPORTED_TYPE,
                    details={"unsupported_type": X.ww.logical_type.type_string},
                ).to_dict(),
            )
            return messages

        (
            is_skew,
            distribution_type,
            skew_value,
            coef,
        ) = _detect_skew_distribution_helper(X)

        if is_skew:
            details = {
                "distribution type": distribution_type,
                "Skew Value": skew_value,
                "Bimodal Coefficient": coef,
            }
            messages.append(
                DataCheckWarning(
                    message="Data may have a skewed distribution.",
                    data_check_name=self.name,
                    message_code=DataCheckMessageCode.SKEWED_DISTRIBUTION,
                    details=details,
                    action_options=[
                        DataCheckActionOption(
                            DataCheckActionCode.TRANSFORM_TARGET,
                            data_check_name=self.name,
                            metadata={
                                "is_skew": True,
                                "transformation_strategy": "yeojohnson",
                            },
                        ),
                    ],
                ).to_dict(),
            )
        return messages


def _detect_skew_distribution_helper(X):
    """Helper method to detect skewed or bimodal distribution. Returns boolean, distribution type, the skew value, and bimodal coefficient."""
    skew_value = np.stats.skew(X)
    coef = diptest.diptest(X)[1]

    if coef < 0.05:
        return True, "bimodal distribution", skew_value, coef
    if skew_value < -0.5:
        return True, "negative skew", skew_value, coef
    if skew_value > 0.5:
        return True, "positive skew", skew_value, coef
    return False, "no skew", skew_value, coef
