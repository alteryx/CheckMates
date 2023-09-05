"""Data check that checks if the target data contains certain distributions that may need to be transformed prior training to improve model performance."""
import numpy as np
import woodwork as ww
import diptest

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
    """Check if the target data contains certain distributions that may need to be transformed prior training to improve model performance. Uses the skew test and yeojohnson transformation"""

    def validate(self, X, y):
        """Check if the target data has a skewed or bimodal distribution.

        Args:
            X (pd.DataFrame, np.ndarray): Features. Ignored.
            y (pd.Series, np.ndarray): Target data to check for underlying distributions.

        Returns:
            dict (DataCheckError): List with DataCheckErrors if certain distributions are found in the target data.

        Examples:
            >>> import pandas as pd

            Targets that exhibit a skewed distribution will raise a warning for the user to transform the target.

            >>> y = [0.946, 0.972, 1.154, 0.954, 0.969, 1.222, 1.038, 0.999, 0.973, 0.897]
            >>> target_check = DistributionDataCheck()
            >>> assert target_check.validate(None, y) == [
            ...     {
            ...         "message": "Target may have a skewed distribution.",
            ...         "data_check_name": "DistributionDataCheck",
            ...         "level": "warning",
            ...         "code": "TARGET_SKEWED_DISTRIBUTION",
            ...         "details": {"normalization_method": "shapiro", "statistic": 0.8, "p-value": 0.045, "columns": None, "rows": None},
            ...         "action_options": [
            ...             {
            ...                 "code": "TRANSFORM_TARGET",
            ...                 "data_check_name": "DistributionDataCheck",
            ...                 "parameters": {},
            ...                 "metadata": {
            ...                     "transformation_strategy": "yeojohnson",
            ...                     "is_target": True,
            ...                     "columns": None,
            ...                     "rows": None
            ...                 }
            ...             }
            ...         ]
            ...     }
            ... ]
            ...
            >>> y = pd.Series([1, 1, 1, 2, 2, 3, 4, 4, 5, 5, 5])
            >>> assert target_check.validate(None, y) == []
            ...
            ...
            >>> y = pd.Series(pd.date_range("1/1/21", periods=10))
            >>> assert target_check.validate(None, y) == [
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
                    message="Target is None",
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
                    details={"unsupported_type": y.ww.logical_type.type_string},
                ).to_dict(),
            )
            return messages

        (
            is_skew,
            distribution_type,
            skew_value,
            coef
        ) = _detect_skew_distribution_helper(y)


        if is_skew:
            details = {
                "distribution type": distribution_type,
                "Skew Value": skew_value,
                "Bimodal Coefficient": coef,
            }
            messages.append(
                DataCheckWarning(
                    message="Target may have a skewed distribution.",
                    data_check_name=self.name,
                    message_code=DataCheckMessageCode.TARGET_SKEWED_DISTRIBUTION,
                    details=details,
                    action_options=[
                        DataCheckActionOption(
                            DataCheckActionCode.TRANSFORM_TARGET,
                            data_check_name=self.name,
                            metadata={
                                "is_target": True,
                                "transformation_strategy": "yeojohnson",
                            },
                        ),
                    ],
                ).to_dict(),
            )
        return messages


def _detect_skew_distribution_helper(y):
    """Helper method to detect skewed or bimodal distribution. Returns boolean, distribution type, the skew value, and bimodal coefficient."""
    skew_value = np.stats.skew(y)
    coef = diptest.diptest(y)[1]

    if coef < 0.05:
        return True, "bimodal distribution", skew_value, coef
    if skew_value < -0.5:
        return True, "negative skew", skew_value, coef
    if skew_value > 0.5:
        return True, "positive skew", skew_value, coef
    return False, "no skew", skew_value, coef
