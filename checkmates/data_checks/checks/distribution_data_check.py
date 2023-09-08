"""Data check that screens data for skewed or bimodal distrbutions prior to model training to ensure model performance is unaffected."""

from diptest import diptest
from scipy.stats import skew

from checkmates.data_checks import (
    DataCheck,
    DataCheckActionCode,
    DataCheckActionOption,
    DataCheckMessageCode,
    DataCheckWarning,
)


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
            ...                 "code": "TRANSFORM_FEATURES",
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
        """
        messages = []

        numeric_X = X.ww.select(["Integer", "Double"])

        for col in numeric_X:
            (
                is_skew,
                distribution_type,
                skew_value,
                coef,
            ) = _detect_skew_distribution_helper(col)

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
                                DataCheckActionCode.TRANSFORM_FEATURES,
                                data_check_name=self.name,
                                metadata={
                                    "is_skew": True,
                                    "transformation_strategy": "yeojohnson",
                                    "columns": col,
                                },
                            ),
                        ],
                    ).to_dict(),
                )
        return messages


def _detect_skew_distribution_helper(X):
    """Helper method to detect skewed or bimodal distribution. Returns boolean, distribution type, the skew value, and bimodal coefficient."""
    skew_value = skew(X)
    coef = diptest(X)[1]

    if coef < 0.05:
        return True, "bimodal distribution", skew_value, coef
    if skew_value < -0.5:
        return True, "negative skew", skew_value, coef
    if skew_value > 0.5:
        return True, "positive skew", skew_value, coef
    return False, "no skew", skew_value, coef


# Testing Data to make sure skews are recognized-- successful
# import numpy as np
# import pandas as pd
# data = {
#     'Column1': np.random.normal(0, 1, 1000),  # Normally distributed data
#     'Column2': np.random.exponential(1, 1000),  # Right-skewed data
#     'Column3': np.random.gamma(2, 2, 1000)  # Right-skewed data
# }

# df = pd.DataFrame(data)
# df.ww.init()
# messages = []

# numeric_X = df.ww.select(["Integer", "Double"])
# print(numeric_X)
# for col in numeric_X:
#     (
#         is_skew,
#         distribution_type,
#         skew_value,
#         coef,
#     ) = _detect_skew_distribution_helper(numeric_X['Column2'])

#     if is_skew:
#         details = {
#             "distribution type": distribution_type,
#             "Skew Value": skew_value,
#             "Bimodal Coefficient": coef,
#         }
#         messages.append(
#             DataCheckWarning(
#                 message="Data may have a skewed distribution.",
#                 data_check_name="Distribution Data Check",
#                 message_code=DataCheckMessageCode.SKEWED_DISTRIBUTION,
#                 details=details,
#                 action_options=[
#                     DataCheckActionOption(
#                         DataCheckActionCode.TRANSFORM_FEATURES,
#                         data_check_name="Distribution Data Check",
#                         metadata={
#                             "is_skew": True,
#                             "transformation_strategy": "yeojohnson",
#                             "columns" : col
#                         },
#                     ),
#                 ],
#             ).to_dict(),
#         )
# print(messages)
