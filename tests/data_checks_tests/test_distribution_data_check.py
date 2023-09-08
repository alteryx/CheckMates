# Testing Data to make sure skews are recognized-- successful
import numpy as np
import pandas as pd
from diptest import diptest
from scipy.stats import skew

from checkmates.data_checks import (
    DataCheckActionCode,
    DataCheckActionOption,
    DataCheckMessageCode,
    DataCheckWarning,
)


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


data = {
    "Column1": np.random.normal(0, 1, 1000),  # Normally distributed data
    "Column2": np.random.exponential(1, 1000),  # Right-skewed data
    "Column3": 1 / (np.random.gamma(2, 2, 1000)),  # Left-skewed data
}

df = pd.DataFrame(data)
df.ww.init()
messages = []

numeric_X = df.ww.select(["Integer", "Double"])
print(numeric_X)
for col in numeric_X:
    (
        is_skew,
        distribution_type,
        skew_value,
        coef,
    ) = _detect_skew_distribution_helper(numeric_X["Column2"])

    if is_skew:
        details = {
            "distribution type": distribution_type,
            "Skew Value": skew_value,
            "Bimodal Coefficient": coef,
        }
        messages.append(
            DataCheckWarning(
                message="Data may have a skewed distribution.",
                data_check_name="Distribution Data Check",
                message_code=DataCheckMessageCode.SKEWED_DISTRIBUTION,
                details=details,
                action_options=[
                    DataCheckActionOption(
                        DataCheckActionCode.TRANSFORM_FEATURES,
                        data_check_name="Distribution Data Check",
                        metadata={
                            "is_skew": True,
                            "transformation_strategy": "yeojohnson",
                            "columns": col,
                        },
                    ),
                ],
            ).to_dict(),
        )
print(messages)
