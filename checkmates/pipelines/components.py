"""Initalizes an transformer that selects specified columns in input data."""
from abc import abstractmethod
from functools import wraps
import pandas as pd
import woodwork as ww
import warnings
from sklearn.impute import SimpleImputer as SkImputer

from woodwork.logical_types import Datetime
from woodwork.statistics_utils import infer_frequency

from checkmates.pipelines.transformers import Transformer
from checkmates.pipelines.transformers import SimpleImputer
from checkmates.exceptions import ComponentNotYetFittedError
from checkmates.pipelines import ComponentBaseMeta
from checkmates.utils import infer_feature_types
from checkmates.utils.nullable_type_utils import (
    _get_new_logical_types_for_imputed_data,
    _determine_fractional_type,
    _determine_non_nullable_equivalent,
)



class ColumnSelector(Transformer):
    """Initalizes an transformer that selects specified columns in input data.

    Args:
        columns (list(string)): List of column names, used to determine which columns to select.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    def __init__(self, columns=None, random_seed=0, **kwargs):
        if columns and not isinstance(columns, list):
            raise ValueError(
                f"Parameter columns must be a list. Received {type(columns)}.",
            )

        parameters = {"columns": columns}
        parameters.update(kwargs)
        super().__init__(
            parameters=parameters,
            component_obj=None,
            random_seed=random_seed,
        )

    def _check_input_for_columns(self, X):
        cols = self.parameters.get("columns") or []
        column_names = X.columns

        missing_cols = set(cols) - set(column_names)
        if missing_cols:
            raise ValueError(f"Columns of type {missing_cols} not found in input data.")

    @abstractmethod
    def _modify_columns(self, cols, X, y=None):
        """How the transformer modifies the columns of the input data."""

    def fit(self, X, y=None):
        """Fits the transformer by checking if column names are present in the dataset.

        Args:
            X (pd.DataFrame): Data to check.
            y (pd.Series, ignored): Targets.

        Returns:
            self
        """
        X = infer_feature_types(X)
        self._check_input_for_columns(X)
        return self

    def transform(self, X, y=None):
        """Transform data using fitted column selector component.

        Args:
            X (pd.DataFrame): The input training data of shape [n_samples, n_features].
            y (pd.Series, optional): The target training data of length [n_samples].

        Returns:
            pd.DataFrame: Transformed data.
        """
        X = infer_feature_types(X)
        self._check_input_for_columns(X)
        cols = self.parameters.get("columns") or []
        modified_cols = self._modify_columns(cols, X, y)
        return infer_feature_types(modified_cols)


class DropColumns(ColumnSelector):
    """Drops specified columns in input data.

    Args:
        columns (list(string)): List of column names, used to determine which columns to drop.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    name = "Drop Columns Transformer"
    hyperparameter_ranges = {}
    """{}"""
    needs_fitting = False

    def _check_input_for_columns(self, X):
        pass

    def _modify_columns(self, cols, X, y=None):
        column_intersection = list(set(cols).intersection(X.columns))
        return X.ww.drop(column_intersection)

    def transform(self, X, y=None):
        """Transforms data X by dropping columns.

        Args:
            X (pd.DataFrame): Data to transform.
            y (pd.Series, optional): Targets.

        Returns:
            pd.DataFrame: Transformed X.
        """
        return super().transform(X, y)


class SelectColumns(ColumnSelector):
    """Selects specified columns in input data.

    Args:
        columns (list(string)): List of column names, used to determine which columns to select. If columns are not present, they will not be selected.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    name = "Select Columns Transformer"
    hyperparameter_ranges = {}
    """{}"""
    needs_fitting = False

    def _check_input_for_columns(self, X):
        pass

    def fit(self, X, y=None):
        """Fits the transformer by checking if column names are present in the dataset.

        Args:
            X (pd.DataFrame): Data to check.
            y (pd.Series, optional): Targets.

        Returns:
            self
        """
        return self

    def _modify_columns(self, cols, X, y=None):
        column_intersection = list(
            sorted(set(cols).intersection(X.columns), key=cols.index),
        )
        return X.ww[column_intersection]


class SelectByType(Transformer):
    """Selects columns by specified Woodwork logical type or semantic tag in input data.

    Args:
        column_types (string, ww.LogicalType, list(string), list(ww.LogicalType)): List of Woodwork types or tags, used to determine which columns to select or exclude.
        exclude (bool): If true, exclude the column_types instead of including them. Defaults to False.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    name = "Select Columns By Type Transformer"
    hyperparameter_ranges = {}
    """{}"""
    needs_fitting = False

    def __init__(self, column_types=None, exclude=False, random_seed=0, **kwargs):
        parameters = {"column_types": column_types, "exclude": exclude}
        parameters.update(kwargs)
        super().__init__(
            parameters=parameters,
            component_obj=None,
            random_seed=random_seed,
        )

    def _modify_columns(self, cols, X, y=None):
        if self.parameters.get("exclude"):
            return X.ww.select(exclude=cols)
        return X.ww.select(include=cols)

    def fit(self, X, y=None):
        """Fits the transformer by checking if column names are present in the dataset.

        Args:
            X (pd.DataFrame): Data to check.
            y (pd.Series, ignored): Targets.

        Returns:
            self
        """
        X = infer_feature_types(X)
        return self

    def transform(self, X, y=None):
        """Transforms data X by selecting columns.

        Args:
            X (pd.DataFrame): Data to transform.
            y (pd.Series, optional): Targets.

        Returns:
            pd.DataFrame: Transformed X.
        """
        X = infer_feature_types(X)
        cols = self.parameters.get("column_types") or []
        modified_cols = self._modify_columns(cols, X, y)
        return infer_feature_types(modified_cols)

"""Transformer to drop rows specified by row indices."""

class DropRowsTransformer(Transformer):
    """Transformer to drop rows specified by row indices.

    Args:
        indices_to_drop (list): List of indices to drop in the input data. Defaults to None.
        random_seed (int): Seed for the random number generator. Is not used by this component. Defaults to 0.
    """

    name = "Drop Rows Transformer"
    modifies_target = True
    training_only = True
    hyperparameter_ranges = {}
    """{}"""

    def __init__(self, indices_to_drop=None, random_seed=0):
        if indices_to_drop is not None and len(set(indices_to_drop)) != len(
            indices_to_drop,
        ):
            raise ValueError("All input indices must be unique.")
        self.indices_to_drop = indices_to_drop
        parameters = {"indices_to_drop": self.indices_to_drop}
        super().__init__(
            parameters=parameters,
            component_obj=None,
            random_seed=random_seed,
        )

    def fit(self, X, y=None):
        """Fits component to data.

        Args:
            X (pd.DataFrame): The input training data of shape [n_samples, n_features].
            y (pd.Series, optional): The target training data of length [n_samples].

        Returns:
            self

        Raises:
            ValueError: If indices to drop do not exist in input features or target.
        """
        X_t = infer_feature_types(X)
        y_t = infer_feature_types(y) if y is not None else None
        if self.indices_to_drop is not None:
            indices_to_drop_set = set(self.indices_to_drop)
            missing_X_indices = indices_to_drop_set.difference(set(X_t.index))
            missing_y_indices = (
                indices_to_drop_set.difference(set(y_t.index))
                if y_t is not None
                else None
            )
            if len(missing_X_indices):
                raise ValueError(
                    "Indices [{}] do not exist in input features".format(
                        list(missing_X_indices),
                    ),
                )
            elif y_t is not None and len(missing_y_indices):
                raise ValueError(
                    "Indices [{}] do not exist in input target".format(
                        list(missing_y_indices),
                    ),
                )
        return self

    def transform(self, X, y=None):
        """Transforms data using fitted component.

        Args:
            X (pd.DataFrame): Features.
            y (pd.Series, optional): Target data.

        Returns:
            (pd.DataFrame, pd.Series): Data with row indices dropped.
        """
        X_t = infer_feature_types(X)
        y_t = infer_feature_types(y) if y is not None else None
        if self.indices_to_drop is None or len(self.indices_to_drop) == 0:
            return X_t, y_t
        schema = X_t.ww.schema

        X_t = X_t.drop(self.indices_to_drop, axis=0)
        X_t.ww.init(schema=schema)

        if y_t is not None:
            y_t = y_t.ww.drop(self.indices_to_drop)
        return X_t, y_t

"""Component that imputes missing data according to a specified imputation strategy per column."""

class PerColumnImputer(Transformer):
    """Imputes missing data according to a specified imputation strategy per column.

    Args:
        impute_strategies (dict): Column and {"impute_strategy": strategy, "fill_value":value} pairings.
            Valid values for impute strategy include "mean", "median", "most_frequent", "constant" for numerical data,
            and "most_frequent", "constant" for object data types. Defaults to None, which uses "most_frequent" for all columns.
            When impute_strategy == "constant", fill_value is used to replace missing data.
            When None, uses 0 when imputing numerical data and "missing_value" for strings or object data types.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    name = "Per Column Imputer"
    hyperparameter_ranges = {}
    """{}"""

    def __init__(
        self,
        impute_strategies=None,
        random_seed=0,
        **kwargs,
    ):
        parameters = {
            "impute_strategies": impute_strategies,
        }
        self.imputers = None
        self.impute_strategies = impute_strategies or dict()
        if not isinstance(self.impute_strategies, dict):
            raise ValueError(
                "`impute_strategies` is not a dictionary. Please provide in Column and {`impute_strategy`: strategy, `fill_value`:value} pairs. ",
            )
        super().__init__(
            parameters=parameters,
            component_obj=None,
            random_seed=random_seed,
        )

    def fit(self, X, y=None):
        """Fits imputers on input data.

        Args:
            X (pd.DataFrame or np.ndarray): The input training data of shape [n_samples, n_features] to fit.
            y (pd.Series, optional): The target training data of length [n_samples]. Ignored.

        Returns:
            self
        """
        X = infer_feature_types(X)
        self.imputers = dict()

        columns_to_impute = self.impute_strategies.keys()
        if len(columns_to_impute) == 0:
            warnings.warn(
                "No columns to impute. Please check `impute_strategies` parameter.",
            )

        for column in columns_to_impute:
            strategy_dict = self.impute_strategies.get(column, dict())
            strategy = strategy_dict["impute_strategy"]
            fill_value = strategy_dict.get("fill_value", None)
            self.imputers[column] = SimpleImputer(
                impute_strategy=strategy,
                fill_value=fill_value,
            )

        for column, imputer in self.imputers.items():
            imputer.fit(X.ww[[column]])

        return self

    def transform(self, X, y=None):
        """Transforms input data by imputing missing values.

        Args:
            X (pd.DataFrame or np.ndarray): The input training data of shape [n_samples, n_features] to transform.
            y (pd.Series, optional): The target training data of length [n_samples]. Ignored.

        Returns:
            pd.DataFrame: Transformed X
        """
        X_ww = infer_feature_types(X)
        original_schema = X_ww.ww.schema

        cols_to_drop = []
        for column, imputer in self.imputers.items():
            transformed = imputer.transform(X_ww.ww[[column]])
            if transformed.empty:
                cols_to_drop.append(column)
            else:
                X_ww.ww[column] = transformed[column]
        X_t = X_ww.ww.drop(cols_to_drop)
        X_t.ww.init(schema=original_schema.get_subset_schema(X_t.columns))
        return X_t

"""Component that imputes missing target data according to a specified imputation strategy."""

class TargetImputerMeta(ComponentBaseMeta):
    """A version of the ComponentBaseMeta class which handles when input features is None."""

    @classmethod
    def check_for_fit(cls, method):
        """`check_for_fit` wraps a method that validates if `self._is_fitted` is `True`.

        Args:
            method (callable): Method to wrap.

        Raises:
            ComponentNotYetFittedError: If component is not fitted.

        Returns:
            The wrapped input method.
        """

        @wraps(method)
        def _check_for_fit(self, X=None, y=None):
            klass = type(self).__name__
            if not self._is_fitted and self.needs_fitting:
                raise ComponentNotYetFittedError(
                    f"This {klass} is not fitted yet. You must fit {klass} before calling {method.__name__}.",
                )
            else:
                return method(self, X, y)

        return _check_for_fit


class TargetImputer(Transformer, metaclass=TargetImputerMeta):
    """Imputes missing target data according to a specified imputation strategy.

    Args:
        impute_strategy (string): Impute strategy to use. Valid values include "mean", "median", "most_frequent", "constant" for
           numerical data, and "most_frequent", "constant" for object data types. Defaults to "most_frequent".
        fill_value (string): When impute_strategy == "constant", fill_value is used to replace missing data.
           Defaults to None which uses 0 when imputing numerical data and "missing_value" for strings or object data types.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    name = "Target Imputer"
    hyperparameter_ranges = {"impute_strategy": ["mean", "median", "most_frequent"]}
    """{
        "impute_strategy": ["mean", "median", "most_frequent"]
    }"""
    modifies_features = False
    modifies_target = True

    def __init__(
        self, impute_strategy="most_frequent", fill_value=None, random_seed=0, **kwargs
    ):
        parameters = {"impute_strategy": impute_strategy, "fill_value": fill_value}
        parameters.update(kwargs)
        imputer = SkImputer(strategy=impute_strategy, fill_value=fill_value, **kwargs)
        super().__init__(
            parameters=parameters,
            component_obj=imputer,
            random_seed=random_seed,
        )

    def fit(self, X, y):
        """Fits imputer to target data. 'None' values are converted to np.nan before imputation and are treated as the same.

        Args:
            X (pd.DataFrame or np.ndarray): The input training data of shape [n_samples, n_features]. Ignored.
            y (pd.Series, optional): The target training data of length [n_samples].

        Returns:
            self

        Raises:
            TypeError: If target is filled with all null values.
        """
        if y is None:
            return self
        y = infer_feature_types(y)
        if all(y.isnull()):
            raise TypeError("Provided target full of nulls.")
        y = y.to_frame()

        # Return early if all the columns are bool dtype, which will never have null values
        if (y.dtypes == bool).all():
            return y

        self._component_obj.fit(y)
        return self

    def transform(self, X, y):
        """Transforms input target data by imputing missing values. 'None' and np.nan values are treated as the same.

        Args:
            X (pd.DataFrame): Features. Ignored.
            y (pd.Series): Target data to impute.

        Returns:
            (pd.DataFrame, pd.Series): The original X, transformed y
        """
        if X is not None:
            X = infer_feature_types(X)
        if y is None:
            return X, None
        y_ww = infer_feature_types(y)
        y_df = y_ww.ww.to_frame()

        # Return early if all the columns are bool dtype, which will never have null values
        if (y_df.dtypes == bool).all():
            return X, y_ww

        transformed = self._component_obj.transform(y_df)
        y_t = pd.Series(transformed[:, 0], index=y_ww.index)

        # Determine logical type to use - should match input data where possible
        new_logical_type_dict = _get_new_logical_types_for_imputed_data(
            self.parameters["impute_strategy"],
            y_df.ww.schema,
        )
        new_logical_type = list(new_logical_type_dict.values())[0]

        return X, ww.init_series(y_t, logical_type=new_logical_type)

    def fit_transform(self, X, y):
        """Fits on and transforms the input target data.

        Args:
            X (pd.DataFrame): Features. Ignored.
            y (pd.Series): Target data to impute.

        Returns:
            (pd.DataFrame, pd.Series): The original X, transformed y
        """
        return self.fit(X, y).transform(X, y)

"""Component that imputes missing data according to a specified timeseries-specific imputation strategy."""
import pandas as pd
import woodwork as ww
from woodwork.logical_types import (
    BooleanNullable,
    Double,
)

class TimeSeriesImputer(Transformer):
    """Imputes missing data according to a specified timeseries-specific imputation strategy.

    This Transformer should be used after the `TimeSeriesRegularizer` in order to impute the missing values that were
    added to X and y (if passed).

    Args:
        categorical_impute_strategy (string): Impute strategy to use for string, object, boolean, categorical dtypes.
            Valid values include "backwards_fill" and "forwards_fill". Defaults to "forwards_fill".
        numeric_impute_strategy (string): Impute strategy to use for numeric columns. Valid values include
            "backwards_fill", "forwards_fill", and "interpolate". Defaults to "interpolate".
        target_impute_strategy (string): Impute strategy to use for the target column. Valid values include
            "backwards_fill", "forwards_fill", and "interpolate". Defaults to "forwards_fill".
        random_seed (int): Seed for the random number generator. Defaults to 0.

    Raises:
        ValueError: If categorical_impute_strategy, numeric_impute_strategy, or target_impute_strategy is not one of the valid values.
    """

    modifies_features = True
    modifies_target = True
    training_only = True

    name = "Time Series Imputer"
    hyperparameter_ranges = {
        "categorical_impute_strategy": ["backwards_fill", "forwards_fill"],
        "numeric_impute_strategy": ["backwards_fill", "forwards_fill", "interpolate"],
        "target_impute_strategy": ["backwards_fill", "forwards_fill", "interpolate"],
    }
    """{
        "categorical_impute_strategy": ["backwards_fill", "forwards_fill"],
        "numeric_impute_strategy": ["backwards_fill", "forwards_fill", "interpolate"],
        "target_impute_strategy": ["backwards_fill", "forwards_fill", "interpolate"],
    }"""
    _valid_categorical_impute_strategies = set(["backwards_fill", "forwards_fill"])
    _valid_numeric_impute_strategies = set(
        ["backwards_fill", "forwards_fill", "interpolate"],
    )
    _valid_target_impute_strategies = set(
        ["backwards_fill", "forwards_fill", "interpolate"],
    )

    # Incompatibility: https://github.com/alteryx/evalml/issues/4001
    # TODO: Remove when support is added https://github.com/alteryx/evalml/issues/4014
    _integer_nullable_incompatibilities = ["X", "y"]
    _boolean_nullable_incompatibilities = ["y"]

    def __init__(
        self,
        categorical_impute_strategy="forwards_fill",
        numeric_impute_strategy="interpolate",
        target_impute_strategy="forwards_fill",
        random_seed=0,
        **kwargs,
    ):
        if categorical_impute_strategy not in self._valid_categorical_impute_strategies:
            raise ValueError(
                f"{categorical_impute_strategy} is an invalid parameter. Valid categorical impute strategies are {', '.join(self._valid_numeric_impute_strategies)}",
            )
        elif numeric_impute_strategy not in self._valid_numeric_impute_strategies:
            raise ValueError(
                f"{numeric_impute_strategy} is an invalid parameter. Valid numeric impute strategies are {', '.join(self._valid_numeric_impute_strategies)}",
            )
        elif target_impute_strategy not in self._valid_target_impute_strategies:
            raise ValueError(
                f"{target_impute_strategy} is an invalid parameter. Valid target column impute strategies are {', '.join(self._valid_target_impute_strategies)}",
            )

        parameters = {
            "categorical_impute_strategy": categorical_impute_strategy,
            "numeric_impute_strategy": numeric_impute_strategy,
            "target_impute_strategy": target_impute_strategy,
        }
        parameters.update(kwargs)
        self._all_null_cols = None
        self._forwards_cols = None
        self._backwards_cols = None
        self._interpolate_cols = None
        self._impute_target = None
        super().__init__(
            parameters=parameters,
            component_obj=None,
            random_seed=random_seed,
        )

    def fit(self, X, y=None):
        """Fits imputer to data.

        'None' values are converted to np.nan before imputation and are treated as the same.
        If a value is missing at the beginning or end of a column, that value will be imputed using
        backwards fill or forwards fill as necessary, respectively.

        Args:
            X (pd.DataFrame, np.ndarray): The input training data of shape [n_samples, n_features]
            y (pd.Series, optional): The target training data of length [n_samples]

        Returns:
            self
        """
        X = infer_feature_types(X)

        nan_ratio = X.isna().sum() / X.shape[0]
        self._all_null_cols = nan_ratio[nan_ratio == 1].index.tolist()

        def _filter_cols(impute_strat, X):
            """Function to return which columns of the dataset to impute given the impute strategy."""
            cols = []
            if self.parameters["categorical_impute_strategy"] == impute_strat:
                if self.parameters["numeric_impute_strategy"] == impute_strat:
                    cols = list(X.columns)
                else:
                    cols = list(X.ww.select(exclude=["numeric"]).columns)
            elif self.parameters["numeric_impute_strategy"] == impute_strat:
                cols = list(X.ww.select(include=["numeric"]).columns)

            X_cols = [col for col in cols if col not in self._all_null_cols]
            if len(X_cols) > 0:
                return X_cols

        self._forwards_cols = _filter_cols("forwards_fill", X)
        self._backwards_cols = _filter_cols("backwards_fill", X)
        self._interpolate_cols = _filter_cols("interpolate", X)

        if y is not None:
            y = infer_feature_types(y)
            if y.isnull().any():
                self._impute_target = self.parameters["target_impute_strategy"]

        return self

    def transform(self, X, y=None):
        """Transforms data X by imputing missing values using specified timeseries-specific strategies. 'None' values are converted to np.nan before imputation and are treated as the same.

        Args:
            X (pd.DataFrame): Data to transform.
            y (pd.Series, optional): Optionally, target data to transform.

        Returns:
            pd.DataFrame: Transformed X and y
        """
        if len(self._all_null_cols) == X.shape[1]:
            df = pd.DataFrame(index=X.index)
            df.ww.init()
            return df, y
        X = infer_feature_types(X)
        if y is not None:
            y = infer_feature_types(y)

        # This will change the logical type of BooleanNullable/IntegerNullable/AgeNullable columns with nans
        # so we save the original schema to recreate it where possible after imputation
        original_schema = X.ww.schema
        X, y = self._handle_nullable_types(X, y)

        X_not_all_null = X.ww.drop(self._all_null_cols)

        # Because the TimeSeriesImputer is always used with the TimeSeriesRegularizer,
        # many of the columns containing nans may have originally been non nullable logical types.
        # We will use the non nullable equivalents where possible
        original_schema = original_schema.get_subset_schema(
            list(X_not_all_null.columns),
        )
        new_ltypes = {
            col: _determine_non_nullable_equivalent(ltype)
            for col, ltype in original_schema.logical_types.items()
        }

        if self._forwards_cols is not None:
            X_forward = X[self._forwards_cols]
            imputed = X_forward.pad()
            imputed.bfill(inplace=True)  # Fill in the first value, if missing
            X_not_all_null[X_forward.columns] = imputed

        if self._backwards_cols is not None:
            X_backward = X[self._backwards_cols]
            imputed = X_backward.bfill()
            imputed.pad(inplace=True)  # Fill in the last value, if missing
            X_not_all_null[X_backward.columns] = imputed

        if self._interpolate_cols is not None:
            X_interpolate = X_not_all_null[self._interpolate_cols]
            imputed = X_interpolate.interpolate()
            imputed.bfill(inplace=True)  # Fill in the first value, if missing
            X_not_all_null[X_interpolate.columns] = imputed

            # Interpolate may add floating point values to integer data, so we
            # have to update those logical types from the ones passed in to a fractional type
            # Note we ignore all other types of columns to maintain the types specified above
            int_cols_to_update = original_schema._filter_cols(
                include=["IntegerNullable", "AgeNullable"],
            )
            new_int_ltypes = {
                col: _determine_fractional_type(ltype)
                for col, ltype in original_schema.logical_types.items()
                if col in int_cols_to_update
            }
            new_ltypes.update(new_int_ltypes)
        X_not_all_null.ww.init(schema=original_schema, logical_types=new_ltypes)

        y_imputed = pd.Series(y)
        if y is not None and len(y) > 0:
            if self._impute_target == "forwards_fill":
                y_imputed = y.pad()
                y_imputed.bfill(inplace=True)
            elif self._impute_target == "backwards_fill":
                y_imputed = y.bfill()
                y_imputed.pad(inplace=True)
            elif self._impute_target == "interpolate":
                y_imputed = y.interpolate()
                y_imputed.bfill(inplace=True)
            # Re-initialize woodwork with the downcast logical type
            y_imputed = ww.init_series(y_imputed, logical_type=y.ww.logical_type)

        return X_not_all_null, y_imputed

    def _handle_nullable_types(self, X=None, y=None):
        """Transforms X and y to remove any incompatible nullable types for the time series imputer when the interpolate method is used.

        Args:
            X (pd.DataFrame, optional): Input data to a component of shape [n_samples, n_features].
                May contain nullable types.
            y (pd.Series, optional): The target of length [n_samples]. May contain nullable types.

        Returns:
            X, y with any incompatible nullable types downcasted to compatible equivalents when interpolate is used. Is NoOp otherwise.
        """
        if self._impute_target == "interpolate":
            # For BooleanNullable, we have to avoid Categorical columns
            # since the category dtype also has incompatibilities with linear interpolate, which is expected
            if isinstance(y.ww.logical_type, BooleanNullable):
                y = ww.init_series(y, Double)
            else:
                _, y = super()._handle_nullable_types(None, y)
        if self._interpolate_cols is not None:
            X, _ = super()._handle_nullable_types(X, None)

        return X, y

"""Transformer that regularizes a dataset with an uninferrable offset frequency for time series problems."""

class TimeSeriesRegularizer(Transformer):
    """Transformer that regularizes an inconsistently spaced datetime column.

    If X is passed in to fit/transform, the column `time_index` will be checked for an inferrable offset frequency. If
    the `time_index` column is perfectly inferrable then this Transformer will do nothing and return the original X and y.

    If X does not have a perfectly inferrable frequency but one can be estimated, then X and y will be reformatted based
    on the estimated frequency for `time_index`. In the original X and y passed:
    - Missing datetime values will be added and will have their corresponding columns in X and y set to None.
    - Duplicate datetime values will be dropped.
    - Extra datetime values will be dropped.
    - If it can be determined that a duplicate or extra value is misaligned, then it will be repositioned to take the
    place of a missing value.

    This Transformer should be used before the `TimeSeriesImputer` in order to impute the missing values that were
    added to X and y (if passed).

    Args:
        time_index (string): Name of the column containing the datetime information used to order the data, required. Defaults to None.
        frequency_payload (tuple): Payload returned from Woodwork's infer_frequency function where debug is True. Defaults to None.
        window_length (int): The size of the rolling window over which inference is conducted to determine the prevalence of uninferrable frequencies.
        Lower values make this component more sensitive to recognizing numerous faulty datetime values. Defaults to 5.
        threshold (float): The minimum percentage of windows that need to have been able to infer a frequency. Lower values make this component more
        sensitive to recognizing numerous faulty datetime values. Defaults to 0.8.
        random_seed (int): Seed for the random number generator. This transformer performs the same regardless of the random seed provided.
        Defaults to 0.

    Raises:
        ValueError: if the frequency_payload parameter has not been passed a tuple
    """

    name = "Time Series Regularizer"
    hyperparameter_ranges = {}
    """{}"""

    modifies_target = True
    training_only = True

    def __init__(
        self,
        time_index=None,
        frequency_payload=None,
        window_length=4,
        threshold=0.4,
        random_seed=0,
        **kwargs,
    ):
        self.time_index = time_index
        self.frequency_payload = frequency_payload
        self.window_length = window_length
        self.threshold = threshold
        self.error_dict = {}
        self.inferred_freq = None
        self.debug_payload = None

        if self.frequency_payload and not isinstance(self.frequency_payload, tuple):
            raise ValueError(
                "The frequency_payload parameter must be a tuple returned from Woodwork's infer_frequency function where debug is True.",
            )

        parameters = {
            "time_index": time_index,
            "window_length": window_length,
            "threshold": threshold,
        }
        parameters.update(kwargs)

        super().__init__(parameters=parameters, random_seed=random_seed)

    def fit(self, X, y=None):
        """Fits the TimeSeriesRegularizer.

        Args:
            X (pd.DataFrame): The input training data of shape [n_samples, n_features].
            y (pd.Series, optional): The target training data of length [n_samples].

        Returns:
            self

        Raises:
            ValueError: if self.time_index is None, if X and y have different lengths, if `time_index` in X does not
                        have an offset frequency that can be estimated
            TypeError: if the `time_index` column is not of type Datetime
            KeyError: if the `time_index` column doesn't exist
        """
        if self.time_index is None:
            raise ValueError("The argument time_index cannot be None!")
        elif self.time_index not in X.columns:
            raise KeyError(
                f"The time_index column `{self.time_index}` does not exist in X!",
            )

        X_ww = infer_feature_types(X)

        if not isinstance(X_ww.ww.logical_types[self.time_index], Datetime):
            raise TypeError(
                f"The time_index column `{self.time_index}` must be of type Datetime.",
            )

        if y is not None:
            y = infer_feature_types(y)
            if len(X_ww) != len(y):
                raise ValueError(
                    "If y has been passed, then it must be the same length as X.",
                )

        if self.frequency_payload:
            ww_payload = self.frequency_payload
        else:
            ww_payload = infer_frequency(
                X_ww[self.time_index],
                debug=True,
                window_length=self.window_length,
                threshold=self.threshold,
            )
        self.inferred_freq = ww_payload[0]
        self.debug_payload = ww_payload[1]

        if self.inferred_freq is not None:
            return self

        if (
            self.debug_payload["estimated_freq"] is None
        ):  # If even WW can't infer the frequency
            raise ValueError(
                f"The column {self.time_index} does not have a frequency that can be inferred.",
            )

        estimated_freq = self.debug_payload["estimated_freq"]
        duplicates = self.debug_payload["duplicate_values"]
        missing = self.debug_payload["missing_values"]
        extra = self.debug_payload["extra_values"]
        nan = self.debug_payload["nan_values"]

        self.error_dict = self._identify_indices(
            self.time_index,
            X_ww,
            estimated_freq,
            duplicates,
            missing,
            extra,
            nan,
        )

        return self

    @staticmethod
    def _identify_indices(
        time_index,
        X,
        estimated_freq,
        duplicates,
        missing,
        extra,
        nan,
    ):
        """Identifies which of the problematic indices is actually misaligned.

        Args:
            time_index (str): The column name of the datetime values to consider.
            X (pd.DataFrame): The input training data of shape [n_samples, n_features].
            estimated_freq (str): The estimated frequency of the `time_index` column.
            duplicates (list): Payload information regarding the duplicate values.
            missing (list): Payload information regarding the missing values.
            extra (list): Payload information regarding the extra values.
            nan (list): Payload information regarding the nan values.

        Returns:
            (dict): A dictionary of the duplicate, missing, extra, and misaligned indices and their datetime values.
        """
        error_dict = {
            "duplicate": {},
            "missing": {},
            "extra": {},
            "nan": {},
            "misaligned": {},
        }

        # Adds the indices for the consecutive range of missing, duplicate, and extra values
        for each_missing in missing:
            # Needed to recreate what the missing datetime values would have been
            temp_dates = pd.date_range(
                pd.to_datetime(each_missing["dt"]),
                freq=estimated_freq,
                periods=each_missing["range"],
            )
            for each_range in range(each_missing["range"]):
                error_dict["missing"][each_missing["idx"] + each_range] = temp_dates[
                    each_range
                ]

        for each_duplicate in duplicates:
            for each_range in range(each_duplicate["range"]):
                error_dict["duplicate"][
                    each_duplicate["idx"] + each_range
                ] = pd.to_datetime(each_duplicate["dt"])

        for each_extra in extra:
            for each_range in range(each_extra["range"]):
                error_dict["extra"][each_extra["idx"] + each_range] = X.iloc[
                    each_extra["idx"] + each_range
                ][time_index]

        for each_nan in nan:
            for each_range in range(each_nan["range"]):
                error_dict["nan"][each_nan["idx"] + each_range] = "No Value"

        # Identify which of the duplicate/extra values in conjunction with the missing values are actually misaligned
        for ind_missing, missing_value in error_dict["missing"].items():
            temp_range = pd.date_range(missing_value, freq=estimated_freq, periods=3)
            window_range = temp_range[1] - temp_range[0]
            missing_range = [missing_value - window_range, missing_value + window_range]
            for ind_duplicate, duplicate_value in error_dict["duplicate"].items():
                if (
                    duplicate_value is not None
                    and missing_range[0] <= duplicate_value <= missing_range[1]
                ):
                    error_dict["misaligned"][ind_duplicate] = {
                        "incorrect": duplicate_value,
                        "correct": missing_value,
                    }
                    error_dict["duplicate"][ind_duplicate] = None
                    error_dict["missing"][ind_missing] = None
                    break
            for ind_extra, extra_value in error_dict["extra"].items():
                if (
                    extra_value is not None
                    and missing_range[0] <= extra_value <= missing_range[1]
                ):
                    error_dict["misaligned"][ind_extra] = {
                        "incorrect": extra_value,
                        "correct": missing_value,
                    }
                    error_dict["extra"][ind_extra] = None
                    error_dict["missing"][ind_missing] = None
                    break

        final_error_dict = {
            "duplicate": {},
            "missing": {},
            "extra": {},
            "nan": {},
            "misaligned": {},
        }
        # Remove duplicate/extra/missing values that were identified as misaligned
        for type_, type_inds in error_dict.items():
            new_type_inds = {
                ind_: date_ for ind_, date_ in type_inds.items() if date_ is not None
            }
            final_error_dict[type_] = new_type_inds

        return final_error_dict

    def transform(self, X, y=None):
        """Regularizes a dataframe and target data to an inferrable offset frequency.

        A 'clean' X and y (if y was passed in) are created based on an inferrable offset frequency and matching datetime values
        with the original X and y are imputed into the clean X and y. Datetime values identified as misaligned are
        shifted into their appropriate position.

        Args:
            X (pd.DataFrame): The input training data of shape [n_samples, n_features].
            y (pd.Series, optional): The target training data of length [n_samples].

        Returns:
            (pd.DataFrame, pd.Series): Data with an inferrable `time_index` offset frequency.
        """
        if self.inferred_freq is not None:
            return X, y

        # The cleaned df will begin at the range determined by estimated_range_start, which will result
        # in dropping of the first consecutive faulty values in the dataset.
        cleaned_df = pd.DataFrame(
            {
                self.time_index: pd.date_range(
                    self.debug_payload["estimated_range_start"],
                    self.debug_payload["estimated_range_end"],
                    freq=self.debug_payload["estimated_freq"],
                ),
            },
        )

        cleaned_x = cleaned_df.merge(X, on=[self.time_index], how="left")
        cleaned_x = cleaned_x.groupby(self.time_index).first().reset_index()

        cleaned_y = None
        if y is not None:
            y_dates = pd.DataFrame({self.time_index: X[self.time_index], "target": y})
            cleaned_y = cleaned_df.merge(y_dates, on=[self.time_index], how="left")
            cleaned_y = cleaned_y.groupby(self.time_index).first().reset_index()

        for index, values in self.error_dict["misaligned"].items():
            to_replace = X.iloc[index]
            to_replace[self.time_index] = values["correct"]
            cleaned_x.loc[
                cleaned_x[self.time_index] == values["correct"]
            ] = to_replace.values
            if y is not None:
                cleaned_y.loc[cleaned_y[self.time_index] == values["correct"]] = y.iloc[
                    index
                ]

        if cleaned_y is not None:
            cleaned_y = cleaned_y["target"]
            cleaned_y = ww.init_series(cleaned_y)

        cleaned_x.ww.init()

        return cleaned_x, cleaned_y