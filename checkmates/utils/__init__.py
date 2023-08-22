"""Utility methods."""
from checkmates.utils.gen_utils import classproperty, safe_repr
from checkmates.utils.woodwork_utils import infer_feature_types
from checkmates.utils.base_meta import BaseMeta
from checkmates.utils.nullable_type_utils import (
    _downcast_nullable_X,
    _downcast_nullable_y,
    _determine_downcast_type,
    _determine_fractional_type,
    _determine_non_nullable_equivalent,
    _get_new_logical_types_for_imputed_data,
)
from checkmates.utils.logger import get_logger, log_subtitle, log_title
