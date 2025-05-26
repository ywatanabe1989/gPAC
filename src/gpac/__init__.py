from ._PAC import PAC
from ._calculate_gpac import calculate_pac
from ._SyntheticDataGenerator import SyntheticDataGenerator
from .tensorpac_compat import (
    calculate_pac_tensorpac_compat,
    TENSORPAC_CONFIGS,
    compare_with_tensorpac
)
from ._differentiable_bucketize import (
    differentiable_bucketize,
    differentiable_bucketize_indices,
    DifferentiableBucketize,
    differentiable_phase_binning,
)

__version__ = "0.2.0"
