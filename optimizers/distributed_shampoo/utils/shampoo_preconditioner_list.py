"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from itertools import chain
from typing import Any, DefaultDict, Optional, Sequence, List, Tuple, Union, Dict

import torch
import torch.distributed as dist
from .shampoo_block_info import BlockInfo
from .shampoo_utils import (
    compress_list,
    get_dtype_size,
)

from ...matrix_functions import (
    check_diagonal,
    compute_matrix_root_inverse_residuals,
    matrix_inverse_root,
    matrix_inverse_root_fast_asymmetric,
    matrix_inverse_root_fast_default
)
from ...optimizer_modules import OptimizerModule
from torch import Tensor
from torch.autograd import profiler


logger: logging.Logger = logging.getLogger(__name__)

RWS_ADAGRAD = "rws_adagrad"
ADAGRAD = "adagrad"
SHAMPOO = "shampoo"


class PreconditionerList(ABC):
    """Preconditioner base class.

    Args:
        block_list (Tuple[Tensor, ...]): List of (blocks of) parameters.

    """

    def __init__(
        self,
        block_list: Tuple[Tensor, ...],
    ) -> None:
        super().__init__()
        self._numel_list: Tuple[int, ...] = (0,) * len(block_list)
        self._dims_list: Tuple[torch.Size, ...] = tuple(
            block.size() for block in block_list
        )
        self._num_bytes_list: Tuple[int, ...] = (0,) * len(block_list)

    @abstractmethod
    def update_preconditioners(
        self,
        masked_grad_list: Tuple[Tensor, ...],
        step: Tensor,
    ) -> None:
        ...

    @abstractmethod
    def precondition(self, masked_grad_list: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        ...

    @abstractmethod
    def compress_preconditioner_list(
        self, local_grad_selector: Tuple[bool, ...]
    ) -> None:
        ...

    @property
    def numel_list(self) -> Tuple[int, ...]:
        return self._numel_list

    @property
    def dims_list(self) -> Tuple[torch.Size, ...]:
        return self._dims_list

    @property
    def num_bytes_list(self) -> Tuple[int, ...]:
        return self._num_bytes_list

    def numel(self) -> int:
        return sum(self._numel_list)

    def num_bytes(self) -> int:
        return sum(self._num_bytes_list)


class SGDPreconditionerList(PreconditionerList):
    """SGD (identity) preconditioners for a list of parameters.

    Args:
        block_list (Tuple[Tensor, ...]): List of (blocks of) parameters.

    """

    def __init__(
        self,
        block_list: Tuple[Tensor, ...],
    ) -> None:
        super().__init__(block_list)

    def update_preconditioners(
        self,
        masked_grad_list: Tuple[Tensor, ...],
        step: Tensor,
    ) -> None:
        return

    def precondition(self, masked_grad_list: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        return masked_grad_list

    def compress_preconditioner_list(
        self, local_grad_selector: Tuple[bool, ...]
    ) -> None:
        return


class RWSAdagradPreconditionerList(PreconditionerList):
    """Row-Wise Adagrad / Adam / RMSProp preconditioners for a list of parameters.

    Operations are performed in-place with foreach operators.

    NOTE: Does not support sparse gradients at this time.

    To enable Adagrad, set beta2 = 1.0.
    To enable RMSProp, set beta2 = 0.999.
    To enable Adam, set beta2 = 0.999, use_bias_correction = True.

    Other variants can also be specified.

    Args:
        block_list (Tuple[Tensor, ...]): List of (blocks of) parameters.
        state (DefaultDict[Parameter, Any]): Dictionary containing optimizer state.
        block_info_list (Tuple[BlockInfo, ...]): List containing corresponding BlockInfo for each block/parameter in block_list.
            Note that this should have the same length as block_list.
        distributor_selector (Tuple[bool, ...]): Distributor selector is a boolean list indicating whether a blocked parameter
            is selected by the current Distributor.
        beta2 (float): Exponential moving average factor for Adam/RMSprop second moment state. If beta2 = 1., will use
            unweighted sum. (Default: 1.0)
        epsilon (float): Epsilon term for regularizing preconditioner to ensure positive definiteness. (Default: 1e-10)
        use_bias_correction (bool): Flag for using bias correction. (Default: False)

    """

    def __init__(
        self,
        block_list: Tuple[Tensor, ...],
        state: DefaultDict[Tensor, Any],
        block_info_list: Tuple[BlockInfo, ...],
        distributor_selector: Tuple[bool, ...],
        beta2: float = 1.0,
        epsilon: float = 1e-10,
        use_bias_correction: bool = True,
    ) -> None:
        super().__init__(block_list)

        # Instantiate scalar hyperparameters.
        self._beta2 = beta2
        self._epsilon = epsilon
        self._use_bias_correction = use_bias_correction
        self._bias_correction2: Tensor = torch.tensor(1.0)

        # Instantiate (blocked) AdaGrad preconditioners and construct preconditioner list.
        # NOTE: We need to instantiate the AdaGrad preconditioner states within the optimizer's state dictionary,
        # and do not explicitly store them as AdagradPreconditionerList attributes here.
        # This is because the optimizer state is defined per-parameter, but AdagradPreconditionerList is defined
        # across each parameter group (which includes multiple parameters).
        preconditioner_list = []
        for block, block_info in zip(block_list, block_info_list, ):
            param_index, block_index = block_info.composable_block_ids
            if block_index not in state[block_info.param]:
                state[block_info.param][block_index] = {}
            block_state = state[block_info.param][block_index]

            # Instantiate AdaGrad optimizer state for this block.
            preconditioner_index = str(param_index) + "." + str(block_index)
            block_state[RWS_ADAGRAD] = block_info.allocate_zeros_tensor(
                block.shape[0], block.dtype, block.device
            )
            preconditioner_list.append(block_info.get_tensor(block_state[RWS_ADAGRAD]))

            logger.info(
                f"Instantiated RWS Adagrad Preconditioner {preconditioner_index} ({block_state[RWS_ADAGRAD].shape}) "
                f"for Parameter {param_index} ({block_info.param.shape}), Block {block_index} ({block.shape})."
            )

        # Masked lists are the list of active preconditioners or values after filtering out gradients with None.
        self._local_preconditioner_list: Tuple[Tensor, ...] = compress_list(
            preconditioner_list, distributor_selector
        )
        self._masked_preconditioner_list: Tuple[
            Tensor, ...
        ] = self._local_preconditioner_list

        # Construct lists of bytes and numels for logging purposes.
        self._numel_list: Tuple[int, ...] = tuple(
            preconditioner.numel() for preconditioner in preconditioner_list
        )
        self._num_bytes_list: Tuple[int, ...] = tuple(
            preconditioner.numel() * preconditioner.element_size()
            for preconditioner in preconditioner_list
        )

        # Log breakdown of number of elements and bytes.
        logger.info(
            f"Rank {dist.get_rank()}: RWSAdaGradPreconditionerList Numel Breakdown: {self._numel_list}"
        )
        logger.info(
            f"Rank {dist.get_rank()}: RWSAdaGradPreconditionerList Bytes Breakdown: {self._num_bytes_list}"
        )
        logger.info(
            f"Rank {dist.get_rank()}: RWSAdaGradPreconditionerList Total Elements: {sum(self._numel_list)}"
        )
        logger.info(
            f"Rank {dist.get_rank()}: RWSAdaGradPreconditionerList Total Bytes: {sum(self._num_bytes_list)}"
        )

    def update_preconditioners(
        self,
        masked_grad_list: Tuple[Tensor, ...],
        step: Tensor,
    ) -> None:
        with profiler.record_function(
            f"## {self.__class__.__name__}:{self.update_preconditioners.__name__} ##"
        ):
            masked_avg_rws_grad_norm_sq_list = tuple(
                torch.mean(grad * grad, axis=tuple(torch.arange(1, grad.dim())))
                for grad in masked_grad_list
            )
            if self._beta2 == 1.0:
                torch._foreach_add_(
                    self._masked_preconditioner_list,
                    masked_avg_rws_grad_norm_sq_list,
                    value=1.0,
                )
            else:
                torch._foreach_mul_(self._masked_preconditioner_list, self._beta2)
                torch._foreach_add_(
                    self._masked_preconditioner_list,
                    masked_avg_rws_grad_norm_sq_list,
                    alpha=1.0 - self._beta2,
                )

            # Update bias correction term based on step list.
            if self._use_bias_correction and self._beta2 < 1.0:
                self._bias_correction2 = torch.tensor(1.0) - self._beta2**step

    def precondition(self, masked_grad_list: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        with profiler.record_function(
            f"## {self.__class__.__name__}:{self.precondition.__name__} ##"
        ):
            masked_bias_corrected_preconditioner_list = torch._foreach_div(
                self._masked_preconditioner_list, self._bias_correction2
            )
            torch._foreach_sqrt_(masked_bias_corrected_preconditioner_list)
            torch._foreach_add_(
                masked_bias_corrected_preconditioner_list, self._epsilon
            )
            return tuple(
                grad / bias_corrected_preconditioner[(...,) + (None,) * (grad.dim() - 1)]
                for grad, bias_corrected_preconditioner in zip(
                    masked_grad_list, masked_bias_corrected_preconditioner_list
                )
            )

    def compress_preconditioner_list(
        self, local_grad_selector: Tuple[bool, ...]
    ) -> None:
        with profiler.record_function(
            f"## {self.__class__.__name__}:{self.compress_preconditioner_list.__name__} ##"
        ):
            self._masked_preconditioner_list = compress_list(
                self._local_preconditioner_list, local_grad_selector
            )



class AdagradPreconditionerList(PreconditionerList):
    """Adagrad / Adam / RMSProp preconditioners for a list of parameters.

    Operations are performed in-place with foreach operators.

    NOTE: Does not support sparse gradients at this time.

    To enable Adagrad, set beta2 = 1.0.
    To enable RMSProp, set beta2 = 0.999.
    To enable Adam, set beta2 = 0.999, use_bias_correction = True.

    Other variants can also be specified.

    Args:
        block_list (Tuple[Tensor, ...]): List of (blocks of) parameters.
        state (DefaultDict[Parameter, Any]): Dictionary containing optimizer state.
        block_info_list (Tuple[BlockInfo, ...]): List containing corresponding BlockInfo for each block/parameter in block_list.
            Note that this should have the same length as block_list.
        distributor_selector (Tuple[bool, ...]): Distributor selector is a boolean list indicating whether a blocked parameter
            is selected by the current Distributor.
        beta2 (float): Exponential moving average factor for Adam/RMSprop second moment state. If beta2 = 1., will use
            unweighted sum. (Default: 1.0)
        epsilon (float): Epsilon term for regularizing preconditioner to ensure positive definiteness. (Default: 1e-10)
        use_bias_correction (bool): Flag for using bias correction. (Default: False)

    """

    def __init__(
        self,
        block_list: Tuple[Tensor, ...],
        state: DefaultDict[Tensor, Any],
        block_info_list: Tuple[BlockInfo, ...],
        distributor_selector: Tuple[bool, ...],
        beta2: float = 1.0,
        epsilon: float = 1e-10,
        use_bias_correction: bool = True,
    ) -> None:
        super().__init__(block_list)

        # Instantiate scalar hyperparameters.
        self._beta2 = beta2
        self._epsilon = epsilon
        self._use_bias_correction = use_bias_correction
        self._bias_correction2: Tensor = torch.tensor(1.0)

        # Instantiate (blocked) AdaGrad preconditioners and construct preconditioner list.
        # NOTE: We need to instantiate the AdaGrad preconditioner states within the optimizer's state dictionary,
        # and do not explicitly store them as AdagradPreconditionerList attributes here.
        # This is because the optimizer state is defined per-parameter, but AdagradPreconditionerList is defined
        # across each parameter group (which includes multiple parameters).
        preconditioner_list = []
        for block, block_info in zip(block_list, block_info_list, ):
            param_index, block_index = block_info.composable_block_ids
            if block_index not in state[block_info.param]:
                state[block_info.param][block_index] = {}
            block_state = state[block_info.param][block_index]

            # Instantiate AdaGrad optimizer state for this block.
            preconditioner_index = str(param_index) + "." + str(block_index)
            block_state[ADAGRAD] = block_info.allocate_zeros_tensor(
                block.size(), block.dtype, block.device
            )
            preconditioner_list.append(block_info.get_tensor(block_state[ADAGRAD]))

            logger.info(
                f"Instantiated Adagrad Preconditioner {preconditioner_index} ({block_state[ADAGRAD].shape}) "
                f"for Parameter {param_index} ({block_info.param.shape}), Block {block_index} ({block.shape})."
            )

        # Masked lists are the list of active preconditioners or values after filtering out gradients with None.
        self._local_preconditioner_list: Tuple[Tensor, ...] = compress_list(
            preconditioner_list, distributor_selector
        )
        self._masked_preconditioner_list: Tuple[
            Tensor, ...
        ] = self._local_preconditioner_list

        # Construct lists of bytes and numels for logging purposes.
        self._numel_list: Tuple[int, ...] = tuple(
            preconditioner.numel() for preconditioner in preconditioner_list
        )
        self._num_bytes_list: Tuple[int, ...] = tuple(
            preconditioner.numel() * preconditioner.element_size()
            for preconditioner in preconditioner_list
        )

        # Log breakdown of number of elements and bytes.
        logger.info(
            f"Rank {dist.get_rank()}: AdaGradPreconditionerList Numel Breakdown: {self._numel_list}"
        )
        logger.info(
            f"Rank {dist.get_rank()}: AdaGradPreconditionerList Bytes Breakdown: {self._num_bytes_list}"
        )
        logger.info(
            f"Rank {dist.get_rank()}: AdaGradPreconditionerList Total Elements: {sum(self._numel_list)}"
        )
        logger.info(
            f"Rank {dist.get_rank()}: AdaGradPreconditionerList Total Bytes: {sum(self._num_bytes_list)}"
        )

    def update_preconditioners(
        self,
        masked_grad_list: Tuple[Tensor, ...],
        step: Tensor,
    ) -> None:
        with profiler.record_function(
            f"## {self.__class__.__name__}:{self.update_preconditioners.__name__} ##"
        ):
            if self._beta2 == 1.0:
                torch._foreach_addcmul_(
                    self._masked_preconditioner_list,
                    masked_grad_list,
                    masked_grad_list,
                    value=1.0,
                )
            else:
                torch._foreach_mul_(self._masked_preconditioner_list, self._beta2)
                torch._foreach_addcmul_(
                    self._masked_preconditioner_list,
                    masked_grad_list,
                    masked_grad_list,
                    value=1 - self._beta2,
                )

            # Update bias correction term based on step list.
            if self._use_bias_correction and self._beta2 < 1.0:
                self._bias_correction2 = torch.tensor(1.0) - self._beta2**step

    def precondition(self, masked_grad_list: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        with profiler.record_function(
            f"## {self.__class__.__name__}:{self.precondition.__name__} ##"
        ):
            masked_bias_corrected_preconditioner_list = torch._foreach_div(
                self._masked_preconditioner_list, self._bias_correction2
            )
            torch._foreach_sqrt_(masked_bias_corrected_preconditioner_list)
            torch._foreach_add_(
                masked_bias_corrected_preconditioner_list, self._epsilon
            )
            return torch._foreach_div(
                masked_grad_list, masked_bias_corrected_preconditioner_list
            )

    def compress_preconditioner_list(
        self, local_grad_selector: Tuple[bool, ...]
    ) -> None:
        with profiler.record_function(
            f"## {self.__class__.__name__}:{self.compress_preconditioner_list.__name__} ##"
        ):
            self._masked_preconditioner_list = compress_list(
                self._local_preconditioner_list, local_grad_selector
            )


@dataclass
class ShampooKroneckerFactors(OptimizerModule):
    """Shampoo Kronecker Factors."""

    factor_matrices: Tuple[Tensor, ...]
    inv_factor_matrices: Tuple[Tensor, ...]
    factor_matrix_indices: Tuple[str, ...]
    is_factor_matrices_diagonal: Tuple[Tensor, ...] = field(init=False)
    eigenvalues: List[Optional[Tensor]] = field(default_factory=list)
    eigenvectors: List[Optional[Tensor]] = field(default_factory=list)

    def __post_init__(self) -> None:
        super().__init__()
        assert (
            len(self.factor_matrices)
            == len(self.inv_factor_matrices)
            == len(self.factor_matrix_indices)
        )
        self.is_factor_matrices_diagonal = tuple(
            torch.tensor(True) for _ in self.factor_matrices
        )
        self.eigenvalues = [None] * len(self.factor_matrices)
        self.eigenvectors = [None] * len(self.factor_matrices)  


class ShampooPreconditionerList(PreconditionerList):
    """Shampoo preconditioners for list of parameters.

    NOTE: Does not support sparse gradients at this time.

    Args:
        block_list (Tuple[Tensor, ...]): List of (blocks of) parameters.
        state (DefaultDict[Parameter, Any]): Dictionary containing optimizer state.
        block_info_list (Tuple[BlockInfo, ...]): List containing corresponding BlockInfo for each block/parameter in block_list.
            Note that this should have the same length as block_list.
        distributor_selector (Tuple[bool, ...]): Distributor selector is a boolean list indicating whether a blocked parameter
            is selected by the current Distributor.
        beta2 (float): Exponential moving average factor for Shampoo factor matrices. If beta2 = 1., will use unweighted sum.
            (Default: 1.0)
        epsilon (float): Epsilon term for regularizing preconditioner to ensure positive definiteness. (Default: 1e-12)
        inv_root_override (int, Tuple[int, ...]): Inverse root to use in Shampoo. If a list [l0, l1, l2, ..., lp], then we will
            use -1 / l0 for 0-D tensors (scalars), -1 / l1 for 1-D tensor (vectors), -1 / l2 for 2-D tensors (matrices), and so on.
            If the order of the tensor exceeds the length of the list, we revert to using the default value. If 0 is used, uses the
            default inverse root -1 / (2 * o), where o is the order of the tensor. (Default: 0)
        exponent_multiplier (float): Number to be multiplied to the numerator of the inverse root, i.e., eta where the
            exponent is -eta / (2 * p). (Default: 1.0)
        use_bias_correction (bool): Flag for using bias correction. (Default: True)
        factor_matrix_dtype (torch.dtype): Data type for accumulating and computing root inverse of preconditioners. (Default: torch.float)
        use_protected_eigh (bool): Flag for using two guards to prevent failures of torch.linalg.eigh. (Default: True)
            1. Attempts to compute root inverse in preconditioner_dtype precision.
            2. Attempts to recompute the eigendecomposition if using lower-precision fails.
            3. Otherwise, re-uses previous inverse factor matrix when both root inverse computations fail.

    """

    def __init__(
        self,
        block_list: Tuple[Tensor, ...],
        state: DefaultDict[Tensor, Any],
        block_info_list: Tuple[BlockInfo, ...],
        distributor_selector: Tuple[bool, ...],
        beta2: float = 1.0,
        epsilon: float = 1e-10,
        epsilon_left : Optional[float] = None,
        epsilon_right : Optional[float] = None,
        use_adaptive_epsilon: bool = False,
        condition_thresholds: Optional[Dict[float, float]] = None,
        is_default_config: bool = False,  # Optimization flag
        inv_root_override: Union[int, Tuple[int, ...]] = 0,
        exponent_multiplier: float = 1.0,
        use_bias_correction: bool = True,
        factor_matrix_dtype: torch.dtype = torch.float,
        use_protected_eigh: bool = True,
        use_trace_correction: bool = False,
        matrix_root_inv_threshold: float = 0.0,
    ) -> None:
        super().__init__(block_list)

        # Initialize parameters.
        self._beta2 = beta2
        self._epsilon = epsilon
        self._use_trace_correction = use_trace_correction
        self._epsilon_left = epsilon_left if epsilon_left is not None else epsilon
        self._epsilon_right = epsilon_right if epsilon_right is not None else epsilon
        self._use_adaptive_epsilon = use_adaptive_epsilon
        self._condition_thresholds = condition_thresholds or (
            {1e6: 1e-5, 1e8: 1e-4} if use_adaptive_epsilon else None
        )
        self._matrix_root_inv_threshold = matrix_root_inv_threshold
        
        # Optimization: Fast path detection
        self._is_default_config = is_default_config
        
        # For adaptive epsilon (unchanged)
        self._thresholds_tensor = None
        self._epsilons_tensor = None
        self._small_positive_tensor = None
        
        # Optimized: Determine configuration type
        self._use_per_dim_epsilon = False
        self._is_asymmetric_non_adaptive = False
        
        if not is_default_config:
            if use_adaptive_epsilon:
                self._use_per_dim_epsilon = True
            elif self._epsilon_left != self._epsilon_right:
                self._use_per_dim_epsilon = True
                self._is_asymmetric_non_adaptive = True
        
        self._inv_root_override = inv_root_override
        self._exponent_multiplier = exponent_multiplier
        self._factor_matrix_dtype = factor_matrix_dtype
        self._use_bias_correction = use_bias_correction
        self._use_protected_eigh = use_protected_eigh
        self._bias_correction2: Tensor = torch.tensor(1.0)
        
        # Instantiate (blocked) Kronecker factors and construct list of Kronecker factors.
        kronecker_factors_list = []
        epsilon_per_dim_list = [] if self._use_per_dim_epsilon else None
        
        def _compute_relative_condition_number(
            self,
            factor_matrix : Tensor,
            prev_eigenvectors : Tensor,
            prev_eigenvalues : Tensor,
            epsilon : float
        ) -> Tensor:
            L_tilde = torch.linalg.multi_dot([prev_eigenvectors.T, factor_matrix, prev_eigenvectors])
            d_term = torch.sqrt(prev_eigenvalues + epsilon)
            denominator = torch.outer(d_term, d_term)
            numerator = L_tilde - torch.diag(prev_eigenvalues)
            scaled_diff = numerator / denominator
            rc_t = torch.linalg.norm(scaled_diff, ord = 'fro')
            return rc_t
        
        # Pre-compute epsilon tensors for asymmetric non-adaptive case
        if self._is_asymmetric_non_adaptive and len(block_list) > 0:
            device = block_list[0].device
            self._epsilon_left_tensor = torch.tensor(self._epsilon_left, device=device, dtype=torch.float32)
            self._epsilon_right_tensor = torch.tensor(self._epsilon_right, device=device, dtype=torch.float32)
        
        for block, block_info, dims in zip(
            block_list, block_info_list, self._dims_list, 
        ):
            param_index, block_index = block_info.composable_block_ids
            if block_index not in state[block_info.param]:
                state[block_info.param][block_index] = {}
            block_state = state[block_info.param][block_index]

            # Only compute per-dim epsilon if needed
            if self._use_per_dim_epsilon:
                block_epsilon_per_dim = []
                for dim_idx, dim in enumerate(dims):
                    if len(dims) == 1:
                        block_epsilon_per_dim.append(self._epsilon)
                    elif dim_idx == 0:
                        block_epsilon_per_dim.append(self._epsilon_left)
                    else:
                        block_epsilon_per_dim.append(self._epsilon_right)
                epsilon_per_dim_list.append(tuple(block_epsilon_per_dim))

            # Instantiate ShampooKroneckerFactors for this block.
            # The factor matrices are instantiated using the determined dtype.
            factor_matrices = tuple(
                block_info.allocate_zeros_tensor(
                    (dim, dim),
                    self._factor_matrix_dtype,
                    block_info.param.device,
                )
                for dim in dims
            )
            # The inverse factor matrices are instantiated using the dtype of the block / gradient.
            inv_factor_matrices = tuple(
                block_info.allocate_zeros_tensor(
                    (dim, dim),
                    block.dtype,
                    block_info.param.device,
                )
                for dim in dims
            )

            preconditioner_index = str(param_index) + "." + str(block_index)
            factor_matrix_indices = tuple(
                preconditioner_index + "." + str(k) for k in range(len(dims))
            )
            block_state[SHAMPOO] = ShampooKroneckerFactors(
                factor_matrices=factor_matrices,
                inv_factor_matrices=inv_factor_matrices,
                factor_matrix_indices=factor_matrix_indices,
            )
            kronecker_factors_list.append(
                ShampooKroneckerFactors(
                    factor_matrices=tuple(
                        block_info.get_tensor(t) for t in factor_matrices
                    ),
                    inv_factor_matrices=tuple(
                        block_info.get_tensor(t) for t in inv_factor_matrices
                    ),
                    factor_matrix_indices=factor_matrix_indices,
                )
            )

            # Optimized logging
            if self._is_default_config:
                logger.info(
                    f"Instantiated Shampoo Preconditioner {preconditioner_index} "
                    f"({[factor_matrix.shape for factor_matrix in block_state[SHAMPOO].factor_matrices]}) "
                    f"with epsilon: {self._epsilon} (default config) "
                    f"for Parameter {param_index} ({block_info.param.shape}), Block {block_index} ({block.shape})."
                )
            elif self._use_per_dim_epsilon and epsilon_per_dim_list:
                logger.info(
                    f"Instantiated Shampoo Preconditioner {preconditioner_index} "
                    f"({[factor_matrix.shape for factor_matrix in block_state[SHAMPOO].factor_matrices]}) "
                    f"with epsilon per dim: {epsilon_per_dim_list[-1]} "
                    f"for Parameter {param_index} ({block_info.param.shape}), Block {block_index} ({block.shape})."
                )
            else:
                logger.info(
                    f"Instantiated Shampoo Preconditioner {preconditioner_index} "
                    f"({[factor_matrix.shape for factor_matrix in block_state[SHAMPOO].factor_matrices]}) "
                    f"with epsilon: {self._epsilon} "
                    f"for Parameter {param_index} ({block_info.param.shape}), Block {block_index} ({block.shape})."
                )

        # Initialize local lists.
        local_block_list = compress_list(block_list, distributor_selector)
        self._local_kronecker_factors_list: Tuple[
            ShampooKroneckerFactors, ...
        ] = compress_list(kronecker_factors_list, distributor_selector)
        
        # Only create epsilon list if needed
        if self._use_per_dim_epsilon:
            self._local_epsilon_per_dim_list: Tuple[Tuple[float, ...], ...] = compress_list(
                epsilon_per_dim_list, distributor_selector
            )
        else:
            self._local_epsilon_per_dim_list = None
            
        self._local_order_list: Tuple[int, ...] = tuple(
            block.dim() for block in local_block_list
        )
        self._local_root_list: Tuple[int, ...] = self._get_inverse_roots_from_override(
            self._inv_root_override, self._local_order_list
        )

        # Masked lists are the list of active preconditioners or values after filtering out gradients with None.
        self._masked_order_list: Tuple[int, ...] = self._local_order_list
        self._masked_root_list: Tuple[int, ...] = self._local_root_list
        self._masked_kronecker_factors_list: Tuple[
            ShampooKroneckerFactors, ...
        ] = self._local_kronecker_factors_list
        self._masked_epsilon_per_dim_list: Tuple[Tuple[float, ...], ...] = (
            self._local_epsilon_per_dim_list if self._use_per_dim_epsilon else None
        )
        
        # Construct lists of bytes and numels for logging purposes.
        # NOTE: These lists are constructed across all blocked parameters.
        self._numel_list: Tuple[int, ...] = tuple(
            sum(2 * dim**2 for dim in dims) for dims in self._dims_list
        )
        self._num_bytes_list: Tuple[int, ...] = tuple(
            numel
            * (get_dtype_size(self._factor_matrix_dtype) + get_dtype_size(block.dtype))
            // 2
            for numel, block in zip(self._numel_list, local_block_list)
        )

        # Log breakdown of number of elements and bytes.
        logger.info(
            f"Rank {dist.get_rank()}: ShampooPreconditionerList Numel Breakdown: {self._numel_list}"
        )
        logger.info(
            f"Rank {dist.get_rank()}: ShampooPreconditionerList Bytes Breakdown: {self._num_bytes_list}"
        )
        logger.info(
            f"Rank {dist.get_rank()}: ShampooPreconditionerList Total Elements: {sum(self._numel_list)}"
        )
        logger.info(
            f"Rank {dist.get_rank()}: ShampooPreconditionerList Total Bytes: {sum(self._num_bytes_list)}"
        )

    def _initialize_adaptive_epsilon_tensors(self) -> None:
        """Initialize GPU tensors for adaptive epsilon computation once."""
        if not self._condition_thresholds or self._thresholds_tensor is not None:
            return
            
        device = self._local_kronecker_factors_list[0].factor_matrices[0].device
        thresholds_keys = sorted(self._condition_thresholds.keys())
        thresholds_values = [self._condition_thresholds[k] for k in thresholds_keys]
        
        self._thresholds_tensor = torch.tensor(
            thresholds_keys,
            dtype=torch.float32,
            device=device
        )
        self._epsilons_tensor = torch.tensor(
            thresholds_values,
            dtype=torch.float32,
            device=device
        )
        self._small_positive_tensor = torch.tensor(
            1e-8,
            dtype=torch.float32,
            device=device
        )

    @staticmethod
    def _get_inverse_roots_from_override(
        inv_root_override: Union[int, Sequence[int]], order_list: Tuple[int, ...]
    ) -> Tuple[int, ...]:
        """Retrieves the appropriate root from the inverse root override parameter
        for a list of tensor orders.

        For example, suppose inv_root_override = (2, 1, 4, 3).
        If order = 0, then we will return 2;
        If order = 1, then we will return 1;
        If order = 2, then we will return 4;
        If order = 3, then we will return 3;
        If order > 3, then we will return 2 * order.

        Args:
            inv_root_override (int, Sequence[int]): Inverse root override int or list.
            order_list (Tuple[int, ...]): List of orders for their corresponding tensors.

        Returns:
            root_list (int): Inverse roots to use in Shampoo for a list of tensors.

        """
        if isinstance(inv_root_override, Sequence):
            return tuple(
                2 * order
                if order >= len(inv_root_override)
                else inv_root_override[order]
                for order in order_list
            )
        else:
            return (
                tuple(2 * order for order in order_list)
                if inv_root_override == 0
                else (inv_root_override,) * len(order_list)
            )

    def update_preconditioners(
        self, masked_grad_list: Tuple[Tensor, ...], step: Tensor
    ) -> None:
        with profiler.record_function(
            f"## {self.__class__.__name__}:{self.update_preconditioners.__name__} ##"
        ):
            # NOTE: Unlike AdagradPreconditionerList, we will loop through each gradient individually.
            # We apply foreach operators onto the list of Kronecker factor matrices (as opposed to the
            # full list of gradients/optimizer states).
            for grad, order, kronecker_factors in zip(
                masked_grad_list,
                self._masked_order_list,
                self._masked_kronecker_factors_list,
            ):
                # Scale Kronecker factors as a list.
                if self._beta2 != 1.0:
                    torch._foreach_mul_(kronecker_factors.factor_matrices, self._beta2)

                # Construct outer product list for updating Kronecker factors.
                outer_product_list = tuple(
                    torch.tensordot(
                        grad,
                        grad,
                        # Contracts across all dimensions except for k.
                        dims=[[*chain(range(k), range(k + 1, order))]] * 2,
                    )
                    for k in range(order)
                )

                # Update Kronecker factors.
                torch._foreach_add_(
                    kronecker_factors.factor_matrices,
                    outer_product_list,
                    alpha=1 - self._beta2 if self._beta2 != 1.0 else 1.0,
                )

            # Update bias correction term based on step list.
            if self._use_bias_correction and self._beta2 < 1.0:
                self._bias_correction2 = torch.tensor(1.0) - self._beta2**step

    def precondition(self, masked_grad_list: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        with profiler.record_function(
            f"## {self.__class__.__name__}:{self.precondition.__name__} ##"
        ):

            def precondition_masked_grad(
                masked_grad: Tensor,
                inv_factor_matrices: Tuple[Tensor, ...],
                factor_matrices : Tuple[Tensor, ...],
                use_trace_correction : bool =True,
            ) -> Tensor:
                
                for inv_factor_matrix in inv_factor_matrices:
                    masked_grad = torch.tensordot(
                        masked_grad, inv_factor_matrix, [[0], [0]]
                    )
                if use_trace_correction and len(factor_matrices) >= 1:
                    L_matrix = factor_matrices[0] / self._bias_correction2
                    trace_L = torch.trace(L_matrix)

                    if trace_L > 1e-10:
                        masked_grad = masked_grad / trace_L

                return masked_grad

            return tuple(
                precondition_masked_grad(
                    masked_grad=masked_grad,
                    inv_factor_matrices=kronecker_factors.inv_factor_matrices,
                    factor_matrices = kronecker_factors.factor_matrices,
                    use_trace_correction = self._use_trace_correction,
                )
                for masked_grad, kronecker_factors in zip(
                    masked_grad_list, self._masked_kronecker_factors_list, 
                )
            )

    def _compute_single_root_inverse(
        self,
        factor_matrix: Tensor,
        inv_factor_matrix: Tensor,
        is_factor_matrix_diagonal: Tensor,
        factor_matrix_index: str,
        root: int,
        epsilon_value: float,
        kronecker_factors : ShampooKroneckerFactors,
        factor_idx : int
    ) -> None:
        """Compute root inverse for a single factor matrix."""
        # For tracking diagonality of the preconditioner.
        # Checks if the preconditioner is currently diagonal, then checks whether or not
        # the update matrix is diagonal.
        if is_factor_matrix_diagonal and not check_diagonal(factor_matrix):
            is_factor_matrix_diagonal.copy_(torch.tensor(False))
            logger.debug(
                f"Factor matrix {factor_matrix_index} is not diagonal."
            )

        # Add epsilon term and incorporate bias correction.
        bias_corrected_factor_matrix = (
            factor_matrix / self._bias_correction2
        )
        should_update = True
        if (self._matrix_root_inv_threshold > 0.0 and
        kronecker_factors.eigenvectors[factor_idx] is not None):
            try:
                rc_t = self._compute_relative_condition_number(
                    bias_corrected_factor_matrix,
                    kronecker_factors.eigenvectors[factor_idx],
                    kronecker_factors.eigenvalues[factor_idx],
                    epsilon_value
                )
                if rc_t < self._matrix_root_inv_threshold:
                    should_update = False
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"Skpping update for {factor_matrix_index} : RC_t {rc_t:.4e} < {self._matrix_root_inv_threshold:.4e}")
                else:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"Updating {factor_matrix_index} : RC_t {rc_t:.4e} > {self._matrix_root_inv_threshold:.4e} ")
            except Exception as e:
                logger.warning(f"Failed to computed RC_t for {factor_matrix_index}, forcing update. Error:{e}")
                should_update = True
        if not should_update:
            return
        # Check for nan or inf values.
        if torch.isnan(bias_corrected_factor_matrix).any():
            raise ValueError(
                f"Encountered nan values in bias-corrected factor matrix {factor_matrix_index}! "
                f"To mitigate, check if nan inputs are being passed into the network or nan gradients "
                f"are being passed to the optimizer."
                f"For debugging purposes, factor_matrix {factor_matrix_index}: "
                f"{torch.min(factor_matrix)=}, {torch.max(factor_matrix)=}, "
                f"{factor_matrix.isinf().any()=}, {factor_matrix.isnan().any()=}."
            )
        if torch.isinf(bias_corrected_factor_matrix).any():
            raise ValueError(
                f"Encountered inf values in bias-corrected factor matrix {factor_matrix_index}! "
                f"In some cases, this may be due to divergence of the algorithm. "
                f"To mitigate, try decreasing the learning rate or increasing grafting epsilon."
                f"For debugging purposes, factor_matrix {factor_matrix_index}: "
                f"{torch.min(factor_matrix)=}, {torch.max(factor_matrix)=}, "
                f"{factor_matrix.isinf().any()=}, {factor_matrix.isnan().any()=}."
            )

        # Compute inverse preconditioner.
        # If reuse_previous_inv_factor_matrix is True, will reuse previous matrix if matrix
        # inverse root computation fails.
        try:
            result = matrix_inverse_root(
                A=bias_corrected_factor_matrix,
                root=root,
                epsilon=epsilon_value,
                use_adaptive_epsilon=self._use_adaptive_epsilon,
                condition_thresholds=self._condition_thresholds if self._use_adaptive_epsilon else None,
                thresholds_tensor=self._thresholds_tensor if self._use_adaptive_epsilon else None,
                epsilons_tensor=self._epsilons_tensor if self._use_adaptive_epsilon else None,
                small_positive_tensor=self._small_positive_tensor if self._use_adaptive_epsilon else None,
                exponent_multiplier=self._exponent_multiplier,
                is_diagonal=is_factor_matrix_diagonal,
                retry_double_precision=self._use_protected_eigh,
            )
            
            computed_inv_factor_matrix, used_epsilon_tensor, L, Q = result
            computed_inv_factor_matrix = computed_inv_factor_matrix.to(
                dtype=inv_factor_matrix.dtype
            )
            if L is not None and Q is not None:
                kronecker_factors.eigenvalues[factor_idx] = L.to(dtype = factor_matrix.dtype)
                kronecker_factors.eigenvectors[factor_idx] = Q.to(dtype = factor_matrix.dtype)
            
            

            if self._use_adaptive_epsilon and logger.isEnabledFor(logging.DEBUG):
                if isinstance(used_epsilon_tensor, torch.Tensor):
                    if abs(float(used_epsilon_tensor) - epsilon_value) > 1e-12:
                        logger.debug(
                            f"Factor matrix {factor_matrix_index}: "
                            f"Original epsilon = {epsilon_value:.2e}, "
                            f"Adjusted epsilon = {float(used_epsilon_tensor):.2e}"
                        )
            
            # Check if we encounter NaN or inf values in computed inverse matrix.
            if (
                torch.isnan(computed_inv_factor_matrix).any()
                or torch.isinf(computed_inv_factor_matrix).any()
            ):
                torch.set_printoptions(threshold=100_000)
                raise ValueError(
                    f"Encountered nan or inf values in inverse factor matrix {factor_matrix_index}! "
                    f"To mitigate, check factor matrix before matrix inverse root computation: "
                    f"{bias_corrected_factor_matrix=}"
                )
            computed_inv_factor_matrix = computed_inv_factor_matrix.to(dtype = inv_factor_matrix.dtype)
            inv_factor_matrix.copy_(computed_inv_factor_matrix)

        except Exception as exception:
            if (
                not self._use_protected_eigh
                or "Encountered nan or inf values in inverse factor matrix"
                in str(exception)
            ):
                raise exception
            else:
                logger.warning(
                    f"Matrix inverse root computation failed for factor matrix {factor_matrix_index} "
                    f"with exception {exception}. Using previous inv_factor_matrix and continuing..."
                )

    def compute_root_inverse(self) -> None:
        # NOTE: This function currently only computes the matrix root inverse based on
        # the masked lists which combines both selection based on the distributor and where
        # grad is not None. Implicitly, this assumes that there are no changes between the
        # selector or masking from iteration-to-iteration within a single precondition_frequency
        # interval.
        with profiler.record_function(
            f"## {self.__class__.__name__}:{self.compute_root_inverse.__name__} ##"
        ):
            # Lazy initialization of adaptive epsilon tensors
            if self._use_adaptive_epsilon and self._thresholds_tensor is None:
                self._initialize_adaptive_epsilon_tensors()
            
            # Optimized: Fast path for default configuration
            if self._is_default_config:
                for kronecker_factors, root in zip(
                    self._local_kronecker_factors_list,
                    self._local_root_list,
                ):
                    for (
                        factor_matrix,
                        inv_factor_matrix,
                        is_factor_matrix_diagonal,
                        factor_matrix_index,
                    ) in zip(
                        kronecker_factors.factor_matrices,
                        kronecker_factors.inv_factor_matrices,
                        kronecker_factors.is_factor_matrices_diagonal,
                        kronecker_factors.factor_matrix_indices,
                    ):
                        # Direct call with single epsilon
                        if is_factor_matrix_diagonal and not check_diagonal(factor_matrix):
                            is_factor_matrix_diagonal.copy_(torch.tensor(False))
                        
                        bias_corrected_factor_matrix = factor_matrix / self._bias_correction2
                        
                        # Fast path for default config
                        try:
                            computed_inv_factor_matrix = matrix_inverse_root_fast_default(
                                A=bias_corrected_factor_matrix,
                                root=root,
                                epsilon=self._epsilon,
                                exponent_multiplier=self._exponent_multiplier,
                                is_diagonal=is_factor_matrix_diagonal,
                                retry_double_precision=self._use_protected_eigh,
                            ).to(dtype=inv_factor_matrix.dtype)
                            
                            inv_factor_matrix.copy_(computed_inv_factor_matrix)
                        except Exception as exception:
                            if not self._use_protected_eigh:
                                raise exception
                            else:
                                logger.warning(
                                    f"Matrix inverse root computation failed for {factor_matrix_index}: {exception}"
                                )
                                
            # Optimized: Fast path for asymmetric non-adaptive configuration            
            elif self._is_asymmetric_non_adaptive:
                for kronecker_factors, root, epsilon_per_dim in zip(
                    self._local_kronecker_factors_list,
                    self._local_root_list,
                    self._local_epsilon_per_dim_list,
                ):
                    for idx, (
                        factor_matrix,
                        inv_factor_matrix,
                        is_factor_matrix_diagonal,
                        factor_matrix_index,
                        epsilon_for_this_dim,
                    ) in enumerate(zip(
                        kronecker_factors.factor_matrices,
                        kronecker_factors.inv_factor_matrices,
                        kronecker_factors.is_factor_matrices_diagonal,
                        kronecker_factors.factor_matrix_indices,
                        epsilon_per_dim,
                    )):
                        if is_factor_matrix_diagonal and not check_diagonal(factor_matrix):
                            is_factor_matrix_diagonal.copy_(torch.tensor(False))
                        
                        bias_corrected_factor_matrix = factor_matrix / self._bias_correction2
                        
                        # Fast path for asymmetric non-adaptive
                        try:
                            computed_inv_factor_matrix = matrix_inverse_root_fast_asymmetric(
                                A=bias_corrected_factor_matrix,
                                root=root,
                                epsilon=epsilon_for_this_dim,
                                exponent_multiplier=self._exponent_multiplier,
                                is_diagonal=is_factor_matrix_diagonal,
                                retry_double_precision=self._use_protected_eigh,
                            ).to(dtype=inv_factor_matrix.dtype)
                            
                            inv_factor_matrix.copy_(computed_inv_factor_matrix)
                        except Exception as exception:
                            if not self._use_protected_eigh:
                                raise exception
                            else:
                                logger.warning(
                                    f"Matrix inverse root computation failed for {factor_matrix_index}: {exception}"
                                )
                                
            # Original path for adaptive epsilon (unchanged)
            elif self._use_per_dim_epsilon:
                for kronecker_factors, root, epsilon_per_dim in zip(
                    self._local_kronecker_factors_list,
                    self._local_root_list,
                    self._local_epsilon_per_dim_list,
                ):
                    for (
                        i,
                        factor_matrix,
                        inv_factor_matrix,
                        is_factor_matrix_diagonal,
                        factor_matrix_index,
                        epsilon_for_this_dim,
                    ) in zip(
                        kronecker_factors.factor_matrices,
                        kronecker_factors.inv_factor_matrices,
                        kronecker_factors.is_factor_matrices_diagonal,
                        kronecker_factors.factor_matrix_indices,
                        epsilon_per_dim,
                    ):
                        self._compute_single_root_inverse(
                            factor_matrix,
                            inv_factor_matrix,
                            is_factor_matrix_diagonal,
                            factor_matrix_index,
                            kronecker_factors = kronecker_factors,
                            factor_idx = i,
                            root = root,
                            epsilon_for_this_dim = epsilon_for_this_dim,
                        )

    def compress_preconditioner_list(
        self, local_grad_selector: Tuple[bool, ...]
    ) -> None:
        with profiler.record_function(
            f"## {self.__class__.__name__}:{self.compress_preconditioner_list.__name__} ##"
        ):
            self._masked_order_list = compress_list(
                self._local_order_list, local_grad_selector
            )
            self._masked_root_list = compress_list(
                self._local_root_list, local_grad_selector
            )
            self._masked_kronecker_factors_list: Tuple[
                ShampooKroneckerFactors, ...
            ] = compress_list(self._local_kronecker_factors_list, local_grad_selector)
            
            if self._use_per_dim_epsilon:
                self._masked_epsilon_per_dim_list = compress_list(
                    self._local_epsilon_per_dim_list, local_grad_selector
                )

    def compute_root_inverse_residuals(
        self,
    ) -> Tuple[Tuple[Tensor, ...], Tuple[Tensor, ...]]:
        relative_errors = []
        relative_residuals = []

        if self._use_per_dim_epsilon:
            for kronecker_factors, root, epsilon_per_dim in zip(
                self._masked_kronecker_factors_list,
                self._masked_root_list,
                self._masked_epsilon_per_dim_list,
            ):
                for factor_matrix, inv_factor_matrix, epsilon_for_this_dim in zip(
                    kronecker_factors.factor_matrices,
                    kronecker_factors.inv_factor_matrices,
                    epsilon_per_dim,
                ):
                    bias_corrected_factor_matrix = factor_matrix / self._bias_correction2
                    (
                        relative_error,
                        relative_residual,
                    ) = compute_matrix_root_inverse_residuals(
                        bias_corrected_factor_matrix,
                        inv_factor_matrix,
                        root,
                        epsilon_for_this_dim,
                        self._exponent_multiplier,
                    )
                    relative_errors.append(relative_error)
                    relative_residuals.append(relative_residual)
        else:
            for kronecker_factors, root in zip(
                self._masked_kronecker_factors_list,
                self._masked_root_list,
            ):
                for factor_matrix, inv_factor_matrix in zip(
                    kronecker_factors.factor_matrices,
                    kronecker_factors.inv_factor_matrices,
                ):
                    bias_corrected_factor_matrix = factor_matrix / self._bias_correction2
                    (
                        relative_error,
                        relative_residual,
                    ) = compute_matrix_root_inverse_residuals(
                        bias_corrected_factor_matrix,
                        inv_factor_matrix,
                        root,
                        self._epsilon,
                        self._exponent_multiplier,
                    )
                    relative_errors.append(relative_error)
                    relative_residuals.append(relative_residual)

        return (
            tuple(relative_errors),
            tuple(relative_residuals),
        )
