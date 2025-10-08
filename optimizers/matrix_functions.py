"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

"""

import enum
import logging
from typing import Tuple, Union, Optional, Dict

import torch
from torch import Tensor

logger: logging.Logger = logging.getLogger(__name__)


class NewtonConvergenceFlag(enum.Enum):
    REACHED_MAX_ITERS = 0
    CONVERGED = 1


class RootInvMethod(enum.Enum):
    EIGEN = 0
    NEWTON = 1


def check_diagonal(A: Tensor) -> bool:
    """Checks if symmetric matrix is diagonal. Throw if the input is not a square matrix."""

    A_shape = A.shape
    if len(A_shape) != 2:
        raise ValueError(f"Matrix is not 2-dimensional! {A_shape=}")

    if A_shape[0] != A_shape[1]:
        raise ValueError(f"Matrix is not square! {A_shape=}")

    # Check both upper triangular part and lower triangular part are all zeros.
    return not A.triu(diagonal=1).any() and not A.tril(diagonal=-1).any()

def compute_condition_based_epsilon(
        eigenvalues : Tensor,
        base_epsilon : float,
        condition_thresholds : Optional[Dict[float, float,]] = None
) -> float:
    """
    조건수에 기반하여 epsilon을 적응적으로 조정함.

    condition_thresholds : 조건수 임계값과 epsilon 매핑 
    기본값 : {1e6 : 1e-05, 1e8 : 5e-5}

    조정된 epsilon 값 반환.
    """
    if condition_thresholds is None:
        condition_thresholds = {
            1e6 : 1e-5,
            1e8: 5e-5,
        }
    max_eigenvalue = torch.max(eigenvalues)
    min_eigenvalue = torch.min(eigenvalues)

    if min_eigenvalue <= 0:
        min_eigenvalue = torch.min(eigenvalues[eigenvalues > 0]) if torch.any(eigenvalues > 0) else 1e-09

    condition_number = max_eigenvalue/min_eigenvalue

    adjusted_epsilon = base_epsilon
    for threshold, epsilon_value in sorted(condition_thresholds.items()):
        if condition_number >= threshold:
            adjusted_epsilon = epsilon_value
        else:
            break
    return adjusted_epsilon, condition_number.item()


def compute_condition_based_epsilon_gpu(
        eigenvalues : Tensor,
        base_epsilon : float,
        thresholds_tensor : Tensor,
        epsilons_tensor : Tensor,
        small_positive_tensor : Optional[Tensor] = None,
) -> Tensor:
    """
    조건수에 기반하여 epsilon을 적응적으로 조정함.

    condition_thresholds : 조건수 임계값과 epsilon 매핑 
    기본값 : {1e6 : 1e-05, 1e8 : 5e-5}

    조정된 epsilon 값 반환.
    """
    if small_positive_tensor is None:
        small_positive_tensor = torch.tensor(1e-8, device = eigenvalues.device)

    eigenvalues_safe = torch.where(
        eigenvalues >0,
        eigenvalues,
        small_positive_tensor
    )
    max_eig = torch.max(eigenvalues_safe)
    min_eig = torch.min(eigenvalues_safe)
    condition_number = max_eig / min_eig
    
    compare_result = condition_number >= thresholds_tensor
    if compare_result.any():
        indices = torch.arange(len(thresholds_tensor), device = thresholds_tensor.device)
        valid_indices = torch.where(compare_result, indices, -1)
        max_valid_idx = valid_indices.max()
        adjusted_epsilon = epsilons_tensor[max_valid_idx]
    else:
        adjusted_epsilon = torch.tensor(base_epsilon, device= eigenvalues.device)

    return adjusted_epsilon

def matrix_inverse_root(
    A: Tensor,
    root: int,
    epsilon: float = 0.0,
    exponent_multiplier: float = 1.0,
    root_inv_method: RootInvMethod = RootInvMethod.EIGEN,
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
    is_diagonal: Union[Tensor, bool] = False,
    retry_double_precision: bool = True,
    use_adaptive_epsilon: bool = False,
    condition_thresholds : Optional[Dict[float, float]] = None,
    thresholds_tensor: Optional[Tensor] = None,
    epsilons_tensor : Optional[Tensor] = None,
    small_positive_tensor: Optional[Tensor] = None,
) -> Tuple[Tensor, Union[float, Tensor]]:
    """Computes matrix root inverse of square symmetric positive definite matrix.

    Args:
        A (Tensor): Square matrix of interest.
        root (int): Root of interest. Any natural number.
        epsilon (float): Adds epsilon * I to matrix before taking matrix root. (Default: 0.0)
        exponent_multiplier (float): exponent multiplier in the eigen method (Default: 1.0)
        root_inv_method (RootInvMethod): Specifies method to use to compute root inverse. (Default: RootInvMethod.EIGEN)
        max_iterations (int): Maximum number of iterations for coupled Newton iteration. (Default: 1000)
        tolerance (float): Tolerance for computing root inverse using coupled Newton iteration. (Default: 1e-6)
        is_diagonal (Tensor, bool): Flag for whether or not matrix is diagonal. If so, will compute root inverse by computing
            root inverse of diagonal entries. (Default: False)
        retry_double_precision (bool): Flag for re-trying eigendecomposition with higher precision if lower precision fails due
            to CuSOLVER failure. (Default: True)
        use_adaptive_epsilon : 조건수 기반 epsilon 조정 사용 여부
        condition_thresholds : 조건수 임계값과 epsilon 매핑

    Returns:
        X (Tensor): Inverse root of matrix A.
        used_epsilon : 실제 사용된 epsilon 값.

    """

    # check if matrix is scalar
    if torch.numel(A) == 1:
        alpha = torch.as_tensor(-exponent_multiplier / root)
        return (A + epsilon) ** alpha, epsilon

    # check matrix shape
    if len(A.shape) != 2:
        raise ValueError("Matrix is not 2-dimensional!")
    elif A.shape[0] != A.shape[1]:
        raise ValueError("Matrix is not square!")
    
    used_epsilon = epsilon

    if is_diagonal:
        X = matrix_root_diagonal(
            A=A,
            root=root,
            epsilon=used_epsilon,
            inverse=True,
            exponent_multiplier=exponent_multiplier,
            return_full_matrix=True,
        )
    elif root_inv_method == RootInvMethod.EIGEN:
        X, _, _, used_epsilon, _ = _matrix_root_eigen(
            A=A,
            root=root,
            epsilon=epsilon,
            inverse=True,
            exponent_multiplier=exponent_multiplier,
            retry_double_precision=retry_double_precision,
            use_adaptive_epsilon = use_adaptive_epsilon,
            condition_thresholds = condition_thresholds,
            thresholds_tensor = thresholds_tensor,
            epsilons_tensor = epsilons_tensor,
            small_positive_tensor = small_positive_tensor,
        )
    elif root_inv_method == RootInvMethod.NEWTON:
        # Newton method는 adaptive epsilon 미지원.
        if exponent_multiplier != 1.0:
            raise ValueError(
                f"Exponent multiplier {exponent_multiplier} must be equal to 1 to use coupled inverse Newton iteration!"
            )

        X, _, termination_flag, _, _ = _matrix_inverse_root_newton(
            A=A,
            root=root,
            epsilon=epsilon,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
        if termination_flag == NewtonConvergenceFlag.REACHED_MAX_ITERS:
            logging.warning(
                "Newton did not converge and reached maximum number of iterations!"
            )
    else:
        raise NotImplementedError(
            f"Root inverse method is not implemented! Specified root inverse method is {str(root_inv_method)}."
        )

    return X, used_epsilon


def matrix_root_diagonal(
    A: Tensor,
    root: int,
    epsilon: float = 0.0,
    inverse: bool = True,
    exponent_multiplier: float = 1.0,
    return_full_matrix: bool = False,
) -> Tensor:
    """Computes matrix inverse root for a diagonal matrix by taking inverse square root of diagonal entries.

    Args:
        A (Tensor): One- or two-dimensional tensor containing either the diagonal entries of the matrix or a diagonal matrix.
        root (int): Root of interest. Any natural number.
        epsilon (float): Adds epsilon * I to matrix before taking matrix root. (Default: 0.0)
        inverse (bool): Returns inverse root matrix. (Default: True)
        return_full_matrix (bool): Returns full matrix by taking torch.diag of diagonal entries. (bool: False)

    Returns:
        X (Tensor): Inverse root of diagonal entries.

    """

    # check order of tensor
    order = len(A.shape)
    if order == 2:
        A = torch.diag(A)
    elif order > 2:
        raise ValueError("Matrix is not 2-dimensional!")

    # check if root is positive integer
    if root <= 0:
        raise ValueError(f"Root {root} should be positive!")

    # compute matrix power
    alpha = exponent_multiplier / root
    if inverse:
        alpha = -alpha

    X = (A + epsilon).pow(alpha)
    return torch.diag(X) if return_full_matrix else X


def _matrix_root_eigen(
    A: Tensor,
    root: int,
    epsilon: float = 0.0,
    inverse: bool = True,
    exponent_multiplier: float = 1.0,
    make_positive_semidefinite: bool = True,
    retry_double_precision: bool = True,
    use_adaptive_epsilon: bool = False,
    condition_thresholds: Optional[Dict[float, float]] = None,
    thresholds_tensor: Optional[Tensor] = None,
    epsilons_tensor: Optional[Tensor] = None,
    small_positive_tensor = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: 
    """
    Returns:
        X: (Inverse) root of matrix
        L: Eigenvalues
        Q: Eigenvectors
        used_epsilon: 실제 사용된 epsilon (Tensor)
        condition_number: 조건수 (Tensor) 
    """

    # Check if root is positive integer
    if root <= 0:
        raise ValueError(f"Root {root} should be positive!")

    # Compute matrix power
    alpha = exponent_multiplier / root
    if inverse:
        alpha = -alpha

    # Compute eigendecomposition
    try:
        L, Q = torch.linalg.eigh(A)
    except Exception as exception:
        if retry_double_precision and A.dtype != torch.float64:
            logger.warning(
                f"Failed to compute eigendecomposition in {A.dtype} precision with exception {exception}! "
                f"Retrying in double precision..."
            )
            L, Q = torch.linalg.eigh(A.double())
        else:
            raise exception

    lambda_min = torch.min(L)

    # Make eigenvalues >= 0 (if necessary)
    if make_positive_semidefinite:
        L += -torch.minimum(lambda_min, torch.as_tensor(0.0, device=L.device))

    #  Condition number 계산 (GPU에서)
    condition_number_tensor = torch.tensor(float('inf'), device=L.device)
    if L.numel() > 1:
        L_safe = torch.where(L > 0, L, torch.tensor(1e-9, device=L.device))
        max_eig = torch.max(L_safe)
        min_eig = torch.min(L_safe)
        condition_number_tensor = max_eig / min_eig  # GPU Tensor

    #  Adaptive epsilon (GPU only)
    if use_adaptive_epsilon and torch.numel(L) > 1:
        if thresholds_tensor is not None and epsilons_tensor is not None:
            # GPU Tensor 사용 (동기화 없음)
            used_epsilon_tensor = compute_condition_based_epsilon_gpu(
                L, epsilon, thresholds_tensor, epsilons_tensor, small_positive_tensor
            )
        else:
            # Fallback
            logger.warning("thresholds_tensor or epsilons_tensor is None, using legacy method")
            adjusted_epsilon, _ = compute_condition_based_epsilon(
                L, epsilon, condition_thresholds
            )
            used_epsilon_tensor = torch.tensor(adjusted_epsilon, device=L.device)
    else:
        used_epsilon_tensor = torch.tensor(epsilon, device=L.device)

    # Add epsilon (GPU only)
    L += used_epsilon_tensor

    # Compute inverse preconditioner
    X = Q * L.pow(alpha).unsqueeze(0) @ Q.T

    return X, L, Q, used_epsilon_tensor, condition_number_tensor  


def _matrix_inverse_root_newton(
    A: Tensor,
    root: int,
    epsilon: float = 0.0,
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
) -> Tuple[Tensor, Tensor, NewtonConvergenceFlag, int, Tensor]:
    """Compute matrix inverse root using coupled inverse Newton iteration.

        alpha <- -1 / p
        X <- 1/c * I
        M <- 1/c^p * A
        repeat until convergence
            M' <- (1 - alpha) * I + alpha * M
            X <- X * M'
            M <- M'^p * M

    where c = (2 |A|_F / (p + 1))^{1/p}. This ensures that |A|_2 <= |A|_F < (p + 1) c^p, which guarantees convergence.
    We will instead use z = (p + 1) / (2 * |A|_F).

    NOTE: Exponent multiplier not compatible with coupled inverse Newton iteration!

    Args:
        A (Tensor): Matrix of interest.
        root (int): Root of interest. Any natural number.
        epsilon (float): Adds epsilon * I to matrix before taking matrix root. (Default: 0.0)
        max_iterations (int): Maximum number of iterations. (Default: 1000)
        tolerance (float): Tolerance. (Default: 1e-6)

    Returns:
        A_root (Tensor): Inverse square root of matrix.
        M (Tensor): Coupled matrix.
        termination_flag (NewtonConvergenceFlag): Specifies convergence.
        iteration (int): Number of iterations.
        error (Tensor): Final error between M and I.

    """

    # initialize iteration, dimension, and alpha
    iteration = 0
    dim = A.shape[0]
    alpha = -1 / root
    identity = torch.eye(dim, dtype=A.dtype, device=A.device)

    # add regularization
    A.add_(identity, alpha=epsilon)

    # initialize matrices
    A_nrm = torch.linalg.norm(A)
    z = (root + 1) / (2 * A_nrm)
    X = z ** (-alpha) * identity
    M = z * A
    error = torch.dist(M, identity, p=torch.inf)

    # main for loop
    while error > tolerance and iteration < max_iterations:
        iteration += 1
        M_p = M.mul(alpha).add_(identity, alpha=(1 - alpha))
        X = X @ M_p
        M = torch.linalg.matrix_power(M_p, root) @ M
        error = torch.dist(M, identity, p=torch.inf)

    # determine convergence flag
    termination_flag = (
        NewtonConvergenceFlag.CONVERGED
        if error <= tolerance
        else NewtonConvergenceFlag.REACHED_MAX_ITERS
    )

    return X, M, termination_flag, iteration, error


def compute_matrix_root_inverse_residuals(
    A: Tensor,
    X_hat: Tensor,
    root: int,
    epsilon: float,
    exponent_multiplier: float,
) -> Tuple[Tensor, Tensor]:
    """Compute residual of matrix root inverse for debugging purposes.

        relative error    = ||X - X_hat||_inf / ||X||_inf
        relative residual = ||A X^r - I||_inf

    Args:
        A (Tensor): Matrix of interest.
        X (Tensor): Computed matrix root inverse.
        root (int): Root of interest.
        epsilon (float): Adds epsilon * I to matrix.
        exponent_multiplier (float): Exponent multiplier to be multiplied to the numerator of the inverse root.

    Returns:
        absolute_error (Tensor): absolute error of matrix root inverse
        relative_error (Tensor): relative error of matrix root inverse
        residual (Tensor): residual of matrix root inverse

    """

    # check shape of matrix
    if len(A.shape) != 2:
        raise ValueError("Matrix is not 2-dimensional!")
    elif A.shape[0] != A.shape[1]:
        raise ValueError("Matrix is not square!")
    elif A.shape != X_hat.shape:
        raise ValueError("Matrix shapes do not match!")

    # compute error by comparing against double precision
    X = matrix_inverse_root(
        A.double(), root, epsilon=epsilon, exponent_multiplier=exponent_multiplier
    )
    relative_error = torch.dist(X, X_hat, p=torch.inf) / torch.norm(X, p=torch.inf)

    # compute residual
    if exponent_multiplier == 1.0:
        X_invr = torch.linalg.matrix_power(X_hat.double(), n=-root)
    else:
        X_invr, _, _ = _matrix_root_eigen(
            X_hat.double(),
            root=1,
            epsilon=0.0,
            inverse=True,
            make_positive_semidefinite=True,
            exponent_multiplier=root / exponent_multiplier,
        )

    A_reg = A.double() + epsilon * torch.eye(
        A.shape[0], dtype=torch.float64, device=A.device
    )
    relative_residual = torch.dist(X_invr, A_reg, p=torch.inf) / torch.norm(
        A_reg, p=torch.inf
    )

    return relative_error, relative_residual
