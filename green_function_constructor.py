from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
from scipy.integrate import solve_ivp

Array = np.ndarray
RhsFunc = Callable[[float, Array], Array]


@dataclass
class BranchSolution:
    r: Array
    y: Array
    sign: str | None = None
    sol_index: int | None = None


@dataclass
class FundamentalMatrix:
    r: Array
    Y: Array
    Yp: Array
    metadata: dict[str, object] | None = None


def integrate_branch(
    rhs: RhsFunc,
    r_start: float,
    r_end: float,
    y0: Array,
    r_eval: Array,
    *,
    method: str = "Radau",
    rtol: float = 1e-7,
    atol: float = 1e-9,
) -> BranchSolution:
    """Integrate a first-order ODE branch on a fixed radial grid."""
    sol = solve_ivp(
        rhs,
        (r_start, r_end),
        np.asarray(y0, dtype=float),
        method=method,
        t_eval=np.asarray(r_eval, dtype=float),
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")
    return BranchSolution(r=sol.t, y=sol.y.T)


def coupled_wronskian_matrix(
    f_minus: Array,
    df_minus: Array,
    f_plus: Array,
    df_plus: Array,
) -> Array:
    """Return the coupled-channel Wronskian matrix profile.

    Inputs are arrays with shape (N_r, n_channels, n_basis).
    The returned array has shape (N_r, n_basis, n_basis).
    """
    f_minus = np.asarray(f_minus)
    df_minus = np.asarray(df_minus)
    f_plus = np.asarray(f_plus)
    df_plus = np.asarray(df_plus)

    if not (f_minus.shape == df_minus.shape == f_plus.shape == df_plus.shape):
        raise ValueError("All fundamental arrays must have the same shape.")
    if f_minus.ndim != 3:
        raise ValueError(
            "Fundamental arrays must have shape (N_r, n_channels, n_basis)."
        )

    return (
        np.einsum("kia,kib->kab", df_minus, f_plus)
        - np.einsum("kia,kib->kab", f_minus, df_plus)
    )


def weighted_wronskian_profile(
    f_minus: Array,
    df_minus: Array,
    f_plus: Array,
    df_plus: Array,
    r_grid: Array,
    *,
    weight_power: int = 3,
) -> Tuple[Array, Array]:
    """Build the raw and weighted Wronskian profiles on a radial grid."""
    r = np.asarray(r_grid, dtype=float)
    w_raw = coupled_wronskian_matrix(f_minus, df_minus, f_plus, df_plus)
    w_scaled = (r[:, None, None] ** weight_power) * w_raw
    return w_raw, w_scaled


def average_wronskian_plateau(
    w_scaled: Array,
    r_grid: Array,
    *,
    r_min_tail: float = 0.05,
    r_max_tail_fraction: float = 0.9,
) -> Tuple[Array, Array, Tuple[int, int]]:
    """Average the Wronskian over a plateau window and invert it."""
    r = np.asarray(r_grid, dtype=float)
    w_scaled = np.asarray(w_scaled)
    if w_scaled.shape[0] != len(r):
        raise ValueError("w_scaled and r_grid must have matching leading length.")
    if len(r) == 0:
        raise ValueError("r_grid must not be empty.")

    i_min = int(np.searchsorted(r, r_min_tail))
    i_max = int(np.searchsorted(r, r_max_tail_fraction * r[-1]))
    if i_min > i_max:
        raise ValueError("Invalid plateau window: i_min > i_max.")

    omega = np.mean(w_scaled[i_min:i_max + 1], axis=0)
    omega_inv = np.linalg.inv(omega)
    return omega, omega_inv, (i_min, i_max)


def diagonal_trace_from_fundamentals(
    f_plus: Array,
    f_minus: Array,
    omega_inv: Array,
) -> Array:
    """Contract the plus/minus bases with Omega^{-1} to get tr G(r,r)."""
    f_plus = np.asarray(f_plus)
    f_minus = np.asarray(f_minus)
    omega_inv = np.asarray(omega_inv)
    if f_plus.shape != f_minus.shape:
        raise ValueError("f_plus and f_minus must have the same shape.")
    if f_plus.ndim != 3:
        raise ValueError(
            "Fundamental arrays must have shape (N_r, n_channels, n_basis)."
        )
    return np.einsum("kia,ab,kib->k", f_plus, omega_inv, f_minus)


def wronskian_trace_bundle(
    f_minus: Array,
    df_minus: Array,
    f_plus: Array,
    df_plus: Array,
    r_grid: Array,
    *,
    weight_power: int = 3,
    r_min_tail: float = 0.05,
    r_max_tail_fraction: float = 0.9,
) -> Tuple[Array, Array, Array, Tuple[int, int]]:
    """Convenience helper returning the common Wronskian ingredients."""
    _, w_scaled = weighted_wronskian_profile(
        f_minus,
        df_minus,
        f_plus,
        df_plus,
        r_grid,
        weight_power=weight_power,
    )
    omega, omega_inv, tail_bounds = average_wronskian_plateau(
        w_scaled,
        r_grid,
        r_min_tail=r_min_tail,
        r_max_tail_fraction=r_max_tail_fraction,
    )
    trace_diag = diagonal_trace_from_fundamentals(f_plus, f_minus, omega_inv)
    return w_scaled, omega, trace_diag, tail_bounds


__all__ = [
    "Array",
    "BranchSolution",
    "FundamentalMatrix",
    "integrate_branch",
    "coupled_wronskian_matrix",
    "weighted_wronskian_profile",
    "average_wronskian_plateau",
    "diagonal_trace_from_fundamentals",
    "wronskian_trace_bundle",
]
