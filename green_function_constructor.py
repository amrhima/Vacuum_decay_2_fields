import numpy as np
from utils import FundamentalMatrix, _rk4_step
from __future__ import annotations
from typing import Callable, Tuple, Optional

Array = np.ndarray

def build_left_fundamental_O4(
    r_grid: Array,
    l: int,
    Upp_of_r: Callable[[float], Array],
) -> FundamentalMatrix:
    """
    Build Y_L, Y_L' for the coupled 2-field O(4) radial operator in partial wave l:
        y'' + (3/r) y' - [l(l+2)/r^2 + U''(r)] y = 0
    where y is a 2-vector and U''(r) is a 2x2 matrix.
    """
    r = np.asarray(r_grid, dtype=float)
    if np.any(np.diff(r) <= 0):
        raise ValueError("r_grid must be strictly increasing.")
    if r[0] <= 0:
        raise ValueError("r_grid[0] must be > 0 (start at small r_min).")

    def ode(rr: float, z: Array) -> Array:
        Y, Yp = _unpack(z)
        Upp = np.asarray(Upp_of_r(rr), dtype=float)
        if Upp.shape != (2, 2):
            raise ValueError("Upp_of_r(r) must return a 2x2 matrix.")

        # Y'' = - (3/r) Y' + [l(l+2)/r^2 + U''] Y
        cent = (l * (l + 2)) / (rr * rr)
        Ypp = -(3.0 / rr) * Yp + (cent * np.eye(2) + Upp) @ Y
        return _pack(Yp, Ypp)

    # initial conditions near r=0: Y ~ r^l I, Y' ~ l r^(l-1) I
    r0 = r[0]
    Y0 = (r0 ** l) * np.eye(2)
    Yp0 = (l * (r0 ** (l - 1))) * np.eye(2) if l > 0 else np.zeros((2, 2))

    z = _pack(Y0, Yp0)
    Y_out = np.zeros((len(r), 2, 2))
    Yp_out = np.zeros((len(r), 2, 2))
    Y_out[0], Yp_out[0] = Y0, Yp0

    for i in range(len(r) - 1):
        h = r[i + 1] - r[i]
        z = _rk4_step(ode, r[i], z, h)
        Yi, Ypi = _unpack(z)
        Y_out[i + 1], Yp_out[i + 1] = Yi, Ypi

    return FundamentalMatrix(r=r, Y=Y_out, Yp=Yp_out, l=l)

def build_right_fundamental_O4(
    r_grid: Array,
    l: int,
    Upp_of_r: Callable[[float], Array],
    Upp_infty: Array,
) -> FundamentalMatrix:
    """
    Build Y_R, Y_R' whose columns are independent solutions decaying at large r.

    We impose asymptotic decaying initial conditions at r_max using Upp_infty.
    Then integrate backwards along r_grid.
    
    We set the 1/r^2 term to be 0 as r goes to infinity (The boundary condition is simply y'' = mu_i^2 y).
    """
    r = np.asarray(r_grid, dtype=float)
    if np.any(np.diff(r) <= 0):
        raise ValueError("r_grid must be strictly increasing.")
    if r[0] <= 0:
        raise ValueError("r_grid[0] must be > 0.")

    Upp_inf = np.asarray(Upp_infty, dtype=float)
    if Upp_inf.shape != (2, 2):
        raise ValueError("Upp_infty must be a 2x2 matrix.")

    # Eigen-decompose Upp_infty to set decay scales
    evals, Q = np.linalg.eig(Upp_inf)
    # We expect evals ~ mu_i^2. Enforce positive real parts for decays.
    mu = np.sqrt(evals.astype(complex))
    # choose branch with Re(mu) >= 0
    mu = np.where(np.real(mu) < 0, -mu, mu)

    # Build initial Y, Y' at r_max using decaying modes along eigenvectors
    rmax = r[-1]

    # columns in eigenbasis: y_i = exp(-mu_i r), y_i' = -mu_i exp(-mu_i r)
    Y_eig = np.diag(np.exp(-mu * rmax))
    Yp_eig = np.diag(-mu * np.exp(-mu * rmax))

    # transform back to field basis
    # Y = Q Y_eig, Y' = Q Yp_eig
    Y0 = (Q @ Y_eig).astype(complex)
    Yp0 = (Q @ Yp_eig).astype(complex)

    # If everything is real, cast back; otherwise keep complex (still fine numerically).
    if np.max(np.abs(np.imag(Y0))) < 1e-12 and np.max(np.abs(np.imag(Yp0))) < 1e-12:
        Y0 = np.real(Y0)
        Yp0 = np.real(Yp0)

    def ode(rr: float, z: Array) -> Array:
        Y, Yp = _unpack(z)
        Upp = np.asarray(Upp_of_r(rr), dtype=float)
        if Upp.shape != (2, 2):
            raise ValueError("Upp_of_r(r) must return a 2x2 matrix.")

        cent = (l * (l + 2)) / (rr * rr)
        Ypp = -(3.0 / rr) * Yp + (cent * np.eye(2) + Upp) @ Y
        return _pack(Yp, Ypp)

    # integrate backward: step with negative h
    z = _pack(np.asarray(Y0), np.asarray(Yp0))
    Y_out = np.zeros((len(r), 2, 2), dtype=z.dtype)
    Yp_out = np.zeros((len(r), 2, 2), dtype=z.dtype)
    Y_out[-1], Yp_out[-1] = Y0, Yp0

    for i in range(len(r) - 1, 0, -1):
        h = r[i - 1] - r[i]  # negative
        z = _rk4_step(ode, r[i], z, h)
        Yi, Ypi = _unpack(z)
        Y_out[i - 1], Yp_out[i - 1] = Yi, Ypi

    return FundamentalMatrix(r=r, Y=Y_out, Yp=Yp_out, l=l)

def wronskian_matrix_O4(
    r: float,
    Y_left: Array,
    Yp_left: Array,
    Y_right: Array,
    Yp_right: Array,
    *,
    weight_power: int = 3,
    transpose: bool = True,
) -> Array:
    YL = np.asarray(Y_left)
    YLp = np.asarray(Yp_left)
    YR = np.asarray(Y_right)
    YRp = np.asarray(Yp_right)

    if YL.shape != YR.shape or YL.shape != YLp.shape or YR.shape != YRp.shape:
        raise ValueError("All Y and Y' inputs must have the same shape (N,N).")
    if YL.ndim != 2 or YL.shape[0] != YL.shape[1]:
        raise ValueError("Y matrices must be square (N,N).")

    p = r ** weight_power

    if transpose:
        return p * (YL.T @ YRp - YLp.T @ YR)
    else:
        return p * (YL @ YRp - YLp @ YR)

def wronskian_profile_O4(
    fundL, fundR,
    *,
    weight_power: int = 3,
    transpose: bool = True,
    return_norms: bool = False,
) -> Tuple[Array, Optional[Array]]:
    r = fundL.r
    if not np.allclose(r, fundR.r):
        raise ValueError("fundL.r and fundR.r must match.")

    Npts = len(r)
    N = fundL.Y.shape[1]
    W_all = np.zeros((Npts, N, N), dtype=np.result_type(fundL.Y, fundR.Y))

    for i in range(Npts):
        W_all[i] = wronskian_matrix_O4(
            r[i],
            fundL.Y[i], fundL.Yp[i],
            fundR.Y[i], fundR.Yp[i],
            weight_power=weight_power,
            transpose=transpose,
        )

    if not return_norms:
        return W_all, None

    W0 = W_all[0]
    norms = np.array([np.linalg.norm(W_all[i] - W0, ord="fro") for i in range(Npts)])
    return W_all, norms

def _pack(Y: Array, Yp: Array) -> Array:
    return np.concatenate([Y.reshape(-1), Yp.reshape(-1)])

def _unpack(z: Array) -> Tuple[Array, Array]:
    Y = z[:4].reshape(2, 2)
    Yp = z[4:].reshape(2, 2)
    return Y, Yp