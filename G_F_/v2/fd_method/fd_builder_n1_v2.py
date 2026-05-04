#!/usr/bin/env python3
"""
fd_builder_n1_ver2.py  --  Finite-difference + Feshbach-projector builder
for the n=1 (zero-mode) sector.

Specialised counterpart of fd_builder_ver2.py: contains only the
machinery used by compute_gbar_n1_fd_ver2.py.

Contents:
  * build_M_tilde_clean          -- discretise M_tilde_n on a uniform grid
  * find_discrete_zero_mode      -- eigenpair closest to lambda = 0
                                    (shift-invert at sigma=0; this is the
                                    discrete translation zero mode, slightly
                                    shifted from 0 by O(dr^2))
  * build_translation_zero_mode  -- analytic continuum translation mode
                                    chi propto r^(3/2) * dphi/dr
                                    (kept as a sanity-check reference)
  * hutchinson_trace_raw         -- Hutchinson trace of A^(-1)
  * hutchinson_trace_projected   -- Hutchinson trace of P A^(-1) P,
                                    with P = I - chi chi^T
  * gbar_raw_fd / gbar_sub_fd    -- per-s^2 wrappers (sparse LU)
  * load_bounce                  -- read a bounce .npz
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, splu
from scipy.interpolate import CubicSpline


# --------------------------------------------------------------------------- #
#   Operator construction                                                     #
# --------------------------------------------------------------------------- #

def build_M_tilde_clean(n, R_bounce, X_prime_bounce, Y_prime_bounce, pot_lin,
                        N=2000, r_min=1e-4, r_max=None, fd_order=2):
    """
    Build the Liouville-transformed radial fluctuation operator M_tilde_n
    as a sparse (2N x 2N) SYMMETRIC matrix.

    Liouville transformation: u_raw(r) = r^(-3/2) * u_tilde(r),
    equivalently u_tilde(r) = r^(3/2) * u_raw(r). This absorbs the r^3 radial
    measure into the function so the inner product is standard Euclidean,
    and the operator M becomes

        M_tilde_n = -d^2/dr^2 + [n(n+2) + 3/4] / r^2 + U''(phi(r))

    with no first-derivative term.

    Dirichlet boundary conditions u_tilde(r_min) = u_tilde(r_max) = 0 are
    applied for each of the two field components by:
      1. Zeroing BOTH rows and columns at boundary indices (preserves symmetry).
      2. Setting diagonal = 1 at boundary indices.
    This yields a symmetric matrix where any solve returns 0 at boundary
    positions (for rhs = 0 at boundaries).
    """
    R_bounce = np.asarray(R_bounce, dtype=float)
    X_prime_bounce = np.asarray(X_prime_bounce, dtype=float)
    Y_prime_bounce = np.asarray(Y_prime_bounce, dtype=float)

    if r_max is None:
        r_max = float(R_bounce[-1])

    r = np.linspace(r_min, r_max, N)
    dr = r[1] - r[0]

    x_spline = CubicSpline(R_bounce, X_prime_bounce, bc_type='natural')
    y_spline = CubicSpline(R_bounce, Y_prime_bounce, bc_type='natural')
    x_r = x_spline(r)
    y_r = y_spline(r)

    U_pp = np.zeros((N, 2, 2))
    for i in range(N):
        phi_prime_i = np.array([x_r[i], y_r[i]])
        U_pp[i] = pot_lin.H(phi_prime_i)

    if fd_order == 2:
        e = np.ones(N)
        L2 = sp.diags([e, -2 * e, e], [-1, 0, 1], shape=(N, N)) / (dr**2)
    else:
        raise ValueError("Only fd_order=2 supported in this prototype.")

    V_radial = (n * (n + 2) + 0.75) / r**2
    U11 = U_pp[:, 0, 0]
    U22 = U_pp[:, 1, 1]
    U12 = U_pp[:, 0, 1]

    M11 = -L2 + sp.diags(V_radial + U11, 0)
    M22 = -L2 + sp.diags(V_radial + U22, 0)
    M12 = sp.diags(U12, 0)

    M_tilde = sp.bmat([[M11, M12], [M12, M22]]).tocsr()

    # symmetric Dirichlet BC enforcement: D @ M @ D + I_boundary
    N2 = 2 * N
    boundary = [0, N - 1, N, 2 * N - 1]
    mask = np.ones(N2, dtype=float)
    for k in boundary:
        mask[k] = 0.0
    D = sp.diags(mask, 0, format='csr')
    M_tilde = D @ M_tilde @ D

    diag_bc = np.zeros(N2)
    for k in boundary:
        diag_bc[k] = 1.0
    M_tilde = M_tilde + sp.diags(diag_bc, 0, format='csr')

    return M_tilde.tocsr(), r, dr, U_pp


# --------------------------------------------------------------------------- #
#   Zero mode (n=1 only)                                                      #
# --------------------------------------------------------------------------- #

def find_discrete_zero_mode(M_tilde, verbose=True, n_eig=5,
                             use_shift_invert=True):
    """
    Return the eigenpair (lambda, chi) of M_tilde whose eigenvalue is
    CLOSEST TO ZERO in magnitude. Used for n=1 to project out the
    discrete zero mode of M_tilde (which is shifted from exactly 0
    by O(dr^2) plus the bounce's own EOM-violation error).

    Two implementation paths:

      use_shift_invert=True  (default, recommended)
        Calls eigsh with sigma=0 and which='LM' -- this is shift-invert
        mode, which finds the eigenpairs of M_tilde whose eigenvalues
        are nearest to sigma=0 (regardless of sign). Robust: it cannot
        miss the true closest-to-zero mode because it directly targets
        zero, not a most-negative bound.

      use_shift_invert=False
        Falls back to which='SA' (smallest algebraic), then picks the
        closest-to-zero among the n_eig returned. This was the previous
        behaviour; safe only if the closest-to-zero eigenvalue lies
        among the n_eig smallest algebraic ones (which is true for our
        F2/T0 bounce, but not guaranteed in general).
    """
    if use_shift_invert:
        # Shift-invert: directly target eigenvalues nearest sigma=0.
        # which='LM' on (M-sigma*I)^{-1} returns largest magnitude
        # eigenvalues of the inverse, i.e. those closest to sigma in M.
        try:
            eigvals, eigvecs = eigsh(M_tilde, k=n_eig, sigma=0.0,
                                      which='LM', tol=1e-9, maxiter=20000)
        except Exception as exc:
            print(f"[find_discrete_zero_mode] shift-invert failed "
                  f"({exc}); falling back to which='SA'.")
            eigvals, eigvecs = eigsh(M_tilde, k=n_eig, which='SA',
                                      tol=1e-9, maxiter=20000)
    else:
        eigvals, eigvecs = eigsh(M_tilde, k=n_eig, which='SA',
                                  tol=1e-9, maxiter=20000)

    # always pick the eigenvalue closest to zero among the returned set
    idx_min = int(np.argmin(np.abs(eigvals)))
    lambda_zm = float(eigvals[idx_min])
    chi_zm = eigvecs[:, idx_min]
    chi_zm = chi_zm / np.linalg.norm(chi_zm)

    if verbose:
        order = np.argsort(np.abs(eigvals))
        mode = "shift-invert sigma=0" if use_shift_invert else "SA"
        print(f"[find_discrete_zero_mode] {n_eig} eigenvalues "
              f"({mode}, sorted by |lambda|):")
        for rank, i in enumerate(order):
            tag = "  <-- picked (closest to 0)" if i == idx_min else ""
            print(f"    rank{rank}: lambda = {eigvals[i]:+.6e}{tag}")
    return lambda_zm, chi_zm


def build_translation_zero_mode(r, R_bounce, X_prime_bounce, Y_prime_bounce):
    """
    Build the analytic continuum translation zero mode in the Liouville basis:
        chi_zm(r) propto r^(3/2) * dphi/dr

    For the v2 pipeline we generally prefer find_discrete_zero_mode instead
    (since the discrete operator's near-zero eigenvector slightly differs
    from the continuum one), but this function is kept as a sanity-check
    reference: the overlap with the discrete chi_zm should be ~1.
    """
    x_spline = CubicSpline(R_bounce, X_prime_bounce, bc_type='natural')
    y_spline = CubicSpline(R_bounce, Y_prime_bounce, bc_type='natural')
    dx = x_spline(r, 1)
    dy = y_spline(r, 1)

    chi_x = r**1.5 * dx
    chi_y = r**1.5 * dy

    chi_zm = np.concatenate([chi_x, chi_y])
    chi_zm = chi_zm / np.linalg.norm(chi_zm)
    return chi_zm


# --------------------------------------------------------------------------- #
#   Hutchinson trace estimators                                               #
# --------------------------------------------------------------------------- #

def hutchinson_trace_raw(lu_factor_obj, N2, K=100, rng=None,
                          boundary_indices=None):
    """
    Estimate trace((M_tilde + s^2 I)^(-1)) via Hutchinson, given a sparse-LU
    object lu_factor_obj (with .solve(b) method).
    """
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")
    if rng is None:
        rng = np.random.default_rng()

    samples = np.empty(K)
    for k in range(K):
        v = rng.choice([-1.0, 1.0], size=N2)
        if boundary_indices is not None:
            v[boundary_indices] = 0.0
        x = lu_factor_obj.solve(v)
        samples[k] = v @ x

    mean = float(np.mean(samples))
    sem = float(np.std(samples, ddof=1) / np.sqrt(K)) if K > 1 else float('nan')
    return mean, sem


def hutchinson_trace_projected(lu_factor_obj, N2, chi, K=100, rng=None,
                                boundary_indices=None):
    """
    Estimate trace(P * A^(-1) * P) via Hutchinson, with sparse-LU lu_factor_obj.
    Here P = I - chi chi^T projects out the discrete zero-mode direction.
    """
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")
    if rng is None:
        rng = np.random.default_rng()

    samples = np.empty(K)
    for k in range(K):
        v = rng.choice([-1.0, 1.0], size=N2)
        if boundary_indices is not None:
            v[boundary_indices] = 0.0
        w = v - chi * (chi @ v)
        x = lu_factor_obj.solve(w)
        y = x - chi * (chi @ x)
        samples[k] = v @ y

    mean = float(np.mean(samples))
    sem = float(np.std(samples, ddof=1) / np.sqrt(K)) if K > 1 else float('nan')
    return mean, sem


# --------------------------------------------------------------------------- #
#   Top-level per-s^2 wrappers                                                #
# --------------------------------------------------------------------------- #

def gbar_raw_fd(M_tilde, s2, dr, N2, K=100, rng=None, boundary_indices=None):
    """
    Unsubtracted gbar(s^2) = trace((M_tilde + s^2 I)^(-1)) restricted to
    the interior DOFs. Uses sparse LU.
    """
    try:
        A = (M_tilde + s2 * sp.eye(N2, format='csr')).tocsc()
        lu = splu(A)
    except Exception:
        return np.nan, np.nan

    try:
        return hutchinson_trace_raw(lu, N2, K=K, rng=rng,
                                     boundary_indices=boundary_indices)
    except Exception:
        return np.nan, np.nan


def gbar_sub_fd(M_tilde, s2, dr, N2, chi, K=100, rng=None,
                boundary_indices=None):
    """
    Subtracted gbar(s^2) = trace(P (M_tilde + s^2 I)^(-1) P) restricted to
    the interior DOFs, with P = I - chi chi^T (chi = discrete zero mode).

    Sparse LU on the bare A = M_tilde + s^2 I. The s^2 grid in
    compute_gbar_n1_fd_ver2.py stays away from the discrete zero-mode pole
    by the skip_radius safeguard, so the LU is well-conditioned at every
    sampled s^2.

    Returns (gbar, sem) or (nan, nan).
    """
    try:
        A = (M_tilde + s2 * sp.eye(N2, format='csr')).tocsc()
        lu = splu(A)
    except Exception:
        return np.nan, np.nan
    try:
        return hutchinson_trace_projected(
            lu, N2, chi, K=K, rng=rng, boundary_indices=boundary_indices
        )
    except Exception:
        return np.nan, np.nan


# --------------------------------------------------------------------------- #
#   Bounce-file loading helper                                                #
# --------------------------------------------------------------------------- #

def load_bounce(bounce_path):
    """
    Load a bounce .npz produced by bounce.py.
    """
    d = np.load(bounce_path, allow_pickle=True)
    return {
        "params":          d["params"],
        "false_vac":       np.asarray(d["false_vac"], dtype=float),
        "R":               d["R"],
        "X_prime":         d["X_bounce_prime"],
        "Y_prime":         d["Y_bounce_prime"],
        "false_index":     int(d["false_index"]),
        "true_index":      int(d["true_index"]),
    }
