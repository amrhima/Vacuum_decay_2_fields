import numpy as np
import scipy.sparse as sp
from scipy.interpolate import CubicSpline

# --------------------------------------------------------------------------- #
#   Liouville-transformed radial fluctuation operator
# --------------------------------------------------------------------------- #
#
# Original O(4) radial operator in partial wave n:
#
#   M_n = -d^2/dr^2 - (3/r)d/dr + n(n+2)/r^2 I + H(phi(r))
#
# With u_raw(r) = r^(-3/2) u_tilde(r):
#
#   M_tilde_n = -d^2/dr^2
#               + [n(n+2) + 3/4]/r^2 I
#               + H(phi(r)).
#
# For Gelfand-Yaglom, the object required by the ODE solver is NOT the full
# sparse differential operator.  It is the 2x2 matrix multiplying Y in
#
#   Y''(r) = [V_tilde_n(r) + mu I] Y(r),
#
# where
#
#   V_tilde_n(r) = [n(n+2) + 3/4]/r^2 I + H(phi(r)).
# --------------------------------------------------------------------------- #


def make_V_matrix(n, R_bounce, X_prime_bounce, Y_prime_bounce, pot_lin):
    """
    Return the 2x2 Liouville-transformed effective-potential function
    V_matrix(r) needed by the GY solver.

    Parameters
    ----------
    n : int
        O(4) partial-wave index.
    R_bounce, X_prime_bounce, Y_prime_bounce : array_like
        Bounce profile data.
    pot_lin :
        Potential object providing pot_lin.H(phi), the 2x2 Hessian.

    Returns
    -------
    V_matrix : callable
        Function of r returning

            [n(n+2)+3/4]/r^2 * I + H(phi_bounce(r)).
    """
    R_bounce = np.asarray(R_bounce, dtype=float)
    X_prime_bounce = np.asarray(X_prime_bounce, dtype=float)
    Y_prime_bounce = np.asarray(Y_prime_bounce, dtype=float)

    x_spline = CubicSpline(R_bounce, X_prime_bounce, bc_type="natural")
    y_spline = CubicSpline(R_bounce, Y_prime_bounce, bc_type="natural")

    centrifugal = n * (n + 2) + 0.75

    def V_matrix(r):
        if r <= 0.0:
            raise ValueError("Liouville-transformed V_matrix requires r > 0.")

        phi = np.array([x_spline(r), y_spline(r)], dtype=float)
        H = np.asarray(pot_lin.H(phi), dtype=float)

        return (centrifugal / r**2) * np.eye(2) + H

    return V_matrix


def make_V_matrix_fv(n, phi_fv, pot_lin):
    """
    Return the false-vacuum Liouville-transformed effective-potential function

        V_FV(r) = [n(n+2)+3/4]/r^2 * I + H(phi_FV).

    The false-vacuum Hessian is constant, but the centrifugal term remains.
    """
    phi_fv = np.asarray(phi_fv, dtype=float)
    H_fv = np.asarray(pot_lin.H(phi_fv), dtype=float)

    centrifugal = n * (n + 2) + 0.75

    def V_matrix_fv(r):
        if r <= 0.0:
            raise ValueError("Liouville-transformed V_matrix_fv requires r > 0.")

        return (centrifugal / r**2) * np.eye(2) + H_fv

    return V_matrix_fv


# --------------------------------------------------------------------------- #
# Optional sparse operator, useful for spectrum checks
# --------------------------------------------------------------------------- #

def M_tilde(n, R_bounce, X_prime_bounce, Y_prime_bounce, pot_lin,
            N=2000, r_min=1e-4, r_max=None, fd_order=2):
    """
    Build a sparse finite-difference representation of M_tilde_n.

    This routine is useful for independent spectrum checks.  It is NOT what
    should be passed to solve_GY_matrix(); for GY use make_V_matrix() above.

    The exact regular Liouville solution behaves as

        u_tilde(r) ~ r^(n + 3/2)

    at the origin.  A finite-difference eigenvalue problem on [r_min, r_max]
    with r_min > 0 only approximates the r=0 condition if one imposes
    Dirichlet at r_min.  For the GY initial-value problem the regular
    small-r behaviour is imposed explicitly (see determinant_evaluation.py).
    """
    R_bounce = np.asarray(R_bounce, dtype=float)
    X_prime_bounce = np.asarray(X_prime_bounce, dtype=float)
    Y_prime_bounce = np.asarray(Y_prime_bounce, dtype=float)

    if r_max is None:
        r_max = float(R_bounce[-1])

    r = np.linspace(r_min, r_max, N)
    dr = r[1] - r[0]

    x_spline = CubicSpline(R_bounce, X_prime_bounce, bc_type="natural")
    y_spline = CubicSpline(R_bounce, Y_prime_bounce, bc_type="natural")
    x_r = x_spline(r)
    y_r = y_spline(r)

    U_pp = np.zeros((N, 2, 2))
    for i in range(N):
        phi_i = np.array([x_r[i], y_r[i]])
        U_pp[i] = pot_lin.H(phi_i)

    if fd_order == 2:
        e = np.ones(N)
        L2 = sp.diags([e, -2 * e, e], [-1, 0, 1], shape=(N, N)) / dr**2
    else:
        raise ValueError("Only fd_order=2 supported in this prototype.")

    V_radial = (n * (n + 2) + 0.75) / r**2
    U11 = U_pp[:, 0, 0]
    U22 = U_pp[:, 1, 1]
    U12 = U_pp[:, 0, 1]

    M11 = -L2 + sp.diags(V_radial + U11, 0)
    M22 = -L2 + sp.diags(V_radial + U22, 0)
    M12 = sp.diags(U12, 0)

    M = sp.bmat([[M11, M12], [M12, M22]]).tocsr()

    # Approximate Dirichlet endpoint treatment for this finite-difference
    # spectrum-checking matrix.  The GY solver does NOT use these BCs.
    N2 = 2 * N
    boundary = [0, N - 1, N, 2 * N - 1]

    mask = np.ones(N2, dtype=float)
    mask[boundary] = 0.0
    D = sp.diags(mask, 0, format="csr")
    M = D @ M @ D

    diag_bc = np.zeros(N2)
    diag_bc[boundary] = 1.0
    M = M + sp.diags(diag_bc, 0, format="csr")

    return M.tocsr(), r, dr, U_pp
