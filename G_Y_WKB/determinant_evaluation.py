import numpy as np
from scipy.integrate import solve_ivp


def regular_liouville_initial_conditions(n, r_min, n_fields=2):
    """
    Regular small-r boundary conditions for the Liouville-transformed
    O(4) radial equation.

    The regular solution behaves as

        Y(r) ~ r^(n + 3/2) I.

    Since an overall constant right-normalization of the fundamental matrix
    cancels between bounce and false-vacuum GY determinants, we rescale by
    r_min^(-(n+3/2)) and use the numerically better-conditioned equivalent:

        Y(r_min)  = I
        Y'(r_min) = (n + 3/2)/r_min * I.

    Bounce and FV must use exactly the same normalization.
    """
    power = n + 1.5

    Y_initial = np.eye(n_fields)
    Yp_initial = (power / r_min) * np.eye(n_fields)

    return Y_initial, Yp_initial


def solve_GY_matrix(mu, n, r_min, Rmax, V_matrix):
    """
    Solve

        (M_tilde_n + mu) Y = 0

    i.e.

        Y'' = [V_matrix(r) + mu I] Y,

    with regular Liouville-transformed initial conditions at r_min > 0.
    """
    n_fields = 2

    if r_min <= 0.0:
        raise ValueError("r_min must be > 0 for the Liouville-transformed equation.")
    if Rmax <= r_min:
        raise ValueError("Rmax must be larger than r_min.")

    def rhs(r, z):
        Y = z[:4].reshape(n_fields, n_fields)
        Yp = z[4:].reshape(n_fields, n_fields)

        Vmat = np.asarray(V_matrix(r), dtype=float)
        Ypp = (Vmat + mu * np.eye(n_fields)) @ Y

        return np.concatenate([Yp.ravel(), Ypp.ravel()])

    Y_initial, Yp_initial = regular_liouville_initial_conditions(
        n, r_min, n_fields=n_fields
    )

    z0 = np.concatenate([Y_initial.ravel(), Yp_initial.ravel()])

    sol = solve_ivp(
        rhs,
        (r_min, Rmax),
        z0,
        rtol=1e-10,
        atol=1e-12
    )

    if not sol.success:
        raise RuntimeError(f"GY ODE solve failed: {sol.message}")

    Y_R = sol.y[:4, -1].reshape(n_fields, n_fields)

    return Y_R


def raw_GY_ratio(mu, n, r_min, Rmax, V_matrix, V_matrix_fv):
    """
    Unprojected radial GY determinant ratio

        D_n(mu) =
            det Y_bounce(Rmax; mu) / det Y_FV(Rmax; mu).

    The common Liouville factor and common origin normalization cancel.
    """
    Y = solve_GY_matrix(mu, n, r_min, Rmax, V_matrix)
    Y_fv = solve_GY_matrix(mu, n, r_min, Rmax, V_matrix_fv)

    sign_Y, logabs_Y = np.linalg.slogdet(Y)
    sign_FV, logabs_FV = np.linalg.slogdet(Y_fv)

    if sign_Y == 0 or sign_FV == 0:
        raise FloatingPointError("Endpoint fundamental matrix is numerically singular.")

    sign_ratio = sign_Y * sign_FV
    return sign_ratio * np.exp(logabs_Y - logabs_FV)


def isolated_mode_factor(mu, isolated_eigenvalues):
    """
    Product of the radial eigenvalue factors being projected out:

        prod_m (lambda_m + mu).

    IMPORTANT:
    In a partial-wave calculation, list only distinct RADIAL eigenvalues in
    this n-sector.  For the O(4) translational sector n=1 there is one radial
    zero eigenvalue; its angular degeneracy d_1 = 4 is applied afterwards,
    not by putting four zeros in this list.
    """
    factor = 1.0

    for eigenvalue in isolated_eigenvalues:
        factor *= (eigenvalue + mu)

    return factor


def determinant_ratio(
    mu,
    n,
    r_min,
    Rmax,
    V_matrix,
    V_matrix_fv,
    isolated_eigenvalues=()
):
    """
    Radial GY determinant ratio with selected radial modes projected out:

        D_reduced(mu) =
            D_n(mu) / prod_m (lambda_m + mu).

    For the n=1 translational zero mode use isolated_eigenvalues=(0.0,).
    If the negative mode is left inside the determinant, do not list it.
    """
    D_mu = raw_GY_ratio(
        mu,
        n,
        r_min,
        Rmax,
        V_matrix,
        V_matrix_fv
    )

    mode_factor = isolated_mode_factor(mu, isolated_eigenvalues)

    return D_mu / mode_factor


def pick_mu_value(
    n,
    r_min,
    Rmax,
    V_matrix,
    V_matrix_fv,
    isolated_eigenvalues=(0.0,),
    mus=None,
    tol=1e-3
):
    """
    Estimate the mu -> 0 reduced determinant ratio by looking for a plateau.

    This is needed when zero modes have been divided out.  For a sector with
    no zero modes, simply evaluate determinant_ratio(mu=0, ...).
    """
    if mus is None:
        mus = np.logspace(-2, -9, 15)

    results = []

    for mu in mus:
        Q_mu = determinant_ratio(
            mu,
            n,
            r_min,
            Rmax,
            V_matrix,
            V_matrix_fv,
            isolated_eigenvalues
        )

        results.append((mu, Q_mu))

    for i in range(1, len(results)):
        mu_prev, Q_prev = results[i - 1]
        mu_curr, Q_curr = results[i]

        scale = max(abs(Q_curr), np.finfo(float).tiny)
        rel_change = abs(Q_curr - Q_prev) / scale

        print(
            f"mu={mu_curr:.2e}, "
            f"ratio={Q_curr:.12e}, "
            f"relative change={rel_change:.3e}"
        )

        if rel_change < tol:
            return mu_curr, Q_curr

    return None, None


def partial_wave_degeneracy(n):
    """O(4) scalar harmonic degeneracy d_n = (n+1)^2."""
    return (n + 1)**2
