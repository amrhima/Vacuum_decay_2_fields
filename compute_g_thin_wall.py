
from scipy.interpolate import CubicSpline
import scipy.sparse as sp
import numpy as np

def M_thin_wall(n, R_bounce, X_prime_bounce, Y_prime_bounce, pot_lin,
                        N=2000, r_min=1e-4, r_max=None):
    """
    Build the thin-wall operator

        M_n = -d^2/dz^2 + n(n+1)/R^2 + U''(phi(R+z)).

    ``R`` is taken to be the wall position, estimated from the largest
    gradient of the supplied bounce profile.  The returned grid is the
    wall-centred coordinate ``z = r - R``.

    """
    R_bounce = np.asarray(R_bounce, dtype=float)
    X_prime_bounce = np.asarray(X_prime_bounce, dtype=float)
    Y_prime_bounce = np.asarray(Y_prime_bounce, dtype=float)

    if r_max is None:
        r_max = float(R_bounce[-1])
        
    # r = np.linspace(r_min, r_max, N)
    r = float(R_bounce[np.argmax(np.hypot(dX_dr, dY_dr))])

    # Locate the wall and use its radius in the thin-wall centrifugal term.
    # The profile arrays are fields (despite the historical *_prime names),
    # so the wall is where their combined radial gradient is largest.
    dX_dr = np.gradient(X_prime_bounce, R_bounce)
    dY_dr = np.gradient(Y_prime_bounce, R_bounce)

    speed = np.sqrt(dX_dr**2 + dY_dr**2)
    i_wall = np.argmax(speed)

    R = r[i_wall]
    
    z = r - R
    dz = z[1] - z[0]

    x_spline = CubicSpline(R_bounce, X_prime_bounce, bc_type='natural')
    y_spline = CubicSpline(R_bounce, Y_prime_bounce, bc_type='natural')
    x_r = x_spline(r)
    y_r = y_spline(r)

    U_pp = np.zeros((N, 2, 2))
    for i in range(N):
        phi_prime_i = np.array([x_r[i], y_r[i]])
        U_pp[i] = pot_lin.H(phi_prime_i)

    e = np.ones(N)
    L2 = sp.diags([e, -2 * e, e], [-1, 0, 1], shape=(N, N)) / (dz**2)

    V_radial = (n * (n + 1)) / R**2
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

    return M_tilde.tocsr(), z, dz, U_pp
