import numpy as np
from scipy.integrate import simpson
from cosmoTransitions.helper_functions import deriv14
from G_Y_WKB.renormalize import wkb_sum_weights_4d


def build_W_profile(
    R_bounce,
    X_bounce,
    Y_bounce,
    phi_fv,
    pot,
):
    """
    Construct

        H_bounce(r) = d^2 U / dphi_a dphi_b |_{phi_b(r)}
        H_fv        = d^2 U / dphi_a dphi_b |_{phi_fv}
        W(r)        = H_bounce(r) - H_fv

    Shapes
    ------
    H_bounce : (N_r, 2, 2)
    H_fv     : (2, 2)
    W        : (N_r, 2, 2)
    """

    r = np.asarray(R_bounce, dtype=float)
    x = np.asarray(X_bounce, dtype=float)
    y = np.asarray(Y_bounce, dtype=float)
    phi_fv = np.asarray(phi_fv, dtype=float)

    if not (r.ndim == x.ndim == y.ndim == 1):
        raise ValueError(
            "R_bounce, X_bounce and Y_bounce must be one-dimensional."
        )

    if not (len(r) == len(x) == len(y)):
        raise ValueError(
            "R_bounce, X_bounce and Y_bounce must have equal lengths."
        )

    if phi_fv.shape != (2,):
        raise ValueError("phi_fv must have shape (2,).")

    if np.any(np.diff(r) <= 0.0):
        raise ValueError("R_bounce must be strictly increasing.")

    if r[0] < 0.0:
        raise ValueError("Radial coordinates must be non-negative.")

    phi_bounce = np.column_stack((x, y))

    # CosmoTransitions-style vectorized Hessian evaluation.
    H_bounce = np.asarray(pot.H(phi_bounce), dtype=float)
    H_fv = np.asarray(pot.H(phi_fv), dtype=float)

    if H_bounce.shape != (len(r), 2, 2):
        raise ValueError(
            "pot.H(phi_bounce) must return an array of shape "
            f"({len(r)}, 2, 2); got {H_bounce.shape}."
        )

    if H_fv.shape != (2, 2):
        raise ValueError(
            "pot.H(phi_fv) must return a (2, 2) matrix; "
            f"got {H_fv.shape}."
        )

    W_profile = H_bounce - H_fv[None, :, :]

    return W_profile, H_fv


def calculate_A1(R_bounce, W_profile):
    """
    A1 = (1/2) integral dr r Tr[W(r)].
    """

    r = np.asarray(R_bounce, dtype=float)
    W = np.asarray(W_profile, dtype=float)

    if W.shape != (len(r), 2, 2):
        raise ValueError(
            f"W_profile must have shape ({len(r)}, 2, 2)."
        )

    trace_W = np.trace(W, axis1=1, axis2=2)
    integrand = r * trace_W

    return 0.5 * simpson(integrand, x=r)


def calculate_A3(R_bounce, W_profile, H_fv):
    """
    A3 = -(1/8) integral dr r^3
         Tr[W(r)^2 + 2 H_fv W(r)].
    """

    r = np.asarray(R_bounce, dtype=float)
    W = np.asarray(W_profile, dtype=float)
    H_fv = np.asarray(H_fv, dtype=float)

    if W.shape != (len(r), 2, 2):
        raise ValueError(
            f"W_profile must have shape ({len(r)}, 2, 2)."
        )

    if H_fv.shape != (2, 2):
        raise ValueError("H_fv must have shape (2, 2).")

    # For every radial point:
    #
    # trace_W2[i] = Tr[W(r_i) @ W(r_i)]
    trace_W2 = np.einsum(
        "rij,rji->r",
        W,
        W,
    )

    # trace_Hfv_W[i] = Tr[H_fv @ W(r_i)]
    trace_Hfv_W = np.einsum(
        "ij,rji->r",
        H_fv,
        W,
    )

    integrand = r**3 * (
        trace_W2 + 2.0 * trace_Hfv_W
    )

    return -0.125 * simpson(integrand, x=r)


def estimate_wkb_tail(
    R_bounce,
    W_profile,
    H_fv,
    l_max,
    start_l=2,
):
    """Estimate the high-partial-wave tail using BubbleDet's WKB series.

    This reproduces BubbleDet's four-dimensional scalar WKB expansion through
    ``j**-9`` (``j = l + 1``).  It returns the per-wave WKB approximation and
    the analytically summed WKB contribution from ``start_l`` to infinity.

    The coefficients are the matrix generalisation of BubbleDet's scalar
    expansion.  Scalar products are promoted to matrix products under a
    trace; this is important because the field-space matrices need not
    commute.  The result therefore supports a genuinely coupled Hessian,
    while reducing exactly to the previous channel-by-channel expression when
    the insertion is diagonal.

    Parameters
    ----------
    R_bounce : array_like, shape (N,)
    W_profile : array_like, shape (N, n_fields, n_fields)
        ``H_bounce - H_false``.
    H_fv : array_like, shape (n_fields, n_fields)
        False-vacuum Hessian.
    l_max : int
        Number of WKB entries to return, indexed by ``l=0,...,l_max-1``.
    start_l : int, optional
        First partial wave included in the infinite WKB sum.

    Returns
    -------
    wkb_by_l : ndarray, shape (l_max,)
        WKB approximation to the logarithmic determinant ratio for each wave.
    wkb_tail : float
        ``1/2 * sum_{l=start_l}^infinity d_l * wkb_by_l`` in the
        BubbleDet/MS-bar convention.
    """
    r = np.asarray(R_bounce, dtype=float)
    W = np.asarray(W_profile, dtype=float)
    H_fv = np.asarray(H_fv, dtype=float)

    if r.ndim != 1 or W.ndim != 3 or H_fv.ndim != 2:
        raise ValueError("Invalid WKB input dimensions.")
    if W.shape[0] != len(r) or W.shape[1:] != H_fv.shape:
        raise ValueError("W_profile and H_fv shapes are inconsistent.")
    if H_fv.shape[0] != H_fv.shape[1]:
        raise ValueError("H_fv must be square.")
    if l_max < 1:
        raise ValueError("l_max must be positive.")
    if start_l < 0 or start_l > l_max:
        raise ValueError("start_l must satisfy 0 <= start_l <= l_max.")

    dim = H_fv.shape[0]
    l1_int = 0.0
    l3_int = 0.0
    l5_int = 0.0
    l7_int = 0.0
    l9_int = 0.0

    def mm(A, B):
        """Matrix product for every radial sample."""
        return np.einsum("rij,rjk->rik", A, B)

    def trace(A):
        return np.trace(A, axis1=1, axis2=2)

    def constant_profile(A):
        return np.broadcast_to(A, W.shape)

    H = constant_profile(H_fv)
    W2 = mm(W, W)
    W3 = mm(W2, W)
    W4 = mm(W3, W)
    H2 = mm(H, H)
    H3 = mm(H2, H)
    H4 = mm(H3, H)
    H5 = mm(H4, H)

    dW = np.empty_like(W)
    for i in range(dim):
        for j in range(dim):
            dW[:, i, j] = deriv14(W[:, i, j], r)
    d2W = np.empty_like(W)
    d3W = np.empty_like(W)
    for i in range(dim):
        for j in range(dim):
            d2W[:, i, j] = deriv14(dW[:, i, j], r)
            d3W[:, i, j] = deriv14(d2W[:, i, j], r)

    dW2 = mm(dW, dW)
    d2W2 = mm(d2W, d2W)
    d3W2 = mm(d3W, d3W)
    dW3 = mm(dW2, dW)
    q = r[:, None, None]

    # The first term at each order is the trace of the corresponding
    # difference of matrix powers.  Writing it this way automatically sums
    # all non-commuting orderings under the trace and agrees with the scalar
    # binomial polynomial in the diagonal limit.
    l1 = 0.5 * r * trace(W)
    l3 = -0.125 * r**3 * trace(W2 + 2.0 * mm(H, W))
    l5 = (
        1.0 / 16.0 * r**5 * trace(
            mm(H + W, mm(H + W, H + W)) - H3
        )
        + 1.0 / 32.0 * (
            r**5 * trace(dW2) - 4.0 * r**3 * trace(W2)
            - 8.0 * r**3 * trace(mm(H, W))
        )
    )
    l7 = (
        -5.0 / 128.0 * r**7 * trace(
            mm(H + W, mm(H + W, mm(H + W, H + W))) - H4
        )
        + 60.0 / 64.0 * r**5 * trace(mm(H2, W))
        + 1.0 / 64.0 * (
            20.0 * r**5 * trace(W3) - 5.0 * r**7 * trace(mm(W, dW2))
        )
        - 32.0 / 128.0 * r**3 * trace(mm(H, W))
        + 1.0 / 128.0 * (
            -r**7 * trace(d2W2) + 13.0 * r**5 * trace(dW2)
            - 16.0 * r**3 * trace(W2)
        )
    )
    l9 = (
        7.0 / 256.0 * r**9 * trace(mm(H + W, mm(H + W, mm(H + W, mm(H + W, H + W)))) - H5)
        - 560.0 / 256.0 * r**7 * trace(mm(H3, W))
        + 35.0 / 256.0 * trace(
            mm(H2, q**9 * dW2 - 24.0 * q**7 * W2)
        )
        + 70.0 / 256.0 * trace(
            mm(H, q**9 * mm(dW2, W) - 8.0 * q**7 * W3)
        )
        + 35.0 / 256.0 * trace(q**9 * mm(dW2, W2) - 4.0 * q**7 * W4)
        + 1008.0 / 256.0 * r**5 * trace(mm(H2, W))
        + 7.0 / 256.0 * trace(
            mm(H, q**9 * d2W2 - 31.0 * q**7 * dW2 + 144.0 * q**5 * W2)
        )
        + 7.0 / 256.0 * trace(
            -5.0 * q**8 * dW3
            + mm(W, q**9 * d2W2 - 31.0 * q**7 * dW2)
            + 48.0 * q**5 * W3
        )
        - 64.0 / 256.0 * r**3 * trace(mm(H, W))
        + 1.0 / 512.0 * trace(
            q**9 * d3W2 - 29.0 * q**7 * d2W2
            + 133.0 * q**5 * dW2 - 64.0 * q**3 * W2
        )
    )

    l1_int = simpson(l1, x=r)
    l3_int = simpson(l3, x=r)
    l5_int = simpson(l5, x=r)
    l7_int = simpson(l7, x=r)
    l9_int = simpson(l9, x=r)

    ell = np.arange(l_max, dtype=float)
    j = ell + 1.0
    wkb_by_l = (
        l1_int / j
        + l3_int / j**3
        + l5_int / j**5
        + l7_int / j**7
        + l9_int / j**9
    )

    # The regulator, MS-bar finite parts, and convergent zeta sums are
    # defined explicitly in renormalize.py. The determinant factor 1/2 is
    # applied here, after the angular sums.
    order_sums = wkb_sum_weights_4d(l_start=2)
    coeffs = np.array([l1_int, l3_int, l5_int, l7_int, l9_int])
    total_wkb = 0.5 * float(np.dot(coeffs, order_sums))

    if start_l > 2:
        total_wkb -= 0.5 * float(
            np.sum((ell[2:start_l] + 1.0) ** 2 * wkb_by_l[2:start_l])
        )
    elif start_l < 2:
        raise ValueError("BubbleDet's four-dimensional WKB tail starts at l=2.")

    return wkb_by_l, total_wkb
