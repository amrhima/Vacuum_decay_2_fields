import numpy as np
from scipy.integrate import simpson
from cosmoTransitions.helper_functions import deriv14


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


def estimate_bubbledet_wkb_tail(
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

    The current implementation is intended for BubbleDet-compatible,
    decoupled backgrounds.  For a matrix background, each diagonal channel is
    treated as an independent scalar.  Nonzero off-diagonal insertions are
    rejected because BubbleDet's scalar WKB formula does not apply to them.

    Parameters
    ----------
    R_bounce : array_like, shape (N,)
    W_profile : array_like, shape (N, n, n)
        ``H_bounce - H_false``.
    H_fv : array_like, shape (n, n)
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

    offdiag = W.copy()
    offdiag[:, range(H_fv.shape[0]), range(H_fv.shape[0])] = 0.0
    if np.max(np.abs(offdiag)) > 1e-10:
        raise ValueError(
            "BubbleDet-style scalar WKB tail requires a diagonal Hessian."
        )

    dim = H_fv.shape[0]
    l1_int = 0.0
    l3_int = 0.0
    l5_int = 0.0
    l7_int = 0.0
    l9_int = 0.0

    for a in range(dim):
        dw = W[:, a, a]
        m0_sq = H_fv[a, a]
        dwr = deriv14(dw, r)
        d2wr = deriv14(dwr, r)
        d3wr = deriv14(d2wr, r)

        l1 = 0.5 * r * dw
        l3 = -0.125 * r**3 * dw * (dw + 2.0 * m0_sq)
        l5 = (
            1.0 / 16.0 * r**5 * dw
            * (3.0 * m0_sq**2 + 3.0 * m0_sq * dw + dw**2)
            + 1.0 / 32.0 * (
                r**5 * dwr**2 - 4.0 * dw**2 * r**3
                - 8.0 * dw * r**3 * m0_sq
            )
        )
        l7 = (
            -5.0 / 128.0 * r**7 * dw * (2.0 * m0_sq + dw)
            * (2.0 * m0_sq**2 + 2.0 * dw * m0_sq + dw**2)
            + 60.0 / 64.0 * m0_sq**2 * r**5 * dw
            + 1.0 / 64.0 * (20.0 * dw**3 * r**5 - 5.0 * r**7 * dw * dwr**2)
            - 32.0 / 128.0 * r**3 * dw * m0_sq
            + 1.0 / 128.0 * (
                -r**7 * d2wr**2 + 13.0 * r**5 * dwr**2
                - 16.0 * r**3 * dw**2
            )
        )
        l9 = (
            7.0 / 256.0 * r**9 * dw
            * (5.0 * m0_sq**4 + 10.0 * m0_sq**3 * dw
               + 10.0 * m0_sq**2 * dw**2 + 5.0 * m0_sq * dw**3 + dw**4)
            - 560.0 / 256.0 * m0_sq**3 * r**7 * dw
            + 35.0 / 256.0 * m0_sq**2 * (r**9 * dwr**2 - 24.0 * r**7 * dw**2)
            + 70.0 / 256.0 * m0_sq * (r**9 * dwr**2 * dw - 8.0 * r**7 * dw**3)
            + 35.0 / 256.0 * (r**9 * dwr**2 * dw**2 - 4.0 * r**7 * dw**4)
            + 1008.0 / 256.0 * m0_sq**2 * r**5 * dw
            + 7.0 / 256.0 * m0_sq * (
                r**9 * d2wr**2 - 31.0 * r**7 * dwr**2 + 144.0 * r**5 * dw**2
            )
            + 7.0 / 256.0 * (
                -5.0 * r**8 * dwr**3
                + dw * (r**9 * d2wr**2 - 31.0 * r**7 * dwr**2)
                + 48.0 * r**5 * dw**3
            )
            - 64.0 / 256.0 * m0_sq * r**3 * dw
            + 1.0 / 512.0 * (
                r**9 * d3wr**2 - 29.0 * r**7 * d2wr**2
                + 133.0 * r**5 * dwr**2 - 64.0 * r**3 * dw**2
            )
        )

        l1_int += simpson(l1, x=r)
        l3_int += simpson(l3, x=r)
        l5_int += simpson(l5, x=r)
        l7_int += simpson(l7, x=r)
        l9_int += simpson(l9, x=r)

    ell = np.arange(l_max, dtype=float)
    j = ell + 1.0
    wkb_by_l = (
        l1_int / j
        + l3_int / j**3
        + l5_int / j**5
        + l7_int / j**7
        + l9_int / j**9
    )

    # BubbleDet's d=4 infinite sums for 1/2 * d_l * j^-p,
    # with d_l = (l+1)^2.  The p=1 and p=3 terms are the MS-bar divergent
    # sums assigned their dimensional-regularisation finite values there.
    order_sums = np.array([
        -3.0,
        -1.5,
        0.07705690315959429,
        0.005677755143369926,
        0.0005367773819228268,
    ])
    coeffs = np.array([l1_int, l3_int, l5_int, l7_int, l9_int])
    total_wkb = 0.5 * float(np.dot(coeffs, order_sums))

    if start_l > 2:
        total_wkb -= 0.5 * float(
            np.sum((ell[2:start_l] + 1.0) ** 2 * wkb_by_l[2:start_l])
        )
    elif start_l < 2:
        raise ValueError("BubbleDet's four-dimensional WKB tail starts at l=2.")

    return wkb_by_l, total_wkb
