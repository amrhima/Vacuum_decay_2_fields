"""Explicit four-dimensional renormalization conventions.

This module collects the dimensional-regularization/MS-bar pieces used
by the WKB determinant calculation. The convention is the one used by
BubbleDet:

* the partial-wave sum starts at l=2;
* j = l + 1 and d_l = j**2;
* the determinant carries the overall factor 1/2;
* the divergent j and 1/j sums are replaced by their d=4
  dimensional-regularization finite parts, -3 and -3/2;
* the remaining local term is the d=4 MS-bar finite remainder.

Keeping these choices here makes the regulator and normalization visible
instead of burying them in numerical arrays or in the determinant driver.
"""

import numpy as np
from scipy.integrate import simpson
from scipy.special import digamma, zeta


def _finite_hurwitz_zeta_at_one(q):
    """MS-bar finite part of Hurwitz zeta at its pole s=1.

    The Laurent expansion is

        zeta(s, q) = 1/(s-1) - digamma(q) + O(s-1).

    MS-bar removes the universal Euler-constant piece, leaving
    -digamma(q) - gamma_E. For integer q this is minus the harmonic
    number H_(q-1).
    """
    return float(-digamma(q) - np.euler_gamma)


def _wkb_weight_4d(p):
    """Compute one regulated d=4 angular sum for WKB power p.

    The convergent cases are ordinary Hurwitz-zeta sums. For p=3 the
    harmonic pole is assigned its MS-bar finite part. For p=1 the
    dimensionally continued angular sum is supplemented by the finite
    Coleman-Weinberg counterterm contribution required by the BubbleDet
    convention:

        zeta(-1, 3) - zeta(-1) = -3.

    Here zeta(-1) is evaluated from the Bernoulli-polynomial identity
    zeta(-1) = -B_2/2, rather than inserted as a numerical weight.
    """
    if p == 1:
        # Hurwitz identity: zeta(-1, q) = -B_2(q)/2, with
        # B_2(q) = q**2 - q + 1/6.
        zeta_minus_one_q3 = -(3.0**2 - 3.0 + 1.0 / 6.0) / 2.0
        zeta_minus_one_q1 = -(1.0**2 - 1.0 + 1.0 / 6.0) / 2.0
        return float(zeta_minus_one_q3 - zeta_minus_one_q1)
    if p == 3:
        return _finite_hurwitz_zeta_at_one(3)
    if p in (5, 7, 9):
        return float(zeta(p - 2, 3))
    raise ValueError("Supported d=4 WKB powers are p=1, 3, 5, 7, and 9.")


def wkb_sum_weights_4d(l_start=2):
    """Return the regulated sums multiplying the d=4 WKB coefficients.

    For a term l_p / j**p the weight before the determinant factor 1/2
    is F_p = sum_{l=l_start}^infinity d_l / j**p, with
    d_l = j**2 and j = l + 1.

    The p=1 and p=3 entries are computed by dimensional regularization
    and MS-bar finite-part extraction. For p=5,7,9 the sums converge and
    are ordinary Hurwitz zeta values.
    """
    if l_start != 2:
        raise ValueError(
            "The d=4 MS-bar divergent finite parts are defined here for "
            "the BubbleDet tail starting at l=2."
        )

    return np.array([
        _wkb_weight_4d(p) for p in (1, 3, 5, 7, 9)
    ], dtype=float)


def msbar_local_term_4d(R, W_profile, H_fv, mu_ren):
    r"""Return the finite local d=4 MS-bar renormalization term.

    This is

        -1/16 int dr r^3 Tr[W^2 + 2 H_fv W]
              [log(mu_ren r / 2) + gamma_E + 1].

    It is the local piece paired with the regulated WKB tail above.
    """
    r = np.asarray(R, dtype=float)
    W = np.asarray(W_profile, dtype=float)
    H = np.asarray(H_fv, dtype=float)

    if r.ndim != 1 or W.ndim != 3 or H.ndim != 2:
        raise ValueError("Invalid dimensions for the d=4 MS-bar term.")
    if W.shape[0] != len(r) or W.shape[1:] != H.shape:
        raise ValueError("W_profile and H_fv shapes are inconsistent.")
    if H.shape[0] != H.shape[1]:
        raise ValueError("H_fv must be square.")
    if mu_ren <= 0.0:
        raise ValueError("mu_ren must be positive.")

    trace_term = np.einsum("rij,rji->r", W, W)
    trace_term += 2.0 * np.einsum("ij,rji->r", H, W)

    log_factor = np.zeros_like(r)
    mask = r > 0.0
    log_factor[mask] = (
        np.log(mu_ren * r[mask] / 2.0) + np.euler_gamma + 1.0
    )

    return float(-1.0 / 16.0 * simpson(
        r**3 * trace_term * log_factor,
        x=r,
    ))
