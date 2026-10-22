#!/usr/bin/env python3
"""fv_analytic_toolbox_vgeneral_phi_v2.py -- the Bessel functions of the
free (false-vacuum) channels: I_nu, K_nu, their logarithms and
log-derivatives and the product I_nu K_nu at the argument kappa_i r, in
closed form.  A library (no stage of its own), used by the h-ODE engine
(greens_function_vgeneral_phi_v2) and by the add-on.

READS
-----
Pure functions of (nu, z): the Bessel order nu = n + 1 and the argument
z = kappa_i r, kappa_i = sqrt(s^2 + m_i^2) the shifted mass of channel i
(GreensFunction.kappa); no files, no project imports (numpy + scipy.special
only).

WRITES (what each helper returns, and who reads it)
---------------------------------------------------
    IK_prod_fv(nu, z)        I_nu(z) K_nu(z), scalar or array z: the FV weight
                          r I_nu K_nu(kappa_i r) on the radial grid
                          (GreensFunction._fv_weight), the slope weight
                          r_max I_nu K_nu(kappa_i r_max) that carries h^- to
                          r = inf (GreensFunction.h_minus_inf) and the free
                          coincident wave of the add-on
                          (add_on_greensfct/greensfct_fv_coincident_v2)
    dlogI_and_logI(nu, z) (d/dz log I_nu, log I_nu) sharing ONE ive(nu, z) call
    dlogK_and_logK(nu, z) (d/dz log K_nu, log K_nu) sharing ONE kve(nu, z) call
                          -- the friction P and the coupling K of the h-ODE
                          (GreensFunction._rhs: the '-' branch reads I_nu,
                          the '+' branch K_nu)
    logI_and_logK_tables(nu, z)   (log I_nu(z), log K_nu(z)) on a GRID array z:
                          the log-Bessel tables of the FV weight
                          W_ij = K_nu(kappa_i r) I_nu(kappa_j r) / r^2 the
                          engine never needs (add_on_greensfct/greensfct_toolbox_v2
                          fv_weight_matrix); same values and same fallback as
                          the scalar helpers, vectorised over z

HOW
---
Every free O(4) partial wave n (nu = n+1) is pure Bessel: with
n(n+2) = nu^2 - 1 the substitution f = w(kappa r)/r maps the free radial
equation exactly onto the modified Bessel equation of order nu, so

    B^-_i(r) = I_nu(kappa_i r) / r         (regular at r = 0)
    B^+_i(r) = K_nu(kappa_i r) / r         (decaying at r -> inf)
    kappa_i     = sqrt(s^2 + m_i^2)        (the shifted mass of channel i)

with the product IK_i(r) = I_nu(kappa_i r) K_nu(kappa_i r) of the FV weight.
These closed forms are never discretized: scipy's SCALED values
ive(nu, z) = e^-z I_nu(z) and kve(nu, z) = e^+z K_nu(z) give
log I_nu = log ive + z, log K_nu = log kve - z and I_nu K_nu = ive kve, and
the log-derivatives follow from the recurrences
    d/dz log I_nu(z) =  nu/z + I_{nu+1}(z)/I_nu(z),
    d/dz log K_nu(z) = -nu/z - K_{nu-1}(z)/K_nu(z).
A scipy value is used only when it is finite and strictly positive (every
Bessel value here is); where the scaled value has left float64 (ive
underflows to 0, kve overflows to inf, once the order has outrun the
argument) the two fallbacks take over:
  * log I_nu and log K_nu fall back to the leading DLMF 10.41 uniform asymptotic,
        t = sqrt(nu^2 + z^2),  eta = t + nu ln(z/(nu + t)),
        log I_nu(z) ~  eta - (1/2) ln(2 pi t),
        log K_nu(z) ~ -eta + (1/2) ln(pi/(2 t)),
    and the product I_nu K_nu ~ 1/(2 t);
  * the ratio inside dlog I / dlog K falls back to the z-DERIVATIVE of the
    same log fallback, d/dz log I ~ t/z - z/(2 t^2), d/dz log K ~ -t/z - z/(2 t^2):
        I_{nu+1}/I_nu ~ z/(nu + t) - z/(2 t^2),   K_{nu-1}/K_nu ~ z/(nu + t) + z/(2 t^2),
    correct to O(1/nu^2) for every z/nu and reducing to the small-argument
    forms z/(2(nu+1)), z/(2(nu-1)) for z << nu.
The fallbacks act only where the order has outrun the argument: at the
innermost radial points of the high waves.  There the leading log I_nu and
log K_nu are off by 1/(12 nu) (z << nu); the coupling K sees that error only
in the radial window where one channel has flushed and another has not
(elsewhere it cancels in the ratio B_j/B_i).

NO false-vacuum term is ever ADDED to lnD_ren: the determinant is the
RATIO to the false vacuum, so the FV enters only as these weights and as
the pointwise subtraction G_bracket - 1 (the FV trace of one wave alone is
GreensFunction.gbar_fv).  The FV matrices of the Green's-function path --
off-coincident, coincident wave by wave, and the renormalised tau_fin_fv
matrix -- live in add_on_greensfct/greensfct_fv_*.py.
"""
import math
import numpy as np
from scipy.special import ive, kve


# ---------------------------------------------------------------------------
# IK_prod_fv: the coincident false-vacuum product I_nu K_nu
# ---------------------------------------------------------------------------
def IK_prod_fv(nu, z):
    """I_nu(z) K_nu(z) for scalar or array z, as ive(nu, z) kve(nu, z) (the
    scalings e^-z and e^+z cancel exactly); where that product has left
    float64 (the order has outrun the argument), the leading uniform
    large-order form 1/(2 sqrt(nu^2 + z^2)), relative error O(1/nu^2)."""
    z = np.asarray(z, float)
    with np.errstate(all='ignore'):
        p = ive(nu, z) * kve(nu, z)
    return np.where(np.isfinite(p) & (p > 0), p, 0.5 / np.sqrt(nu * nu + z * z))

# ---------------------------------------------------------------------------
# scalar Bessel helpers: (d/dz log, log) of I_nu or K_nu from ONE scaled
# ive/kve call, with the DLMF 10.41 fallback where the scaled value flushes
# ---------------------------------------------------------------------------
def _log_bessel_fallback(branch, nu, z):
    """log I_nu(z) ('-') or log K_nu(z) ('+') from the leading DLMF 10.41
    uniform asymptotic (module header), used where scipy's SCALED Bessel value
    has left the float64 range."""
    t = np.hypot(nu, z)
    eta = t + nu * np.log(z / (nu + t))
    return (eta - 0.5 * np.log(2 * np.pi * t)) if branch == '-'\
        else (-eta + 0.5 * np.log(np.pi / (2 * t)))


def _ratio_fallback(branch, nu, z):
    """I_{nu+1}(z)/I_nu(z) ('-') or K_{nu-1}(z)/K_nu(z) ('+') from the
    z-derivative of _log_bessel_fallback (module header): with t = sqrt(nu^2 + z^2)
    the ratio is z/(nu + t) -/+ z/(2 t^2), used where a scaled Bessel value
    has left float64."""
    t = math.hypot(nu, z)
    return (z / (nu + t) - z / (2.0 * t * t)) if branch == '-'\
        else (z / (nu + t) + z / (2.0 * t * t))


def dlogI_and_logI(nu, z):
    """(d/dz log I_nu(z), log I_nu(z)) from ONE ive(nu, z) evaluation,
        d/dz log I_nu(z) = nu/z + I_{nu+1}(z)/I_nu(z).
    The scaled value is used where it is finite and strictly positive
    (I_nu > 0), and so is the I_ratio; where either has left float64 the two
    module-header fallbacks replace it."""
    I_scaled = ive(nu, z)
    if math.isfinite(I_scaled) and I_scaled > 0.0:
        logI = np.log(I_scaled) + z
        I_ratio = ive(nu + 1, z) / I_scaled
        if not (math.isfinite(I_ratio) and I_ratio > 0.0):
            I_ratio = _ratio_fallback('-', nu, z)
    else:                                  # ive flushed to 0 (or inf, nan)
        logI = _log_bessel_fallback('-', nu, z)
        I_ratio = _ratio_fallback('-', nu, z)
    return nu / z + I_ratio, logI


def dlogK_and_logK(nu, z):
    """(d/dz log K_nu(z), log K_nu(z)) from ONE kve(nu, z) evaluation,
        d/dz log K_nu(z) = -nu/z - K_{nu-1}(z)/K_nu(z),
    under the same finite-and-positive rule (kve overflows to +inf rather
    than flushing, so that test is what catches it)."""
    K_scaled = kve(nu, z)
    if math.isfinite(K_scaled) and K_scaled > 0.0:
        logK = np.log(K_scaled) - z
        K_ratio = kve(nu - 1, z) / K_scaled
        if not (math.isfinite(K_ratio) and K_ratio > 0.0):
            K_ratio = _ratio_fallback('+', nu, z)
    else:                                  # kve overflowed to inf (or nan)
        logK = _log_bessel_fallback('+', nu, z)
        K_ratio = _ratio_fallback('+', nu, z)
    return -nu / z - K_ratio, logK


def logI_and_logK_tables(nu, z):
    """(log I_nu(z), log K_nu(z)) for an ARRAY of arguments z -- the log-Bessel
    tables of the add-on (add_on_greensfct/greensfct_toolbox_v2
    log_bessel_tables): behind the FV weight W_ij(r) =
    K_nu(kappa_i r) I_nu(kappa_j r) / r^2 (fv_weight_matrix; the cross
    channels i != j make the PRODUCT overflow float64, so the weight must be
    assembled in log space), the free Bessel solutions B^pm of the radial wave
    (build_G_n), the free off-coincident wave and the eigenfunction of
    greensfct_dangerous_modes_v2.  Same rule as the
    scalar helpers above -- log of the scaled scipy value where it is finite
    and strictly positive, the DLMF 10.41 uniform asymptotic where it has
    flushed -- vectorised over z (np.where takes the same branch, entry by
    entry, that the scalar helpers take)."""
    z = np.asarray(z, float)
    with np.errstate(all='ignore'):
        iv, kv = ive(nu, z), kve(nu, z)
        logI = np.where(np.isfinite(iv) & (iv > 0.0), np.log(iv) + z,
                        _log_bessel_fallback('-', nu, z))
        logK = np.where(np.isfinite(kv) & (kv > 0.0), np.log(kv) - z,
                        _log_bessel_fallback('+', nu, z))
    return logI, logK
