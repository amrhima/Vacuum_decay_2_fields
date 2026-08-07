#!/usr/bin/env python3
"""fv_analytic_coupled_toy_model.py -- the ANALYTIC false-vacuum Green-function
background used on ALL partial waves.

WHAT IT COMPUTES
----------------
The false-vacuum mode functions of every O(4) partial wave n (nu = n+1)
are pure Bessel functions (notes eq (43) / notes/false_vacuum.pdf sec 9,
boxed equations):

    B-_i(r) = I_nu(kap_i r) / r        (regular at r = 0)
    B+_i(r) = K_nu(kap_i r) / r        (decaying at r -> inf)
    kap_i   = sqrt(s^2 + m_i^2)        (channel i = 1..N)

and the coincident false-vacuum Green-function weight is their product

    IK_i(r) = I_nu(kap_i r) K_nu(kap_i r) .

The FV trace is never discretized -- only these closed
Bessel forms appear (scaled ive/kve, float64-safe).  This module holds the ONE
certified implementation of them; the band engine
(delta_g_bar_greater_equal_3_coupled_toy_model) imports IK_prod and the fast scalar
helpers from here, and the n = 0, 1 sector stages reach them only through the
band engine's eval_point / h_inf.

WHAT EACH HELPER RETURNS
------------------------
    IK_prod(nu, z)        I_nu(z) K_nu(z)  (array form; DLMF large-order
                          fallback 0.5/sqrt(nu^2+z^2) where the scaled
                          product under/overflows)
    dlogI_fast(nu, z)     scalar d/dz log I_nu(z) resp. log K_nu(z)
    dlogK_fast(nu, z)     (guarded branches; scalar fast path)
    dlogI_and_logI_fast(nu, z)   (dlogI, logI) sharing ONE ive(nu, z) call
    dlogK_and_logK_fast(nu, z)   (dlogK, logK) sharing ONE kve(nu, z) call,
                          both with the DLMF 10.41 uniform-asymptotic
                          fallback where the scaled Bessel value flushes

INPUTS / OUTPUT: pure functions of (nu, z); no files, no project
imports (numpy + scipy.special only).
"""
import math
import numpy as np
from scipy.special import ive, kve


# ---------------------------------------------------------------------------
# IK_prod: the dense-grid coincident FV weight the band engine imports
# ---------------------------------------------------------------------------
def IK_prod(nu, z):
    z = np.asarray(z, float)
    with np.errstate(all='ignore'):
        p = ive(nu, z) * kve(nu, z)
    return np.where(np.isfinite(p) & (p > 0), p, 0.5 / np.sqrt(nu * nu + z * z))

# ---------------------------------------------------------------------------
# fast scalar Bessel helpers (guarded branches, no array machinery);
# the *_and_log*_fast variants also return log B reusing the already-computed
# scaled Bessel value, with the DLMF 10.41 uniform-asymptotic fallback where
# it flushes.
# ---------------------------------------------------------------------------
def _dlogI(nu, z, den):
    """Guarded I-ratio from a PRECOMPUTED den = ive(nu, z) (one guard, one
    implementation -- shared by dlogI_fast and dlogI_and_logI_fast)."""
    if den > 0:
        r = ive(nu + 1, z) / den
        if not math.isfinite(r):
            r = z / (2.0 * (nu + 1.0))
    else:                                  # covers den <= 0 and den = nan
        r = z / (2.0 * (nu + 1.0))
    return nu / z + r

def _dlogK(nu, z, den):
    """Guarded K-ratio from a PRECOMPUTED den = kve(nu, z)."""
    if den > 0:
        r = kve(nu - 1, z) / den
        if not math.isfinite(r):
            r = z / (2.0 * max(nu - 1.0, 1.0))
    else:
        r = z / (2.0 * max(nu - 1.0, 1.0))
    return -nu / z - r

def dlogI_fast(nu, z):
    return _dlogI(nu, z, ive(nu, z))

def dlogK_fast(nu, z):
    return _dlogK(nu, z, kve(nu, z))

def _logB_fallback(branch, nu, z):
    t = np.hypot(nu, z)
    eta = t + nu * np.log(z / (nu + t))
    return (eta - 0.5 * np.log(2 * np.pi * t)) if branch == '-' \
        else (-eta + 0.5 * np.log(np.pi / (2 * t)))

def dlogI_and_logI_fast(nu, z):
    """(dlogI, logI) sharing one ive(nu, z) evaluation; identical values."""
    den = ive(nu, z)
    dlog = _dlogI(nu, z, den)
    if math.isfinite(den) and den > 0:     # logB's own guard (finite AND > 0)
        lb = np.log(den) + z
    else:
        lb = _logB_fallback('-', nu, z)
    return dlog, lb

def dlogK_and_logK_fast(nu, z):
    """(dlogK, logK) sharing one kve(nu, z) evaluation; identical values."""
    den = kve(nu, z)
    dlog = _dlogK(nu, z, den)
    if math.isfinite(den) and den > 0:
        lb = np.log(den) - z
    else:
        lb = _logB_fallback('+', nu, z)
    return dlog, lb
