#!/usr/bin/env python3
"""pipeline_quadrature_coupled_toy_model.py -- the s2 integration scheme: how the s2
stepsize/grid is built and how every s2 integral (sectors, band, plots) is
taken; NOT a physics stage.

This module owns the ONE production s2 quadrature (the log-Simpson
scheme), the dedicated geometric resolvent sector s2 grid, and the shared
pole-subtracted resolvent sector integral that both n=0 and n=1 use.  It is
purely numerical: nothing here knows about a specific physics stage.

CONTENTS
--------
    S2_LOG_SPLIT                shared endpoint (s2 = 50) of the log- and
                                linear-Simpson quadrature segments
    s2_integral(s2, delta)    I(L2) = -Int_0^{L2} delta ds2, log-Simpson
                                scheme (the ONE production s2 quadrature:
                                sectors, band, plots)
    s2_tail_shape_ratio(...)         the Method-I analytic completion shape ratio
                                R_n: the s2 integral is numerical below the
                                cutoff and completed analytically above it
                                with T_n = -delta_n(Lam2) * R_n (the exact
                                integral of the uniform O(U^3) tail shape,
                                normalised at Lam2) -- the closing piece of
                                the same s2 integration scheme
    odd_tail_fit(...)           the n-direction completion scheme: the
                                three-term odd-family fit a/nu^3 + c/nu^5 +
                                e/nu^7 on [fit_lo, fit_hi] + the Hurwitz-zeta
                                tail over all deeper waves (enters lnD_ren in
                                stage 5; the driver's watcher reads the same
                                fit for its measurements); carries the
                                N-WINDOW guard
    sector_s2_grid(...)         the dedicated geometric resolvent s2 grid
                                (odd node count, exact endpoint snap)
    pole_subtracted_sector_integral(...)
                                the production resolvent sector integral
                                I8res_n = -Int [delta_n - deg/(lam+s2)] ds2
                                with the SECTOR-STEP every-2nd-point meter
    adaptive_sector_integral(...)
                                the above, auto-doubling nodes until
                                SECTOR-STEP passes (pole-aware,
                                model-independent)
    sector_tail_error(...)      the lnD truncation error Int_{s2_max}^inf
                                delta_n ds2 from cutting the raw trace at the
                                cutoff -- the estimator behind the assembler's
                                SECTOR-CUTOFF guard (is s2_sector_max high
                                enough?  aborts; does NOT enter lnD)

DEPENDENCY: pole_subtracted_sector_integral uses require_finite, imported
FROM pipeline_helpers_coupled_toy_model (one-way dependency pipeline_quadrature -> pipeline_helpers;
pipeline_helpers must NOT import this module -- there is no cycle).
"""
import numpy as np
from scipy.integrate import simpson

from pipeline_helpers_coupled_toy_model import require_finite


# --------------------------------------------------------------------------- #
#  Band s2 quadrature: the LOG-SIMPSON scheme                                  #
# --------------------------------------------------------------------------- #
S2_LOG_SPLIT = 50.0   # shared endpoint of the log- and linear-Simpson segments


def s2_integral(s2, delta):
    """I(L2) = -Int_0^L2 delta ds2 on the EXISTING band grid, evaluated with
    the log-Simpson scheme (verified per-wave against exact J3plus
    endpoints for ALL n = 2..150: total band error -0.0073, max 0.0009/wave):

      [s2_min, 50]   Simpson in u = ln s2 of delta*s2
                     (Int delta ds2 = Int delta*s2 dln s2; the integrand is
                     smooth in ln s2, which removes the +1.19 low-s2 bias of
                     linear Simpson on the 6-pts/decade geometric grid; scipy
                     simpson handles the non-uniform u-spacing of a non-uniform
                     grid),
      [50, L2]       linear Simpson in s2 (the grid is converged there),
      [0, s2_min]    rectangle sliver delta(s2_min)*s2_min
                     (delta(0) is finite -- a low-s2 plateau -- so the
                     rectangle closes this segment, worth +0.30 if dropped,
                     to -0.0003 total).

    The split point s2 = 50 is the SHARED endpoint of both Simpson segments.
    s2 must be a 1-D ascending grid; delta holds the band values on it (last
    axis if multi-dimensional)."""
    s2 = np.asarray(s2, float)
    delta = np.asarray(delta, float)
    if s2.ndim != 1 or s2.size < 3 or np.any(np.diff(s2) <= 0.0):
        raise RuntimeError('[ABORT] s2_integral needs an ascending 1-D s2 '
                           'grid with >= 3 points.')
    # largest grid index with s2 <= split -> shared endpoint of the segments
    i_split = int(np.searchsorted(s2, S2_LOG_SPLIT * (1.0 + 1e-12),
                                  side='right')) - 1
    i_split = min(max(i_split, 0), len(s2) - 1)
    I_low = 0.0
    if i_split >= 1:                       # log-Simpson segment [s2_min, split]
        I_low = simpson(delta[..., :i_split + 1] * s2[:i_split + 1],
                        x=np.log(s2[:i_split + 1]), axis=-1)
    I_high = 0.0
    if i_split <= len(s2) - 2:             # linear-Simpson segment [split, L2]
        I_high = simpson(delta[..., i_split:], x=s2[i_split:], axis=-1)
    sliver = delta[..., 0] * s2[0]         # rectangle for the omitted [0, s2_min]
    return -(I_low + I_high + sliver)


def s2_tail_shape_ratio(nu, Lam2, mbar2, R, trDW3):
    """Method-I calibrated uniform-shape tail ratio
        R_n = (2/5) [Int dr r^5 trU3 chiL^-5] / [Int dr r^7 trU3 chiL^-7],
    chiL(r) = sqrt(nu^2 + (Lam2 + mbar2) r^2), trU3 = tr(DW^3).
    Exact s2-integral of the uniform O(U^3) tail shape from Lam2 to inf,
    normalised to the shape's value at Lam2 (np.trapezoid on the bounce
    grid).  The completion stage forms T_n = -delta_n(Lam2) * R_n with it;
    the watchers evaluate the same formula for their measurements."""
    chiL = np.sqrt(nu * nu + (Lam2 + mbar2) * R**2)
    num = np.trapezoid(R**5 * trDW3 * chiL**-5, R)
    den = np.trapezoid(R**7 * trDW3 * chiL**-7, R)
    return (2.0 / 5.0) * num / den


def odd_tail_fit(ns, I_inf, fit_lo, fit_hi, min_points=12):
    """The n-direction completion scheme (companion of s2_tail_shape_ratio, which
    completes the s2 direction): the odd-family least-squares fit on the
    window [fit_lo, fit_hi] (nu = n+1),

        I_n(inf) ~ a/nu^3 + c/nu^5 + e/nu^7,

    and the closed-form Hurwitz-zeta tail over ALL waves beyond fit_hi.
    Three terms: measured on the certified tables, the two-term fit leaves a
    2.9e-2 (bdet) / 4.8e-3 (F2) tail error at the SAME window where three
    terms leave ~1e-4 -- the e/nu^7 term is kept.  Only `a` is analytic (A3);
    c and e are interpolation coefficients absorbing the window's remaining
    approach.  Returns (a, c, e, tail, subleading_bound) with
        tail = a z3 + c z5 + e z7,        z_k = zeta(k, fit_hi + 2)
        subleading_bound = |c| z5 + |e| z7
    `tail` ENTERS lnD_ren (stage 5); subleading_bound is only what the
    N-TAIL-BOUND watcher/guard gate on.
    ---- GUARD N-WINDOW (aborts; does NOT enter lnD): the window must be ----
    ---- contiguous and hold >= min_points waves (3-parameter fit).      ----"""
    from scipy.special import zeta
    ns = np.asarray(ns, int)
    m = (ns >= fit_lo) & (ns <= fit_hi)
    win = ns[m]
    if win.size < min_points:
        raise RuntimeError(
            f"[ABORT N-WINDOW] fit window [{fit_lo},{fit_hi}] holds {win.size} "
            f"waves < min_points={min_points} -- too few for a stable "
            f"3-parameter fit; generate deeper waves.")
    if not np.array_equal(win, np.arange(fit_lo, fit_hi + 1)):
        raise RuntimeError(
            f"[ABORT N-WINDOW] fit window [{fit_lo},{fit_hi}] is not "
            f"contiguous ({win.size}/{fit_hi - fit_lo + 1} waves present) -- "
            f"a missing wave would bias the fit.")
    v = win + 1.0
    X = np.vstack([v**-3.0, v**-5.0, v**-7.0]).T
    (a, c, e), *_ = np.linalg.lstsq(X, np.asarray(I_inf, float)[m], rcond=None)
    z3, z5, z7 = (float(zeta(k, fit_hi + 2.0)) for k in (3, 5, 7))
    tail = a * z3 + c * z5 + e * z7
    sub = abs(c) * z5 + abs(e) * z7
    return float(a), float(c), float(e), float(tail), float(sub)


# --------------------------------------------------------------------------- #
#  Production resolvent sector grid + integral (shared by n = 0 and n = 1)     #
# --------------------------------------------------------------------------- #
def sector_s2_grid(s2_sec_lo, s2_sec_max, n_nodes, pole=None):
    """The dedicated resolvent sector s2 grid on [s2_sec_lo, s2_sec_max].

    PLAIN (pole is None or OUTSIDE the range): a single geometric grid,
    integrated by the certified s2_integral (log-Simpson).  Covers every
    wave whose bound-state pole sits below the sector floor (e.g. the n=1
    zero mode at s2 ~ 1e-5).

    POLE-AWARE (pole = -lam INSIDE the range, e.g. a coupled potential's n=0
    negative mode landing mid-range): the pole-subtracted integrand
    delta_n - deg/(lam+s2) has its curvature peak AT s2 = -lam, which a plain
    geometric grid resolves only to FIRST order (~3800 nodes for 1e-3).  The
    grid is instead SPLIT at the pole and made geometric in the DISTANCE
    |s2 - pole| on each side -- dense at the pole, spreading outward -- with the
    innermost node held 1% of the pole away so the delta_n ~ deg/(lam+s2)
    near-cancellation keeps full float64 precision.  Each side is Simpson-
    integrated (pole_subtracted_sector_integral); measured ~8x fewer nodes for
    the same accuracy.  Model-independent: it centres on wherever the eigenvalue
    puts the pole.

    n_nodes is forced ODD.  Returns (s2_grid, split): split is None (plain) or
    the index of the first above-pole node (clustered)."""
    s2_sec_max = float(s2_sec_max)
    if s2_sec_max <= 0.0:
        raise RuntimeError('[ABORT] --s2-sector-max must be > 0.')
    n_nodes = int(n_nodes)
    if n_nodes < 61:
        raise RuntimeError('[ABORT SECTOR] --sector-nodes must be >= 61 '
                           '(production-grade resolvent sector grid).')
    if n_nodes % 2 == 0:
        n_nodes += 1                      # odd -> [::2] keeps both endpoints
    if pole is None or not (s2_sec_lo < pole < s2_sec_max):
        s2_sec = np.geomspace(s2_sec_lo, s2_sec_max, n_nodes)
        s2_sec[-1] = s2_sec_max          # exact endpoint (the fict logs use it)
        return s2_sec, None
    # pole-clustered: geometric in |s2 - pole|, split at the pole (the small jump
    # in the subtracted integrand then sits ON the split, not straddled).
    dmin = 1e-2 * pole                    # innermost node distance (glitch-safe)
    if dmin >= pole - s2_sec_lo:          # pole not comfortably above the floor:
        s2_sec = np.geomspace(s2_sec_lo, s2_sec_max, n_nodes)  # below-side would
        s2_sec[-1] = s2_sec_max          # invert -> fall back to the plain grid
        return s2_sec, None
    n_each = n_nodes // 2                 # nodes on each side of the pole
    if n_each % 2 == 0:                    # force ODD per side so the meter's
        n_each += 1                        # half grid ([::2]) keeps BOTH the
                                           # innermost pole node AND the outer
                                           # endpoint -- fine and half then span
                                           # the SAME domain, so the node-doubling
                                           # convergence check is an honest
                                           # Richardson step (not a comparison of
                                           # two different integration domains)
    below = (pole - np.geomspace(dmin, pole - s2_sec_lo, n_each))[::-1]
    above = pole + np.geomspace(dmin, s2_sec_max - pole, n_each)
    above[-1] = s2_sec_max               # exact endpoint (the fict logs use it)
    s2_sec = np.concatenate([below, above])
    return s2_sec, int(len(below))


def _sector_simpson_split(s2_grid, dsub, split):
    """I8res = -Int dsub ds2 on a POLE-SPLIT clustered grid: Simpson on the
    below-pole side [:split] + the above-pole side [split:], plus the small
    [pole-dmin, pole+dmin] sliver closed by the trapezoid of the two innermost
    boundary values.  Each side is smooth; the tiny jump AT the pole sits on the
    split (not straddled), so Simpson recovers high order.  Same -Int sign
    convention as s2_integral."""
    b, a = s2_grid[:split], s2_grid[split:]
    db, da = dsub[:split], dsub[split:]
    sliver = 0.5 * (db[-1] + da[0]) * (a[0] - b[-1])   # [pole-dmin, pole+dmin]
    return -(simpson(db, x=b) + simpson(da, x=a) + sliver)


def pole_subtracted_sector_integral(band, n, lam, s2_grid, split=None):
    """The PRODUCTION resolvent sector integral for wave n (Carosi Eq. 5.10,
    bounce side):
        delta_n(s2)  = GfullmFV - gbar1 - gbar2   (all orders >= 3)
        delta_sub    = delta_n(s2) - deg_n/(lam + s2)   (bounce-side pole removed
                       at the NUMERICAL continuum eigenvalue lam)
        I8res        = -Int delta_sub ds2
    `split` (from sector_s2_grid): None -> plain grid, s2_integral
    (log-Simpson); an int -> pole-clustered grid, Simpson per side.
    Returns (delta_raw, delta_sub, I8res, conv_meter); conv_meter = |I(full) -
    I(every-2nd-point)| is the SECTOR-STEP estimate the caller gates on -- this
    function does NOT abort (adaptive_sector_integral owns the gate)."""
    import time as _time
    n_nodes = len(s2_grid)
    deg = (n + 1.0) ** 2
    vals = np.empty(n_nodes)
    t_sec0 = _time.time()
    for k, s2p in enumerate(s2_grid):
        vals[k] = band.eval_point(n, float(s2p))['delta_geq3']
        if (k + 1) % 80 == 0 or k == n_nodes - 1:
            print(f'       [SECTOR n={n}] node {k+1}/{n_nodes} '
                  f'(s2={s2p:.6g})  [{_time.time()-t_sec0:.0f}s]', flush=True)
    require_finite(vals, f'[ABORT SECTOR] non-finite delta_{n}(s2) '
                         f'resolvent nodes')
    dsub = vals - deg / (lam + s2_grid)      # bounce-side pole removed
    require_finite(dsub, '[ABORT SECTOR] non-finite pole-subtracted delta_n')
    if split is None:                        # plain geometric grid (unchanged)
        I_fine = float(s2_integral(s2_grid, dsub))
        I_half = float(s2_integral(s2_grid[::2], dsub[::2]))
    else:                                    # pole-clustered grid, Simpson/side
        I_fine = float(_sector_simpson_split(s2_grid, dsub, split))
        b2, a2 = s2_grid[:split][::2], s2_grid[split:][::2]
        db2, da2 = dsub[:split][::2], dsub[split:][::2]
        I_half = float(_sector_simpson_split(np.concatenate([b2, a2]),
                                             np.concatenate([db2, da2]), len(b2)))
    conv = abs(I_fine - I_half)
    return vals, dsub, I_fine, conv


def adaptive_sector_integral(band, n, lam, s2_sec_lo, s2_sec_max, n0, conv_tol,
                             n_cap=8001):
    """Build the (pole-aware) sector grid and integrate; if the SECTOR-STEP
    meter exceeds conv_tol, DOUBLE the node count and retry until it passes or
    n_cap is reached.
    Model-independent + automatic: any potential's pole -- wherever the
    eigenvalue puts it (low edge -> plain grid; mid-range -> pole-clustered) --
    converges with NO manual --sector-nodes and no mid-run abort.  Returns
    (s2_grid, delta_raw, delta_sub, I8res, conv)."""
    pole = -lam
    n_nodes = int(n0)
    while True:
        s2_grid, split = sector_s2_grid(s2_sec_lo, s2_sec_max, n_nodes, pole=pole)
        vals, dsub, I8res, conv = pole_subtracted_sector_integral(
            band, n, lam, s2_grid, split)
        kind = 'pole-clustered' if split is not None else 'plain'
        print(f'       [SECTOR-STEP n={n}] {conv:.3e} (limit {conv_tol:g}; '
              f'{kind} grid {len(s2_grid)} nodes)', flush=True)
        if conv <= conv_tol:
            return s2_grid, vals, dsub, I8res, conv
        if n_nodes >= n_cap:
            raise RuntimeError(
                f'[ABORT SECTOR-STEP] resolvent sector n={n} not converged at '
                f'{len(s2_grid)} nodes (meter {conv:.3e} > {conv_tol:g}); node '
                f'cap {n_cap} reached -- raise it or check the bounce/eigenvalue.')
        n_nodes = min(n_cap, 2 * n_nodes + 1)


def sector_tail_error(s2_grid, delta_raw, fit_frac=0.1):
    """The lnD truncation error from cutting the RAW resolvent trace delta_n at
    the sector cutoff s2_sector_max -- the piece the closed-form T_sec8 does NOT
    carry (T_sec8 continues only the pole -deg/(lam+s2) to infinity; the physical
    Born-subtracted trace delta_n beyond the cutoff is simply dropped).

    The exact sector-cutoff dependence of the assembled result is
    d(lnD_ren)/d(s2_hi) = -delta_n(s2_hi) (the fict + T_sec8 pieces cancel the
    subtracted pole part identically), so the tail error is Int_{s2_max}^inf
    delta_n ds2.  delta_n(s2) decays as a power law C s2^p at large s2 (the
    coincident third-order-Born trace); fit p from the top decade of the grid
    (s2 >= fit_frac*s2_max) and integrate the fitted tail analytically:
        tail = C s2_max^{p+1} / (-(p+1)),   C = |delta_n(s2_max)| / s2_max^p .
    A converged cutoff has delta_n already decaying (p < -1) and a tiny tail.
    p >= -1 (the raw trace is NOT decaying -- cutoff far too low, or the Born
    subtraction is failing for this potential) returns inf, which the assembler's
    SECTOR-CUTOFF guard turns into an abort (the guard only aborts; nothing here
    enters lnD).  Model-independent: it reads wherever the
    trace of THIS potential sits at the cutoff.  Returns
    (tail_estimate, decay_power_p, |delta_n(s2_max)|)."""
    s2 = np.asarray(s2_grid, float)
    raw = np.abs(np.asarray(delta_raw, float))
    s2_max = s2[-1]
    m = (s2 >= fit_frac * s2_max) & (raw > 0.0)
    if m.sum() < 3:
        return float('inf'), 0.0, float(raw[-1])
    p = float(np.polyfit(np.log(s2[m]), np.log(raw[m]), 1)[0])
    if p >= -1.0:                        # trace not integrably decaying
        return float('inf'), p, float(raw[-1])
    C = raw[m][-1] / s2[m][-1] ** p
    tail = C * s2_max ** (p + 1.0) / (-(p + 1.0))
    return float(tail), p, float(raw[-1])
