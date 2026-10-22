#!/usr/bin/env python3
"""pipeline_quadrature_vgeneral_phi_v2.py -- library of the determinant
path (no stage lives here; used by stages 1, 2a, 2b, 3, 4 and by the plot
codes): the ONE s2 integration scheme, the band integrals with their
high-s2 and high-n tails, and the sector grids with their pole-subtracted
integral.
"First the s2-integral, then the n-sum" is the production organisation:
after subtraction through Born order 2, delta_geq3 is O((n+1)^-5) at
fixed s2, and its s2-integral is O((n+1)^-3), so the n-sum converges
in either order.  Born orders 1 and 2 do not: Gbar_1 falls only like
(n+1)^-1 already at fixed s2, and Gbar_2 falls like (n+1)^-3 at fixed s2
but its s2-integral only like (n+1)^-1.  Both are subtracted wave by wave
and their MS-bar finite parts restored once in closed form at assembly
(A1_fin and -A2_fin/2).

READS (arrays handed in by the calling stage; this module opens no file)
    delta_geq3(n, s2) of the band    the residual_band slices, assembled by
                                     band_adaptive (stage 1) and
                                     band_integrals (stage 3)
    delta_geq3(s2) of a sector       the engine's gbar_traces, called per
                                     node by sector_traces_of_wave below for
                                     gbar_sector (stages 2a/2b)
    I_n(inf) of the band waves       band_integrals_<tag>.npz, via
                                     tail_high_n_zeta (stage 4), or the band
                                     driver's own candidate arrays
    R, tr_U_int3, m_min2; omega2     load_background (pipeline_helpers); the
                                     eig_n0/eig_n1 npz, via gbar_sector
WRITES (returned to the caller, who stores them)
    I_n(s2_max), T_n(s2_max),        band_candidate_sums -> band_integrals
    I_n(inf), the repair flags,      (stage 3) and the watchers of
    the candidate band sums          band_adaptive (stage 1)
    a, b, c, Zcal, the zeta bound    -> tail_high_n_<tag>.npz (stage 4: keys
                                     fit_a, fit_b, fit_c, Zcal,
                                     subleading_zeta_bound)
    s2_sector, delta_geq3, delta_sub, Delta_n, the SECTOR-STEP
    meter, the Gbar traces           -> sector_n0/_n1_<tag>.npz (stages 2a/2b)

HOW (in code order)
    I_n_from_delta(s2, delta)   I_n(s2_max) = -Int_0^{s2_max} delta_geq3 ds2,
                                log-Simpson scheme (THE production quadrature)
    low_s2_strip(s2, delta)     the [0, s2_min] strip, closed by the quadratic
                                through the three lowest nodes (both paths)
    high_s2_remainder_ratios(...) the s2-continuation of the high-s2 tail:
                                g_n(Lambda^2) = missing fraction above Lambda^2,
                                with GUARD SHAPE-ANCHOR (used only by
                                plot_computation/lnD_ren_parts_common: the
                                high-s2 and high-n tails of the lnD_ren
                                part figures against Lambda^2)
    high_s2_tail(...)     T_n(s2_max) per wave, the high-s2 tail
                                above s2_max: the exact s2-integral of the
                                uniform O(U_int^3) tail shape, times the
                                wave's own delta_geq3(n, s2_max) over the
                                shape's value there.  GUARD SHAPE-RATIO inside
    band_candidate_sums(...)    I_n(s2_max), T_n(s2_max) and I_n(inf) of every
                                band wave at the trailing candidate cutoffs of
                                the s2 grid, and the two band sums the s2
                                watcher fits; at the full cutoff these are the
                                production numbers stage 3 stores
    odd_family_high_n_fit(...)  high-n tail: fit I_n(inf) ~ a/nu^3 + b/nu^5
                                + c/nu^7 on [fit_lo, fit_hi] + Hurwitz-zeta
                                tail Zcal (enters lnD_ren,
                                tail_high_n_zeta_vgeneral_phi_v2).  GUARD
                                N-WINDOW
    sector_s2_lo_default(...)   model-derived default of the sector grid floor
    sector_s2_grid(...)         the resolvent sector s2 grid (plain geometric,
                                or pole-clustered when the pole is in range).
                                GUARD SECTOR-NODES
    sector_traces_of_wave(...)  the engine's traces of ONE wave on the
                                sector grid (delta_geq3 + the Gbar traces),
                                node-memoised across the node-doubling
                                retries.  GUARD SECTOR-TRACE
    pole_subtracted_sector_integral(...) / adaptive_sector_integral(...)
                                Delta_n = -Int [delta_geq3 -
                                deg_n/(omega2+s2)] ds2, with the SECTOR-STEP
                                halving meter; adaptive = node-doubling
                                retry.  GUARD SECTOR-POLESUB

DEPENDENCY: one-way pipeline_quadrature -> pipeline_helpers (require_finite);
pipeline_helpers must NOT import this module.
"""
import time as _time
import numpy as np
from scipy.integrate import simpson
from scipy.special import zeta

from pipeline_helpers_vgeneral_phi_v2 import require_finite


# --------------------------------------------------------------------------- #
#  Band s2 quadrature: the LOG-SIMPSON scheme                                  #
# --------------------------------------------------------------------------- #
def I_n_from_delta(s2, delta):
    """I_n(s2_max) = -Int_0^{s2_max} delta ds2 on the existing grid, where
    delta is the trace delta_geq3(n, s2) (a sector passes its delta_sub).
    The MINUS: lnD(s2) = ln[det(M_b + s2)/det(M_FV + s2)] has d lnD/ds2 =
    Tr[G(s2) - G_FV(s2)], whose Born orders >= 3 in wave n are delta_geq3;
    that part of lnD(s2) vanishes as s2 -> inf, so a wave n >= 2 (no mode
    pole) contributes -Int_0^inf delta_geq3 ds2 to it at s2 = 0.
    Log-Simpson scheme:
      [s2_min, s2_max]  Simpson in u = ln s2 of delta*s2 (ds2 = s2 du): a
                        geometric s2 grid (the band ladder's fine bottom, a
                        plain sector grid) is evenly spaced in u, and delta*s2
                        stays smooth in u across all decades of s2,
      [0, s2_min]       low_s2_strip (quadratic through the 3 lowest nodes;
                        delta(0) is a finite plateau, real area if dropped).
    The rule carries no s2 scale, so it is the same for every model.  The
    per-run check is the S2-STEP grid-halving meter
    (band_integrals_vgeneral_phi_v2).  s2 must be 1-D, ascending and > 0;
    delta holds the values on it (last axis if multi-dimensional)."""
    s2 = np.asarray(s2, float)
    delta = np.asarray(delta, float)
    if (s2.ndim != 1 or s2.size < 3 or not np.all(np.isfinite(s2))
            or s2[0] <= 0.0 or np.any(np.diff(s2) <= 0.0)):
        raise RuntimeError('[ABORT] I_n_from_delta needs an ascending 1-D s2 '
                           'grid of finite values > 0 with >= 3 points.')
    I_log_simpson = simpson(delta * s2, x=np.log(s2), axis=-1)
    return -(I_log_simpson + low_s2_strip(s2, delta))


def low_s2_strip(s2, delta):
    """Int_0^{s2[0]} delta ds2 -- the strip BELOW the first grid point, closed
    by the exact integral of the quadratic through the three lowest nodes.
    delta(0) is finite (low-s2 plateau), so the strip is ordinary area.
    Why a quadratic, not a rectangle: a rectangle misses the strip's
    curvature; the quadratic through nodes 0,1,2 is what the [::2] meter can
    see (nodes 0,1,2 on the fine grid against 0,2,4 on the half grid).
    The quadratic extrapolates below s2[0], so the strip must hold no
    structure: the band delta_geq3 is flat there, and a sector integrates
    delta_sub (its pole removed) on a floor at least a decade below any
    resolved pole scale |omega2| (sector_s2_lo_default)."""
    x0, x1, x2 = float(s2[0]), float(s2[1]), float(s2[2])

    def moment(a, b):                  # Int_0^{x0} (s2-a)(s2-b) ds2
        return x0**3 / 3.0 - 0.5 * (a + b) * x0**2 + a * b * x0

    w0 = moment(x1, x2) / ((x0 - x1) * (x0 - x2))    # Lagrange weights of the
    w1 = moment(x0, x2) / ((x1 - x0) * (x1 - x2))    # three lowest nodes
    w2 = moment(x0, x1) / ((x2 - x0) * (x2 - x1))
    return w0 * delta[..., 0] + w1 * delta[..., 1] + w2 * delta[..., 2]


# --------------------------------------------------------------------------- #
#  High-s2 tail of the band: the uniform O(U_int^3) shape ratios               #
#  (from R, tr_U_int3, m_min2; band_candidate_sums uses them for T_n(s2_max))  #
# --------------------------------------------------------------------------- #
# GUARD SHAPE-RATIO's tolerance, and the ONE owner of it: a moment of the
# sign-indefinite tr_U_int3 counts as CANCELLED when less than this fraction of
# its unsigned size survives (GUARD SHAPE-ANCHOR below applies the same rule
# to the r^5 twin of the same moment).
SHAPE_COND_TOL = 0.05


def _shape_moments(nu_values, s2_cutoff, m_min2, R, tr_U_int3):
    """The two radial moments of the uniform O(U_int^3) tail shape at the
    cutoff s2_cutoff, and the unsigned r^7 moment (its cancellation scale),
    for every wave of nu_values at once (one array entry per wave):
        moment_r5     = Int dr r^5 tr_U_int3 chi^-5 = Int dr tr_U_int3 q^-5
        moment_r7     = Int dr r^7 tr_U_int3 chi^-7 = Int dr tr_U_int3 q^-7
        moment_r7_abs = Int dr |r^7 tr_U_int3 chi^-7|
    chi(r) = sqrt(nu^2 + (s2_cutoff + m_min2) r^2) = r q(r), one row per wave;
    q(r) = sqrt(nu^2/r^2 + s2_cutoff + m_min2) is the local decay rate of the
    free wave, G_n^fv(r, r') ~ exp(-q |r - r'|) (the chi form stays finite
    at r = 0).  tr_U_int3 can change sign inside the wall, so
    |moment_r7|/moment_r7_abs measures how much of moment_r7 survives the
    cancellation -- what GUARD SHAPE-RATIO gates on."""
    chi = np.sqrt(np.asarray(nu_values, float)[:, None] ** 2
                  + (s2_cutoff + m_min2) * R**2)          # (waves, radii)
    integrand_r5_chi5 = R**5 * tr_U_int3 * chi**-5
    integrand_r7_chi7 = R**7 * tr_U_int3 * chi**-7
    return (np.trapezoid(integrand_r5_chi5, R, axis=-1),
            np.trapezoid(integrand_r7_chi7, R, axis=-1),
            np.trapezoid(np.abs(integrand_r7_chi7), R, axis=-1))


def high_s2_remainder_ratios(nu_values, s2_cutoff, L2_grid, m_min2, R, tr_U_int3):
    """The s2-continuation of the band's high-s2 tail: for every wave nu and
    every cutoff Lambda^2 in L2_grid,
        g_n(Lambda^2) = moment_r5(Lambda^2) / moment_r5(s2_max),
    moment_r5 = the r^5 moment of the uniform tail shape (_shape_moments),
    i.e. the fraction of each wave's high-s2 tail still missing above
    Lambda^2 (g = 1 at s2_max, -> 0 as Lambda^2 -> inf).
    GUARD SHAPE-ANCHOR (the r^5 twin of GUARD SHAPE-RATIO, same tolerance):
    tr_U_int3 can change sign inside the wall, so the anchor moment_r5(s2_max)
    can cancel; a mostly-cancelled anchor (|moment_r5| < SHAPE_COND_TOL x
    its unsigned size) makes the ratio meaningless for that wave, and g_n is
    repaired from the neighbouring waves.  Aborts when EVERY anchor cancels.
    Returns (remainder_ratio_n (len(L2_grid), len(nu_values)),
    anchor_repaired (len(nu_values),) bool)."""
    nu_values = np.asarray(nu_values, float)
    moment_r5_at_s2_max = _shape_moments(nu_values, float(s2_cutoff), m_min2, R,
                                         tr_U_int3)[0]
    # the UNSIGNED anchor moment: r^5 chi^-5 > 0, so integrating |tr_U_int3| with
    # the same kernel gives Int |r^5 tr_U_int3 chi^-5| -- the cancellation scale
    # GUARD SHAPE-ANCHOR gates on (the r^5 twin of moment_r7_abs in _shape_moments).
    moment_r5_abs = _shape_moments(nu_values, float(s2_cutoff), m_min2, R,
                                   np.abs(tr_U_int3))[0]
    anchor_repaired = ~np.isfinite(moment_r5_at_s2_max) | (np.abs(moment_r5_at_s2_max) < SHAPE_COND_TOL * moment_r5_abs)
    if anchor_repaired.all():
        raise RuntimeError('[ABORT SHAPE-ANCHOR] the r^5 tr_U_int3 anchor moment cancels '
                           'for EVERY wave -- the uniform tail shape cannot '
                           'continue this band.')
    remainder_ratio_n = np.empty((len(L2_grid), len(nu_values)))
    for i, L2 in enumerate(L2_grid):
        remainder_ratio_at_L2 = (_shape_moments(nu_values, float(L2), m_min2, R,
                                                tr_U_int3)[0] / moment_r5_at_s2_max)
        remainder_ratio_at_L2[anchor_repaired] = np.interp(nu_values[anchor_repaired], nu_values[~anchor_repaired],
                                    remainder_ratio_at_L2[~anchor_repaired])
        if np.any(~np.isfinite(remainder_ratio_at_L2)):
            raise RuntimeError(f'[ABORT SHAPE-ANCHOR] non-finite tail remainder ratio at '
                               f'Lambda^2 = {float(L2):g} (fail-loud).')
        remainder_ratio_n[i] = remainder_ratio_at_L2
    return remainder_ratio_n, anchor_repaired


def high_s2_tail(nu_values, s2_cutoff, m_min2, R, tr_U_int3,
                       delta_at_cutoff, quiet=False):
    """The high-s2 tail T_n = -Int_{s2_cutoff}^inf delta_geq3 ds2 of
    every wave of nu_values (nu = n+1) above the cutoff s2_cutoff.  Above the
    cutoff delta_geq3 is given the s2-dependence of its third Born order, with
    the free waves in their uniform large-nu / large-s2 form and the lightest
    mass m_min2 in every channel,
        shape_n(s2) = -(5/32) nu^2 Int dr r^7 tr_U_int3 chi^-7,
        chi(r)      = sqrt(nu^2 + (s2 + m_min2) r^2),
    scaled to the wave's own computed value delta_at_cutoff =
    delta_geq3(n, s2_cutoff) (only the SHAPE of the formula is used):
        T_n         = -delta_at_cutoff * shape_ratio,
        shape_ratio = Int_{s2_cutoff}^inf shape_n ds2 / shape_n(s2_cutoff)
                    = (2/5) moment_r5 / moment_r7          (units of s2),
    with the moments of _shape_moments at s2_cutoff (the 2/5 and r^7 -> r^5
    come from the exact s2-integral; nothing is fitted).  shape_ratio may be
    NEGATIVE: tr_U_int3 can change sign inside the wall, and where moment_r5
    and moment_r7 have opposite signs the shape changes sign above the
    cutoff -- a correct value.
    band_candidate_sums calls this at every candidate cutoff; at the full
    cutoff the result is the production high-s2 tail
    band_integrals_vgeneral_phi_v2 stores.

    ---- GUARD SHAPE-RATIO (repairs + logs loudly; never a silent gate) ----
    Where the denominator moment_r7 cancels (|moment_r7| < SHAPE_COND_TOL *
    moment_r7_abs, or a non-finite ratio), shape_n(s2_cutoff) is near zero
    and cannot be scaled to delta_at_cutoff (the ratio has a pole there).
    Such a wave takes the analytic amplitude instead, delta_at_cutoff ->
    shape_n(s2_cutoff), so that
        T_n = -Int_{s2_cutoff}^inf shape_n ds2 = nu^2 moment_r5 / 16
    (at s2_cutoff -> 0 and large nu this is A3/nu^3; printed per repair).
    The repair is safe only while the flagged waves carry a negligible
    share of the tail sum; the REPAIR-SHARE guard in
    band_integrals_vgeneral_phi_v2 measures that share and aborts above its
    ceiling.  Aborts when EVERY wave is flagged.

    quiet=True suppresses only the per-repair lines (same flags, same
    repair, same abort): band_candidate_sums passes it below the full
    cutoff and prints the repair count once; the plot helper
    lnD_ren_parts_common passes it for its one re-check of the stored
    high-s2 tail at s2_max.
    Returns (T_n, repaired): the high-s2 tail of every wave and the
    flags of the repaired waves."""
    nu_values = np.asarray(nu_values, float)
    delta_at_cutoff = np.asarray(delta_at_cutoff, float)
    moment_r5, moment_r7, moment_r7_abs = _shape_moments(nu_values, s2_cutoff, m_min2, R, tr_U_int3)
    shape_ratio = (2.0 / 5.0) * moment_r5 / moment_r7
    moment_r7_survival_fraction = np.abs(moment_r7) / moment_r7_abs
    repaired = ~np.isfinite(shape_ratio) | (moment_r7_survival_fraction < SHAPE_COND_TOL)
    if repaired.all():
        raise RuntimeError(
            f'[ABORT SHAPE-RATIO] the high-s2 tail shape ratio is unusable for '
            f'EVERY one of the {nu_values.size} waves at s2_max={s2_cutoff:g} (largest '
            f'|moment_r7|/moment_r7_abs = {moment_r7_survival_fraction.max():.2e}): '
            f'the uniform O(U^3) tail shape does not calibrate on this '
            f'bounce -- inspect tr(U_int^3).')
    T_n = -delta_at_cutoff * shape_ratio
    if repaired.any():
        T_n[repaired] = (nu_values**2 * moment_r5 / 16.0)[repaired]
        if not quiet:           # a probe call repairs the same way, silently
            for k in np.flatnonzero(repaired):
                print(f'[GUARD SHAPE-RATIO] s2_max={s2_cutoff:g} wave '
                      f'n={int(nu_values[k])-1}: shape ratio = {shape_ratio[k]:+.6g} '
                      f'rejected (|moment_r7|/moment_r7_abs = {moment_r7_survival_fraction[k]:.2e}, the '
                      f'tr(U_int^3) moment cancels) -> high-s2 tail {T_n[k]:+.6g} '
                      f'from the analytic amplitude', flush=True)
    return T_n, repaired


def band_candidate_sums(s2_values, delta_table, band_n, m_min2, R, tr_U_int3,
                        n_cand=6):
    """The band s2 integrals, their high-s2 tails and the candidate
    sums of the s2 watcher/guard pair -- computed ONCE here, so S2-PLATEAU
    (band_adaptive) and S2-BAACKE (band_integrals_vgeneral_phi_v2) gate
    on the numbers that enter lnD_ren.  For each trailing candidate cutoff
    L2 (the last n_cand ladder points in the top 65% of the grid -- the
    points watch_s2_plateau fits; the LAST candidate is the full grid,
    L2 = s2_max) the SAME band table delta_table
    (n_waves, n_slices of delta_geq3, aligned with (band_n, s2_values)) is integrated
    (I_n_from_delta) and completed (high_s2_tail) at that cutoff:
        I_n_L2      = -Int_0^{L2} delta_geq3 ds2                  per wave,
        T_n_L2      = high_s2_tail at L2                    per wave,
        I_n_inf_L2  = I_n_L2 + T_n_L2                             per wave,
        band_sum_to_cutoff(L2) = sum_n I_n_L2,   band_sum_with_high_s2_tail(L2) = sum_n I_n_inf_L2.
    The full-cutoff candidate IS the production computation: its per-wave
    arrays come back as `per_wave_at_s2_max` = dict(I_n_s2_max,
    shape_ratio_repaired, T_n_s2_max, I_n_inf), which band_integrals stores and the
    n-direction watchers read -- the ONE creation point of I_n(inf).  The
    probe cutoffs below it feed the two candidate sums only.
    Returns (s2_cutoff_candidates, band_sum_to_cutoff, band_sum_with_high_s2_tail, per_wave_at_s2_max)."""
    s2_values = np.asarray(s2_values, float)
    delta_table = np.asarray(delta_table, float)
    nu = np.asarray(band_n, float) + 1.0
    # trailing candidates: the last n_cand grid points in the top 65% of the
    # grid (a dimensionless fraction)
    i_cutoff_candidates = [k for k in range(len(s2_values)) if s2_values[k] >= 0.35 * s2_values[-1]][-n_cand:]
    # each candidate re-integrates s2_values[:k+1], which needs >= 3 grid points
    i_cutoff_candidates = [k for k in i_cutoff_candidates if k >= 2]
    if not i_cutoff_candidates:
        raise RuntimeError('[ABORT] the band grid is too small for the s2 '
                           'watcher (every candidate cutoff needs >= 3 grid '
                           'points below it) -- extend the s2 grid.')
    i_full_grid = len(s2_values) - 1     # the last candidate: the full grid, L2 = s2_max
    band_sum_to_cutoff, band_sum_with_high_s2_tail, n_repair = [], [], 0
    per_wave_at_s2_max = None
    for k in i_cutoff_candidates:
        L2 = s2_values[k]                     # the candidate cutoff Lambda^2
        I_n_L2 = I_n_from_delta(s2_values[:k + 1], delta_table[:, :k + 1])
        # GUARD SHAPE-RATIO inside high_s2_tail: its per-repair lines
        # are printed for the production cutoff only; the probe cutoffs are
        # quiet and their repair COUNT is the one line printed below
        T_n_L2, shape_ratio_repaired_L2 = high_s2_tail(       # tail above L2
            nu, L2, m_min2, R, tr_U_int3, delta_table[:, k], quiet=(k != i_full_grid))
        if k != i_full_grid:
            n_repair += int(np.count_nonzero(shape_ratio_repaired_L2))
        I_n_inf_L2 = I_n_L2 + T_n_L2
        if k == i_full_grid:                  # the production cutoff: these ARE stage 3's arrays
            per_wave_at_s2_max = dict(I_n_s2_max=I_n_L2,
                                      shape_ratio_repaired=shape_ratio_repaired_L2,
                                      T_n_s2_max=T_n_L2, I_n_inf=I_n_inf_L2)
        band_sum_to_cutoff.append(float(np.sum(I_n_L2)))
        band_sum_with_high_s2_tail.append(float(np.sum(I_n_inf_L2)))
    if n_repair:
        print(f'[GUARD SHAPE-RATIO] {n_repair} wave/cutoff pair(s) repaired across '
              f'the {len(i_cutoff_candidates) - 1} PROBE cutoffs of the s2 watcher ladder '
              f'(per-repair lines suppressed; the production cutoff logs its '
              f'own)', flush=True)
    return s2_values[i_cutoff_candidates], np.array(band_sum_to_cutoff), np.array(band_sum_with_high_s2_tail), per_wave_at_s2_max


# --------------------------------------------------------------------------- #
#  High-n tail of the band: the odd-family fit + Hurwitz-zeta sum             #
#  (from I_n(inf) of the band waves; tail_high_n_zeta stores a, b, c, Zcal)    #
# --------------------------------------------------------------------------- #
N_WINDOW_MIN_WAVES = 12   # GUARD N-WINDOW: the fewest waves a fit window may
                          # hold (3-parameter fit) -- the one number the band
                          # driver's n stop and stage 4 share


def odd_family_high_n_fit(band_n, I_n_inf, fit_lo, fit_hi):
    """The high-n tail scheme (companion of high_s2_tail, the high-s2
    tail): the odd-family high-n least-squares fit
    on the window [fit_lo, fit_hi] (nu = n+1),

        I_n(inf) ~ a/nu^3 + b/nu^5 + c/nu^7,

    (only the odd powers occur), and the closed-form Hurwitz-zeta tail over
    ALL waves beyond fit_hi.  All three coefficients are fitted.  Only a has
    an analytic prediction, A3 (third Born order; compared by N-ASYMPTOTE);
    b and c carry the window's remaining approach to the asymptote
    (interpolation coefficients).
    Returns (a, b, c, Zcal, subleading_zeta_bound) with
        Zcal = a z3 + b z5 + c z7,        z_k = zeta(k, fit_hi + 2)
        subleading_zeta_bound = |b| z5 + |c| z7
    (the missing waves have nu = fit_hi+2, fit_hi+3, ...).  `Zcal` ENTERS
    lnD_ren (tail_high_n_zeta_vgeneral_phi_v2); subleading_zeta_bound is only
    what the N-TAIL-BOUND watcher/guard gate on.  `I_n_inf` is the 1-D
    sequence of the s2-completed per-wave numbers I_n(inf), one per wave of
    band_n.
    ---- GUARD N-WINDOW (aborts; does NOT enter lnD): the window must be ----
    ---- contiguous, hold >= N_WINDOW_MIN_WAVES waves, and the 3-column  ----
    ---- design matrix must come back with full rank 3.                  ----"""
    band_n = np.asarray(band_n, int)
    in_window = (band_n >= fit_lo) & (band_n <= fit_hi)
    window_n = band_n[in_window]
    if window_n.size < N_WINDOW_MIN_WAVES:
        raise RuntimeError(
            f"[ABORT N-WINDOW] fit window [{fit_lo},{fit_hi}] holds {window_n.size} "
            f"waves < N_WINDOW_MIN_WAVES={N_WINDOW_MIN_WAVES} -- too few for a stable "
            f"3-parameter fit; generate deeper waves.")
    if not np.array_equal(window_n, np.arange(fit_lo, fit_hi + 1)):
        raise RuntimeError(
            f"[ABORT N-WINDOW] fit window [{fit_lo},{fit_hi}] is not "
            f"contiguous ({window_n.size}/{fit_hi - fit_lo + 1} waves present) -- "
            f"a missing wave would bias the fit.")
    nu_values = window_n + 1.0                                     # the Bessel order
    nu_power_columns = np.column_stack([nu_values**-3.0, nu_values**-5.0, nu_values**-7.0])   # one column per power
    # The columns nu^-3, nu^-5, nu^-7 differ by the factors nu^2 and nu^4, a
    # SCALING disparity, not a real degeneracy: on a deep window it can make
    # lstsq drop the third column (rank 2) and silently lose c.  Dividing each
    # column by its first entry removes the disparity; the least-squares
    # solution is unchanged up to rounding (a column rescaling, undone on the
    # coefficients below).
    scale = nu_power_columns[0].copy()
    I_n_inf_window = np.asarray(I_n_inf, float)[in_window]
    fit_coeffs_scaled, _, rank, _ = np.linalg.lstsq(nu_power_columns / scale, I_n_inf_window, rcond=None)
    if rank < 3:
        raise RuntimeError(
            f"[ABORT N-WINDOW] the odd-family high-n fit's design matrix on "
            f"[{fit_lo},{fit_hi}] came back with rank {rank} < 3 even after "
            f"column scaling: the window cannot separate nu^-3, nu^-5 and "
            f"nu^-7 (it is too deep and/or too short) -- widen it "
            f"(lower fit_lo).")
    a, b, c = fit_coeffs_scaled / scale                                # unscaled
    z3, z5, z7 = (float(zeta(k, fit_hi + 2.0)) for k in (3, 5, 7))
    Zcal = a * z3 + b * z5 + c * z7
    subleading_zeta_bound = abs(b) * z5 + abs(c) * z7
    return (float(a), float(b), float(c), float(Zcal),
            float(subleading_zeta_bound))


# --------------------------------------------------------------------------- #
#  Production resolvent sector grid + integral (shared by n = 0 and n = 1)     #
# --------------------------------------------------------------------------- #
def sector_s2_lo_default(m_min2, omega2):
    """Default floor sector_s2_lo of the sector s2 grid, derived from the
    model (s2 carries mass^2 units, so an absolute floor would presuppose
    the model's mass scale).  The lower of two scales:
      - 1e-4 * m_min2, a fixed fraction of the lightest channel mass^2
        (kappa_i^2 = s2 + m_i^2);
      - |omega2|/10, taken ONLY when the eigenvalue is resolved above that
        floor (|omega2| > 1e-4 m_min2): the pole scale s2 ~ |omega2| then
        lies at least a decade above the floor, on the grid (pole-clustered by
        sector_s2_grid when omega2 < 0), not inside the extrapolated
        [0, sector_s2_lo] strip (low_s2_strip).
    A numerically zero eigenvalue (the n=1 zero mode of a good bounce,
    omega2_10 = 0 to bounce error) keeps the mass-scaled floor: |omega2_10|/10
    would put the floor at noise level, where det C ~ 0 makes delta_geq3 noise.
    --sector-s2-lo overrides this default."""
    lo = 1e-4 * float(m_min2)
    if abs(float(omega2)) > lo:
        lo = min(lo, abs(float(omega2)) / 10.0)
    return float(lo)


def sector_s2_grid(sector_s2_lo, s2_max, n_nodes, pole):
    """The resolvent sector s2 grid on [sector_s2_lo, s2_max].  `pole` is
    the s2 LOCATION of the subtracted pole, pole = -omega2 (the subtraction
    deg_n/(omega2 + s2) blows up there).

    PLAIN (the pole not inside the range with the clearance pole_clearance
    = pole/100 on BOTH sides): one geometric grid, integrated by
    I_n_from_delta.  POLE-CLUSTERED (the pole inside with that clearance --
    the n=0 negative mode; the n=1 zero mode only when omega2_10 < 0 puts
    its pole there): split AT the pole, geometric in the DISTANCE |s2 - pole| per
    side, innermost node pole_clearance away from the pole so the
    delta_geq3 ~ deg_n/(omega2+s2) near-cancellation keeps float64 precision.
    Which branch a sector takes is decided HERE, per run, by that run's
    eigenvalue.

    The clearance keeps each side's geomspace ascending (the
    strictly-ascending abort below guards the grid).  Node counts: n_nodes
    arrives = 1 mod 4 from adaptive_sector_integral (the owner of the node
    count and of its 2N-1 doubling), so the plain grid is odd and each
    clustered side holds the odd (n_nodes+1)/2: every [::2] half grid keeps
    the innermost pole node and the endpoint, and every doubled grid
    contains the previous one.
    Returns (s2_sector, i_above_pole): None (plain) or the index of the first
    above-pole node."""
    s2_max = float(s2_max)
    if s2_max <= 0.0:
        raise RuntimeError('[ABORT] s2_max must be > 0.')
    n_nodes = int(n_nodes)
    if n_nodes < 61:
        raise RuntimeError('[ABORT SECTOR-NODES] --sector-nodes must be >= 61 '
                           '(production-grade resolvent sector grid).')
    pole_clearance = 1e-2 * pole      # distance of the innermost clustered nodes from the pole
    plain = not (sector_s2_lo + pole_clearance < pole < s2_max - pole_clearance)
    if plain:
        s2_sector = np.geomspace(sector_s2_lo, s2_max, n_nodes)
        s2_sector[-1] = s2_max          # exact endpoint (the fict logs use it)
        i_above_pole = None
    else:
        # pole-clustered: geometric in |s2 - pole|, split at the pole (the small
        # jump in the subtracted integrand then sits ON the split, not straddled)
        n_each = (n_nodes + 1) // 2       # nodes per side: odd, n_nodes = 1 mod 4
        below = (pole - np.geomspace(pole_clearance, pole - sector_s2_lo, n_each))[::-1]
        below[0] = sector_s2_lo      # exact floor (pole - (pole - floor) rounds)
        above = pole + np.geomspace(pole_clearance, s2_max - pole, n_each)
        above[-1] = s2_max           # exact endpoint (the fict logs use it)
        s2_sector = np.concatenate([below, above])
        i_above_pole = int(len(below))
    # every quadrature below (and every meter) assumes a strictly ascending grid
    # that ends at the cutoff: fail loud rather than integrate a folded grid
    if np.any(np.diff(s2_sector) <= 0.0):
        raise RuntimeError(
            f'[ABORT SECTOR-GRID] the sector s2 grid is not strictly ascending '
            f'({int(np.sum(np.diff(s2_sector) <= 0.0))} of {len(s2_sector)-1} steps '
            f'non-positive; range {s2_sector.min():g}..{s2_sector.max():g} for '
            f'[{sector_s2_lo:g}, {s2_max:g}], pole={pole}) -- Simpson on it '
            f'would return a silently wrong integral.')
    return s2_sector, i_above_pole


def _sector_integral_pole_split(s2_sector, delta_sub, i_above_pole, pole):
    """-Int delta_sub ds2 on a POLE-SPLIT clustered grid.  Each side is
    geometric in the distance |s2 - pole|, so each side takes the log rule of
    I_n_from_delta with the pole as origin: Simpson in u = ln|s2 - pole| of
    delta_sub (s2 - pole)  (ds2 = (s2 - pole) du on both sides).  The sliver
    [pole - pole_clearance, pole + pole_clearance] between the two innermost
    nodes is closed by their trapezoid, the [0, sector_s2_lo] strip by
    low_s2_strip (the SAME strip the plain path adds).  The tiny jump AT the
    pole sits ON the split, so Simpson keeps high order.  Same -Int sign as
    I_n_from_delta."""
    s2_below_pole, s2_above_pole = s2_sector[:i_above_pole], s2_sector[i_above_pole:]
    delta_sub_below, delta_sub_above = delta_sub[:i_above_pole], delta_sub[i_above_pole:]
    I_below = simpson(delta_sub_below * (s2_below_pole - pole), x=np.log(pole - s2_below_pole))
    I_above = simpson(delta_sub_above * (s2_above_pole - pole), x=np.log(s2_above_pole - pole))
    sliver = 0.5 * (delta_sub_below[-1] + delta_sub_above[0]) * (s2_above_pole[0] - s2_below_pole[-1])   # across the pole
    return -(I_below + I_above + sliver + low_s2_strip(s2_sector, delta_sub))


def sector_traces_of_wave(engine, n, s2_sector, trace_memo):
    """The engine's coincident traces of wave n on the sector grid, ONE
    gbar_traces call per NEW node:
        delta_geq3(s2)  the Born-order >= 3 part, traced from that part
                        itself (= Gbar_minus_fv - Gbar_1 - Gbar_2, never
                        formed as that difference)
    plus Gbar_traces = {Gbar_minus_fv, Gbar_1, Gbar_2, Gbar_fv} arrays on the
    grid -- the same gbar_traces call returns them all (zero extra ODE
    cost); the sector stage stores them RAW, nothing here integrates them.
    `trace_memo` = {s2: {name: value}} node memo, keyed on s2 ALONE -- valid
    for ONE wave only (adaptive_sector_integral makes a fresh dict per call).
    The 2N-1 node-doubling retry NESTS exactly, so every old node is a memo
    hit instead of an h-ODE re-solve.  Returns (delta_geq3, Gbar_traces)."""
    n_nodes = len(s2_sector)
    Gbar_trace_keys = ('Gbar_minus_fv', 'Gbar_1', 'Gbar_2', 'Gbar_fv')
    keys = ('delta_geq3', *Gbar_trace_keys)
    delta_geq3 = np.empty(n_nodes)
    Gbar_traces = {key: np.empty(n_nodes) for key in Gbar_trace_keys}
    t_sec0 = _time.time()
    for k, s2_node in enumerate(s2_sector):
        s2_node = float(s2_node)
        if s2_node not in trace_memo:             # else: solved on a coarser grid
            traces_at_node = engine.gbar_traces(n, s2_node)
            trace_memo[s2_node] = {key: traces_at_node[key] for key in keys}
        traces_at_node = trace_memo[s2_node]
        delta_geq3[k] = float(traces_at_node['delta_geq3'])
        for key in Gbar_trace_keys:
            Gbar_traces[key][k] = float(traces_at_node[key])
        if (k + 1) % 80 == 0 or k == n_nodes - 1:
            print(f'       [SECTOR n={n}] node {k+1}/{n_nodes} '
                  f'(s2={s2_node:.6g})  [{_time.time()-t_sec0:.0f}s]', flush=True)
    require_finite(delta_geq3, f'[ABORT SECTOR-TRACE] non-finite '
                              f'delta_geq3(n={n}, s2) resolvent nodes')
    return delta_geq3, Gbar_traces


def pole_subtracted_sector_integral(n, omega2, s2_sector, delta_geq3, i_above_pole):
    """The production resolvent sector integral of wave n from its trace
    delta_geq3 (sector_traces_of_wave):
        delta_sub         = delta_geq3 - deg_n/(omega2 + s2)   (pole removed)
        Delta_n           = -Int delta_sub ds2
    `omega2` is the EIGENVALUE of the subtracted mode (omega2_00 < 0 for n=0,
    omega2_10 ~ 0 of either sign for n=1); the pole sits at s2 = -omega2 and
    the residue is the angular degeneracy deg_n = (n+1)^2 of the ONE crossing
    at that eigenvalue (the census selection gate admits one such crossing).  `i_above_pole` from
    sector_s2_grid: None -> plain grid via I_n_from_delta; int ->
    pole-clustered grid, the same log rule per side in the distance to the
    pole (_sector_integral_pole_split).
    Returns (delta_sub, Delta_n, sector_step_meter); the meter
    |Delta_n - Delta_n_half| (Delta_n_half =
    the same integral on the every-2nd-point grid) is the SECTOR-STEP
    estimate the CALLER gates on (adaptive_sector_integral owns the abort)."""
    deg = (n + 1.0) ** 2                     # (n+1)^2 angular degeneracy
    delta_sub = delta_geq3 - deg / (omega2 + s2_sector)   # pole removed
    require_finite(delta_sub, '[ABORT SECTOR-POLESUB] non-finite pole-subtracted '
                              'delta_geq3 (delta_sub)')
    if i_above_pole is None:                        # plain geometric grid
        Delta_n = float(I_n_from_delta(s2_sector, delta_sub))
        Delta_n_half = float(I_n_from_delta(s2_sector[::2],
                                            delta_sub[::2]))
    else:                                    # pole-clustered grid, log rule per side
        Delta_n = float(_sector_integral_pole_split(s2_sector, delta_sub,
                                                        i_above_pole, -omega2))
        below_half = np.arange(0, i_above_pole, 2)       # every 2nd node, per side
        above_half = np.arange(i_above_pole, len(s2_sector), 2)
        half_grid_idx = np.concatenate([below_half, above_half])
        Delta_n_half = float(_sector_integral_pole_split(
            s2_sector[half_grid_idx], delta_sub[half_grid_idx], len(below_half), -omega2))
    sector_step_meter = abs(Delta_n - Delta_n_half)
    return delta_sub, Delta_n, sector_step_meter


def adaptive_sector_integral(engine, n, omega2, sector_s2_lo, s2_max,
                             n_nodes_start, sector_step_tol, sector_nodes_cap=8001):
    """Build the (pole-aware) sector grid and integrate; if the SECTOR-STEP
    meter exceeds sector_step_tol, DOUBLE the node count and retry until it passes or
    sector_nodes_cap is reached.  `omega2` is the eigenvalue omega^2 of the subtracted
    mode, so the pole sits at the s2 location -omega2.  The grid needs only the
    pole's LOCATION.
    Any potential's pole -- wherever the eigenvalue puts it (low edge ->
    plain grid; mid-range -> pole-clustered) -- goes through this one loop;
    only a meter still above sector_step_tol at the cap aborts.  ONE node memo
    is carried across the retries, so a doubled grid only pays for the s2 values
    it did not already solve (see sector_traces_of_wave).  Returns
    (s2_sector, delta_geq3, delta_sub, Delta_n, sector_step_meter,
    Gbar_traces)."""
    pole = -omega2
    n_nodes = int(n_nodes_start)
    n_nodes += (1 - n_nodes) % 4   # up to N = 1 mod 4: sector_s2_grid's half grids
                                   # then keep their end nodes, and 2N-1 is again
                                   # 1 mod 4, so each doubled grid nests
    trace_memo = {}                    # s2 -> the traces at that node, shared by retries
    while True:
        s2_sector, i_above_pole = sector_s2_grid(sector_s2_lo, s2_max, n_nodes, pole)
        delta_geq3, Gbar_traces = sector_traces_of_wave(engine, n, s2_sector, trace_memo)
        delta_sub, Delta_n, sector_step_meter = pole_subtracted_sector_integral(
            n, omega2, s2_sector, delta_geq3, i_above_pole)
        kind = 'pole-clustered' if i_above_pole is not None else 'plain'
        print(f'       [SECTOR-STEP n={n}] {sector_step_meter:.3e} (limit {sector_step_tol:g}; '
              f'{kind} grid {len(s2_sector)} nodes)', flush=True)
        if sector_step_meter <= sector_step_tol:
            return (s2_sector, delta_geq3, delta_sub, Delta_n,
                    sector_step_meter, Gbar_traces)
        if n_nodes >= sector_nodes_cap:
            raise RuntimeError(
                f'[ABORT SECTOR-STEP] resolvent sector n={n} not converged at '
                f'{len(s2_sector)} nodes (meter {sector_step_meter:.3e} > '
                f'{sector_step_tol:g}); node '
                f'cap {sector_nodes_cap} reached -- raise it or check the bounce/eigenvalue.')
        # 2N-1, not 2N+1: this is the density doubling whose grid CONTAINS the
        # previous one (geomspace(lo,hi,2N-1)[::2] == geomspace(lo,hi,N)), so
        # every already-solved node is a memo hit instead of a repeat solve
        # (a count clipped at the cap no longer contains the previous grid).
        n_nodes = min(sector_nodes_cap, 2 * n_nodes - 1)
