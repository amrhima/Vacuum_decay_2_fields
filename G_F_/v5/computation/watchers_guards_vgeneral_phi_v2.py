#!/usr/bin/env python3
"""watchers_guards_vgeneral_phi_v2.py -- library of the band (used by
stages 1, 3, 4 and 7): the four watcher/guard pairs of the band cutoffs
(table under HOW).  Three pairs live here in full, with the two analytic
moments they test against; of the fourth, SECTOR-CUTOFF, the estimator
and its limit live here.  A WATCHER
compares a computed object against its asymptotic prediction while band
data is generated and DECIDES when to stop; a GUARD re-checks the same
condition on the final data and ABORTS on failure.  Watcher and guard
evaluate the SAME estimator: the three band pairs on the same band data
(each guard repeats a number the band driver's stop already tested),
SECTOR-CUTOFF on different grids (the watcher on the band s2 grid, the
guard on the sector grids of stages 2a/2b), so its two numbers differ
slightly and a run near the limit can pass the watcher and still abort
in the guard.  Their measurements (misses, ratios, predicted errors)
never enter lnD_ren; what the watchers decide are the band cutoffs s2_max,
n_max and the fit-window start fit_lo, which fix what does.

READS (arrays handed in by the calling stage; this module opens no file)
    the band sums at the candidate  band_candidate_sums (pipeline_quadrature),
    cutoffs, I_n(inf) per band wave via stages 1 and 3
    R, tr_U_int3, m_min2            load_background (pipeline_helpers)
    fit_a, fit_b, fit_c, the subleading zeta bound
                                    odd_family_high_n_fit (pipeline_quadrature),
                                    via stages 1 and 4
    delta_geq3(s2) of the sectors   the band's base slices (stage 1) and the
    n = 0, 1                        sector_n0/_n1 npz (stage 7)
WRITES (returned to the caller, who stores them)
    A3, baacke_c and their unsigned scales A3_scale, baacke_c_scale
                                    analytic_moments -> band_adaptive (stage
                                    1: the moment-cancellation aborts; A3,
                                    baacke_c and baacke_c_scale stored in
                                    band_cutoffs) and
                                    assemble_lnD (stage 7: A3 of the
                                    reference profile, for eps_bounce)
    the decisions s2_max, fit_lo, n_max   the watchers -> band_cutoffs npz
                                    (band_adaptive)
    the sector remainder bound      sector_high_s2_remainder_bound -> the
                                    SECTOR-CUTOFF watcher (stage 1) and
                                    guard (stage 7); never in lnD_ren
    aborts                          the guards, in stages 3 and 4

HOW
  quantity   watcher             guard (consuming stage)      labels printed
  --------   ------------------  ---------------------------  ------------------
  s2_max     watch_s2_plateau    guard_s2_baacke              S2-PLATEAU /
                                 (band_integrals_vgeneral_     S2-BAACKE
                                  phi_v2)
  fit_lo     watch_n_onset       guard_n_asymptote            N-ONSET /
                                 (tail_high_n_zeta_vgeneral_   N-ASYMPTOTE
                                  phi_v2)
  n_max      watch_n_max_needed  guard_n_tail_bound           N-TAIL-BOUND
                                 (tail_high_n_zeta_vgeneral_   (one label for
                                  phi_v2)                      both)
  s2_max     band_adaptive       assemble_lnD_vgeneral_phi_v2 SECTOR-CUTOFF
  (sectors   evaluate_watchers   (stage 7)                    (one label for
   n = 0, 1)                                                   both)
The SECTOR-CUTOFF estimator sector_high_s2_remainder_bound and its limit
SECTOR_REMAINDER_TOL live at the end of this module; its watcher runs in
band_adaptive (evaluate_watchers, on the n = 0, 1 traces of the band's base
slices) and its guard in assemble_lnD (on the sector grids of sector_n0/_n1).
The one guard that CORRECTS instead of aborting, SHAPE-RATIO, lives with the
high-s2 tail it repairs (pipeline_quadrature_vgeneral_phi_v2.high_s2_tail,
called from band_candidate_sums); its repaired value IS that wave's
T_n(s2_max).  Every other model-dependent quantity is determined adaptively
where it lives and gated by its own guard there (S2-STEP, SECTOR-STEP,
RADIAL-GRID, BAND-EIG, MANIFEST, ...).

THE TWO ANALYTIC LAWS of the three band pairs (S2-PLATEAU/S2-BAACKE test the
s2 law; N-ONSET/N-ASYMPTOTE the n law, whose fitted subleading terms
N-TAIL-BOUND bounds):
  n-direction:   I_n(inf) -> A3/nu^3,   A3 = (1/16) Int dr r^5 tr_U_int3
  s2-direction:  sum_n I_n(s2_max) -> const + baacke_c/(s2_max + m_min2),
                 baacke_c = -(1/48) Int dr r^3 tr_U_int3  (the Baacke law of
                 the raw band sum; baacke_c/(s2_max + m_min2) is minus the
                 third-Born high-s2 tail summed over all waves)
SECTOR-CUTOFF tests no analytic law: its estimator fits a power law to the
sector traces themselves.
Both moments belong to the bounce the band integrates (analytic_moments).
The Baacke law holds when the fitted s2 coefficient lies within half the
unsigned moment baacke_c_scale of baacke_c (baacke_law_holds,
BAACKE_MISS_TOL); the n law is judged relative to A3 (N-ONSET:
nu^3 I_n(inf)/A3, N-ASYMPTOTE: fit_a/A3).  Both moments can cancel
(tr_U_int3 can change sign):
band_adaptive aborts before generating anything when either moment is below
MOMENT_REL_TOL of its unsigned scale, so every watcher and guard below works
with a resolved moment.
"""
import numpy as np
from scipy.special import zeta


# --------------------------------------------------------------------------- #
#  Analytic guard objects (bounce moments; the prediction side of every        #
#  watcher/guard)                                                              #
# --------------------------------------------------------------------------- #
def analytic_moments(R, tr_U_int3):
    """The analytic asymptotic-regime objects of THIS bounce, the moments of
    tr_U_int3 = tr(U_int^3):
        A3       = (1/16) Int dr r^5 tr_U_int3     (n-direction law)
        A3_scale = the same with |tr_U_int3|       (unsigned moment: tr_U_int3
                   can change sign, so |A3|/A3_scale says whether a small A3
                   is real or a cancellation -- the band driver's A3 test)
        baacke_c = -(1/48) Int dr r^3 tr_U_int3    (s2-direction Baacke law)
        baacke_c_scale = the same with |tr_U_int3| (its cancellation scale --
                   the band driver's baacke_c test, exactly as A3_scale)
    """
    R = np.asarray(R, float)
    tr_U_int3 = np.asarray(tr_U_int3, float)
    A3 = np.trapezoid(R**5 * tr_U_int3, R) / 16.0
    A3_scale = np.trapezoid(R**5 * np.abs(tr_U_int3), R) / 16.0
    baacke_c = -np.trapezoid(R**3 * tr_U_int3, R) / 48.0
    baacke_c_scale = np.trapezoid(R**3 * np.abs(tr_U_int3), R) / 48.0
    return dict(A3=float(A3), A3_scale=float(A3_scale), baacke_c=float(baacke_c),
                baacke_c_scale=float(baacke_c_scale))


# --------------------------------------------------------------------------- #
#  WATCHER S2-PLATEAU / GUARD S2-BAACKE  (quantity: s2_max)                    #
# --------------------------------------------------------------------------- #
def watch_s2_plateau(s2_cutoff_candidates, band_sum_with_high_s2_tail, band_sum_to_cutoff, baacke_c,
                     baacke_c_scale, m_min2):
    """WATCHER S2-PLATEAU (decides s2_max; never enters lnD; guard_s2_baacke
    re-checks exactly this on the final tables).  Over the candidate cutoffs
    L2 of band_candidate_sums, with x = 1/(L2 + m_min2) and the envelope
    baacke_envelope = |baacke_c| x:
        baacke_c_measured   = d band_sum_to_cutoff / dx: must be ~ baacke_c,
                              i.e. the raw band sum follows the Baacke law
                              const + baacke_c x (asymptotic-regime signal);
        baacke_c_miss       = |baacke_c_measured - baacke_c| / baacke_c_scale:
                              how far the fitted slope lies from baacke_c, in
                              units of the unsigned moment (baacke_law_holds);
        completed_sum_drift = |d band_sum_with_high_s2_tail / d baacke_envelope|:
                              the drift left in the completed sum per unit
                              envelope;
        s2_trunc_err_pred   = completed_sum_drift * baacke_envelope(s2_max):
                              the predicted remaining s2-truncation error in
                              lnD.
    Slopes are least-squares over >= 3 candidates: the band ladder
    (band_adaptive.s2_ladder) has a node ratio of at most 1.25, so the top
    65% of any grid of >= 5 points holds >= 5 nodes and band_candidate_sums
    keeps >= 3 of them (its k >= 2 filter); on a shorter grid the watcher is
    undecidable and returns miss nan / error inf, so the driver extends and
    the guard aborts.  band_adaptive aborts on a cancelled baacke_c before any
    band exists, so baacke_c != 0 here.
    Returns dict(s2_trunc_err_pred, baacke_c_miss, completed_sum_drift,
    baacke_envelope)."""
    s2_cutoff_candidates = np.asarray(s2_cutoff_candidates, float)
    band_sum_with_high_s2_tail = np.asarray(band_sum_with_high_s2_tail, float)
    band_sum_to_cutoff = np.asarray(band_sum_to_cutoff, float)
    if s2_cutoff_candidates.size < 3:
        return dict(s2_trunc_err_pred=np.inf, baacke_c_miss=np.nan, completed_sum_drift=np.inf,
                    baacke_envelope=np.inf)
    # polyfit(x, y, 1)[0] = the least-squares slope dy/dx
    x = 1.0 / (s2_cutoff_candidates + m_min2)
    baacke_envelope = abs(baacke_c) * x
    baacke_c_measured = float(np.polyfit(x, band_sum_to_cutoff, 1)[0])
    completed_sum_drift = abs(float(np.polyfit(baacke_envelope, band_sum_with_high_s2_tail, 1)[0]))
    s2_trunc_err_pred = completed_sum_drift * baacke_envelope[-1]
    # the finite-n and finite-cutoff corrections to the slope are set by the
    # UNSIGNED structure of tr U^3, not by baacke_c, so the miss is measured in
    # units of baacke_c_scale: a partly cancelled baacke_c is not judged on its
    # own small size
    baacke_c_miss = abs(baacke_c_measured - baacke_c) / baacke_c_scale
    return dict(s2_trunc_err_pred=float(s2_trunc_err_pred), baacke_c_miss=baacke_c_miss,
                completed_sum_drift=float(completed_sum_drift), baacke_envelope=float(baacke_envelope[-1]))


# The Baacke law holds when the fitted slope lies within half the unsigned
# moment of baacke_c:  |baacke_c_measured - baacke_c| <= BAACKE_MISS_TOL *
# baacke_c_scale.
# baacke_c/(s2_max + m_min2) is the leading large-cutoff term of the raw
# band sum over ALL waves, with the single mass m_min2; the band holds
# n = 2..n_max only and its cutoff is finite, so the measured coefficient
# approaches baacke_c without equalling it -- a tolerance, not a match.  For
# a sign-definite tr U^3 (baacke_c_scale = |baacke_c|) this is
# baacke_c_measured/baacke_c in [0.5, 1.5]; for a strongly cancelled baacke_c
# (|baacke_c| < baacke_c_scale/2) it also admits a slope of the opposite
# sign: it then says only that the slope lies within baacke_c_scale/2 of
# baacke_c.
BAACKE_MISS_TOL = 0.5


def baacke_law_holds(plateau):
    """The ONE Baacke-law test (`plateau` = a watch_s2_plateau dict), used
    by the S2-PLATEAU stop and the wave extension of the band driver
    (band_adaptive) and by the S2-BAACKE guard: the fitted slope lies within
    BAACKE_MISS_TOL baacke_c_scale of baacke_c.  False when the watcher was
    undecidable (miss nan)."""
    return plateau['baacke_c_miss'] <= BAACKE_MISS_TOL


def guard_s2_baacke(plateau, eps_s2):
    """GUARD S2-BAACKE (aborts; does NOT enter lnD): the delivered s2_max must
    satisfy the same two conditions the S2-PLATEAU watcher stopped on
    (`plateau` = the watch_s2_plateau dict on the delivered tables):
      (i)  the raw band sum follows the Baacke law -- the fitted
           slope lies within BAACKE_MISS_TOL baacke_c_scale of baacke_c
           (baacke_law_holds; why a tolerance: the comment at
           BAACKE_MISS_TOL);
      (ii) the predicted remaining error s2_trunc_err_pred = completed_sum_drift * envelope is
           at most eps_s2."""
    if not baacke_law_holds(plateau):
        raise RuntimeError(
            f"[ABORT S2-BAACKE] the raw band truncation does not follow the "
            f"Baacke law at this cutoff: its fitted slope misses baacke_c by "
            f"|baacke_c_measured - baacke_c| / baacke_c_scale = {plateau['baacke_c_miss']:.3f}, "
            f"not within {BAACKE_MISS_TOL:g}.  The law holds for the sum over "
            f"all waves that carry the truncation, up to the centrifugal edge "
            f"r_W sqrt(s2_max + m_min2): either the band stops below that edge -- raise "
            f"n_max -- or the s2 grid top is not yet in the asymptotic regime -- raise s2_max.")
    if plateau['s2_trunc_err_pred'] > eps_s2:
        raise RuntimeError(
            f"[ABORT S2-BAACKE] predicted remaining s2-truncation error "
            f"s2_trunc_err_pred = completed_sum_drift x envelope = {plateau['completed_sum_drift']:.2e} x "
            f"{plateau['baacke_envelope']:.3f} = {plateau['s2_trunc_err_pred']:.2e} > eps_s2 = "
            f"{eps_s2:g} -- s2_max is too low for this potential.")


# --------------------------------------------------------------------------- #
#  WATCHER N-ONSET / GUARD N-ASYMPTOTE  (quantity: fit_lo, the fit window)     #
# --------------------------------------------------------------------------- #
def watch_n_onset(band_n, I_n_inf, A3, onset_tol=0.2):
    """WATCHER N-ONSET (decides the fit-window start; does NOT enter lnD):
    the first wave from which the computed nu^3 I_n(inf)
    matches the analytic A3 within onset_tol, SUSTAINED for every deeper wave.
    Comparing against the leading 1/nu^3 term alone is sufficient because the
    deviation from it IS the subleading (interpolation) content and decays as
    1/nu^2 -- once it is inside onset_tol the odd-family high-n form has taken
    over.  The sub-leading terms still belong in the FIT
    (pipeline_quadrature_vgeneral_phi_v2.odd_family_high_n_fit); they are
    only absent from this onset comparison.  Returns the onset n, or None if
    no sustained match exists in the data yet (generate deeper waves).
    A3 is resolved (band_adaptive's cancellation abort ran before any band
    was generated)."""
    nu3_I_n_inf_over_A3 = (np.asarray(band_n, float) + 1.0) ** 3 * np.asarray(I_n_inf, float) / A3
    dev = np.abs(nu3_I_n_inf_over_A3 - 1.0)
    ok = dev <= onset_tol
    # sustained: onset index i has ok[i:] all True
    for i in range(len(band_n)):
        if ok[i:].all():
            return int(band_n[i])
    return None


def guard_n_asymptote(fit_a, A3, tol=0.10):
    """GUARD N-ASYMPTOTE (aborts; does NOT enter lnD): the fitted leading
    coefficient fit_a of the odd-family fit must agree with the analytic
    third-Born moment A3 (only the third Born order produces 1/nu^3) to
    within tol -- the check that the fit window sits on the I_n(inf)
    asymptote.  A3 is resolved (band_adaptive aborts on a cancelled A3
    before any band exists)."""
    dev = abs(fit_a / A3 - 1.0)
    if dev > tol:
        raise RuntimeError(
            f"[ABORT N-ASYMPTOTE] fitted a = {fit_a:.5g} deviates from the "
            f"analytic A3 = {A3:.5g} by {100*dev:.1f}% > {100*tol:.0f}% -- the "
            f"fit window is not on the I_n(inf) asymptote; raise n_max (or "
            f"inspect the high-s2 tail).")


# --------------------------------------------------------------------------- #
#  WATCHER + GUARD N-TAIL-BOUND  (quantity: n_max)                             #
# --------------------------------------------------------------------------- #
def watch_n_max_needed(fit_b, fit_c, eps_n):
    """WATCHER N-TAIL-BOUND, prediction step (proposes the next n_max; does
    NOT enter lnD): the smallest n_max at which the subleading tail bound
    |b| zeta(5, n_max+2) + |c| zeta(7, n_max+2) -- the bound GUARD
    N-TAIL-BOUND checks -- is at most eps_n.  The bound falls monotonically
    to 0 in n_max, so the upward scan ends for every eps_n > 0.  The fitted
    (b, c) come from the production odd-family high-n fit
    (pipeline_quadrature_vgeneral_phi_v2.odd_family_high_n_fit); this
    watcher only READS their magnitudes."""
    if not eps_n > 0.0:
        raise RuntimeError(f'[ABORT N-TAIL-BOUND] eps_n = {eps_n:g} must be > 0.')
    n_max = 0
    while abs(fit_b) * zeta(5, n_max + 2.0) + abs(fit_c) * zeta(7, n_max + 2.0) > eps_n:
        n_max += 1
    return n_max


def guard_n_tail_bound(subleading_zeta_bound, eps_n):
    """GUARD N-TAIL-BOUND (aborts; does NOT enter lnD): at the delivered
    n_max, the interpolation coefficients' whole contribution to the zeta
    tail must be below eps_n (the subleading_zeta_bound returned by
    pipeline_quadrature_vgeneral_phi_v2.odd_family_high_n_fit) -- the check
    that n_max is deep enough for the fitted subleading part (the b and c
    terms) of the tail to be negligible."""
    if subleading_zeta_bound > eps_n:
        raise RuntimeError(
            f"[ABORT N-TAIL-BOUND] the fitted subleading tail contribution "
            f"|b| zeta5 + |c| zeta7 = {subleading_zeta_bound:.2e} > eps_n = "
            f"{eps_n:g} at this n_max -- the interpolation part of the tail "
            f"is not yet negligible; raise n_max.")


# --------------------------------------------------------------------------- #
#  SECTOR-CUTOFF estimator + limit  (quantity: s2_max of the sectors n = 0, 1) #
#  (its watcher runs in band_adaptive, its guard in assemble_lnD)              #
# --------------------------------------------------------------------------- #
# SECTOR-CUTOFF limit (lnD units) on the dropped raw trace of the sectors
# n = 0, 1 above s2_max (their two sector_high_s2_remainder_bound values,
# summed): the band's s2 watcher stops only below it (band_adaptive), the
# assembly's SECTOR-CUTOFF guard verifies it (assemble_lnD).
SECTOR_REMAINDER_TOL = 1e-3


def sector_high_s2_remainder_bound(s2_sector, delta_geq3, fit_frac=0.1,
                                   max_resid=0.5):
    """Cutoff-truncation estimate behind the SECTOR-CUTOFF watcher
    (band_adaptive, on the band s2 grid) and guard (assemble_lnD, on the
    sector grid); never enters lnD.  The exact cutoff dependence is
    d(lnD_ren)/d(s2_max) = -delta_geq3(n, s2_max) (the closed-form tail T_fict
    continues only the subtracted pole to infinity; the raw trace beyond the
    cutoff is dropped), so the error is Int_{cutoff}^inf delta_geq3 ds2.
    |delta_geq3| decays as a power law s2^p at large s2: p is fitted in log-log
    on TWO trailing windows (s2 >= fit_frac and 2*fit_frac of the cutoff,
    and both above twice the s2 of the first node past the trace's last
    sign change),
    the amplitude is |delta_geq3(s2_max)| itself, so the remainder is
    |delta_geq3(s2_max)| s2_max / (-(p+1)).  A window counts only when its fit
    is good (max ln-residual <= max_resid) and integrable (p < -1).  Report
    the LARGER surviving remainder bound (conservative); inf if none survives
    (the guard aborts).
    Returns (remainder_bound, decay_power, abs_delta_at_s2_max,
    max_log_resid)."""
    s2 = np.asarray(s2_sector, float)
    delta_geq3 = np.asarray(delta_geq3, float)
    abs_delta_geq3 = np.abs(delta_geq3)
    s2_max = s2[-1]
    # the trace may change sign (across its pole, and where its low-s2 part
    # and its third-Born tail differ in sign): next to a zero ln|delta_geq3|
    # dives to -inf and no power law fits, so the fit starts well above the
    # LAST sign change, at twice the s2 of the first node past it.  The
    # signs are compared on the finite nonzero nodes (a band node exactly on
    # a pole stores NaN; a node exactly at zero has no sign)
    nonzero = np.flatnonzero(np.isfinite(delta_geq3) & (delta_geq3 != 0.0))
    sign_changes = np.flatnonzero(np.diff(np.sign(delta_geq3[nonzero])) != 0.0)
    s2_fit_lo = 2.0 * s2[nonzero[sign_changes[-1] + 1]] if sign_changes.size else 0.0
    if not np.isfinite(delta_geq3[-1]):      # no amplitude at the cutoff: no estimate
        return float('inf'), 0.0, float('nan'), float('inf')
    usable_bounds, decay_powers, max_log_resids = [], [], []
    for frac in (fit_frac, 2.0 * fit_frac):
        in_window = (s2 >= max(frac * s2_max, s2_fit_lo)) & (abs_delta_geq3 > 0.0)
        if in_window.sum() < 3:
            continue
        log_s2, log_abs_delta = np.log(s2[in_window]), np.log(abs_delta_geq3[in_window])
        decay_power, log_intercept = np.polyfit(log_s2, log_abs_delta, 1)
        resid = float(np.max(np.abs(log_abs_delta - (decay_power * log_s2 + log_intercept))))
        decay_powers.append(float(decay_power))
        max_log_resids.append(resid)
        if resid > max_resid or decay_power >= -1.0:
            continue                     # misfit, or trace not integrably
                                         # decaying (decay_power >= -1)
        # |delta_geq3(s2)| = |delta_geq3(s2_max)| (s2/s2_max)^p above the
        # cutoff, so Int_{s2_max}^inf ds2 |delta_geq3(s2)|
        #   = |delta_geq3(s2_max)| s2_max / (-(p+1))
        usable_bounds.append(
            (abs_delta_geq3[-1] * s2_max / (-(decay_power + 1.0)),
             float(decay_power)))
    if not max_log_resids:               # no window holds 3 non-zero points
        return float('inf'), 0.0, float(abs_delta_geq3[-1]), float('inf')
    if not usable_bounds:
        return (float('inf'), decay_powers[0], float(abs_delta_geq3[-1]),
                max(max_log_resids))
    remainder_bound, decay_power = max(usable_bounds)   # the larger bound
                                                        # (conservative)
    return (float(remainder_bound), decay_power, float(abs_delta_geq3[-1]),
            max(max_log_resids))
