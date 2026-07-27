#!/usr/bin/env python3
"""watchers_guards_coupled_toy_model.py -- ALL watchers and the cutoff guards of the coupled_toy_model
pipeline, in one module, so the architecture is visible in one place.

DEFINITIONS (the pipeline's autonomy architecture)
--------------------------------------------------
A WATCHER watches a computed object while band data is being generated and
compares it against its analytic asymptotic prediction; when the two match,
the asymptotic regime is reached and data generation in that direction stops.
Watchers only DECIDE how much data to generate -- nothing a watcher computes
ever enters lnD_ren.

A GUARD re-checks the same computed-vs-analytic condition on the final data
and ABORTS the pipeline if it fails.  Guards only gate -- nothing a guard
computes ever enters lnD_ren either.

One watcher and one guard per model-dependent quantity, and the KIND is in
every function name: watch_* functions are watchers, guard_* functions are
guards; nothing in this module computes anything that enters lnD_ren (the
production odd-family fit lives in pipeline_quadrature_coupled_toy_model.odd_tail_fit,
next to the s2 completion s2_tail_shape_ratio):

  quantity        watcher (in band_adaptive)   guard (in the stage that uses it)
  --------        --------------------------   ---------------------------------
  s2_max          watch_s2_plateau             guard_s2_baacke (tail_s2_completion)
  fit_lo (onset)  watch_n_onset                guard_n_asymptote (tail_high_n_zeta)
  n_max           watch_n_max_needed           guard_n_tail_bound (tail_high_n_zeta)

The remaining model-dependent quantities need no watcher because the stage
that owns them determines them while computing (adaptive node doubling, the
det-crossing eigenvalue search, the kappa-scaled radial grid); each is gated
by its own guard where it lives: S2-STEP and SECTOR-STEP (quadrature step
size), SECTOR-CUTOFF (sector s2 cutoff), EIG-RESIDUAL (eigenvalue location),
COLEMAN-N0/COLEMAN-N1 (mode census; reject a multi-wall index>=2 bounce),
RADIAL-GRID (radial resolution),
BAND-EIG (no in-band eigenvalue), MANIFEST (slice-set integrity),
FICT-INDEP (fictitious-scale independence).

THE TWO ASYMPTOTIC LAWS (notes/F2_T0.pdf Part II, eqs. (5) and (6))
-------------------------------------------------------------------
n-direction (eq. 5):    I_n(inf) -> A3/nu^3,  A3 = (1/16) Int dr r^5 tr U^3
s2-direction (eq. 6):   raw band truncation -> c/(Lam2 + mbar2),
                        c = -(1/48) Int dr r^3 tr U^3,  mbar2 = min m^2
Both A3 and c are moments of the SAME bounce the band integrates, computed
in analytic_moments() below -- no reference to any previous run.

MEASURED CALIBRATION (replayed on the certified F2 110/300, F2 124/400 and
bdet 250/2000 band tables; see F2_T0.pdf Part III):
  - the completed band sum plateaus to 1e-2 by Lam2 = 145 (F2) / 450 (bdet);
  - c_eff/c sits at 0.88..1.05 across the plateau approach (the band up to
    n_max carries most but not all of the full-n Baacke law, hence the wide
    [0.5, 1.5] guard bracket);
  - the pointwise 0.2-onset lands at n = 23 (F2) / 42 (bdet), and the
    subleading-bound rule then picks n_max = 109 (F2) / 243 (bdet) at its
    defaults -- reproducing the hand-tuned production 110/250 from analytic
    objects alone, with true tail errors 50-90x inside the bound.
"""
import numpy as np

from pipeline_helpers_coupled_toy_model import wall_radius
from pipeline_quadrature_coupled_toy_model import s2_integral, s2_tail_shape_ratio


# --------------------------------------------------------------------------- #
#  Analytic guard objects (bounce moments; the prediction side of every        #
#  watcher/guard)                                                              #
# --------------------------------------------------------------------------- #
def analytic_moments(R, trDW3, Phi, m2):
    """The analytic asymptotic-regime objects of THIS bounce:
        A3    = (1/16) Int dr r^5 tr U^3      (n-direction, eq. 5)
        c     = -(1/48) Int dr r^3 tr U^3     (s2-direction, eq. 6)
        mbar2 = min m^2                       (lightest channel; the resolvent
                                               variable is kap^2 = s2 + m^2, so
                                               the slowest-decaying channel
                                               sets the tail variable)
        r_W   = wall radius = R at max ||dPhi'/dr|| (where the two-field
                                               profile VECTOR moves fastest;
                                               reduces to max |dX'/dr| for the
                                               decoupled one-direction bounce)
        edge(s2) = r_W sqrt(s2 + mbar2)       (centrifugal edge, eq. 4)
    """
    R = np.asarray(R, float)
    trDW3 = np.asarray(trDW3, float)
    A3 = np.trapezoid(R**5 * trDW3, R) / 16.0
    c = -np.trapezoid(R**3 * trDW3, R) / 48.0
    mbar2 = float(np.asarray(m2, float).min())
    r_W = wall_radius(R, Phi)
    return dict(A3=float(A3), c=float(c), mbar2=mbar2, r_W=r_W,
                edge=lambda s2: r_W * np.sqrt(s2 + mbar2))


# --------------------------------------------------------------------------- #
#  WATCHER S2-PLATEAU / GUARD S2-BAACKE  (quantity: s2_max)                    #
# --------------------------------------------------------------------------- #
def band_candidate_sums(s2s, D, ns, mbar2, R, trDW3, n_cand=8):
    """The measurement input of the s2 watcher/guard pair, computed ONCE here
    so the S2-PLATEAU watcher (band_adaptive) and the S2-BAACKE guard
    (tail_s2_completion) gate on the same numbers.  For each trailing
    candidate cutoff (ladder points in the top 65% of the grid, at most
    n_cand) the SAME band tables are re-integrated (s2_integral) and
    re-completed (s2_tail_shape_ratio) at that cutoff:

        B_raw(L2) = sum_n I_n(L2),   B_c(L2) = sum_n [I_n(L2) + T_n(L2)]

    D is the (n_waves, n_slices) delta_geq3 matrix aligned with (ns, s2s).
    Returns (L2_cand, B_raw, B_completed, I_inf_at_full_cutoff).  Nothing
    returned here enters lnD_ren -- the completion stage recomputes its
    production I_n(inf) itself."""
    s2s = np.asarray(s2s, float)
    D = np.asarray(D, float)
    nu = np.asarray(ns, float) + 1.0
    cand = [k for k in range(len(s2s)) if s2s[k] >= 0.35 * s2s[-1]][-n_cand:]
    if len(cand) < 3:
        cand = list(range(len(s2s)))[-3:]
    # each candidate re-integrates s2s[:k+1], which needs >= 3 grid points
    cand = [k for k in cand if k >= 2]
    if not cand:
        raise RuntimeError('[ABORT] the band grid is too small for the s2 '
                           'watcher (every candidate cutoff needs >= 3 grid '
                           'points below it) -- extend the s2 grid.')
    Braw, Bc, Iinf_last = [], [], None
    for k in cand:
        sk = s2s[:k + 1]
        Idat = s2_integral(sk, D[:, :k + 1])
        Rn = np.array([s2_tail_shape_ratio(v, sk[-1], mbar2, R, trDW3) for v in nu])
        Iinf = Idat - D[:, k] * Rn
        Braw.append(float(np.sum(Idat)))
        Bc.append(float(np.sum(Iinf)))
        Iinf_last = Iinf
    return s2s[cand], np.array(Braw), np.array(Bc), Iinf_last


def watch_s2_plateau(L2, B_completed, B_raw, c_analytic, mbar2, n_trail=6):
    """WATCHER S2-PLATEAU (decides data generation; does NOT enter lnD).
    The s2-direction measurement, shared verbatim with the S2-BAACKE guard
    (guard_s2_baacke re-checks exactly this on the final tables).  Inputs are
    the completed and raw band sums evaluated at
    the trailing candidate cutoffs L2 (ascending; each candidate re-uses the
    same band tables, re-integrated and re-completed at that cutoff).

    Eq. (6): the RAW truncation approaches  c/(Lam2 + mbar2).  The completion
    removes exactly the part that follows this law, so what is LEFT in the
    completed sum is the law's shape-miss.  Measured over the trailing
    candidates (envelope variable e = |c|/(Lam2 + mbar2)):

        c_eff      = d B_raw / d (1/(Lam2+mbar2))     -- must be ~ c  (the
                     asymptotic-regime signal: the raw tail follows the law)
        shape_miss = |d B_completed / d e|            -- the completed sum's
                     residual drift per unit envelope
        E_pred     = shape_miss * e(Lam2_max)         -- the predicted
                     remaining s2-truncation error in lnD at the cutoff

    Returns dict(E_pred, c_eff_ratio, shape_miss, envelope, n_used).
    Slopes are least-squares over the trailing n_trail candidates (>= 3
    required); with a vanishing analytic c the completion magnitude is zero
    and E_pred is reported as the bare completed drift."""
    L2 = np.asarray(L2, float)
    Bc = np.asarray(B_completed, float)
    Br = np.asarray(B_raw, float)
    if L2.size < 3:
        return dict(E_pred=np.inf, c_eff_ratio=np.nan, shape_miss=np.inf,
                    envelope=np.inf, n_used=int(L2.size))
    k = min(int(n_trail), L2.size)
    x = 1.0 / (L2[-k:] + mbar2)
    env = abs(c_analytic) * x
    c_eff = float(np.polyfit(x, Br[-k:], 1)[0])
    if c_analytic != 0.0:
        shape_miss = abs(float(np.polyfit(env, Bc[-k:], 1)[0]))
        E_pred = shape_miss * env[-1]
        ratio = c_eff / c_analytic
    else:                       # degenerate potential: tr U^3 moment vanishes
        shape_miss = np.inf
        E_pred = float(np.max(np.abs(Bc[-k:] - Bc[-1])))
        ratio = np.nan
    return dict(E_pred=float(E_pred), c_eff_ratio=ratio,
                shape_miss=float(shape_miss), envelope=float(env[-1]),
                n_used=k)


def guard_s2_baacke(measure, eps_s2, ratio_lo=0.5, ratio_hi=1.5):
    """GUARD S2-BAACKE (aborts; does NOT enter lnD): the delivered s2_max must
    satisfy the same two conditions the S2-PLATEAU watcher stopped on:
      (i)  the raw band truncation follows the Baacke law -- c_eff/c inside
           [ratio_lo, ratio_hi] (the band up to n_max carries most, not all,
           of the full-n law: measured 0.88..1.05 on the certified tables);
      (ii) the predicted remaining error E_pred = shape_miss * envelope is
           at most eps_s2.
    A NaN c_eff_ratio (analytic c = 0) skips (i) and gates on (ii) alone."""
    ok_ratio = (np.isnan(measure['c_eff_ratio'])
                or ratio_lo <= measure['c_eff_ratio'] <= ratio_hi)
    if not ok_ratio:
        raise RuntimeError(
            f"[ABORT S2-BAACKE] the raw band truncation does not follow the "
            f"Baacke law at this cutoff: c_eff/c = {measure['c_eff_ratio']:.3f} "
            f"outside [{ratio_lo}, {ratio_hi}].  The s2 grid top is not in the "
            f"asymptotic regime -- raise s2_max.")
    if measure['E_pred'] > eps_s2:
        raise RuntimeError(
            f"[ABORT S2-BAACKE] predicted remaining s2-truncation error "
            f"E_pred = shape_miss x envelope = {measure['shape_miss']:.2e} x "
            f"{measure['envelope']:.3f} = {measure['E_pred']:.2e} > eps_s2 = "
            f"{eps_s2:g} -- s2_max is too low for this potential.")


# --------------------------------------------------------------------------- #
#  WATCHER N-ONSET / GUARD N-ASYMPTOTE  (quantity: fit_lo, the fit window)     #
# --------------------------------------------------------------------------- #
def watch_n_onset(ns, I_inf, A3, onset_tol=0.2):
    """WATCHER N-ONSET (decides the fit-window start; does NOT enter lnD):
    the first wave from which the computed nu^3 I_n(inf)
    matches the analytic A3 within onset_tol, SUSTAINED for every deeper wave.
    Comparing against the leading 1/nu^3 term alone is sufficient because the
    deviation from it IS the subleading (interpolation) content and decays as
    1/nu^2 -- once it is inside onset_tol the odd family has taken over.  The
    sub-leading terms still belong in the FIT (pipeline_quadrature_coupled_toy_model.odd_tail_fit
   ); they are
    only absent from this onset comparison.  Returns the onset n, or None if
    no sustained match exists in the data yet (generate deeper waves)."""
    if A3 == 0.0:
        return None
    q = (np.asarray(ns, float) + 1.0) ** 3 * np.asarray(I_inf, float) / A3
    dev = np.abs(q - 1.0)
    ok = dev <= onset_tol
    # sustained: onset index i has ok[i:] all True
    for i in range(len(ns)):
        if ok[i:].all():
            return int(ns[i])
    return None


def guard_n_asymptote(a_fitted, A3, tol=0.10):
    """GUARD N-ASYMPTOTE (aborts; does NOT enter lnD): the fitted leading
    coefficient must agree with the analytic third-Born moment A3 (eq. 5) to
    within tol -- the certificate that the fit window sits on the I_n(inf)
    asymptote.  A vanishing A3 skips the check (no asymptote to certify)."""
    if A3 == 0.0:
        return
    dev = abs(a_fitted / A3 - 1.0)
    if dev > tol:
        raise RuntimeError(
            f"[ABORT N-ASYMPTOTE] fitted a = {a_fitted:.5g} deviates from the "
            f"analytic A3 = {A3:.5g} by {100*dev:.1f}% > {100*tol:.0f}% -- the "
            f"fit window is not on the I_n(inf) asymptote; raise n_max (or "
            f"inspect the completion).")


# --------------------------------------------------------------------------- #
#  WATCHER + GUARD N-TAIL-BOUND  (quantity: n_max)                             #
# --------------------------------------------------------------------------- #
def watch_n_max_needed(c_fit, e_fit, eps_n):
    """WATCHER N-TAIL-BOUND (decides n_max; does NOT enter lnD), prediction
    step: the nu0 = n_max + 2 at which the subleading tail bound
    |c| zeta(5,nu0) + |e| zeta(7,nu0) falls to eps_n, using
    zeta(5,nu0) ~ 1/(4 nu0^4) and zeta(7,nu0) ~ 1/(6 nu0^6).  Solved on
    the c-term first (dominant in every measured case), then nudged up while
    the two-term bound still exceeds eps_n.  The fitted (c, e) come from the
    production odd-family fit (pipeline_quadrature_coupled_toy_model.odd_tail_fit); this
    watcher only READS their magnitudes."""
    nu0 = max(14.0, (abs(c_fit) / (4.0 * eps_n)) ** 0.25)
    for _ in range(200):
        if abs(c_fit) / (4.0 * nu0**4) + abs(e_fit) / (6.0 * nu0**6) <= eps_n:
            break
        nu0 *= 1.05
    return int(np.ceil(nu0)) - 2 + 1          # n_max = nu0 - 2, +1 safety wave


def guard_n_tail_bound(subleading_bound, eps_n):
    """GUARD N-TAIL-BOUND (aborts; does NOT enter lnD): at the delivered
    n_max, the interpolation coefficients' whole contribution to the zeta
    tail must be below eps_n (the subleading_bound returned by
    pipeline_quadrature_coupled_toy_model.odd_tail_fit) -- the certificate that n_max is deep
    enough that the non-analytic part of the tail is negligible."""
    if subleading_bound > eps_n:
        raise RuntimeError(
            f"[ABORT N-TAIL-BOUND] the fitted subleading tail contribution "
            f"|c| zeta5 + |e| zeta7 = {subleading_bound:.2e} > eps_n = "
            f"{eps_n:g} at this n_max -- the interpolation part of the tail "
            f"is not yet negligible; raise n_max.")
