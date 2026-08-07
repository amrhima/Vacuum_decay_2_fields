#!/usr/bin/env python3
"""eigenvalue_neg_coupled_toy_model.py -- stage 0a: locate the n = 0 NEGATIVE-mode
eigenvalue w_neg2 as the det(1+M) zero-crossing (the det-crossing
eigenvalue condition, kept in its OWN file, separate from the RK resolvent
of gbar_n0 and from the determinant validator in crosschecks/).

METHOD MANDATE (user, standing): ONE method per code.  The det-crossing IS
the eigenvalue condition -- no determinant VALUE is
formed here; only the SIGN/ZERO of the 2x2 solution-matching determinant
(pipeline_helpers_coupled_toy_model.sector_matching_det) is used to LOCATE the pole.

WHAT IT COMPUTES
----------------
The CONTINUUM negative-mode eigenvalue lam_neg = w_neg2 (model-dependent;
-9.42e-2 for the decoupled bdet toy potential) from the det(1+M) zero-crossing
of the h-ODE:
  1. bracketing scan on geomspace(1e-3, s2_sector_max, 33), take the first
     sign flip -> a bracket [s2_lo, s2_hi] straddling the crossing;
  2. brentq(det, s2_lo, s2_hi, xtol=1e-12) -> the crossing s2-root s0;
  3. the eigenvalue is w_neg2 = -s0 (the crossing is at positive s2 ~ +0.094,
     so the eigenvalue is negative).
No polynomial fit: production = the bracketed brentq root itself.

GUARDS (none enters lnD)
  COLEMAN-N0    the n=0 spectrum must be exactly one SIMPLE negative mode
                (Coleman: exactly one negative mode).  Two independent checks:
                (i) MORE THAN ONE det sign change on the scan = several negative
                modes at DISTINCT s2 (a generic multi-wall bounce), and
                (ii) at the located crossing the matching matrix must have a
                one-dimensional null space -- a DEGENERATE crossing (several
                coincident modes at the SAME s2, as a symmetric multi-wall
                composite gives; an ODD multiplicity shows only one sign change
                and would otherwise be miscounted as index-1) is caught by
                crossing_multiplicity.  Either => Morse index >= 2, the bounce
                does not contribute to the decay rate, so it is REJECTED -- the
                stage prints "multiple zero crossings in sector n=0" and exits
                with code 3 so the driver skips it and continues.  ZERO sign
                changes = no negative mode found = a hard error (invalid bounce;
                note an EVEN coincident multiplicity shows zero sign changes and
                lands here -- still not computed, so no wrong rate).
  EIG-RESIDUAL  |det(1+M)(-w_neg2)| = |det(1+M)(s0)| must be
                <= --crossing-resid-tol (default 1e-8), else the brentq root
                does not sit on the zero (hard error).
n=1 (eigenvalue_zero_coupled_toy_model) uses the IDENTICAL method shape (bracket ->
brentq -> EIG-RESIDUAL guard).

INPUTS   --bounce-npz (R, X'/Y', params, false_vac) -- the potential is
         rebuilt from the coupling vector stored in it; no other stage output
         is needed.
OUTPUT   eig_n0_<tag>.npz (stage_paths key 'eig_n0') with keys
    lam_neg_cont   ()   w_neg2 = -(brentq crossing root) (PRODUCTION)
    crossing_resid ()   |det(1+M)(-w_neg2)| (the EIG-RESIDUAL guard value)
    s2_sector_max  ()   the scan upper end used
    bounce_sha, potential_id, code_version   metadata
"""
import argparse
import os
import sys
import numpy as np
from scipy.optimize import brentq

sys.dont_write_bytecode = True
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from delta_g_bar_greater_equal_3_coupled_toy_model import ResidualBand                          # noqa: E402
from pipeline_helpers_coupled_toy_model import (add_standard_cli, atomic_savez,          # noqa: E402
                                 crossing_multiplicity, load_bounce,
                                 provenance_stamp, sector_matching_det,
                                 stable_crossing_scan, stage_paths)


def _parking_report(bounce_npz):
    """On a multi-crossing REJECTION: name the vacua the background PARKS at
    (|Phi(r) - vacuum| < 0.05 over an r-interval) -- the walls of a multi-wall
    bounce sit between parked stretches, so this tells you which intermediate
    vacuum produced the extra crossing.  The vacua are the HARDCODED ones of
    whichever toy model the bounce belongs to (matched on the stored coupling
    vector), so nothing is searched for here either.  Diagnostic only."""
    try:
        import potential_coupled_toy_model as pm
        b = np.load(bounce_npz, allow_pickle=True)
        R = np.asarray(b['R'], float)
        Phi = np.asarray(b['Phi_bounce_orig'], float)
        if Phi.ndim == 1:
            Phi = Phi[:, None]
        params = np.asarray(b['params'], float)
        model = None
        for nm in pm.model_names():
            cand = pm.get_model(nm)
            if np.allclose(cand['params'], params, rtol=0, atol=1e-12):
                model = cand
                break
        if model is None:
            return ('\n  (parking report unavailable: this bounce was solved '
                    'for neither hardcoded model)')
        lines = []
        for i, pt in enumerate(model['minima']):
            dist = np.sqrt(np.sum((Phi - pt) ** 2, axis=1))
            sel = R[dist < 0.05]
            if sel.size:
                lines.append(f'    M{i} = {np.round(pt, 4)} '
                             f'(V = {model["V_min"][i]:+.5f}): parked for r in '
                             f'[{sel.min():.2f}, {sel.max():.2f}]')
        if not lines:
            return ''
        return ('\n  Parking report (vacua the background sits at; the '
                'outermost is the false vacuum,\n  the innermost the core; '
                'anything in between is an INTERMEDIATE vacuum =\n  one '
                'extra wall = one extra crossing):\n' + '\n'.join(lines))
    except Exception as e:
        return f'\n  (parking report unavailable: {e})'


def find_crossing_eigenvalue(band, s2_sec_max, bounce_npz=None):
    """Continuum negative-mode eigenvalue (n=0): det(1+M)(s2) crosses zero
    once at s2 = -lam_neg > 0.  SELF-CONTAINED extraction (no prior needed):
    bracket the first sign change on a geometric scan of [1e-3, s2_sec_max]
    at SELF-CERTIFIED density (stable_crossing_scan doubles the density until
    the crossing count stabilizes), then root it with brentq (xtol 1e-12).
    Returns
    (w_neg2_cont, crossing_resid): w_neg2_cont = -(brentq root)
    (PRODUCTION), crossing_resid = |det(1+M)(-w_neg2_cont)| is how far the
    located root sits from the true zero (the EIG-RESIDUAL guard value)."""
    def _det0(s2):
        return sector_matching_det(band, 0, float(s2))

    # Self-certifying census (stable_crossing_scan): sign changes at DOUBLED
    # density until the count stabilizes, PLUS the svmin hidden-root sweep --
    # roots the det sign cannot see (a near-split or tangential even pair)
    # still make M lose rank and are caught by the smallest-singular-value dip.
    scan, dets, flips, n_hidden = stable_crossing_scan(
        band, 0, lambda npts: np.geomspace(1e-3, s2_sec_max, npts))
    n_roots = int(flips.size) + int(n_hidden)
    # ---- GUARD COLEMAN-N0 --------------------------------------------------
    # A valid decay bounce has EXACTLY ONE n=0 negative mode, so the matching
    # problem has exactly one root (at s2 = -lam_neg > 0).
    #   * MORE THAN ONE root = Morse index >= 2 (a multi-wall bounce, one
    #     bubble-radius negative mode per resolved wall).  Such a bounce does NOT
    #     contribute to the decay rate (Callan-Coleman: exactly one negative mode;
    #     Picard-Lefschetz: >=2 negative modes are off-thimble).  We REJECT it and
    #     exit with code 3 ("rejected", NOT an error), so the production driver can
    #     skip this bounce and continue with the others; no lnD_ren is computed.
    #   * ZERO roots = no negative mode found -- not a valid bounce (wrong
    #     potential/bounce, or the mode is below the 1e-3 scan floor): a hard error.
    if n_roots > 1:
        print(f"[REJECTED] multiple zero crossings in sector n=0: {n_roots} "
              f"roots ({int(flips.size)} sign changes + {int(n_hidden)} hidden) "
              f"= {n_roots} negative modes (Morse index {n_roots} >= 2).  This "
              f"bounce has more than one negative mode and does NOT contribute "
              f"to the decay rate; skipping it (no lnD_ren).  Only index-1 "
              f"bounces (exactly one n=0 root) are computed."
              + _parking_report(bounce_npz))
        sys.exit(3)
    if flips.size == 0:
        raise RuntimeError(
            f'[ABORT COLEMAN-N0] n=0 det(1+M) has no sign change on the scan '
            f'[1e-3, {s2_sec_max:g}]: no negative mode found -- wrong '
            f'potential/bounce, or the negative mode sits below the 1e-3 scan '
            f'floor.  A genuine bounce has exactly one n=0 negative mode.')
    i0 = int(flips[0])
    s0 = float(brentq(_det0, scan[i0], scan[i0 + 1], xtol=1e-12))
    w_neg2_cont = -s0
    crossing_resid = abs(_det0(s0))
    # ---- GUARD COLEMAN-N0, MULTIPLICITY --------------------------------------
    # A single SIGN change of the SCALAR det cannot see EXACTLY degenerate modes:
    # a symmetric multi-wall composite has several coincident n=0 negative modes
    # at the SAME s2, and for ODD multiplicity the det still flips sign once, so
    # the sign count alone would falsely certify it index-1.  Count the coincident
    # modes directly (null-space dim of the matching matrix at the crossing) and
    # REJECT (exit 3) anything but a simple, multiplicity-1 crossing.
    mult = crossing_multiplicity(band, 0, s0)
    if mult > 1:
        print(f"[REJECTED] multiple zero crossings in sector n=0: the det(1+M) "
              f"crossing at s2={s0:.4e} is {mult}-fold DEGENERATE = {mult} "
              f"coincident negative modes (Morse index >= {mult}).  A symmetric "
              f"multi-wall bounce whose modes coincide shows only one sign change "
              f"but is NOT index-1; it does NOT contribute to the decay rate; "
              f"skipping it (no lnD_ren)." + _parking_report(bounce_npz))
        sys.exit(3)
    print(f"[COLEMAN-N0] 1 simple det(1+M) crossing on the scan "
          f"[1e-3, {s2_sec_max:g}] (index 1: exactly one negative mode, "
          f"multiplicity 1)")
    return w_neg2_cont, crossing_resid


def main():
    ap = argparse.ArgumentParser(
        description='coupled toy model stage 0a: n=0 negative-mode eigenvalue '
                    '(det-crossing locator)')
    add_standard_cli(ap)
    ap.add_argument('--s2-sector-max', type=float, default=300.0,
                    help='upper end of the geomspace(1e-3, .) scan used to '
                         'bracket the n=0 det(1+M) zero-crossing')
    ap.add_argument('--crossing-resid-tol', type=float, default=1e-8,
                    help='EIG-RESIDUAL guard: |det(1+M)(-w_neg2)| at the '
                         'located crossing must not exceed this (else the '
                         'brentq root is not on the zero)')
    args = ap.parse_args()

    bg = load_bounce(args.bounce_npz)
    band = ResidualBand(bg['R'], bg['DW'], bg['m2'])   # DEFAULT tolerances
    s2_sec_max = float(args.s2_sector_max)

    w_neg2_cont, crossing_resid = find_crossing_eigenvalue(
        band, s2_sec_max, bounce_npz=args.bounce_npz)
    # ---- GUARD EIG-RESIDUAL (aborts; does NOT enter lnD) ---------------------
    if crossing_resid > args.crossing_resid_tol:
        raise RuntimeError(
            f'[ABORT EIG-RESIDUAL] n=0 det-crossing residual too large: '
            f'|det(1+M)(-w_neg2)| = {crossing_resid:.2e} > '
            f'{args.crossing_resid_tol:g} -- the brentq root at '
            f'w_neg2 = {w_neg2_cont:+.10e} is not on the det zero.')
    print(f"[EIG-RESIDUAL] |det(1+M)(-w_neg2)| = {crossing_resid:.2e} "
          f"(limit {args.crossing_resid_tol:g})")

    out = stage_paths(args.data_dir, args.tag)['eig_n0']
    atomic_savez(out,
             lam_neg_cont=w_neg2_cont,
             crossing_resid=crossing_resid,
             s2_sector_max=s2_sec_max,
             **provenance_stamp(args.bounce_npz, bg['m2']))
    print(f"[EIG n=0] lam_neg_cont = w_neg2 = {w_neg2_cont:+.10e} "
          f"(brentq det-crossing eigenvalue; scan [1e-3, {s2_sec_max:g}])")
    print(f"[OK] wrote {out}")


if __name__ == '__main__':
    main()
