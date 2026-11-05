"""eigenvalue_zero_vgeneral_phi_v2.py -- level 2 of the README tree, stage 0b:
the n = 1 sector of the SELECTED bounce -- its translation zero-mode crossing
s2_c (omega2_10 = -s2_c), the n=1 half of the selection-gate backstop, and the
ZERO-MODE gauge of the numerical quality (bounce solve, h-ODE, domain).

THE OBJECT.  The four n=1 modes d_mu phi_b are EXACT zero modes of any true
bounce (translation invariance): omega^2_{1,0} = 0 (wave n = 1, radial number
0; stored as omega2_10), so an eigenvalue of the constant Wronskian C(n=1, s2)
(mode_census.constant_wronskian_and_floor) is zero at s2_c = 0.  Numerically the
crossing is displaced (by the bounce solve, the h-ODE integration and the
finite domain), so the census (mode_census.zero_mode_census) scans a window
straddling zero and takes the located root NEAREST s2 = 0 as the
translation crossing.

WHAT THE STAGE DOES
  1. READS the census bounce_vgeneral_phi_v2.bounce_census located and stored
     in the bounce npz (keys census_*, read by mode_census.stored_census):
     the census runs ONCE per bounce, there.  Only a bounce npz without the
     census keys (stored_census returns None) is censused here, by the same
     locator mode_census.zero_mode_census;
  2. GUARD LEVEL-STAMP: the ZERO-MODE levels derived here
     (mode_census.zero_mode_levels) must equal the ones bounce_vgeneral_phi_v2
     derived and stamped into the census (tol is the level its
     dangerous-crossing rule mode_census.dangerous_n1_mask used; trust is
     stored for this gauge);
  3. the SELECTION-GATE BACKSTOP: exactly ONE DANGEROUS n=1 crossing (the
     translation zero mode; mode_census.dangerous_n1_mask) shared by ONE
     mode is the index-1 bounce -- positive bound states are allowed;
     anything else aborts with OPEN_PROBLEM_MSG;
  4. GUARD ZERO-MODE, the numerical-quality gauge: s2_c > +tol means
     omega^2_{1,0} < 0, FOUR SPURIOUS NEGATIVE modes -- a wrong Morse index,
     ABORT; |s2_c| > tol on the negative side, or above trust, only WARNS
     (accuracy is reported, not gated; the bounce's sensitivity indicator
     in lnD units is eps_bounce, computed by assemble_lnD_vgeneral_phi_v2
     against the reference profile the bounce npz stores);
  5. stores s2_c and omega2_10 with the crossing residual |det C(s2_c)| (the
     stored census_n1_resid, reported) and the det-crossing curve on the
     scanned window.

NAMES
  s2_c             the located translation crossing (~0)
  omega2_10        omega^2_{1,0} = -s2_c, the eigenvalue the sector stage's
                   pole subtraction uses
  n_modes_at_root  how many eigenvalues of C are zero at one located crossing
                   (the stored key census_n1_modes)

READS   the bounce npz (--bounce-npz): through load_background its R, phi_b,
        m2, m_min2 and U_int -> the h-ODE engine, the variational ceiling and
        the zero-mode levels, and its params and fields -> the
        decoupled-potential pointer and the provenance stamp; and the census
        keys census_* the bounce stage stored in it.
WRITES  eig_n1_<tag>.npz, written ONLY on the index-1 path; read by
        gbar_sector (stage 2b: omega2_10 sets the pole it subtracts, deg 4),
        by the add-on greensfct_dangerous_modes_v2 (the translation mode at
        the crossing) and by the plot codes plot_det_C (the det-crossing
        curve) and plot_core_eq_fictitious_vs_s2 (omega2_10).  Keys:
    s2_c, omega2_10     ()    the translation crossing and its eigenvalue
    omega2_10_census    (L,)  the eigenvalue of every located n=1 crossing
    crossing_resid      ()    |det C(s2_c)|
    s2_scan, det_scan, lambda_scan, max_abs_imag   the det-crossing curve
    bounce_sha, potential_id, code_version         metadata
"""
import argparse
import os
import sys
import numpy as np

sys.dont_write_bytecode = True
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from bounce_polish_vgeneral_phi_v2 import POLISH_TOL
from greens_function_vgeneral_phi_v2 import GreensFunction
from pipeline_helpers_vgeneral_phi_v2 import (add_standard_cli, atomic_savez,
                                 load_background, provenance_stamp,
                                 stage_paths)
from potential_vgeneral_phi_v2 import is_decoupled
from mode_census_vgeneral_phi_v2 import (CONTINUUM_WIDEN_ROUNDS,
                                         DET_SCAN_PTS,
                                         LEG_POINTER_MSG, OPEN_PROBLEM_MSG,
                                         dangerous_n1_mask,
                                         det_C_scan,
                                         print_census_inventory,
                                         stored_census,
                                         variational_s2_ceiling,
                                         zero_mode_census,
                                         zero_mode_levels,
                                         zero_mode_window_lo)


def main():
    ap = argparse.ArgumentParser(
        description='vgeneral_phi_v2 stage 0b: the n=1 zero-mode eigenvalue, '
                    'read from the stored census, and the n=1 half of the '
                    'selection gate')
    add_standard_cli(ap)
    args = ap.parse_args()

    background = load_background(args.bounce_npz)
    # the h-ODE engine on this bounce: one solve per (n, s2) gives C(n, s2)
    engine = GreensFunction(background['R'], background['U_int'], background['m2'])
    # the two ZERO-MODE levels (mode_census.zero_mode_levels): fractions of
    # the n=1 window's reach below zero, this model's own s2 scale
    s2_ceiling = variational_s2_ceiling(background['U_int'], background['m2'])
    m_min2 = float(background['m_min2'])
    zero_mode_tol, zero_mode_trust = zero_mode_levels(s2_ceiling, m_min2)
    print(f"[ZERO-MODE] guard levels, fractions of the n=1 window's reach "
          f"below zero (variational ceiling {s2_ceiling:.6g}, lightest "
          f"channel m_min2 {m_min2:.6g}):\n"
          f"  warn above |s2_c| = {zero_mode_tol:.4e} (ABORT if the crossing "
          f"is POSITIVE), trust below {zero_mode_trust:.4e}")

    # ---- the census: located by bounce_vgeneral_phi_v2.bounce_census, stored
    # keys census_* -- READ here.  A bounce npz without the census keys
    # (stored_census returns None) is censused by the same locator. ----------
    with np.load(args.bounce_npz, allow_pickle=True) as b:
        census = stored_census(b)
    if census is not None:
        # ---- GUARD LEVEL-STAMP (aborts; does NOT enter lnD): the levels this
        # stage derived above must equal the ones bounce_vgeneral_phi_v2
        # stamped into the census, so the tol its dangerous-crossing rule used
        # at solve time is provably the tol this stage gates on now, and the
        # trust level of this gauge is the stamped one (a changed level
        # constant would otherwise re-gate a stored bounce silently).
        stamped_levels = (census['census_zero_mode_tol'],
                          census['census_zero_mode_trust'])
        if not np.allclose((zero_mode_tol, zero_mode_trust), stamped_levels,
                           rtol=1e-12, atol=0.0):
            raise RuntimeError(
                f'[ABORT LEVEL-STAMP] derived zero-mode levels '
                f'(tol {zero_mode_tol:.6e}, trust {zero_mode_trust:.6e}) '
                f'do not match the census stamp of the bounce npz '
                f'(tol {stamped_levels[0]:.6e}, trust {stamped_levels[1]:.6e}) '
                f'-- the level constants changed since the bounce was solved; '
                f're-run the pipeline into a NEW --data-dir (a bounce stored '
                f'in this data folder is always reused).')
        zero_mode_tol, zero_mode_trust = stamped_levels
        s2_roots = [float(v) for v in census['census_n1_s2']]
        # the translation crossing: the stored census_s2_c (the crossing
        # nearest s2 = 0, the rule of zero_mode_census), one of the crossings
        s2_c = census['census_s2_c']
        n_modes_at_root = [int(v) for v in census['census_n1_modes']]
        crossing_resid = census['census_n1_resid']   # |det C| at s2_c
        n_unresolved = census['census_n1_unresolved']
        print_census_inventory(census, 'selected bounce (stored census)',
                               phi_b=background['phi_b'])
    else:
        print('[CENSUS] the bounce npz lacks a census key -- running the n=1 '
              'census on it')
        s2_c, s2_roots, n_modes_at_root, crossing_resid, n_unresolved =\
            zero_mode_census(engine, s2_ceiling, zero_mode_tol)
    # ---- THE n=1 SELECTION-GATE BACKSTOP (nothing here enters lnD) ----------
    # The index-1 bounce has ONE dangerous n=1 crossing -- the translation
    # zero mode, shared by ONE radial mode (mode_census.dangerous_n1_mask, the
    # one owner of the rule: a crossing below -zero_mode_tol is a positive
    # bound state and stays inside the resolvent).  bounce_vgeneral_phi_v2
    # gates every solved bounce at selection with the same rule; this is the
    # backstop.
    dangerous = dangerous_n1_mask(s2_roots, s2_c, zero_mode_tol)
    dangerous_modes = np.asarray(n_modes_at_root, int)[dangerous]
    if dangerous_modes.size != 1 or int(dangerous_modes[0]) != 1 or n_unresolved:
        # the crossings were printed above (the census inventory), one line
        # each with its label.  If the translation crossing ALSO misses its
        # quality level the reader has to be told here -- what looks like
        # extra walls may simply be an under-converged bounce.
        if abs(s2_c) > zero_mode_tol:
            print(f"[SELECTION] note: the translation crossing itself sits at "
                  f"|s2_c| = {abs(s2_c):.2e}, ABOVE the "
                  f"{zero_mode_tol:.4e} ZERO-MODE quality level -- "
                  f"its exact value is 0, so this bounce is ALSO under-converged. "
                  f"Re-solve it (a tighter VGENERAL_PHI_POLISH_TOL, into a NEW "
                  f"--data-dir) before reading this census as physics.")
        pointer = (LEG_POINTER_MSG
                   if is_decoupled(background['params'], background['fields'])
                   else '')
        raise SystemExit(
            f'[ABORT SELECTION] the selected bounce carries '
            f'{int(np.count_nonzero(dangerous))} dangerous n=1 crossing(s), '
            f'sharing {dangerous_modes.tolist()} mode(s), and {n_unresolved} '
            f'unresolved near-zero region(s) (census above; sectors n >= 2 '
            f'were not scanned, so this is the n <= 1 inventory, '
            f'not a total Morse index).\n' + OPEN_PROBLEM_MSG + pointer)
    # ---- GUARD ZERO-MODE (does NOT enter lnD): the numerical-quality gauge --
    # The exact value is s2_c = 0; the measured offset is numerical (the
    # bounce solve, the h-ODE integration at LSODA_RTOL, the finite domain):
    #   s2_c  >  +zero_mode_tol : ABORT, omega^2_{1,0} < 0 is FOUR SPURIOUS
    #                             NEGATIVE modes (a wrong Morse index)
    #   |s2_c| > zero_mode_tol on the negative side, or > zero_mode_trust :
    #                             WARN, the run continues
    # No lnD-unit number is inferred from |s2_c| (it weights the profile error
    # by d_r phi_b, which peaks at the wall, while a core offset is invisible
    # to it): the lnD-unit sensitivity indicator is eps_bounce, which
    # assemble_lnD_vgeneral_phi_v2 computes against the reference profile the
    # bounce npz stores.
    omega2_10 = -s2_c
    abs_s2_c = abs(s2_c)
    remedy = ("  The exact value is 0; the offset is numerical, and the bounce "
              "solve is the first suspect: compare a run at a tighter boundary-value tolerance "
              f"(VGENERAL_PHI_POLISH_TOL, e.g. {POLISH_TOL / 10:g}, a tenth of "
              f"bounce_polish_vgeneral_phi_v2.POLISH_TOL = {POLISH_TOL:g}) into "
              "a NEW --data-dir -- a bounce already stored in this data folder "
              "(bounce_data_<tag>.npz, bounce_gen/) is always reused, even "
              "under a new --tag, so a re-run here changes nothing.")
    if s2_c > zero_mode_tol:   # omega^2_(1,0) = -s2_c < -tol: four spurious negative modes
        raise RuntimeError(
            f'[ABORT ZERO-MODE] omega^2_(1,0) = {omega2_10:+.6e} < 0: the n=1 '
            f'sector carries FOUR SPURIOUS NEGATIVE modes, so this profile has '
            f'the wrong Morse index and its determinant is not lnD_ren.\n'
            + remedy)
    if abs_s2_c > zero_mode_tol:      # only the NEGATIVE side reaches this line
        print(f"[WARN ZERO-MODE] |s2_c| = {abs_s2_c:.2e} exceeds the "
              f"{zero_mode_tol:.4e} quality level (s2_c = "
              f"{s2_c:+.6e}, omega^2_(1,0) = {omega2_10:+.6e}).  The run "
              f"CONTINUES -- judge the result by eps_bounce "
              f"(assemble_lnD_vgeneral_phi_v2).\n" + remedy, file=sys.stderr)
    elif abs_s2_c > zero_mode_trust:
        print(f"[WARN ZERO-MODE] |s2_c| = {abs_s2_c:.2e} exceeds the trust level "
              f"{zero_mode_trust:.4e} (abort level "
              f"{zero_mode_tol:.4e}).  The run CONTINUES -- quote the result "
              f"with eps_bounce (assemble_lnD_vgeneral_phi_v2).\n" + remedy,
              file=sys.stderr)
    else:
        print(f"[ZERO-MODE] |s2_c| = {abs_s2_c:.2e} is within the trust level "
              f"{zero_mode_trust:.4e}: the translation modes are "
              f"where they should be, so the bounce is QUALITATIVELY right (no "
              f"spurious negative modes).  That is all this gauge establishes "
              f"-- the bounce's sensitivity indicator in lnD is eps_bounce "
              f"(assemble_lnD_vgeneral_phi_v2)")
    print(f"[ZERO-MODE] s2_c = {s2_c:+.6e}, "
          f"omega2_10 = omega^2_(1,0) = {omega2_10:+.6e} (abort level "
          f"{zero_mode_tol:.4e})")

    # ---- the det-crossing curve (plot_det_C's data; diagnostic): the LINEAR
    # window straddling zero of the census (zero_mode_census) at its DEEPEST
    # reach, after all CONTINUUM_WIDEN_ROUNDS widenings -- so every crossing
    # the census can locate lies inside the stored scan.
    s2_scan_lo = zero_mode_window_lo(s2_ceiling, m_min2, CONTINUUM_WIDEN_ROUNDS)
    s2_scan = np.linspace(s2_scan_lo, s2_ceiling, DET_SCAN_PTS)
    det_scan, lambda_scan, max_abs_imag = det_C_scan(engine, 1, s2_scan)

    out = stage_paths(args.data_dir, args.tag)['eig_n1']
    atomic_savez(out,
             s2_c=s2_c,
             omega2_10=omega2_10,
             omega2_10_census=np.array([-s2_root for s2_root in s2_roots], float),
             crossing_resid=crossing_resid,
             s2_scan=s2_scan, det_scan=det_scan, lambda_scan=lambda_scan,
             max_abs_imag=max_abs_imag,
             **provenance_stamp(args.bounce_npz, background['m2'],
                                background['params'], background['fields']))
    print(f"[EIG n=1] omega2_10 = -s2_c = {omega2_10:+.6e} "
          f"(the translation crossing of the census, |det C| there "
          f"{crossing_resid:.2e})")
    print(f"[OK] wrote {out}")


if __name__ == '__main__':
    main()
