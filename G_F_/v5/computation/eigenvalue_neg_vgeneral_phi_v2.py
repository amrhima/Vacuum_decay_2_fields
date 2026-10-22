"""eigenvalue_neg_vgeneral_phi_v2.py -- level 2 of the README tree, stage 0a:
the n = 0 sector of the SELECTED bounce -- its negative mode omega2_00 and
the selection-gate backstop.

THE OBJECT.  The n=0 sector of the fluctuation operator has the bounce's one
negative mode, omega^2_{0,0} < 0 (spherically symmetric: it expands or shrinks
the bubble).  At s2 = -omega^2_{0,0} > 0 an eigenvalue of the constant
Wronskian C(n=0, s2) (mode_census.constant_wronskian_and_floor) is zero, so
det C = 0 there: the mode census of stage 0 (mode_census.negative_mode_census)
locates this crossing.

WHAT THE STAGE DOES
  1. READS the census bounce_vgeneral_phi_v2.bounce_census located and stored
     in the bounce npz (keys census_*, read by mode_census.stored_census):
     the census runs ONCE per bounce, there.  Only a bounce npz without the
     census keys (stored_census returns None) is censused here, by the same
     locator on the same derived window (mode_census.negative_mode_window);
  2. the SELECTION-GATE BACKSTOP: exactly ONE n=0 crossing shared by ONE mode
     is the index-1 bounce the lnD assembly is written for; anything else
     aborts with OPEN_PROBLEM_MSG (+ LEG_POINTER_MSG for a decoupled
     potential), the census printed above it as evidence;
  3. stores omega2_00 with its crossing residual |det C| at s2 = -omega2_00
     (the stored census_n0_resid, reported) and the det-crossing curve on a
     plot-density scan of that window, so the det-crossing figure never
     re-solves the h-ODE.

NAMES
  omega2_00        omega^2_{0,0} = -s2 of the ONE n=0 crossing the backstop
                   admits: the negative-mode eigenvalue whose pole
                   1/(omega2_00 + s2) the sector stage subtracts
  s2               -omega^2, the scan variable
  n_modes_at_root  how many eigenvalues of C are zero at one located crossing
                   (the stored key census_n0_modes)

READS   the bounce npz (--bounce-npz): through load_background its R, phi_b,
        m2 and U_int -> the h-ODE engine and the derived scan window, and its
        params and fields -> the decoupled-potential pointer and the
        provenance stamp; and the census keys census_* the bounce stage
        stored in it.
WRITES  eig_n0_<tag>.npz, written ONLY on the index-1 path; read by
        gbar_sector (stage 2a: omega2_00 sets the pole it subtracts), by the
        add-on greensfct_dangerous_modes_v2 (the eigenfunction at the crossing)
        and by the plot codes plot_det_C (the det-crossing curve) and
        plot_core_eq_fictitious_vs_s2 (omega2_00).  Keys:
    omega2_00           ()    omega^2_{0,0}
    omega2_00_census    (L,)  the eigenvalue of every located n=0 crossing
    crossing_resid      ()    |det C(-omega2_00)|
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

from greens_function_vgeneral_phi_v2 import GreensFunction
from potential_vgeneral_phi_v2 import is_decoupled
from pipeline_helpers_vgeneral_phi_v2 import (add_standard_cli, atomic_savez,
                                 load_background, provenance_stamp,
                                 stage_paths)
from mode_census_vgeneral_phi_v2 import (DET_SCAN_PTS, LEG_POINTER_MSG,
                                         OPEN_PROBLEM_MSG,
                                         det_C_scan,
                                         negative_mode_census,
                                         print_census_inventory,
                                         negative_mode_window,
                                         stored_census)


def main():
    ap = argparse.ArgumentParser(
        description='vgeneral_phi_v2 stage 0a: the n=0 negative-mode '
                    'eigenvalue, read from the stored census')
    add_standard_cli(ap)
    args = ap.parse_args()

    background = load_background(args.bounce_npz)
    # the h-ODE engine on this bounce: one solve per (n, s2) gives C(n, s2)
    engine = GreensFunction(background['R'], background['U_int'], background['m2'])
    # The window is DERIVED from the bounce (mode_census.negative_mode_window)
    s2_lo, s2_hi = negative_mode_window(engine, background['U_int'],
                                        background['m2'])
    print(f"[EIG-WINDOW] derived n=0 scan window [{s2_lo:.6e}, {s2_hi:.6e}]: "
          f"lower end = the accuracy floor of C's eigenvalue nearest zero times the "
          f"variational ceiling, upper end = that ceiling")

    # ---- the census: located by bounce_vgeneral_phi_v2.bounce_census, stored
    # keys census_* -- READ here.  A bounce npz without the census keys
    # (stored_census returns None) is censused by the same locator on the
    # derived window. ---------------------------------------------------------
    with np.load(args.bounce_npz, allow_pickle=True) as b:
        census = stored_census(b)
    if census is not None:
        s2_roots = [float(v) for v in census['census_n0_s2']]
        n_modes_at_root = [int(v) for v in census['census_n0_modes']]
        crossing_resid = census['census_n0_resid']   # |det C| at s2_roots[0]
        n_unresolved = census['census_n0_unresolved']
        print_census_inventory(census, 'selected bounce (stored census)',
                               phi_b=background['phi_b'])
    else:
        print('[CENSUS] the bounce npz lacks a census key -- running the n=0 '
              'census on it')
        s2_roots, n_modes_at_root, crossing_resid, n_unresolved =\
            negative_mode_census(engine, s2_lo, s2_hi)
    # ---- THE SELECTION-GATE BACKSTOP (nothing here enters lnD) --------------
    # The index-1 bounce is ONE crossing shared by ONE mode; a second crossing,
    # or one at which TWO eigenvalues of C are zero, means a non-index-1 bounce
    # was selected; an unresolved near-zero region may hide such a mode.  The
    # census was printed above as the evidence.
    if (len(s2_roots) != 1 or any(m != 1 for m in n_modes_at_root)
            or n_unresolved):
        pointer = (LEG_POINTER_MSG
                   if is_decoupled(background['params'], background['fields'])
                   else '')
        raise SystemExit(
            f'[ABORT SELECTION] the selected bounce carries {len(s2_roots)} '
            f'n=0 crossing(s), sharing '
            f'{", ".join(str(m) for m in n_modes_at_root)} mode(s), and '
            f'{n_unresolved} unresolved near-zero region(s) (census above; '
            f'sectors n >= 2 were not scanned, so this is the n <= 1 '
            f'inventory, not a total Morse index).\n'
            + OPEN_PROBLEM_MSG + pointer)
    # the ONE n=0 crossing the gate just admitted IS the negative-mode
    # eigenvalue omega^2_{0,0}, whose pole 1/(omega2_00 + s2) the sector stage
    # subtracts (census_n0_resid is |det C| at this same crossing)
    omega2_00 = -s2_roots[0]

    # ---- the det-crossing curve (plot_det_C's data; diagnostic) ----
    s2_scan = np.geomspace(s2_lo, s2_hi, DET_SCAN_PTS)
    det_scan, lambda_scan, max_abs_imag = det_C_scan(engine, 0, s2_scan)

    out = stage_paths(args.data_dir, args.tag)['eig_n0']
    atomic_savez(out,
             omega2_00=omega2_00,
             omega2_00_census=np.array([-s2c for s2c in s2_roots], float),
             crossing_resid=crossing_resid,
             s2_scan=s2_scan, det_scan=det_scan, lambda_scan=lambda_scan,
             max_abs_imag=max_abs_imag,
             **provenance_stamp(args.bounce_npz, background['m2'],
                                background['params'], background['fields']))
    print(f"[EIG n=0] omega2_00 = omega^2_(0,0) = {omega2_00:+.10e} "
          f"(the one n=0 crossing of the census, |det C| there "
          f"{crossing_resid:.2e}; det-crossing curve on "
          f"[{s2_lo:.6e}, {s2_hi:.6e}])")
    print(f"[OK] wrote {out}")


if __name__ == '__main__':
    main()
