"""gbar_sector_vgeneral_phi_v2.py -- Level 3, stages 2a/2b: Delta_n,
the s2-integral of the traced Born-order >= 3 Green's function of a dangerous
wave with the pole of its mode subtracted -- n = 0 (the negative mode) or
n = 1 (the four translation zero modes).  ONE file, selected by --sector-n.

READS
    bounce npz (--bounce-npz)   R, phi_b -> U_int, m2, m_min2
                                (pipeline_helpers.load_background)
    eig_n<sector>_<tag>.npz     omega2_00 | omega2_10 (+ s2_c), crossing_resid:
                                the eigenvalue of the subtracted pole, written
                                by eigenvalue_neg / eigenvalue_zero (stages
                                0a/0b) after their selection-gate backstop;
                                --eig-npz overrides the path
    band_cutoffs_<tag>.npz      s2_max, the band cutoff the band driver
                                certified, and s2_min, the lowest band node
                                (stage 1, their one creation point)
WRITES   sector_n<sector>_<tag>.npz, read by assemble_lnD (stage 7, this
         wave's part of (T1) Sigma_dang) and by the plot codes (its sector
         picture plot_Gbar_n<sector>, the lnD_ren part figures), with keys
    s2_sector         (M,)  the resolvent s2 grid (M = final adaptive count)
    delta_geq3        (M,)  the Born-subtracted trace on it
    delta_sub         (M,)  delta_geq3 - deg_n/(omega2 + s2)
    Gbar_minus_fv     (M,)  bounce-minus-FV coincident trace
    Gbar_1, Gbar_2    (M,)  the traced Born orders 1 / 2 (tadpole / fish)
    Gbar_fv           (M,)  the analytic false-vacuum trace (box-dependent:
                            radial box [r_start, r_max]); the bounce trace
                            alone is Gbar_minus_fv + Gbar_fv
    omega2_00 | omega2_10   omega^2_{n,0}, the eigenvalue of the subtracted
                            mode (wave n, radial number 0); n=1 also stores
                            the crossing s2_c = -omega2_10
    crossing_resid    ()    |det C| at the crossing (reported by the eig stage)
    Delta_n           ()    the sector integral (enters lnD_ren)
    sector_step_meter, sector_step_tol   the SECTOR-STEP meter and its limit
    deg               ()    (n+1)^2 angular degeneracy
    s2_max, sector_s2_lo, sector_nodes (FINAL), runtime_s
    sector_s2_lo_is_default ()  True when the floor was derived, False under
                            --sector-s2-lo
    (s2_max, sector_s2_lo_is_default, sector_s2_lo and sector_step_tol are
    value-matched by the driver's resume skip-guard)
    bounce_sha, potential_id, code_version                    metadata

HOW
    delta_geq3(s2)  = (n+1)^2 sum_i Int dr r^3 [G_n^(>=3)(r, r; s2)]_ii, the
                      traced Born-order >= 3 part G_n - G_n^fv - G_n^(1)
                      - G_n^(2) of the wave's coincident Green's function,
                      from the engine (gbar_traces, called per node of this
                      sector's own s2 grid by sector_traces_of_wave),
    delta_sub       = delta_geq3 - deg_n/(omega2 + s2),   deg_n = (n+1)^2,
    Delta_n = -Int_0^{s2_max} delta_sub ds2 .
The pole term deg_n/(omega2 + s2) is the dangerous mode's whole share of the
traced Green's function (deg_n angular copies, unit residue each), so
subtracting it projects the mode out.  The pole s2 = -omega2 is the ONE pole
this sector subtracts (the index-1 bounce: the census selection gate and its
backstop in the eigenvalue stage admit no second crossing whose pole lies on
the integration path); the eigenvalue is READ from the eig npz, never located
here.  The grid is pole-clustered when the pole falls inside
[sector_s2_lo, s2_max] with a clearance of pole/100 on both sides, and plain
geometric otherwise (sector_s2_grid); the node count is doubled from
--sector-nodes until the SECTOR-STEP meter passes --sector-step-tol
(adaptive_sector_integral).  The [0, sector_s2_lo] strip is closed by
low_s2_strip.  Beyond s2_max the closed-form fictitious-mode terms of
assemble_lnD continue the subtracted pole; the raw trace there is dropped and
only bounded (SECTOR-CUTOFF).
"""
import argparse
import os
import sys
import time

sys.dont_write_bytecode = True
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from greens_function_vgeneral_phi_v2 import GreensFunction
from pipeline_helpers_vgeneral_phi_v2 import (add_standard_cli, atomic_savez,
                                 load_background, load_stage_npz,
                                 provenance_stamp, stage_paths)
from pipeline_quadrature_vgeneral_phi_v2 import (adaptive_sector_integral,
                                                 sector_s2_lo_default)

# per-sector wiring: (eig stage_paths key, eig npz eigenvalue key, the stage
# that writes it, this sector's output stage_paths key)
SECTOR_N0_N1 = {
    0: ('eig_n0', 'omega2_00',
        'eigenvalue_neg_vgeneral_phi_v2', 'sector_n0'),
    1: ('eig_n1', 'omega2_10',
        'eigenvalue_zero_vgeneral_phi_v2', 'sector_n1'),
}


def main():
    ap = argparse.ArgumentParser(
        description='vgeneral_phi_v2 stages 2a/2b: the s2 integral of the '
                    'Born-order >= 3 trace delta_geq3 of wave n = 0 or 1, '
                    'its pole subtracted')
    add_standard_cli(ap)
    ap.add_argument('--sector-n', type=int, choices=(0, 1), required=True,
                    help='which dangerous sector: 0 (negative mode) or 1 '
                         '(zero mode)')
    ap.add_argument('--eig-npz', default=None,
                    help='eigenvalue npz (default: <data-dir>/eig_n<sector>_'
                         '<tag>.npz; run eigenvalue_neg / eigenvalue_zero_'
                         'vgeneral_phi_v2 first)')
    ap.add_argument('--sector-s2-lo', type=float, default=None,
                    help='low end of the sector s2 grid; default: derived '
                         'from the model, min(sector_s2_lo_default(m_min2, '
                         'omega2), s2_min of band_cutoffs) -- the '
                         '[0, sector_s2_lo] strip is closed by low_s2_strip')
    ap.add_argument('--sector-nodes', type=int, default=321,
                    help='STARTING node count (min 61; rounded up to 1 mod 4 '
                         'so every doubled grid contains the previous one); '
                         'doubled until the SECTOR-STEP meter passes')
    ap.add_argument('--sector-step-tol', type=float, default=1e-3,
                    help='SECTOR-STEP limit: |I(full grid) - I(every-2nd-'
                         'point grid)| of the sector integral')
    args = ap.parse_args()
    n = args.sector_n
    eig_key, omega2_key, eig_stage, out_key = SECTOR_N0_N1[n]

    # ---- the background of this bounce (R, U_int, m2: load_background) and
    # the h-ODE engine on it, called per grid node by sector_traces_of_wave --
    background = load_background(args.bounce_npz)
    engine = GreensFunction(background['R'], background['U_int'],
                        background['m2'])          # DEFAULT tolerances
    # the upper end of the sector grid = the band cutoff s2_max the band
    # driver certified (its one creation point: band_cutoffs_<tag>.npz)
    paths = stage_paths(args.data_dir, args.tag)
    band_cutoffs_npz = load_stage_npz(paths['band_cutoffs'], args.bounce_npz,
                            'band_adaptive_vgeneral_phi_v2')
    s2_max = float(band_cutoffs_npz['s2_max'])
    s2_min = float(band_cutoffs_npz['s2_min'])

    # ---- READ omega^2_{n,0} from the eig npz that eigenvalue_neg (n=0) /
    # eigenvalue_zero (n=1) wrote after their selection-gate backstop; its
    # pole s2 = -omega2 is the ONE pole this sector subtracts ----------------
    eig_npz = load_stage_npz(args.eig_npz or paths[eig_key], args.bounce_npz,
                             eig_stage)
    omega2 = float(eig_npz[omega2_key])
    crossing_resid = float(eig_npz['crossing_resid'])
    print(f"       omega2_{n}0 (READ from {eig_key}) = {omega2:+.10e} "
          f"(det residual {crossing_resid:.2e})")

    # grid floor: explicit --sector-s2-lo, else derived from the model
    # (m_min2) and this sector's own eigenvalue, and never above the band's
    # lowest node s2_min (the lnD_ren part figures carry this sector's
    # running integral onto the band nodes, so the band nodes must lie
    # inside the sector grid)
    sector_s2_lo = (float(args.sector_s2_lo) if args.sector_s2_lo is not None
                    else min(sector_s2_lo_default(background['m_min2'], omega2),
                             s2_min))

    # ---- Delta_n = -Int [delta_geq3 - deg_n/(omega2 + s2)] ds2 on
    # the adaptive sector grid (pipeline_quadrature.adaptive_sector_integral),
    # with its grid, traces and meter: this wave's part of the term (T1) -----
    t_sec0 = time.time()
    (s2_sector, delta_geq3, delta_sub, Delta_n, sector_step_meter,
     Gbar_traces) = adaptive_sector_integral(
        engine, n, omega2, sector_s2_lo, s2_max,
        args.sector_nodes, args.sector_step_tol)
    runtime_s = time.time() - t_sec0

    # n=1 also carries the located CROSSING s2_c = -omega2_10 (an eigenvalue
    # is never a crossing -- both are stored, distinguishably)
    pole_eigenvalue_keys = ({'omega2_00': omega2} if n == 0 else
             {'omega2_10': omega2, 's2_c': float(eig_npz['s2_c'])})
    # ---- WRITE sector_n<sector>_<tag>.npz, read by assemble_lnD (stage 7) ---
    sector_npz_path = paths[out_key]
    atomic_savez(sector_npz_path,
             s2_sector=s2_sector,
             delta_geq3=delta_geq3, delta_sub=delta_sub,
             Gbar_minus_fv=Gbar_traces['Gbar_minus_fv'],
             Gbar_1=Gbar_traces['Gbar_1'], Gbar_2=Gbar_traces['Gbar_2'],
             Gbar_fv=Gbar_traces['Gbar_fv'],
             crossing_resid=crossing_resid,
             Delta_n=Delta_n,
             sector_step_meter=sector_step_meter,
             sector_step_tol=float(args.sector_step_tol),
             deg=float((n + 1) ** 2),        # (n+1)^2 angular degeneracy
             s2_max=s2_max,
             sector_s2_lo=sector_s2_lo,
             sector_s2_lo_is_default=(args.sector_s2_lo is None),
             sector_nodes=int(len(s2_sector)),
             runtime_s=runtime_s,
             **pole_eigenvalue_keys,
             **provenance_stamp(args.bounce_npz, background['m2'],
                                background['params'], background['fields']))
    print(f"[SECTOR n={n}] SUMMARY: sector grid [{sector_s2_lo:g}, "
          f"{s2_max:g}], {len(s2_sector)} nodes; runtime {runtime_s:.1f} s")
    print(f"       omega2_{n}0 = {omega2:+.10e} "
          f"(pole-subtraction eigenvalue; READ from {eig_key})")
    print(f"       Delta_n = {Delta_n:+.6f} "
          f"(SECTOR-STEP meter {sector_step_meter:.2e}; "
          f"limit {args.sector_step_tol:g})")
    print(f"[OK] wrote {sector_npz_path}")


if __name__ == '__main__':
    main()
