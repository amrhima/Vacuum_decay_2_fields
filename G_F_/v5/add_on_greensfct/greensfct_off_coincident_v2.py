#!/usr/bin/env python3
"""greensfct_off_coincident_v2.py -- the position-space off-coincident
Green's-function matrix [G(x, x', s^2)]_ij at ONE s^2 and ONE angle Theta
between x and x', over the two radii (r, r'); add-on README tree: level 3,
branch (d) (needs the bounce and the wave cutoff n_max of the band):

    python3 greensfct_off_coincident_v2.py --s2 <s2> [--n-max <n_max>]   (same ray)
    python3 greensfct_off_coincident_v2.py --s2 <s2> [--n-max <n_max>] --theta <Theta>

READS   bounce_data_<tag>.npz (stage 0, level 1) through load_background, on
        which the production h-ODE engine is built (background_and_engine);
        band_cutoffs_<tag>.npz (stage 1, level 2) for the wave cutoff n_max
        (read_n_max) unless --n-max is given.  Every wave [G_n(r, r')]_ij is
        assembled by build_G_n of greensfct_off_coincident_radial_v2.py (the
        full resolvent of the wave; the n = 0, 1 pole terms are contained).

WRITES  greensfct_off_coincident_v2_<tag>_s2<v>.npz (the npz carries THIS
        code's name) with r_grid (nr,), G_off_coincident (nr, nr, N, N)
        [axes r, r', i, j; NaN on the diagonal r = r'], theta, n_max, s2,
        last_wave_magnitude [the magnitude of the last added term, wave
        n_max, off the diagonal: a convergence diagnostic, not a bound on
        the omitted waves].  Read, in the thin-wall test, by
        thinwall_greensfct_off_coincident_v2 (--pipeline-npz).

HOW     THE FORMULA (the angular partial-wave sum over the radial Green's function),
with Theta the angle between x and x' (cos Theta = e_r . e_r'):

    [G(x,x',s^2)]_ij = 1/(2 pi^2) sum_{n=0}^{n_max} (n+1) U_n(cos Theta)
                                        [G_n(r, r', s^2)]_ij ,

where (n+1) U_n(cos Theta)/(2 pi^2) = sum_{l,m} Y_nlm(e_r) Y_nlm(e_r')^* is the
sum over the (n+1)^2 hyperspherical harmonics of wave n; U_n = Chebyshev of
the second kind (U_n(1) = n+1; chebyshev_U), summed wave by wave in main.

THE n_max CUTOFF -- WHEN IT IS SAFE AND WHEN IT IS NOT:
  * r != r' :  the radial factor I_nu(kappa r_<) K_nu(kappa r_>) falls like
               (r_</r_>)^nu -- the summand decays EXPONENTIALLY in n
               and the truncated sum converges fast in n_max.
  * r == r', Theta != 0 :  I_nu K_nu -> 1/(2 nu), the summand ~ U_n(cos Theta)
               does NOT decay; the series is only Abel-summable and a plain
               truncated sum oscillates with O(1) amplitude.  This code
               therefore masks the diagonal r = r' with NaN and prints a
               warning -- do NOT compare anything on that diagonal.
  * the coincident limit (r'=r AND Theta=0) is not taken here: its wave sum
    diverges -- greensfct_coincident_unrenorm_v2.py exhibits that divergence,
    greensfct_coincident_renorm_v2.py gives the renormalised matrix.
"""
import argparse

import numpy as np

from greensfct_toolbox_v2 import (add_greensfct_cli, background_and_engine, radial_grid,
                                  resolve_bounce, save_greensfct_npz,
                                  check_cli, chebyshev_U,
                                  read_n_max)
from greensfct_off_coincident_radial_v2 import build_G_n
from pipeline_helpers_vgeneral_phi_v2 import s2_filetag


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_greensfct_cli(ap)
    ap.add_argument('--theta', type=float, default=0.0,
                    help='angle Theta between x and x-prime, in radians '
                         '(default 0: both points on the same ray)')
    ap.add_argument('--n-max', type=int, default=None,
                    help='wave cutoff of the sum (default: the certified '
                         'n_max of the run\'s band_cutoffs_<tag>.npz)')
    args = ap.parse_args()

    bounce_npz = resolve_bounce(args)

    n_max = (args.n_max if args.n_max is not None
             else read_n_max(args.data_dir, args.tag, bounce_npz))
    background, engine = background_and_engine(bounce_npz)
    check_cli(args, background)
    r_grid = radial_grid(engine, args.n_r)
    nr, N = len(r_grid), engine.N
    cosT = float(np.cos(args.theta))
    U_n = chebyshev_U(n_max, cosT)                # U_0 .. U_n_max at cos Theta

    print(f'[GF-ADDON] off-coincident G(x,x\'):  s2={args.s2:g}  '
          f'Theta={args.theta:g}  n_max={n_max}  N={N}')
    if args.theta != 0.0:
        print('[WARN] Theta != 0: on the sphere r = r\' the n-sum is only '
              'Abel-summable -- the diagonal is masked with NaN and the '
              'near-diagonal converges slowly.')

    # the wave sum 1/(2 pi^2) sum_n (n+1) U_n(cos Theta) G_n(r, r'), every
    # wave from build_G_n; the last term is kept for the diagnostic below
    G_off_coincident = np.zeros((nr, nr, N, N))
    for n in range(n_max + 1):
        G_n, _, _, _ = build_G_n(engine, n, args.s2, r_grid)
        term = (n + 1) * U_n[n] * G_n / (2.0 * np.pi**2)
        G_off_coincident += term
        last_wave_term = term
        print(f'  wave n={n:3d} done  (weight (n+1)U_n = {(n+1)*U_n[n]:+.3g})')

    # last-wave magnitude: the magnitude of the last added term (wave
    # n_max) off the diagonal, where the summand decays exponentially in n
    equal_radius = np.eye(nr, dtype=bool)                    # r = r' pairs
    last_wave_magnitude = float(np.max(np.abs(last_wave_term[~equal_radius])))

    # the diagonal r = r', where the summand does not decay (the divergent
    # coincident limit at Theta = 0, only Abel-summable at Theta != 0) --
    # mask it
    G_off_coincident[equal_radius] = np.nan

    out = (f'greensfct_off_coincident_v2_{args.tag}'
           f'_s2{s2_filetag(args.s2)}.npz')
    save_greensfct_npz(f'{args.data_dir}/{out}', bounce_npz, background,
                r_grid=r_grid, G_off_coincident=G_off_coincident,
                theta=args.theta, n_max=n_max,
                s2=args.s2, last_wave_magnitude=last_wave_magnitude)
    print(f'[INFO] last-wave magnitude off the diagonal (|term of '
          f'wave n_max|): {last_wave_magnitude:.3e}')


if __name__ == '__main__':
    main()
