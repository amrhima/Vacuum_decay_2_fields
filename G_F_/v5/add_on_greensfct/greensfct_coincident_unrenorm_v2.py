#!/usr/bin/env python3
"""greensfct_coincident_unrenorm_v2.py -- the UNRENORMALISED coincident
Green's-function matrix [G(x, x, s^2)]_ij at ONE s^2 (add-on README tree:
level 3, branch (f)): the divergence exhibit beside the end product
greensfct_coincident_renorm_v2.py, which removes that divergence.

    python3 greensfct_coincident_unrenorm_v2.py --s2 <s2> --n-max <n_max>

READS   bounce_data_<tag>.npz (level 1): R, U_int, m2 and the engine on them
        (toolbox background_and_engine); band_cutoffs_<tag>.npz (level 2):
        n_max, unless --n-max is given.  Each wave [G_n(r,r)]_ij comes from
        the toolbox's coincident_wave.
WRITES  greensfct_coincident_unrenorm_v2_<tag>_s2<v>.npz: r_grid (nr,),
        G_unren (nr, N, N), G_unren_half (nr, N, N) [the same partial sum
        at n_max/2 = n_half], n_max, n_half, s2; read by
        plot_add_on_greensfct/plot_greensfct_coincident_unrenorm_v2.py and
        by the thin-wall crosscheck (--pipeline-npz).

THE FORMULA (the degeneracy-weighted coincident partial-wave sum):  with
nu = n+1, kappa_i = sqrt(m_i^2 + s^2),

    [G(x,x,s^2)]_ij = 1/(2 pi^2) sum_{n=0}^{n_max} (n+1)^2
                        K_nu(kappa_i r) I_nu(kappa_j r) / r^2  [G_bracket(r)]_ij ,

    [G_bracket]_ik = sum_ab [1+h^+]_ia [C^-1]_ba [1+h^-]_kb   (code G_bracket;
                     the identity in the false vacuum, where h^pm = 0 and C = 1).

THIS OBJECT DIVERGES WITH n_max.  For large n at fixed r the deg-weighted
summand is
    (n+1)/(4 pi^2 r^2) delta_ij
        - (kappa_i^2 delta_ij + [U_int(r)]_ij)/(8 pi^2 (n+1)) + O((n+1)^-3),
the free wave (Dyson order 0) giving the first term and the kappa_i^2 part,
the tadpole wave (Dyson order 1) the U_int part, so the partial sums grow
like n_max^2/(8 pi^2 r^2), with a linear and a logarithmic part: n_max is an
ULTRAVIOLET REGULATOR here, not the truncation of a convergent series.  The
code exhibits the divergence that greensfct_coincident_renorm_v2.py removes.
HOW: main applies the degeneracy weight (n+1)^2/(2 pi^2) to every wave
n = 0..n_max, sums, and copies the running sum at n_half (G_unren_half),
which makes the regulator dependence visible.
"""
import argparse

import numpy as np

from greensfct_toolbox_v2 import (add_greensfct_cli, background_and_engine, radial_grid,
                                  resolve_bounce, save_greensfct_npz,
                                  check_cli, coincident_wave,
                                  read_n_max)
# the toolbox import above is what puts computation/ on sys.path
from pipeline_helpers_vgeneral_phi_v2 import s2_filetag


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_greensfct_cli(ap)
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

    print(f'[GF-ADDON] UNrenormalised coincident G(x,x):  s2={args.s2:g}  '
          f'n_max={n_max}  N={N}')
    print('[NOTE] this object DIVERGES with n_max (summand ~ (n+1)/4pi^2r^2 '
          'on the diagonal); the partial sum at n_half = n_max/2, stored '
          'alongside, makes the regulator dependence visible.')

    # the wave sum: every wave n = 0..n_max from the toolbox's coincident_wave,
    # deg-weighted by (n+1)^2/(2 pi^2) into G_unren; the running sum is copied
    # at n_half, the half-cutoff exhibit stored beside it
    G_unren = np.zeros((nr, N, N))
    n_half = n_max // 2
    for n in range(n_max + 1):
        deg_weight = (n + 1)**2 / (2.0 * np.pi**2)
        G_n_coinc, _ = coincident_wave(engine, n, args.s2, r_grid)
        G_unren += deg_weight * G_n_coinc             # the full wave, deg-weighted
        if n == n_half:
            G_unren_half = G_unren.copy()             # the partial sum at n_half
        print(f'  wave n={n:3d} done')

    out = (f'greensfct_coincident_unrenorm_v2_{args.tag}'
           f'_s2{s2_filetag(args.s2)}.npz')
    save_greensfct_npz(f'{args.data_dir}/{out}', bounce_npz, background,
                r_grid=r_grid, G_unren=G_unren, G_unren_half=G_unren_half,
                n_max=n_max, n_half=n_half, s2=args.s2)


if __name__ == '__main__':
    main()
