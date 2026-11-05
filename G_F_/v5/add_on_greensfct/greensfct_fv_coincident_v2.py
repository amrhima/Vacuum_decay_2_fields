#!/usr/bin/env python3
"""greensfct_fv_coincident_v2.py -- the COINCIDENT false-vacuum matrix,
WAVE-RESOLVED (add-on README tree: level 3, branch (h)): the free (Bessel)
waves whose n-sum is the divergent [G_FV(x, x, s^2)]_ij, at ONE s^2:

    python3 greensfct_fv_coincident_v2.py --s2 <s2> --n-max <n_max>

READS   bounce_data_<tag>.npz (level 1): m2 and the radial box of the
        bounce (the engine is built for its grid rule and its kappa_i
        only; no h-ODE is solved); band_cutoffs_<tag>.npz (level 2): n_max,
        unless --n-max is given.  IK_prod_fv, the same-channel product
        I_nu K_nu, from fv_analytic_toolbox_vgeneral_phi_v2.
WRITES  greensfct_fv_coincident_v2_<tag>_s2<v>.npz: r_grid (nr,),
        G_n_fv_coinc_diag (n_max+1, nr, N) [the RAW per-wave FV waves
        I K / r^2 per channel -- no degeneracy weight], G_fv_sum (nr, N, N),
        G_fv_sum_half (nr, N, N) [the deg-weighted partial sums at n_max and
        n_max/2 = n_half], kappa_i (N,), n_max, n_half, s2.  No code
        reads the file: it is the divergence exhibit.

THE WAVE.  In the false vacuum every wave is free: the coincident bracket is
the identity, so the wave-n coincident FV matrix is the pure Bessel wave

    [G_FV_n(r)]_ij = delta_ij I_nu(kappa_i r) K_nu(kappa_i r) / r^2 ,
    nu = n+1,  kappa_i = sqrt(m_i^2 + s^2),

the diagonal of the FV weight of every coincident wave
(greensfct_toolbox_v2.fv_weight_matrix), taken here directly from the
bounded same-channel product IK_prod_fv.  These are the subtraction waves of
Dyson order 0 that the renormalised code greensfct_coincident_renorm_v2.py
leaves out of its sum, wave by wave.

THE n-SUM DIVERGES.  For large n at fixed r the deg-weighted summand is
(n+1)/(4 pi^2 r^2) - kappa_i^2/(8 pi^2 (n+1)) + O((n+1)^-3) on the diagonal,
so the partial sums grow like n_max^2/(8 pi^2 r^2), with a linear and a
logarithmic part: the Dyson-order-0 UV divergence.  What dim reg + MSbar
leaves of it is the finite matrix of greensfct_fv_coincident_renorm_v2.py.
HOW: main forms every wave n = 0..n_max (fv_coincident_wave), stores its
raw diagonal and sums it with the weight (n+1)^2/(2 pi^2), copying the
running sum at n_half, which shows the growth with the cutoff.
"""
import argparse

import numpy as np

from greensfct_toolbox_v2 import (add_greensfct_cli, background_and_engine,
                                  radial_grid, resolve_bounce,
                                  check_cli, save_greensfct_npz,
                                  read_n_max)
from pipeline_helpers_vgeneral_phi_v2 import s2_filetag
# I_nu(z) K_nu(z): the same-channel product the engine's traced FV weight
# w_fv_i = r I_nu K_nu(kappa_i r) is built from
from fv_analytic_toolbox_vgeneral_phi_v2 import IK_prod_fv


def fv_coincident_wave(nu, kappa_i, r_grid):
    """ONE wave of the coincident FV matrix, (nr, N, N): the diagonal
    delta_ij I_nu K_nu(kappa_i r)/r^2 (FV bracket = identity), zeros off the
    diagonal -- the Dyson-order-0 subtraction wave.  The same-channel product
    I_nu K_nu is bounded, so it is taken directly (IK_prod_fv), not in log
    space.  No degeneracy weight: the caller applies (n+1)^2/(2 pi^2) when
    summing."""
    IK_over_r2_fv = IK_prod_fv(nu, np.outer(r_grid, kappa_i)) / r_grid[:, None]**2   # (nr, N)
    return IK_over_r2_fv[:, :, None] * np.eye(len(kappa_i))   # delta_ij I_nu K_nu(kappa_i r)/r^2


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
    # the engine is built only for its shared radial-grid rule (engine.r_start
    # = R[1]/2, engine.r_max) and its kappa_i; NO h-ODE is solved here -- the
    # false vacuum is free, its waves are the closed Bessel forms above
    background, engine = background_and_engine(bounce_npz)
    check_cli(args, background)
    r_grid = radial_grid(engine, args.n_r)
    nr, N = len(r_grid), engine.N
    kappa_i = engine.kappa(args.s2)       # sqrt(m_i^2 + s2), the shifted masses

    print(f'[GF-ADDON] coincident FV matrix, wave-resolved:  s2={args.s2:g}'
          f'  n_max={n_max}  N={N}  (analytic Bessel waves, no solves)')
    print('[NOTE] the n-sum DIVERGES like n_max^2 '
          '(Dyson-order-0 UV divergence); the partial sum at '
          'n_half = n_max/2 makes the regulator dependence visible.')

    # the waves n = 0..n_max (fv_coincident_wave, nu = n+1): the raw diagonal
    # of each is stored, the deg-weighted sum runs into G_fv_sum and is
    # copied at n_half -- the two partial sums of the divergence exhibit
    G_n_fv_coinc_diag = np.empty((n_max + 1, nr, N))
    G_fv_sum = np.zeros((nr, N, N))
    n_half = n_max // 2
    for n in range(n_max + 1):
        G_n_fv_coinc = fv_coincident_wave(n + 1.0, kappa_i, r_grid)  # (nr,N,N)
        G_n_fv_coinc_diag[n] = np.diagonal(G_n_fv_coinc, axis1=1, axis2=2)   # the raw waves
        G_fv_sum += (n + 1)**2 / (2.0 * np.pi**2) * G_n_fv_coinc  # deg-weighted
        if n == n_half:
            G_fv_sum_half = G_fv_sum.copy()        # the partial sum at n_half

    out = (f'greensfct_fv_coincident_v2_{args.tag}'
           f'_s2{s2_filetag(args.s2)}.npz')
    save_greensfct_npz(f'{args.data_dir}/{out}', bounce_npz, background,
                r_grid=r_grid, G_n_fv_coinc_diag=G_n_fv_coinc_diag,
                G_fv_sum=G_fv_sum, G_fv_sum_half=G_fv_sum_half,
                kappa_i=kappa_i, n_max=n_max, n_half=n_half,
                s2=args.s2)


if __name__ == '__main__':
    main()
