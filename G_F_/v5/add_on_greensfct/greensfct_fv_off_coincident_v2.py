#!/usr/bin/env python3
"""greensfct_fv_off_coincident_v2.py -- the position-space OFF-COINCIDENT
false-vacuum matrix [G_FV(x, x', s^2)]_ij at ONE s^2 and ONE angle Theta,
the free counterpart of greensfct_off_coincident_v2.py; add-on README tree:
level 3, branch (g) (needs the bounce and the wave cutoff n_max of the band):

    python3 greensfct_fv_off_coincident_v2.py --s2 <s2> [--n-max <n_max>] (same ray)
    python3 greensfct_fv_off_coincident_v2.py --s2 <s2> [--n-max <n_max>] --theta <Theta>

READS   bounce_data_<tag>.npz (stage 0, level 1) through load_background --
        only the channel masses m2 (the shifted masses kappa_i = engine.kappa)
        and the radial box [r_start, r_max] of the engine are used; NO h-ODE
        is solved, every wave is the closed Bessel form below;
        band_cutoffs_<tag>.npz (stage 1, level 2) for the wave cutoff n_max
        (read_n_max) unless --n-max is given.

WRITES  greensfct_fv_off_coincident_v2_<tag>_s2<v>.npz (the npz carries
        THIS code's name) with
        r_grid (nr,), G_fv (nr, nr, N, N) [axes r, r', i, j; NaN on the
        diagonal], theta, n_max, s2, kappa_i (N,), free_propagator_fv_dev (NaN
        when no grid pair is converged at n_max: not measured).  No plot code
        reads it: its exhibit is the printed and stored free_propagator_fv_dev.

HOW     THE OBJECT (finite and diagonal).  In the false vacuum every wave is
free Bessel, so the radial wave matrix is diagonal in the field channels,

    [G_FV_n(r, r')]_ij = delta_ij I_nu(kappa_i r_<) K_nu(kappa_i r_>) / (r r'),
    nu = n+1,  kappa_i = sqrt(m_i^2 + s^2),  r_< = min(r,r'), r_> = max(r,r'),

(fv_off_coincident_wave) and the position-space matrix is the same angular
partial-wave sum the interacting sibling (greensfct_off_coincident_v2.py)
performs, with Theta the angle between x and x' (cos Theta = e_r . e_r') and
U_n = Chebyshev of the second kind (chebyshev_U):

    [G_FV(x,x',s^2)]_ij = 1/(2 pi^2) sum_{n=0}^{n_max} (n+1) U_n(cos Theta)
                                                       [G_FV_n(r, r')]_ij .

Off the diagonal r = r' the radial factor falls like (r_</r_>)^nu, so the
truncated sum converges exponentially; ON it the summand does not decay --
at Theta = 0 the sum is the divergent coincident object (that exhibit is
greensfct_fv_coincident_v2.py), at Theta != 0 it is only Abel-summable
although the propagator itself is finite there -- so the diagonal is masked
with NaN, the same convention as the interacting sibling.

THE EXHIBIT: THE WAVE SUM IS THE FREE PROPAGATOR.  The same object has the
closed form of the massive Euclidean 4D propagator at the shifted mass,

    [G_FV(x,x')]_ii = kappa_i K_1(kappa_i d) / (4 pi^2 d),   d = |x - x'|,

(d^2 = r^2 + r'^2 - 2 r r' cos Theta), and this code CHECKS its wave sum
against that closed form on every grid pair where the truncation is
converged -- the printed free_propagator_fv_dev is the exhibit that the
partial-wave machinery reassembles the known propagator exactly.  The false
vacuum has no dangerous modes: this matrix has no pole term anywhere.
"""
import argparse

import numpy as np
from scipy.special import kv

from greensfct_toolbox_v2 import (add_greensfct_cli, background_and_engine,
                                  check_cli, chebyshev_U, radial_grid,
                                  resolve_bounce, save_greensfct_npz,
                                  log_bessel_tables,
                                  read_n_max)
from pipeline_helpers_vgeneral_phi_v2 import s2_filetag


def fv_off_coincident_wave(nu, kappa_i, r_grid):
    """ONE free wave [G_FV_n(r, r')]_ij on r_grid x r_grid, (nr, nr, N, N):
    delta_ij I_nu(kappa_i r_<) K_nu(kappa_i r_>)/(r r'), assembled in log
    space.  Same-channel and ORDERED (I at the smaller radius, K at the
    larger), so the exponent logI(kappa r_<) + logK(kappa r_>) is bounded --
    unlike the cross entries (i != j) of the coincident FV weight, no
    overflow guard is needed."""
    nr, N = len(r_grid), len(kappa_i)
    logI, logK = log_bessel_tables(nu, kappa_i, r_grid)       # (N, nr) each
    a = np.arange(nr)
    amin = np.minimum(a[:, None], a[None, :])                 # index of r_<
    amax = np.maximum(a[:, None], a[None, :])                 # index of r_>
    rr = r_grid[:, None] * r_grid[None, :]
    G_n_fv = np.zeros((nr, nr, N, N))
    for i in range(N):
        G_n_fv[:, :, i, i] = np.exp(logI[i][amin] + logK[i][amax]) / rr
    return G_n_fv


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
    # the engine is built only for its shared radial-grid rule and its
    # kappa_i; NO h-ODE is solved -- every FV wave is the closed Bessel form
    # above
    background, engine = background_and_engine(bounce_npz)
    check_cli(args, background)
    r_grid = radial_grid(engine, args.n_r)
    nr, N = len(r_grid), engine.N
    kappa_i = engine.kappa(args.s2)       # sqrt(m_i^2 + s2), the shifted masses
    cosT = float(np.cos(args.theta))
    U_n = chebyshev_U(n_max, cosT)                # U_0 .. U_n_max at cos Theta

    print(f'[GF-ADDON] off-coincident FV matrix:  s2={args.s2:g}  '
          f'Theta={args.theta:g}  n_max={n_max}  N={N}  '
          f'(analytic Bessel waves, no solves)')

    # ---- the wave sum over the free waves ---------------------------------
    #   G_FV(x,x') = 1/(2 pi^2) sum_n (n+1) U_n(cos Theta) G_FV_n(r, r')
    G_fv = np.zeros((nr, nr, N, N))
    for n in range(n_max + 1):
        G_fv += ((n + 1) * U_n[n] / (2.0 * np.pi**2)
                 * fv_off_coincident_wave(n + 1.0, kappa_i, r_grid))

    # ---- the closed-form CHECK: wave sum vs the free 4D propagator ---------
    # kappa_i K_1(kappa_i d)/(4 pi^2 d), d = |x - x'|.  Compared only where
    # the truncated sum is converged: each wave falls off like its radial
    # factor (r_</r_>)^nu, so pairs with (r_</r_>)^(n_max+1) above 1e-10 are
    # excluded (near the diagonal r = r' the deviation would measure the
    # truncation, not the propagator).
    r_lt = np.minimum(r_grid[:, None], r_grid[None, :])
    r_gt = np.maximum(r_grid[:, None], r_grid[None, :])
    d = np.sqrt(np.maximum(r_grid[:, None]**2 + r_grid[None, :]**2
                           - 2.0 * r_grid[:, None] * r_grid[None, :] * cosT, 0.0))
    converged = ((r_lt / r_gt) ** (n_max + 1) < 1e-10) & (d > 0.0)
    if not converged.any():
        # a low wave cutoff (a small --n-max) converges no grid pair: the
        # partial sum is still stored, the check is reported as not made
        free_propagator_fv_dev = float('nan')
        print(f'[CHECK] FV wave sum vs closed form: no grid pair is converged at '
              f'n_max = {n_max} (the check needs (r_</r_>)^(n_max+1) < 1e-10); '
              f'free_propagator_fv_dev is stored as NaN, not measured.')
    else:
        free_propagator_fv_dev_i = []
        with np.errstate(all='ignore'):
            for i in range(N):
                free_propagator_fv = kappa_i[i] * kv(1, kappa_i[i] * d) / (4.0 * np.pi**2 * d)
                rel_dev_fv = np.abs(G_fv[:, :, i, i] - free_propagator_fv) / np.abs(free_propagator_fv)
                free_propagator_fv_dev_i.append(float(np.max(rel_dev_fv[converged])))
        free_propagator_fv_dev = max(free_propagator_fv_dev_i)
        print(f'[CHECK] FV wave sum vs closed form kappa_i K_1(kappa_i d)/(4 pi^2 d)'
              f': max relative deviation = {free_propagator_fv_dev:.3e} on '
              f'{int(converged.sum())}/{converged.size} converged pairs '
              f'(per channel: {", ".join(f"{v:.2e}" for v in free_propagator_fv_dev_i)}) -- the '
              f'partial-wave sum reassembles the free propagator.')

    # the diagonal r = r', where the summand does not decay -- mask it
    # (same convention as the interacting sibling)
    G_fv[np.eye(nr, dtype=bool)] = np.nan

    out = (f'greensfct_fv_off_coincident_v2_{args.tag}'
           f'_s2{s2_filetag(args.s2)}.npz')
    save_greensfct_npz(f'{args.data_dir}/{out}', bounce_npz, background,
                r_grid=r_grid, G_fv=G_fv, theta=args.theta,
                n_max=n_max, s2=args.s2, kappa_i=kappa_i,
                free_propagator_fv_dev=free_propagator_fv_dev)


if __name__ == '__main__':
    main()
