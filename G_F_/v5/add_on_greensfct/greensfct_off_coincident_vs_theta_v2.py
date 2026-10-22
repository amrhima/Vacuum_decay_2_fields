#!/usr/bin/env python3
"""greensfct_off_coincident_vs_theta_v2.py -- the position-space
off-coincident Green's-function matrix [G(x, x', s^2)]_ij at ONE s^2 as a
function of the ANGLE Theta between the two points, at ONE fixed pair of
radii; add-on README tree: level 3, branch (e) (needs the bounce and the wave
cutoff n_max of the band):

    python3 greensfct_off_coincident_vs_theta_v2.py --s2 <s2>                 (r_W pair)
    python3 greensfct_off_coincident_vs_theta_v2.py --s2 <s2> --r <radius> --r-prime <radius>

READS   bounce_data_<tag>.npz (stage 0, level 1) through load_background, on
        which the production h-ODE engine is built (background_and_engine)
        and from which the wall radius r_W of the default radii is read
        (wall_radius); band_cutoffs_<tag>.npz (stage 1, level 2) for the wave
        cutoff n_max (read_n_max; always the run's own).  Every wave
        [G_n(r0, r0')]_ij is read off the grid assembled by build_G_n of
        greensfct_off_coincident_radial_v2.py (the master formula, the full
        resolvent of every wave), sampled at the two evaluation radii and
        both integration endpoints (LSODA's steps do not depend on these
        samples).

WRITES  greensfct_off_coincident_vs_theta_v2_<tag>_s2<v>.npz (the npz carries THIS
        code's name) with
        theta_grid (nt,), G_theta (nt, N, N)  [the assembled curves],
        G_n_at_r0_pair (n_max+1, N, N)  [the per-wave values at (r0, r0')],
        G_partial_sums_theta0 (n_max+1, N, N)  [the partial sums of the Theta = 0 wave sum,
                              G_partial_sums_theta0[K] = 1/(2 pi^2) sum_{n<=K} (n+1)^2 G_n(r0, r0'):
                              the convergence in n],
        r0, r0_prime, r_W, s2, n_max, last_wave_magnitude [the last-wave
        magnitude diagnostic].

HOW     WHAT Theta IS: by O(4) symmetry G(x, x', s^2) depends on the positions
only through THREE independent coordinates -- the two radii r = |x|,
r' = |x'| and the angle Theta between the two points' directions
(cos Theta = e_r . e_r').  The companion code
greensfct_off_coincident_v2.py scans the two radii at ONE Theta; THIS code
scans Theta at ONE pair of radii (r0, r0'), so the pair shows all three
coordinates.  Theta runs over the full range [0, pi] on a uniform grid;
Theta = 0 is the same-ray configuration (both points on one ray from the
bubble centre, separation purely radial), Theta = pi the antipodal one.

THE FORMULA (the angular partial-wave sum over the radial Green's function):

    [G(x,x',s^2)]_ij = 1/(2 pi^2) sum_{n=0}^{n_max} (n+1) U_n(cos Theta)
                                        [G_n(r0, r0', s^2)]_ij ,

U_n = Chebyshev polynomials of the second kind (U_n(1) = n+1; chebyshev_U):
main solves the waves first (G_n_at_r0_pair), then forms the angular
weights (n+1) U_n(cos Theta) on the Theta grid and contracts the two over n.

CONVERGENCE: at separated radii (r0 != r0') the radial factor falls like
(r_</r_>)^nu, so the summand decays EXPONENTIALLY in n and the
truncated sum is the answer at EVERY Theta -- this is why the code refuses
equal radii (there the summand does not decay and the plain sum is only
Abel-summable).  n_max is the production wave cutoff of the run
(band_cutoffs npz); the last-wave magnitude is printed as a convergence
diagnostic.  It is not a bound on the sum of omitted waves.

DEFAULT RADII (printed): r0 = r_W (the background wall radius, R at
max ||dphi_b/dr|| -- the pipeline's one wall rule) and r0' = r_W / 2, so
the default scan correlates a point ON the wall with one halfway in.
"""
import argparse

import numpy as np

from greensfct_toolbox_v2 import (add_greensfct_cli, background_and_engine,
                                  check_cli, chebyshev_U, read_n_max,
                                  resolve_bounce, save_greensfct_npz)
from greensfct_off_coincident_radial_v2 import build_G_n
from pipeline_helpers_vgeneral_phi_v2 import s2_filetag, wall_radius


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # the folder's uniform CLI (--s2 REQUIRED, --data-dir, --bounce-npz,
    # --tag; no --n-r: the radial samples are fixed by the two radii), plus the fixed-radii
    # selectors of the angle scan
    add_greensfct_cli(ap, need_r_grid=False)
    ap.add_argument('--r', type=float, default=None,
                    help='first radius r0 (default: the background wall '
                         'radius r_W, printed)')
    ap.add_argument('--r-prime', type=float, default=None,
                    help='second radius r0\' (default: r_W / 2, printed); '
                         'must differ from --r (at equal radii the wave sum '
                         'does not converge)')
    ap.add_argument('--n-theta', type=int, default=361,
                    help='number of Theta grid points on [0, pi] '
                         '(default 361)')
    args = ap.parse_args()

    bounce_npz = resolve_bounce(args)
    background, engine = background_and_engine(bounce_npz)
    check_cli(args, background)
    N = engine.N

    # ---- the production wave cutoff of this run ---------------------------
    n_max = read_n_max(args.data_dir, args.tag, bounce_npz)

    # ---- the fixed radii: defaults from the background wall radius --------
    r_W = wall_radius(background['R'], background['phi_b'])
    r0 = float(args.r) if args.r is not None else float(r_W)
    r0_prime = (float(args.r_prime) if args.r_prime is not None
                else float(r_W) / 2.0)
    print(f'[GF-ADDON] G(x,x\') vs Theta:  s2={args.s2:g}  n_max={n_max}  '
          f'N={N}')
    print(f'[RADII] background wall radius r_W = {r_W:.6f};  evaluating at '
          f'r0 = {r0:.6f}' + (' (= r_W)' if args.r is None else '')
          + f',  r0\' = {r0_prime:.6f}'
          + (' (= r_W/2)' if args.r_prime is None else ''))
    if abs(r0 - r0_prime) <= 1e-9 * max(abs(r0), abs(r0_prime)):
        raise RuntimeError('[ABORT] --r equals --r-prime: at equal radii the '
                           'wave summand does not decay (only Abel-summable) '
                           '-- pick two different radii.')
    for lab, v in (('--r', r0), ('--r-prime', r0_prime)):
        if not (engine.r_start <= v <= engine.r_max):
            raise RuntimeError(f'[ABORT] {lab} = {v:g} outside the radial box '
                               f'[{engine.r_start:g}, {engine.r_max:.4f}].')

    # ---- output samples: integration endpoints and evaluation radii ------
    # LSODA controls its internal steps independently of this output grid;
    # the samples are the two requested radii plus both endpoints, which the
    # engine needs: the h-ODE solve runs from r_start to r_max over these
    # samples and C is read at the last one, r_max.
    r_grid = np.unique(np.asarray(
        [engine.r_start, r0, r0_prime, engine.r_max], float))
    # r0 and r0' are entries of r_grid, so each is an exact grid node
    i_r0 = int(np.searchsorted(r_grid, r0))
    i_r0_prime = int(np.searchsorted(r_grid, r0_prime))

    # ---- the wave loop: one pair of branch solves per n -------------------
    G_n_at_r0_pair = np.empty((n_max + 1, N, N))       # at (r0, r0')
    # every wave from build_G_n (greensfct_off_coincident_radial_v2.py), a
    # (len(r_grid), len(r_grid), N, N) array on the samples; keep (r0, r0')
    for n in range(n_max + 1):
        G_n, _, _, _ = build_G_n(engine, n, args.s2, r_grid)
        G_n_at_r0_pair[n] = G_n[i_r0, i_r0_prime]
        print(f'  wave n={n:3d} done  max|G_n(r0,r0\')| = '
              f'{np.max(np.abs(G_n_at_r0_pair[n])):.3e}', flush=True)

    # last-wave magnitude: |U_n(cos Theta)| <= n+1, so the last term
    # is bounded by (n_max+1)^2 |G_{n_max}| / (2 pi^2) at every Theta (the
    # summand decays exponentially in n at separated radii)
    last_wave_magnitude = float((n_max + 1)**2 * np.max(np.abs(G_n_at_r0_pair[-1]))
                          / (2.0 * np.pi**2))
    print(f'[INFO] last-wave magnitude over all Theta: {last_wave_magnitude:.3e}')

    # ---- assemble the Theta curves ----------------------------------------
    theta_grid = np.linspace(0.0, np.pi, int(args.n_theta))
    n_plus_1 = np.arange(1, n_max + 2)                        # nu = n + 1 of every wave
    angular_weight = np.empty((len(theta_grid), n_max + 1))   # (n+1) U_n(cos Theta)
    for t, th in enumerate(theta_grid):
        angular_weight[t] = n_plus_1 * chebyshev_U(n_max, float(np.cos(th)))
    # G_theta[t] = 1/(2 pi^2) sum_n angular_weight[t, n] G_n(r0, r0'): the sum
    # runs over the wave index n only
    G_theta = np.einsum('tn,nij->tij', angular_weight, G_n_at_r0_pair) / (2.0 * np.pi**2)
    # the partial sums of the same-ray slice Theta = 0, where U_n(1) = n+1 and
    # every wave enters with the weight (n+1)^2:
    #   G_partial_sums_theta0[K] = 1/(2 pi^2) sum_{n=0}^{K} (n+1)^2 G_n(r0, r0')   (numpy cumsum over n)
    G_partial_sums_theta0 = (np.cumsum((n_plus_1**2)[:, None, None] * G_n_at_r0_pair,
                     axis=0) / (2.0 * np.pi**2))

    out = (f'greensfct_off_coincident_vs_theta_v2_{args.tag}'
           f'_s2{s2_filetag(args.s2)}.npz')
    save_greensfct_npz(f'{args.data_dir}/{out}', bounce_npz, background,
                theta_grid=theta_grid, G_theta=G_theta, G_n_at_r0_pair=G_n_at_r0_pair,
                G_partial_sums_theta0=G_partial_sums_theta0, r0=r0, r0_prime=r0_prime, r_W=r_W, s2=args.s2, n_max=n_max,
                last_wave_magnitude=last_wave_magnitude)


if __name__ == '__main__':
    main()
