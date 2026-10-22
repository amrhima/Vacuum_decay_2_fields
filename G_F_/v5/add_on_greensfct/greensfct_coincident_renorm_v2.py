#!/usr/bin/env python3
"""greensfct_coincident_renorm_v2.py -- the RENORMALISED coincident
Green's-function matrix [G_ren(x, x, s^2)]_ij at ONE s^2, the add-on's END
PRODUCT (add-on README tree: level 3, branch (j)); dim reg + MSbar, scale
mu = sum_i m_i unless --mu overrides it:

    python3 greensfct_coincident_renorm_v2.py --s2 <s2> --n-max <n_max>
    python3 greensfct_coincident_renorm_v2.py --s2 <s2> --n-max <n_max> --mu <scale>

READS   bounce_data_<tag>.npz (level 1): R, U_int, m2, mu and the engine on
        them (toolbox background_and_engine); band_cutoffs_<tag>.npz
        (level 2): n_max, unless --n-max is given.  Functions of other
        codes: coincident_wave (toolbox), fv_coincident_renorm_matrix
        (greensfct_fv_coincident_renorm_v2), Utilde_matrix and Pi_fin_fv_q of
        the fish add-back stage (addback_fish_vgeneral_phi_v2).
WRITES  greensfct_coincident_renorm_v2_<tag>_s2<v>.npz: r_grid, G_ren
        (nr,N,N) and its three pieces G0_fv_ren_addback (N,N), G1_ren_addback
        (nr,N,N), G_remainder (nr,N,N) [the deg-weighted Dyson-order >= 2
        remainders of the waves n = 0..n_max, summed], n_max, s2, mu; read
        by plot_add_on_greensfct/plot_greensfct_coincident_renorm_v2.py
        and by the thin-wall crosschecks (--pipeline-npz, --run-npz).
        The file name carries the tag and s2 only, so a check run with a
        larger --n-max or another --mu overwrites this file: run it into
        another --data-dir, with --bounce-npz and --n-max given explicitly.

THE FORMULA (the wave sum stops at n_max):

  [G_ren]_ij =   delta_ij tau_fin_fv(kappa_i^2)             <- Dyson-order-0 add-back
               + [G1_ren]_ij(r)                          <- Dyson-order-1 add-back
               + 1/(2 pi^2) sum_{n=0}^{n_max} (n+1)^2    <- numerical remainder,
                   K_nu(kappa_i r) I_nu(kappa_j r)/r^2      Dyson orders >= 2
                   [G_bracket_born_order_2 + G_bracket_geq3]_ij(r)
                   (= G_bracket - 1 - G_bracket_born_order_1)

with (all masses SHIFTED, kappa_i^2 = m_i^2 + s^2):
  tau_fin_fv(W^2)     = W^2/(16 pi^2) [ln(W^2/mu^2) - 1]
  [G1_ren]_ij(r)   = 1/(64 pi^4 r) Int dq q^2 J1(qr) Utilde_ij(q)
                                     Int_0^1 dx ln(Delta_ij(x,q^2)/mu^2)
  Utilde_ij(q)     = 4 pi^2 Int dr r^3 J1(qr)/(qr) [U_int]_ij(r)
                     (the 4D Fourier transform of the insertion)
  Delta_ij(x,q^2)  = (1-x) kappa_i^2 + x kappa_j^2 + x(1-x) q^2
                     (the denominator of the two propagators kappa_i, kappa_j
                     combined with the Feynman parameter x, as in Pi_fin_fv_q of
                     addback_fish_vgeneral_phi_v2)
  [G_bracket_born_order_1]_ik = [h1^+]_ik + [h1^-]_ki - [C_born_order_1]_ki
                     (Dyson order 1 of the bracket, from the engine's
                     h-ODE chain; C_born_order_1 = h1^-(inf))

HOW (main does the two renormalisation steps in this order):
1. The subtraction waves, wave by wave inside the sum: every wave G_n(r, r)
   loses its free wave (the bracket's 1, Dyson order 0: the waves of
   greensfct_fv_coincident_v2.py) and its tadpole wave
   (G_bracket_born_order_1, Dyson order 1), the two orders whose wave sums
   diverge, each removed whole, divergent and finite part together.  No
   number is subtracted: the remainder is taken directly from the engine's
   Born orders 2 and >= 3 (G_n_coinc_dyson_geq2 of coincident_wave), so
   nothing large cancels, and every wave is solved once.  The deg-weighted
   remainder falls like (n+1)^-3, so n_max is a truncation, not a regulator;
   its error is this code's own (the band watchers check the determinant,
   not this sum): compare a larger --n-max at the radii and s2 of interest.
2. The add-backs, once in x space: the MSbar finite parts of the two
   subtracted orders, order 0 from fv_coincident_renorm_matrix, order 1 from
   the Hankel quadrature G1_ren below; they carry all the mu-dependence.
G_ren is the full resolvent, not projected: its waves n = 0 (negative mode)
and n = 1 (zero modes) keep their poles at s2 = -omega^2; the pipeline
removes them only at the determinant level, and
greensfct_dangerous_modes_v2.py computes the pole terms.
"""
import argparse

import numpy as np
from scipy.special import j1
from scipy.integrate import trapezoid

from greensfct_toolbox_v2 import (add_greensfct_cli, background_and_engine, radial_grid,
                                  resolve_bounce, save_greensfct_npz,
                                  check_cli, coincident_wave,
                                  read_n_max)
# the toolbox import above is what puts computation/ on sys.path
from pipeline_helpers_vgeneral_phi_v2 import s2_filetag
from addback_fish_vgeneral_phi_v2 import Utilde_matrix, Pi_fin_fv_q
# the Dyson-order-0 add-back delta_ij tau_fin_fv(kappa_i^2): the renormalised
# coincident FV matrix of greensfct_fv_coincident_renorm_v2
from greensfct_fv_coincident_renorm_v2 import fv_coincident_renorm_matrix


# --------------------------------------------------------------------------- #
#  The Dyson-order-1 MSbar add-back G1_ren(r), from the fish stage's          #
#  Utilde_ij(q) and Pi_fin_fv_ij(q^2) at the shifted masses kappa_i^2         #
# --------------------------------------------------------------------------- #
def G1_ren(r_grid, R, U_int, m2, kappa_i, mu, nodes_per_period=16):
    """The Dyson-order-1 add-back [G1_ren]_ij(r): the MSbar value of the
    first Dyson (tadpole) term -G_FV U_int G_FV at coincidence, which the
    wave-by-wave subtraction of G_bracket_born_order_1 removed whole:
        1/(64 pi^4 r) Int dq q^2 J1(qr) Utilde_ij(q)
                       Int_0^1 dx ln(Delta_ij(x, q^2)/mu^2)
    (equivalently  - inverse-Hankel of Utilde_ij * Pi_fin_fv_ij).  Its scale
    dependence, d G1_ren/d ln mu = -U_int(r)/(8 pi^2), tests this quadrature
    (two runs at different --mu): mu enters only through the logarithm, and
    the inverse Hankel transform of Utilde_ij is U_int itself.  Traced over
    spacetime it is where this add-back meets the determinant path: with the
    tadpole moments M_i = 2 pi^2 Int dr r^3 [U_int]_ii,
        Int d^4x [G1_ren]_ii = -Utilde_ii(0) Pi_fin_fv_ii(0)
                             = M_i ln(kappa_i^2/mu^2)/(16 pi^2)
                             = d/ds2 [M_i tau_fin_fv(m_i^2 + s2)];
    assemble_lnD_vgeneral_phi_v2 books the s2-integral of this trace in
    closed form (the part beyond s2_max in dim reg) as its terms (T2) + (T4),
    which sum to the tadpole add-back
    A1_fin = sum_i M_i tau_fin_fv(m_i^2) (addback_tadpole_vgeneral_phi_v2).
    U_int here is the insertion on the bounce grid, zero beyond
    r_max = R[-1]: the M_i of this trace are the grid part of the moments
    (the stored tadpole_moment adds the bounce's tail beyond r_max in
    closed form), and the jump of U_int at r_max rings in G1_ren at the
    relative size of the bounce's tail there (r_max lies several decay
    lengths past the wall), visibly only at the two ends of r_grid.
    Returns (n_r, N, N)."""
    N = len(kappa_i)
    # the momentum window belongs to the insertion: U_int(r) varies on lengths
    # no shorter than about 1/m_hessian_max (m_hessian_max^2 = the largest
    # |eigenvalue| of the Hessian diag(m_i^2) + U_int(r) along the bounce), so
    # Utilde_ij(q) has died off long before q = 12 m_hessian_max; s2 and mu do
    # not change it.  12 is a dimensionless accuracy factor for every model.
    m_hessian_max = float(np.sqrt(np.max(np.abs(
        np.linalg.eigvalsh(U_int + np.diag(m2))))))
    q_max = 12.0 * m_hessian_max
    # the inverse kernel J1(q r) oscillates in q with the period 2 pi / r,
    # fastest at the largest radius: nodes_per_period nodes into that period
    n_q = int(np.ceil(nodes_per_period * q_max * float(r_grid[-1])
                      / (2.0 * np.pi))) + 1
    # the grid starts at q = 0: Utilde_matrix takes the q -> 0 kernel
    # exactly, and the q^2 measure below vanishes at that node anyway
    q_grid = np.linspace(0.0, q_max, n_q)
    # the forward transform, on its owner's uniform r grid resolved by the
    # same nodes_per_period rule
    Utilde = Utilde_matrix(q_grid, R, U_int, nodes_per_period)     # (nq, N, N)
    # -Pi_fin_fv_ij = +1/(16 pi^2) Int_0^1 dx ln(Delta_ij/mu^2) (Pi_fin_fv_q of
    # addback_fish at the shifted masses kappa_i^2), one curve per UNORDERED
    # pair: Pi_fin_fv_ij = Pi_fin_fv_ji (x -> 1 - x in the integral)
    minus_Pi_fin_fv = np.empty((len(q_grid), N, N))
    for i in range(N):
        for j in range(i, N):
            minus_Pi_fin_fv_ij = -Pi_fin_fv_q(q_grid, kappa_i[i]**2, kappa_i[j]**2, mu)
            minus_Pi_fin_fv[:, i, j] = minus_Pi_fin_fv_ij
            minus_Pi_fin_fv[:, j, i] = minus_Pi_fin_fv_ij
    integrand = q_grid[:, None, None]**2 * Utilde * minus_Pi_fin_fv  # (nq, N, N)
    G1_ren_addback = np.empty((len(r_grid), N, N))
    # the inverse Hankel transform, radius by radius: 1/(4 pi^2 r) times the
    # 1/(16 pi^2) inside -Pi_fin_fv = the 1/(64 pi^4 r) of the docstring
    # (scipy.integrate.trapezoid over q)
    for k, r in enumerate(r_grid):
        G1_ren_addback[k] = trapezoid(j1(q_grid * r)[:, None, None] * integrand,
                                      q_grid, axis=0) / (4.0 * np.pi**2 * r)
    return G1_ren_addback


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_greensfct_cli(ap)
    ap.add_argument('--n-max', type=int, default=None,
                    help='wave cutoff of the sum (default: the certified '
                         'n_max of the run\'s band_cutoffs_<tag>.npz)')
    ap.add_argument('--mu', type=float, default=None,
                    help='MSbar scale (default: the pipeline scale '
                         'mu = sum_i m_i from the background)')
    args = ap.parse_args()

    bounce_npz = resolve_bounce(args)

    n_max = (args.n_max if args.n_max is not None
             else read_n_max(args.data_dir, args.tag, bounce_npz))
    background, engine = background_and_engine(bounce_npz)
    check_cli(args, background)
    r_grid = radial_grid(engine, args.n_r)
    nr, N = len(r_grid), engine.N
    mu = float(args.mu) if args.mu is not None else float(background['mu'])
    kappa_i = engine.kappa(args.s2)       # sqrt(m_i^2 + s2): the add-backs' shifted masses

    print(f'[GF-ADDON] RENORMALISED coincident G(x,x):  s2={args.s2:g}  '
          f'mu={mu:g}  n_max={n_max}  N={N}')

    # ---- the subtracted wave sum: every wave n = 0..n_max contributes only
    #      its Dyson orders >= 2, its two subtraction waves left out
    #      (coincident_wave; 'geq2' counts Dyson orders, not waves) --------
    G_remainder = np.zeros((nr, N, N))
    for n in range(n_max + 1):
        deg_weight = (n + 1)**2 / (2.0 * np.pi**2)
        _, G_n_coinc_dyson_geq2 = coincident_wave(engine, n, args.s2, r_grid)
        G_remainder += deg_weight * G_n_coinc_dyson_geq2
        print(f'  wave n={n:3d} done')

    # ---- the two analytic MSbar add-backs, applied once in x space -------
    # Dyson order 0:  delta_ij tau_fin_fv(kappa_i^2)  (r-independent, diagonal),
    # the renormalised coincident FV matrix
    G0_fv_ren_addback = fv_coincident_renorm_matrix(kappa_i, mu)
    # Dyson order 1:  the Hankel-quadrature matrix [G1_ren]_ij(r)
    print('[GF-ADDON] computing the Dyson-order-1 add-back (Hankel quadrature) ...')
    G1_ren_addback = G1_ren(r_grid, background['R'], background['U_int'],
                            background['m2'], kappa_i, mu)

    G_ren = G0_fv_ren_addback[None, :, :] + G1_ren_addback + G_remainder

    out = (f'greensfct_coincident_renorm_v2_{args.tag}'
           f'_s2{s2_filetag(args.s2)}.npz')
    save_greensfct_npz(f'{args.data_dir}/{out}', bounce_npz, background,
                r_grid=r_grid, G_ren=G_ren,
                G0_fv_ren_addback=G0_fv_ren_addback,
                G1_ren_addback=G1_ren_addback,
                G_remainder=G_remainder,
                n_max=n_max, s2=args.s2, mu=mu)
    print(f'[NOTE] the wave sum stops at n_max = {n_max}.')


if __name__ == '__main__':
    main()
