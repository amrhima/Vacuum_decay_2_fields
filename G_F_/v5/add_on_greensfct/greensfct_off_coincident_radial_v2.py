#!/usr/bin/env python3
"""greensfct_off_coincident_radial_v2.py -- the radial Green's-function
matrix [G_n(r, r', s^2)]_ij of ONE wave n at ONE s^2, both from the command
line; add-on README tree: level 2, branch (f) (needs only the bounce), and
the ONE assembly of a wave (build_G_n) that every off-coincident code of the
folder reuses:

    python3 greensfct_off_coincident_radial_v2.py --s2 <s2> --n <n>

READS   bounce_data_<tag>.npz (stage 0, level 1) through load_background --
        the radial grid R, the bounce phi_b, the channel masses m2 and the
        interaction Hessian U_int(r) = H'(phi_b(r)) - diag(m2) -- on which the
        production h-ODE engine is built (background_and_engine).  The two
        h-branches of the wave are solved by that engine (solve_branches ->
        GreensFunction.branch_objects); this code only assembles.

WRITES  greensfct_off_coincident_radial_v2_<tag>_n<n>_s2<v>.npz in
        --data-dir (the npz carries THIS code's name) with
        r_grid (nr,), G_n (nr, nr, N, N)  [axes: r, r', i, j],
        C (N,N), kappa_i (N,), n, s2, coincident_sym_dev (the continuity
        meter), and -- for the plot layer, which must not solve --
        the two h-branches h_minus, h_plus (nr, N, N) and the free Bessel
        solutions B^pm they dress, stored as their logarithms
        log_B_minus_fv = log[I_nu(kappa_i r)/r], log_B_plus_fv = log[K_nu(kappa_i r)/r]
        (nr, N) (the bare functions overflow float64 at small r for large nu).
        Read by plot_greensfct_off_coincident_radial_v2, plot_h_functions_v2
        and plot_fv_free_modes_v2, and in the thin-wall test by
        thinwall_greensfct_off_coincident_radial_v2 (--pipeline-npz).
        build_G_n is imported by greensfct_off_coincident_v2 and
        greensfct_off_coincident_vs_theta_v2 (their wave sums) and by
        test_greensequation/greens_equation_test_vgeneral_phi_v2.

HOW     THE MASTER FORMULA (the radial Green's function from the two h-branches): the
free Bessel solutions B^-_i(r) = I_nu(kappa_i r)/r (regular at r = 0) and
B^+_i(r) = K_nu(kappa_i r)/r (decaying), dressed into the basis-function
matrices Phi^pm[i,a](r) = B^pm_i(r) [1+h^pm(r)]_ia (component i, basis
function a), matched through C^-1; with nu = n+1, kappa_i = sqrt(m_i^2 + s^2),

  r > r':  [G_n]_ij = sum_ab Phi^+[i,a](r) [C^-1]_ba Phi^-[j,b](r')
                    = K_nu(kappa_i r) I_nu(kappa_j r') / (r r')
                        * sum_ab [1+h^+(r)]_ia [C^-1]_ba [1+h^-(r')]_jb
  r < r':  [G_n]_ij = sum_ab Phi^-[i,a](r) [C^-1]_ab Phi^+[j,b](r')
                    = I_nu(kappa_i r) K_nu(kappa_j r') / (r r')
                        * sum_ab [1+h^-(r)]_ia [C^-1]_ab [1+h^+(r')]_jb

  (in both terms the '-' label sits in the FIRST slot of C^-1, the '+' label
  in the SECOND: bracket_outer_pairs / bracket_inner_pairs);
  C[a, b] = r^3 sum_i (Phi^+[i,a] Phi^-[i,b]' - Phi^+[i,a]' Phi^-[i,b]), the
  constant Wronskian of the two branches (the same at every r), evaluated
  at r_max where h^+ = h^+' = 0:
      C[a, b] = delta_ab + h^-[a, b](r_max)
                + r_max I_nu K_nu(kappa_a r_max) h^-'[a, b](r_max)
              = delta_ab + h^-[a, b](r -> inf)   (U_int = 0 beyond r_max);
  det C(n, s2) = 0 exactly when -s2 is an eigenvalue of wave n.

The r < r' term is the r > r' term with (r, i) and (r', j) exchanged, so the
reciprocity [G_n(r,r')]_ij = [G_n(r',r)]_ji holds by construction off the
equal-radius slice.  What the two terms must still satisfy is continuity at
r' = r, i.e. a symmetric equal-radius slice [G_n(r,r)]_ij = [G_n(r,r)]_ji;
it holds when C is the Wronskian of the two branches, and
coincident_symmetry_dev measures it.  The overall normalisation of C is
fixed by the unit source (derivative jump), tested by
test_greensequation/greens_equation_test_vgeneral_phi_v2.py (S_bulk + S_jump = 1).

THE OBJECT IS THE FULL RESOLVENT of the wave: for the two dangerous waves
n = 0, 1 it contains the pole term u_i(r) u_j(r')/(omega^2 + s^2) of the
negative mode / the translation zero mode (exhibited separately by
greensfct_dangerous_modes_v2.py on this same grid); at s2 = -omega^2 itself
solve_branches aborts.
"""
import argparse

import numpy as np

from greensfct_toolbox_v2 import (add_greensfct_cli, background_and_engine, radial_grid,
                                  resolve_bounce, save_greensfct_npz,
                                  solve_branches, log_bessel_tables,
                                  check_cli)
# the toolbox import above is what puts computation/ on sys.path
from pipeline_helpers_vgeneral_phi_v2 import (bracket_outer_pairs,
                                              bracket_inner_pairs, radius_first,
                                              s2_filetag)


def build_G_n(engine, n, s2, r_grid):
    """Assemble [G_n(r,r',s^2)]_ij on r_grid x r_grid (the master formula of
    the docstring) from ONE solve_branches call.  Returns
    (G_n, branch_objects, log_B_minus_fv, log_B_plus_fv): the matrix (nr, nr, N, N)
    [axes r, r', i, j], the outer term on r >= r' and the inner term on
    r < r'; the solve_branches dict it was built from; and the logarithms of
    the free Bessel solutions B^- and B^+ the h-branches dress, (nr, N) each."""
    branch_objects = solve_branches(engine, n, s2, r_grid)
    logI, logK = log_bessel_tables(branch_objects['nu'], branch_objects['kappa_i'],
                                   r_grid)                                  # (N, nr) each

    # the free Bessel solutions in log space (the master formula is the product
    # B^+_i(r) B^-_j(r') outer / B^-_i(r) B^+_j(r') inner, times the bracket):
    #   log B^-_i(r) = log I_nu(kappa_i r) - log r,  log B^+_i(r) = log K_nu(kappa_i r) - log r
    # laid out (radius, channel) by radius_first
    log_B_minus_fv = radius_first(logI) - np.log(r_grid)[:, None]              # (nr, N)
    log_B_plus_fv = radius_first(logK) - np.log(r_grid)[:, None]

    # the two numerical bracket stacks (one per orientation of r vs r'):
    #   outer (r > r'):  sum_ab [1+h^+(r)]_ia [C^-1]_ba [1+h^-(r')]_jb
    #   inner (r < r'):  sum_ab [1+h^-(r)]_ia [C^-1]_ab [1+h^+(r')]_jb
    one_plus_h_plus = branch_objects['one_plus_h_plus']                      # (nr, N, N)
    one_plus_h_minus = branch_objects['one_plus_h_minus']                     # (nr, N, N)
    bracket_out = bracket_outer_pairs(one_plus_h_plus, branch_objects['C_inv'], one_plus_h_minus)  # (nr, nr, N, N)
    bracket_in = bracket_inner_pairs(one_plus_h_minus, branch_objects['C_inv'], one_plus_h_plus)

    # the products of the free solutions in log space, on (r, r', i, j):
    #   outer:  log B^+_i(r) + log B^-_j(r')        inner:  log B^-_i(r) + log B^+_j(r')
    # Where each one is used (outer for r >= r', inner for r < r') it cannot
    # overflow: it is at most I_nu K_nu (< 1/(2 nu)) at the smaller radius
    # times the channel ratio I_nu(kappa_j .)/I_nu(kappa_i .), which the
    # engine's BESSEL-RATIO guard keeps below exp(BESSEL_LOG_RATIO_MAX) for
    # every pair in one block of U_int (engine.same_block); -log(r r') adds at
    # most 2 |log r_start|.  A pair in different blocks is not coupled (G_n is
    # zero there, its bracket entry round-off): its weight is set to 0
    # (exp(-inf)).  The unused corner (K at small r times I at large r) may
    # overflow to inf; np.where discards it.
    log_fv_weight_outer = np.where(engine.same_block, log_B_plus_fv[:, None, :, None]
                                + log_B_minus_fv[None, :, None, :], -np.inf)
    log_fv_weight_inner = np.where(engine.same_block, log_B_minus_fv[:, None, :, None]
                                + log_B_plus_fv[None, :, None, :], -np.inf)
    outer_mask = (r_grid[:, None] >= r_grid[None, :])   # r >= r' (at r = r' both terms agree)
    with np.errstate(over='ignore'):                    # the unused corner
        G_n = np.where(outer_mask[:, :, None, None],
                       np.exp(log_fv_weight_outer) * bracket_out,
                       np.exp(log_fv_weight_inner) * bracket_in)
    return G_n, branch_objects, log_B_minus_fv, log_B_plus_fv


def coincident_symmetry_dev(G_n):
    """The continuity meter of an assembled G_n.
    At r' = r the two terms of the master formula must agree,
    Phi^+ C^-T Phi^-T = Phi^- C^-1 Phi^+T, and the second is the transpose
    of the first, so continuity IS the symmetry of the equal-radius slice
    [G_n(r,r)]_ij = [G_n(r,r)]_ji.  It holds when C is the Wronskian of the
    two branches the brackets are built from (value AND slope of h^- at
    r_max); an overall factor of C leaves it unchanged (the unit jump fixes
    that factor), and at N = 1 the slice is a single number, so the meter
    reads 0.  Returns the largest |[G_n(r,r)]_ij - [G_n(r,r)]_ji| on the
    grid, at each radius relative to the largest entry of G_n(r,r) there."""
    idx = np.arange(G_n.shape[0])
    G_n_coinc = G_n[idx, idx]                                  # (nr, N, N): r' = r
    asymmetry = np.abs(G_n_coinc - np.einsum('aij->aji', G_n_coinc))
    return float(np.max(np.max(asymmetry, axis=(1, 2))
                        / np.max(np.abs(G_n_coinc), axis=(1, 2))))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_greensfct_cli(ap, need_n=True)
    args = ap.parse_args()

    bounce_npz = resolve_bounce(args)
    background, engine = background_and_engine(bounce_npz)
    check_cli(args, background)
    r_grid = radial_grid(engine, args.n_r)

    print(f'[GF-ADDON] radial off-coincident G_n:  n={args.n}  s2={args.s2:g}'
          f'  N={engine.N}  grid {len(r_grid)} points on '
          f'[{r_grid[0]:.4g}, {r_grid[-1]:.4g}]')
    G_n, branch_objects, log_B_minus_fv, log_B_plus_fv = build_G_n(engine, args.n, args.s2, r_grid)
    coincident_sym_dev = coincident_symmetry_dev(G_n)
    print(f'[CHECK] continuity at r\' = r, max|G_ij(r,r) - G_ji(r,r)| / '
          f'max|G(r,r)| = {coincident_sym_dev:.3e} (tests C as the Wronskian '
          f'of the two branches; identically 0 at N = 1)')

    out = (f'greensfct_off_coincident_radial_v2_{args.tag}_n{args.n}'
           f'_s2{s2_filetag(args.s2)}.npz')
    save_greensfct_npz(f'{args.data_dir}/{out}', bounce_npz, background,
                r_grid=r_grid, G_n=G_n, C=branch_objects['C'], kappa_i=branch_objects['kappa_i'],
                n=args.n, s2=args.s2,
                coincident_sym_dev=coincident_sym_dev,
                h_minus=branch_objects['h_minus'], h_plus=branch_objects['h_plus'],
                log_B_minus_fv=log_B_minus_fv, log_B_plus_fv=log_B_plus_fv)


if __name__ == '__main__':
    main()
