#!/usr/bin/env python3
"""greensfct_dangerous_modes_v2.py -- the pole terms of the two DANGEROUS
MODES in the radial Green's function: the negative mode (wave n = 0,
eigenvalue omega2_00 < 0) and the translation zero mode (wave n = 1,
eigenvalue omega2_10 ~ 0), each at coincident points r' = r and off them;
add-on README tree: level 3, branch (i) (needs the bounce and the two
eigenvalue files):

    python3 greensfct_dangerous_modes_v2.py --s2 <s2> [--n-r 300] [--data-dir <dir>] [--tag <tag>]

READS   bounce_data_<tag>.npz (stage 0, level 1) through load_background, on
        which the production h-ODE engine is built (background_and_engine);
        the radial grid is the toolbox's radial_grid, so with the same --n-r
        it is the grid of greensfct_off_coincident_radial_v2 and the two
        outputs can be subtracted point by point; eig_n0_<tag>.npz,
        eig_n1_<tag>.npz (stages 0a, 0b, level 2): the eigenvalues omega2_00,
        omega2_10 written by eigenvalue_neg_vgeneral_phi_v2 and
        eigenvalue_zero_vgeneral_phi_v2, opened by the stage-file read rule
        pipeline_helpers.load_stage_npz (bounce fingerprint checked).

WRITES  greensfct_dangerous_modes_v2_<tag>_s2<v>.npz with
    r_grid (nr,), s2 ()
    omega2_00 (), omega2_10 ()          the pipeline's eigenvalues (eig npz)
    u_n0 (nr, N), u_n1 (nr, N)          the radial eigenfunctions
    pole_term_coinc_n0, pole_term_coinc_n1        (nr, N, N)      the pole terms at r' = r
    pole_term_offc_n0, pole_term_offc_n1          (nr, nr, N, N)  the pole terms at every (r, r')
    null_quality_n0/n1, match_dev_n0/n1 ()  meters of the null-vector construction (record only)
    overlap_dphi_n1 ()                   overlap of u_1 with the normalised phi_b' (record only)
    and the provenance stamp (bounce_sha, potential_id, code_version).

HOW     per wave n in (0, 1), at the requested s2:
  1. THE EIGENVALUE omega^2_{n,0} (wave n, radial number 0; npz key
     omega2_00 for n = 0, omega2_10 for n = 1) from the eig npz: the
     census's root of det C(n, -omega^2) = 0 on this same engine, so C is
     singular to solver precision here (checked; the code aborts if not).
  2. THE EIGENFUNCTION u_n(r) by eigenfunction_from_null_vectors: no eigenvalue
     problem is solved -- at s2 = -omega^2 the null vectors of C give the one
     combination of the regular basis functions that is also a combination
     of the decaying ones; normalised to sum_i Int dr r^3 u_i^2 = 1.  For
     n = 1 its overlap with the translation mode phi_b'(r) (translation_mode)
     is printed as a health meter (near 1).
  3. THE POLE TERMS, the mode's single term of the spectral representation
     of the radial Green's function G_n:
       coincident      pole_term_coinc_n<n>[r, i, j]    = u_i(r) u_j(r)  / (omega^2 + s2)
       off-coincident  pole_term_offc_n<n>[r, r', i, j] = u_i(r) u_j(r') / (omega^2 + s2)
     G_n minus its pole term is G_n^perp, the wave with this mode
     projected out.
     NO factor (n+1)^2 at the radial level: the (n+1)^2 degenerate modes of
     wave n share the radial function u_n(r); the degeneracy enters only when
     waves are summed into position space (weight (n+1) U_n(cos Theta)/(2 pi^2))
     or traced (deg = (n+1)^2).  By the normalisation of u_n the traced term is
     the trunk's pole term:  deg Int dr r^3 sum_i pole_term_coinc[r,i,i] = deg/(omega^2 + s2).

WHERE THE PIPELINE SUBTRACTS THESE MODES.  Nowhere at the Green's-function
level: every add-on code stores the FULL (unsubtracted) resolvent.  The
production determinant removes the two poles at the determinant level, with
the eigenvalue alone -- gbar_sector_vgeneral_phi_v2 integrates
delta_sub = delta_geq3 - deg/(omega2 + s2), deg = (n+1)^2 (the pole subtraction
in pipeline_quadrature_vgeneral_phi_v2), and assemble_lnD_vgeneral_phi_v2
carries ln|omega2_00| and the fictitious-mode block.  This side code only
shows what those poles look like in the radial Green's function.
"""
import argparse

import numpy as np
from scipy.interpolate import CubicSpline

from greensfct_toolbox_v2 import (add_greensfct_cli, background_and_engine,
                                  check_cli, log_bessel_tables, radial_grid,
                                  resolve_bounce,
                                  save_greensfct_npz)
# the toolbox import above is what puts computation/ on sys.path
from pipeline_helpers_vgeneral_phi_v2 import (load_stage_npz, radius_first,
                                              s2_filetag, stage_paths,
                                              wall_radius)
# the census's threshold for a located root of det C = 0
from mode_census_vgeneral_phi_v2 import VERIFY_FRAC

# the two dangerous waves: wave n -> (eig npz key of the pipeline, its
# eigenvalue key, the stage that writes it)
DANGEROUS_WAVE_EIG_SOURCES_N0_N1 = {0: ('eig_n0', 'omega2_00', 'eigenvalue_neg_vgeneral_phi_v2'),
                                    1: ('eig_n1', 'omega2_10', 'eigenvalue_zero_vgeneral_phi_v2')}


# --------------------------------------------------------------------------- #
#  The eigenfunction at an eigenvalue, from the null vectors of C             #
# --------------------------------------------------------------------------- #
def eigenfunction_from_null_vectors(engine, n, s2_star, r_grid, r_W):
    """The normalised eigenfunction u of wave n at its eigenvalue
    omega^2 = -s2_star (the eig npz root), from the two branches solved at
    s2_star (no eigenvalue problem is solved):

      1. solve both branches at (n, s2_star) -> h^-, h^+, C; abort unless C is
         singular to the census's threshold for a located root: its smallest
         singular value sigma_min at most VERIFY_FRAC times 1 + max|C - 1|,
         the size of the two terms of C = 1 + h^-(inf) before they cancel;
      2. at an eigenvalue ONE function psi is a combination of the regular
         basis functions (columns of Phi^-) AND of the decaying ones
         (columns of Phi^+):
             psi(r) = Phi^-(r) v = c Phi^+(r) w,
             Phi^-[i,a] = I_nu(kappa_i r)/r [1+h^-]_ia,
             Phi^+[i,a] = K_nu(kappa_i r)/r [1+h^+]_ia;
         psi lies in both branches, so its Wronskian with every basis function
         vanishes: C v = 0 and w^T C = 0 (C[a,b] = r^3 times the Wronskian of
         the columns Phi^+[:,a] and Phi^-[:,b]) -- v and w are the right and
         left null vectors of C;
      3. use each representation where it is accurate (regular inside,
         decaying outside), matched where |psi| is largest inside the wall
         r <= r_W;
      4. normalise in the measure of the radial operator, u = psi / norm with
         norm^2 = sum_i Int dr r^3 psi[i]^2: the traced radial pole term then
         has residue exactly 1; the (n+1)^2 degenerate copies are angular.

    Returns dict(u (nr, N), match_dev, null_quality)."""
    # -- 1. both branches at the crossing, from the engine directly: AT the
    #    crossing C is singular by construction, which is exactly what
    #    solve_branches (the resolvent path) aborts on
    branch_objects = engine.branch_objects(n, s2_star, r_grid)
    nu, kappa_i = branch_objects['nu'], branch_objects['kappa_i']

    # -- 2. the two null vectors from the SVD C = U diag(S) Vt (S descending,
    #    so the last singular pair is the smallest): the last right singular
    #    vector solves C v = 0, the last left one w^T C = 0
    C = branch_objects['C']
    U, S, Vt = np.linalg.svd(C)
    basis_coeffs_minus = Vt[-1, :]                   # v: regular representation
    basis_coeffs_plus = U[:, -1]                    # w: decaying representation
    # ~0 when the crossing was hit: sigma_min against 1 + max|C - 1|, the
    # size of the two terms of C (the identity and h^-(inf)) before they
    # cancel -- a scale outside C's own singular values, so the test holds
    # at N = 1 too, where C is a single number
    null_quality = float(S[-1] / (1.0 + np.max(np.abs(C - np.eye(engine.N)))))
    if null_quality > VERIFY_FRAC:
        raise RuntimeError(
            f'[ABORT DANG-MODES] wave n={n}: C(n, s2={s2_star:+.9e}) is not '
            f'singular (sigma_min/(1 + max|C - 1|) = {null_quality:.2e} > '
            f'{VERIFY_FRAC:g}) -- the eig npz root does not reproduce on this '
            f'engine; re-run the eig stage.')

    # -- the two representations of psi on the grid: Phi^- v and Phi^+ w
    logI, logK = log_bessel_tables(nu, kappa_i, r_grid)          # (N, nr)
    coef_in = branch_objects['one_plus_h_minus'] @ basis_coeffs_minus       # (nr, N): sum_b [1+h^-(r)]_ib v_b
    coef_out = branch_objects['one_plus_h_plus'] @ basis_coeffs_plus       # (nr, N): sum_a [1+h^+(r)]_ia w_a
    # layout: the Bessel tables are stored (channel, radius); radius_first
    # puts the radius axis first, (radius, channel)
    psi_in = radius_first(np.exp(logI)) / r_grid[:, None] * coef_in    # sum_b Phi^-[i,b] v_b
    psi_out = radius_first(np.exp(logK)) / r_grid[:, None] * coef_out  # sum_a Phi^+[i,a] w_a

    # -- 3. match the two representations where |psi| is largest inside the
    #    wall, r <= r_W (the first n_inside grid points): further out the
    #    regular representation can grow back -- the solver-precision
    #    remainder of C v at the root rides on the growing I_nu(kappa_i r) of
    #    a heavier channel -- and overtake the mode before r_max
    n_inside = np.count_nonzero(r_grid <= r_W)
    i_match_radius = int(np.argmax(np.max(np.abs(psi_in[:n_inside]), axis=1)))
    i_match_channel = int(np.argmax(np.abs(psi_in[i_match_radius])))
    match_coeff = psi_in[i_match_radius, i_match_channel] / psi_out[i_match_radius, i_match_channel]
    # the relative jump of the stitched mode at the seam: the two
    # representations at the first radius outside the matching radius (at
    # the matching radius itself they agree by construction in the matching
    # channel, which at N = 1 is the only channel)
    i_seam = min(i_match_radius + 1, len(r_grid) - 1)
    match_dev = float(np.max(np.abs(psi_in[i_seam] - match_coeff * psi_out[i_seam]))
                      / np.max(np.abs(psi_in[i_seam])))
    psi = np.where((np.arange(len(r_grid)) <= i_match_radius)[:, None],
                   psi_in, match_coeff * psi_out)

    # -- 4. normalise in the r^3 dr measure (trapezoid rule) and fix the sign
    #       the SVD leaves open: the largest component of u at the matching
    #       radius is made positive (the pole terms u u do not depend on it)
    norm = np.sqrt(np.trapezoid(r_grid**3 * np.sum(psi**2, axis=1), r_grid))
    u = psi / norm * np.sign(psi[i_match_radius, i_match_channel])
    return dict(u=u, match_dev=match_dev, null_quality=null_quality)


def translation_mode(background, r_grid):
    """The exact-bounce zero mode u_i ~ phi_b,i'(r), radially normalised:
    differentiating the bounce equation gives M_b (d_mu phi_b) = 0, an n = 1
    mode with radial part phi_b'(r) -- scipy CubicSpline derivative of the
    stored profile + one trapezoid quadrature for the r^3 dr norm."""
    dphi = CubicSpline(background['R'], background['phi_b'], axis=0)(r_grid, nu=1)
    norm = np.sqrt(np.trapezoid(r_grid**3 * np.sum(dphi**2, axis=1), r_grid))
    return dphi / norm


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_greensfct_cli(ap)                       # --s2, --n-r, --data-dir, --bounce-npz, --tag
    args = ap.parse_args()

    bounce_npz = resolve_bounce(args)
    background, engine = background_and_engine(bounce_npz)
    check_cli(args, background)
    r_grid = radial_grid(engine, args.n_r)   # the radial code's grid at the same --n-r
    r_W = wall_radius(background['R'], background['phi_b'])   # the matching region r <= r_W
    s2 = float(args.s2)
    print(f'[DANG-MODES] s2={s2:g}  N={engine.N}  nr={len(r_grid)}')

    # the smallest s^2 scale the model resolves: 1e-8 of its lightest
    # channel mass^2 (s^2 carries mass^2 units)
    s2_resolution = 1e-8 * float(np.min(engine.m2))
    npz_payload = dict(r_grid=r_grid, s2=s2)
    for n, (eig_key, omega2_key, stage) in DANGEROUS_WAVE_EIG_SOURCES_N0_N1.items():
        # ---- 1. the eigenvalue: the pipeline's (eig npz) ---------------------
        eig_npz = load_stage_npz(stage_paths(args.data_dir, args.tag)[eig_key],
                                 bounce_npz, rerun_hint=stage)
        omega2 = float(eig_npz[omega2_key])
        # |omega2 + s2| below the smallest s2 scale the model resolves means
        # s2 sits AT the crossing: the pole term is 1/0 there
        if abs(omega2 + s2) <= s2_resolution:
            raise RuntimeError(
                f'[ABORT DANG-MODES] s2={s2:g} sits at the n={n} crossing '
                f's2 = -omega2 = {-omega2:.9e}: the pole term '
                f'u u/(omega2 + s2) is infinite there -- choose another s2.')
        print(f'[DANG-MODES n={n}] eigenvalue omega^2 = {omega2:+.8e} (eig npz)')

        # ---- 2. the eigenfunction from the null vector of C ----------------
        mode = eigenfunction_from_null_vectors(engine, n, -omega2, r_grid, r_W)
        u = mode['u']                                          # (nr, N)
        print(f'[CHECK n={n}] null quality sigma_min/(1 + max|C - 1|) = '
              f'{mode["null_quality"]:.2e} (small = crossing hit); '
              f'two-representation match deviation = {mode["match_dev"]:.2e}')

        # ---- 3. the two pole terms at this s2 ------------------------------
        # coincident:     pole_term[r, i, j]     = u_i(r) u_j(r)  / (omega2 + s2)
        # off-coincident: pole_term[r, r', i, j] = u_i(r) u_j(r') / (omega2 + s2)
        pole_term_coinc = np.einsum('ai,aj->aij', u, u) / (omega2 + s2)
        pole_term_offc = np.einsum('ai,bj->abij', u, u) / (omega2 + s2)

        npz_payload[omega2_key] = omega2
        npz_payload[f'u_n{n}'] = u
        npz_payload[f'pole_term_coinc_n{n}'] = pole_term_coinc
        npz_payload[f'pole_term_offc_n{n}'] = pole_term_offc
        npz_payload[f'null_quality_n{n}'] = float(mode['null_quality'])
        npz_payload[f'match_dev_n{n}'] = float(mode['match_dev'])
        if n == 1:
            u_dphi = translation_mode(background, r_grid)
            overlap = float(np.trapezoid(r_grid**3 * np.sum(u * u_dphi, axis=1), r_grid))
            npz_payload['overlap_dphi_n1'] = abs(overlap)
            print(f'[CHECK n=1] overlap of u_1 with the normalised phi_b\' = '
                  f'{abs(overlap):.6f} (near 1 for a translation zero mode)')

    npz_name = (f'greensfct_dangerous_modes_v2_{args.tag}'
                f'_s2{s2_filetag(s2)}.npz')
    save_greensfct_npz(f'{args.data_dir}/{npz_name}', bounce_npz, background,
                       **npz_payload)


if __name__ == '__main__':
    main()
