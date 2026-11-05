#!/usr/bin/env python3
"""greensfct_toolbox_v2.py -- the shared toolbox of the nine add_on_greensfct
codes (writes no file): the two h-branches, the constant Wronskian C and the
coincident wave of one (n, s2).  The add-on has no physics of its own: the
objects below come from the production engine's assembly
GreensFunction.branch_objects (computation/greens_function_vgeneral_phi_v2.py),
solved on the same bounce background (load_background) as the determinant.

READS   bounce_data_<tag>.npz (stage 0, README level 1) through
        load_background: R, U_int, m2, mu, N -- the background of every
        add-on code (background_and_engine); band_cutoffs_<tag>.npz
        (stage 1, level 2): n_max, the wave cutoff of the run (read_n_max,
        opened by the trunk's fingerprint-checked load_stage_npz).
WRITES  nothing itself; save_greensfct_npz is the ONE writer every add-on
        code calls (atomic_savez plus the pipeline's provenance stamp).
HOW     the functions, in the order a code calls them:
        add_greensfct_cli + resolve_bounce (the shared options; the bounce
        npz and the run tag) -> background_and_engine (load_background ->
        GreensFunction) -> check_cli (--s2 above -min_i m_i^2, --n and
        --n-max not negative) -> radial_grid (the output grid r_start .. r_max)
        -> solve_branches (engine.branch_objects at one (n, s2): h^pm, C,
        C^-1, G_bracket and its Dyson orders; aborts when s2 sits AT a
        crossing of the wave) -> coincident_wave, the summand of the two
        coincident codes: [G_n(r,r)]_ij = W_ij [G_bracket]_ij and its
        Dyson-order >= 2 remainder, with the FV weight
        [W]_ij = K_nu(kappa_i r) I_nu(kappa_j r)/r^2 of fv_weight_matrix
        (formed in log space from log_bessel_tables).  log_bessel_tables
        also serves the off-coincident and pole codes; chebyshev_U gives
        the angular weight (n+1) U_n(cos Theta) of the off-coincident wave
        sums; save_greensfct_npz writes every output.

THE OBJECTS (all from branch_objects), and what each one is:

    h^{a,pm}_{n,i}(r,s^2)   the two h-matrix branches: the deviation of the
                            exact radial solutions from the free Bessel
                            solutions, '-' grown outward from r_start ~ 0
                            (regular), '+' grown inward from r_max (decaying;
                            h^+ = h^+' = 0 there, U_int = 0 beyond r_max)
    C[a, b] = delta_ab + h^-[a, b](inf)
            = delta_ab + h^-[a, b](r_max)
              + r_max I_nu K_nu(kappa_a r_max) h^-'[a, b](r_max)
                            the constant Wronskian of the two branches, read
                            off at r_max as value + slope of the regular
                            branch (engine.h_minus_inf; U_int = 0 beyond
                            r_max); det C(n, s2) = 0 exactly when -s2 is an
                            eigenvalue of wave n
    [G_bracket]_ik = sum_ab [1+h^+]_ia [C^-1]_ba [1+h^-]_kb
                            the coincident bracket (code bracket_outer): the
                            radial Green's function at r' = r without its
                            FV weight,
        [G_n(r, r)]_ik = K_nu(kappa_i r) I_nu(kappa_k r)/r^2 [G_bracket]_ik
    [G_bracket_born_order_1]_ik = [h1^+]_ik + [h1^-]_ki - [C_born_order_1]_ki
                            the first Dyson order of the bracket, from the
                            h-ODE linearised in U_int (h1^pm, and
                            C_born_order_1 = h1^-(inf)): the bracket of the
                            tadpole term -G_FV U_int G_FV at r' = r, whose
                            n-sum diverges logarithmically
    G_bracket_born_order_2, G_bracket_geq3
                            the second Dyson order and the exact remainder
                            of orders >= 3 of the bracket (the engine's
                            Born orders 2 and >= 3); their sum is the
                            Dyson-order >= 2 bracket that coincident_wave
                            hands to the renormalised code
    BORN AND DYSON ORDERS.  The engine splits the h-ODE and the bracket by
    powers of U_int and calls the parts Born orders (C_born_order_1/2,
    G_bracket_born_order_1/2, G_bracket_geq3); for the Green's function they
    are its Dyson orders: Dyson order k of the coincident wave G_n(r, r) is
    the FV weight K_nu I_nu / r^2 times Born order k of the bracket.  Objects
    the add-on forms itself are named by their Dyson order
    (G_n_coinc_dyson_geq2, G0_fv_ren_addback, G1_ren_addback).
    kappa_i = sqrt(m2_i + s2), nu = n+1   (engine.kappa)
    U_int   the interaction Hessian U_int(r) = H'(phi_b(r)) - diag(m2)
              (pipeline_helpers.insertion, inside load_background): the
              primed Hessian on the bounce minus its false-vacuum value, the
              local insertion every interacting object is built from

The add-on keeps the pole terms of the two dangerous modes (n = 0 negative
mode, n = 1 translation zero modes) inside its n = 0, 1 waves: they are
removed by the trunk at the determinant level only
(gbar_sector_vgeneral_phi_v2), and greensfct_dangerous_modes_v2.py exhibits
them beside the full waves.  solve_branches therefore aborts when s2 sits AT
a crossing of the wave (the full resolvent has its pole there).
"""
import glob
import os
import re
import sys

import numpy as np

# ---- make the pipeline's computation/ importable ---------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_COMP = os.path.join(os.path.dirname(_HERE), 'computation')
for _p in (_COMP, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.dont_write_bytecode = True

from greens_function_vgeneral_phi_v2 import GreensFunction
from pipeline_helpers_vgeneral_phi_v2 import (load_background,
                                              load_stage_npz, stage_paths,
                                              atomic_savez,
                                              provenance_stamp,
                                              radius_first)
from fv_analytic_toolbox_vgeneral_phi_v2 import logI_and_logK_tables
from mode_census_vgeneral_phi_v2 import constant_wronskian_and_floor


# --------------------------------------------------------------------------- #
#  CLI                                                                         #
# --------------------------------------------------------------------------- #
def add_greensfct_cli(parser, need_n=False, need_r_grid=True):
    """The uniform CLI of every add-on code:
    --s2 (REQUIRED, the spectral parameter), --data-dir, --bounce-npz, --tag,
    --n-r, and --n (REQUIRED) for the single-wave radial code.
    need_r_grid=False drops --n-r (for codes without a user-chosen radial
    grid: the r-independent FV matrix, the angle scan's fixed radii)."""
    parser.add_argument('--s2', type=float, required=True,
                        help='spectral parameter s^2 (REQUIRED; shifts '
                             'every channel mass: '
                             'kappa_i = sqrt(m_i^2 + s^2))')
    if need_n:
        parser.add_argument('--n', type=int, required=True,
                            help='angular-momentum wave n (REQUIRED; nu = n+1)')
    parser.add_argument('--data-dir',
                        default=os.environ.get('VGENERAL_PHI_DATA', '.'),
                        help='data directory (output npz written here; '
                             'default $VGENERAL_PHI_DATA)')
    parser.add_argument('--bounce-npz',
                        default=os.environ.get('VGENERAL_PHI_BOUNCE_NPZ'),
                        help='bounce npz (default $VGENERAL_PHI_BOUNCE_NPZ, '
                             'else the unique bounce_data_*.npz in '
                             '--data-dir)')
    parser.add_argument('--tag', default=None,
                        help='run tag used in the output filename (default: '
                             'auto-discovered from the bounce npz filename '
                             'bounce_data_<tag>.npz, or from a unique '
                             'bounce_data_*.npz in --data-dir)')
    if need_r_grid:
        parser.add_argument('--n-r', type=int, default=300,
                            help='number of radial output points (default 300)')
    return parser


def resolve_bounce(args):
    """Resolve the bounce npz AND the run tag, with nothing model-specific:
    the bounce is --bounce-npz (or $VGENERAL_PHI_BOUNCE_NPZ) if given, else
    the unique bounce_data_*.npz in --data-dir; a missing --tag is derived
    from the bounce filename bounce_data_<tag>.npz.  Ambiguity aborts and
    lists the candidates -- nothing is ever guessed."""
    if args.bounce_npz:
        bounce_npz = args.bounce_npz
    else:
        hits = sorted(glob.glob(os.path.join(args.data_dir,
                                             'bounce_data_*.npz')))
        if len(hits) == 1:
            bounce_npz = hits[0]
        elif not hits:
            raise RuntimeError(
                '[ABORT] no --bounce-npz given, $VGENERAL_PHI_BOUNCE_NPZ '
                f'unset, and no bounce_data_*.npz in {args.data_dir!r}.')
        else:
            raise RuntimeError(
                '[ABORT] --bounce-npz not given and the data dir holds '
                'several bounce files -- pass one explicitly:\n  '
                + '\n  '.join(hits))
    if args.tag is None:
        m = re.match(r'bounce_data_(.+)\.npz$', os.path.basename(bounce_npz))
        if not m:
            raise RuntimeError(
                f'[ABORT] --tag not given and it cannot be derived from the '
                f'bounce filename {os.path.basename(bounce_npz)!r} -- pass --tag.')
        args.tag = m.group(1)
        print(f'[TAG] auto-discovered run tag: {args.tag}')
    return bounce_npz


def read_n_max(data_dir, tag, bounce_npz):
    """The wave cutoff n_max of this run, where the add-on's wave sums stop
    unless --n-max is given: the n_max the band driver (stage 1) stores in
    band_cutoffs_<tag>.npz, read by load_stage_npz, which checks that the
    file belongs to this bounce."""
    band_cutoffs_npz = load_stage_npz(
        stage_paths(data_dir, tag)['band_cutoffs'], bounce_npz,
        'the band driver (band_adaptive_vgeneral_phi_v2) on this bounce')
    return int(band_cutoffs_npz['n_max'])


# --------------------------------------------------------------------------- #
#  Background + engine                                                        #
# --------------------------------------------------------------------------- #
def background_and_engine(bounce_npz):
    """load_background -> the shared background dict (R, U_int, m2, mu, N)
    and the production h-ODE engine built on it."""
    background = load_background(bounce_npz)
    engine = GreensFunction(background['R'], background['U_int'], background['m2'])
    return background, engine


def check_cli(args, background):
    """Abort on a command line no add-on code can evaluate, right after the
    background is loaded: every kappa_i = sqrt(m_i^2 + s2) must be real and
    positive, so --s2 must exceed -min_i m_i^2 (the continuum threshold,
    background m_min2), and the wave --n and the wave cutoff --n-max must
    not be negative."""
    s2_threshold = -background['m_min2']
    if not args.s2 > s2_threshold:
        raise SystemExit(f'[ABORT] --s2 must exceed -min_i m_i^2 = '
                         f'{s2_threshold:g} (got --s2 {args.s2:g}).')
    for option, value in (('--n', getattr(args, 'n', None)),
                          ('--n-max', getattr(args, 'n_max', None))):
        if value is not None and value < 0:
            raise SystemExit(f'[ABORT] {option} must be >= 0 (got {value}).')


def radial_grid(engine, n_r):
    """The output radial grid r_start .. r_max (the h-solve is sampled ON it;
    LSODA's internal accuracy does not depend on the sampling density)."""
    return np.linspace(engine.r_start, engine.r_max, int(n_r))


# --------------------------------------------------------------------------- #
#  The two branch solves at ONE (n, s2)                                        #
# --------------------------------------------------------------------------- #
def solve_branches(engine, n, s2, r_grid):
    """Solve the bundled h-ODE (h, h1, h2, h_geq3 per branch) at one (n, s2),
    sampled on r_grid, and assemble the objects every code needs -- the
    engine's ONE assembly, GreensFunction.branch_objects.

    Returns its dict (npts = len(r_grid), N = field number):
      h_plus, h_minus          (npts, N, N)  the full h of each branch
      one_plus_h_plus          (npts, N, N)  1 + h^+(r)
      one_plus_h_minus         (npts, N, N)  1 + h^-(r)
      C, C_inv                 (N, N)        the constant Wronskian and its inverse
      G_bracket                (npts, N, N)  sum_ab [1+h^+]_ia [C^-1]_ba [1+h^-]_kb
      G_bracket_born_order_1   (npts, N, N)  first Dyson order of G_bracket - 1
      G_bracket_born_order_2   (npts, N, N)  second Dyson order
      G_bracket_geq3           (npts, N, N)  exact remainder, orders >= 3
      kappa_i                  (N,)          shifted masses sqrt(m_i^2 + s2)
      nu                       ()            n + 1
    Index order: every stack is [radius, row, column]; for h^pm the row is
    the component i and the column the basis function a.

    Aborts when s2 sits AT a crossing of wave n: det C(n, s2) = 0 exactly
    when -s2 is an eigenvalue of the wave, and there the full resolvent has
    its pole u_i(r) u_j(r')/(omega^2 + s^2).  As in the census (mode_census),
    a crossing is an eigenvalue of C at zero (det C is their product), here
    of the C of branch_objects, whose inverse every add-on object carries.
    An eigenvalue counts as zero below crossing_floor, the smaller of two
    estimates of its accuracy:
      * census_floor, the census's own floor
        (mode_census.constant_wronskian_and_floor) -- too large at large n,
        where |C - 1| grows without bound while the eigenvalues of C stay
        O(1);
      * engine.rtol, the engine's relative accuracy, times the size of C,
        its largest |eigenvalue| but at least 1 (the identity part of C) --
        too large at small n when a heavier channel feeds a lighter one (the
        eigenvalues of a regular C then spread over many orders of
        magnitude).
    The census, which works only in n = 0 and 1, uses its floor alone; the
    add-on builds every wave up to n_max and needs the smaller of the two.
    """
    branch_objects = engine.branch_objects(n, s2, r_grid)
    _, lam_C, census_floor = constant_wronskian_and_floor(
        engine, n, s2, branch_objects['C'])
    lam = np.abs(lam_C)
    crossing_floor = np.minimum(census_floor,
                                engine.rtol * max(1.0, float(np.max(lam))))
    k = int(np.argmin(lam / crossing_floor))    # the eigenvalue nearest its floor
    if lam[k] <= crossing_floor[k]:
        raise RuntimeError(
            f'[ABORT CROSSING] wave n={n}: det C(n, s2={s2:g}) = 0 within the '
            f'solver precision (an eigenvalue of C, {lam[k]:.2e}, '
            f'is below its accuracy floor {crossing_floor[k]:.2e}) -- -s2 is an '
            f'eigenvalue of this wave and the full resolvent has a pole there.  '
            f'Choose another s2 (the pole term itself is exhibited by '
            f'greensfct_dangerous_modes_v2.py).')
    return branch_objects


# --------------------------------------------------------------------------- #
#  Log-Bessel tables, the FV weight, the angular weight, the coincident wave #
# --------------------------------------------------------------------------- #
def log_bessel_tables(nu, kappa_i, r_grid):
    """(log I_nu(kappa_i r), log K_nu(kappa_i r)) for every channel i and grid
    radius r, (N, nr) each: logI_and_logK_tables of the FV analytic toolbox
    (with its large-order fallback where scipy's scaled value leaves the
    float64 range) on the table of arguments kappa_i r."""
    return logI_and_logK_tables(nu, np.outer(kappa_i, r_grid))


def fv_weight_matrix(nu, kappa_i, r_grid, same_block):
    """The FV weight of every coincident wave, (nr, N, N):

        [W(r)]_ij = K_nu(kappa_i r) I_nu(kappa_j r) / r^2 = B^+_i(r) B^-_j(r),

    the free decaying solution of channel i times the free regular solution
    of channel j, so that [G_n(r, r)]_ij = [W]_ij [G_bracket]_ij entry by
    entry.  Its diagonal is the free coincident wave (the false-vacuum
    bracket is the identity; greensfct_fv_coincident_v2.fv_coincident_wave);
    off the diagonal W is no Green's function, only the factor the bracket
    multiplies.  Formed in log space: for i != j the product grows like
    e^{(kappa_j - kappa_i) r} and could overflow float64 before the bracket
    multiplies it.  Its exponent, log[I_nu K_nu](kappa_i r) < 0 plus
    log I_nu(kappa_j r) - log I_nu(kappa_i r), stays below the cap
    BESSEL_LOG_RATIO_MAX that the engine's BESSEL-RATIO guard has checked, in
    the branch solve at this (n, s2), for every pair in the same block of
    U_int.  A pair in different blocks (engine.same_block) has a bracket
    entry of round-off size and no bound on its exponent: its weight is 0."""
    logI, logK = log_bessel_tables(nu, kappa_i, r_grid)      # (N, nr) each
    log_fv_weight = np.where(same_block[:, :, None],
                             logK[:, None, :] + logI[None, :, :], -np.inf)   # (i, j, r)
    fv_weight = np.exp(log_fv_weight) / r_grid[None, None, :]**2
    return radius_first(fv_weight)            # layout: (nr, N, N)


def chebyshev_U(n_max, x):
    """U_0..U_n_max at scalar x by the stable recurrence
    U_0 = 1, U_1 = 2x, U_{n+1} = 2x U_n - U_{n-1} -- the angular weight
    (n+1) U_n(cos Theta) of every position-space wave sum (U_n(1) = n+1)."""
    U = np.empty(n_max + 1)
    U[0] = 1.0
    if n_max >= 1:
        U[1] = 2.0 * x
    for n in range(1, n_max):
        U[n + 1] = 2.0 * x * U[n] - U[n - 1]
    return U


def coincident_wave(engine, n, s2, r_grid):
    """ONE wave of the COINCIDENT matrix, (nr, N, N) each: the radial wave of
    greensfct_off_coincident_radial_v2.build_G_n at r' = r, computed
    directly on the diagonal,

        G_n_coinc            = [G_n(r,r,s^2)]_ij
                             = K_nu(kappa_i r) I_nu(kappa_j r)/r^2 [G_bracket]_ij
        G_n_coinc_dyson_geq2 = K_nu(kappa_i r) I_nu(kappa_j r)/r^2
                               [G_bracket_born_order_2 + G_bracket_geq3]_ij

    Returns (G_n_coinc, G_n_coinc_dyson_geq2): the full wave (summed by the
    unrenormalised code) and its exact Dyson-order >= 2 remainder (summed by
    the renormalised code), formed from the engine's Born orders, never as the
    full wave minus its free and tadpole waves.  The caller applies the
    weight (n+1)^2/(2 pi^2).  Traced, sum_i Int dr 2 pi^2 r^3 (n+1)^2/(2 pi^2)
    [.]_ii, these are the waves of the determinant path: G_n_coinc minus the
    free wave gives the engine's Gbar_minus_fv(n, s2), and
    G_n_coinc_dyson_geq2 gives Gbar_2 + delta_geq3 (GreensFunction
    .gbar_traces, whose weight r I_nu K_nu(kappa_i r) is r^3 times the
    diagonal of fv_weight_matrix); the determinant also subtracts Gbar_2 and
    integrates delta_geq3 over s2."""
    branch_objects = solve_branches(engine, n, s2, r_grid)
    fv_weight_matrix_n = fv_weight_matrix(branch_objects['nu'],
                                              branch_objects['kappa_i'],
                                              r_grid, engine.same_block)  # (nr, N, N)
    G_n_coinc = fv_weight_matrix_n * branch_objects['G_bracket']
    G_n_coinc_dyson_geq2 = fv_weight_matrix_n * (
        branch_objects['G_bracket_born_order_2'] + branch_objects['G_bracket_geq3'])
    return G_n_coinc, G_n_coinc_dyson_geq2


# --------------------------------------------------------------------------- #
#  Output                                                                      #
# --------------------------------------------------------------------------- #
def save_greensfct_npz(path, bounce_npz, background, **arrays):
    """atomic_savez with the pipeline's provenance stamp; `background` is
    the load_background dict of bounce_npz (m2, params, fields)."""
    atomic_savez(path, **arrays, **provenance_stamp(bounce_npz, background['m2'],
                                                    background['params'], background['fields']))
    print(f'[WROTE] {path}')
