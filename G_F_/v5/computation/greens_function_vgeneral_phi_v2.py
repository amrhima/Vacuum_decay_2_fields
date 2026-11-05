#!/usr/bin/env python3
"""greens_function_vgeneral_phi_v2.py -- the h-ODE -> Green's-function engine:
per O(4) wave n and spectral parameter s^2 it solves the h-matrix ODE on the
bounce background, matches the two branches through the constant Wronskian C
and traces the coincident Green's function against the analytic false-vacuum
weight -- the Gbar family and, for the determinant, delta_geq3(n, s^2), the
>= O(U^3) remainder trace.  Level 2 of the README tree: run as a program it
is the band worker of stage 1 (band_adaptive_vgeneral_phi_v2 starts it as
parallel workers, each with its share of the s2 slices in --s2-list); as a
library (the class GreensFunction) it serves the mode census of stages 0,
0a, 0b, the sector stages 2a, 2b and the add-on.
N = the number of scalar fields, read from the background -- never
hardcoded; every matrix below is (N, N).

READS
-----
The bounce npz named by $VGENERAL_PHI_BOUNCE_NPZ (env, never a CLI path --
every parallel worker must see the same background), through
pipeline_helpers load_background: R (the bounce mesh; r_max = R[-1], the
solve start r_start = R[1]/2), U_int(r) (the insertion, from
phi_b and the potential), m2 (the channel masses^2).  The band
driver passes --s2-list, --n-min, --n-max (and --data-dir, --tag,
--file-suffix, which place and name the output); a library caller passes the
three arrays to GreensFunction(R, U_int, m2).

WRITES
------
As the band worker: residual_band_<tag>_s2<v><suffix>.npz per slice
(v = pipeline_helpers.band_s2_text(s2), 7 significant digits, '.' -> 'p';
suffix = --file-suffix, empty for a base slice)
with n_values, s2, Gbar_minus_fv, Gbar_1, Gbar_2,
delta_geq3 (one value per wave); on base slices also the exact n = 0, 1
traces on the band s2 grid, Gbar_minus_fv_n0 / _n1 for the plot layer (which
must not solve) and delta_geq3_n0 / _n1 for the band driver's SECTOR-CUTOFF
watcher; and the provenance keys m2, engine, rtol, atol,
bounce_sha, potential_id, code_version -- read through read_band_tables by
the band driver (stage 1: watchers and resume), by band_integrals (stage 3)
and by the plot codes.
As a library: gbar_traces(n, s2) -> the traces of one wave at one s2 (the
sector stages through pipeline_quadrature sector_traces_of_wave; its
'integrands' are kept for inspection, nothing reads them);
state_minus_r_max(n, s2) with
h_minus_inf -> the constant Wronskian C of the census (mode_census
constant_wronskian_and_floor); branch_objects(n, s2, r_grid) -> the branch
objects of the add-on (greensfct_toolbox_v2 solve_branches, and
greensfct_dangerous_modes_v2 at a crossing); gbar_fv(n, s2)
-> the FV trace of one wave alone (kept for inspection: nothing in the
pipeline calls it).

HOW (the physics, in the order the code does it)
------------------------------------------------
Radial fluctuation operator (nu = n+1, deg = nu^2 = (n+1)^2 degeneracy):
    [M_n(s^2)]_ij = [-d^2/dr^2 - (3/r)d/dr + n(n+2)/r^2 + s^2 + m_i^2] d_ij
                    + [U_int(r)]_ij,
    [U_int(r)]_ij = [H(phi_b(r))]_ij - m_i^2 d_ij   (the interaction Hessian,
    or insertion: the deviation of the Hessian on the bounce from its
    false-vacuum value diag(m_i^2), both in the primed field frame).
1. kappa, U_int_at: the shifted masses kappa_i = sqrt(s^2 + m_i^2), and
   U_int(r) from ONE cubic spline through the bounce-mesh values.
2. _rhs: the free Bessel solutions B^-_i = I_nu(kappa_i r)/r (regular at 0)
   and B^+_i = K_nu(kappa_i r)/r (decaying at inf); the basis functions carry
   all interaction in the h-matrix (array h[i, a]: component i first, basis
   function a second -- the convention of every stack in this pipeline),
       phi^pm_{ia} = B^pm_i [ delta_ia + h^pm[i, a] ] ,
   which solves the h-ODE
       h'' + P h' = K (1 + h)
       [P]_ii = 3/r + 2 (ln B^pm_i)' = 1/r + 2 kappa_i (ln I_nu or K_nu)'(kappa_i r)
                friction matrix ((ln B^pm_i)' = kappa_i (ln Bessel)'(kappa_i r)
                - 1/r because B^pm_i = Bessel/r; _rhs evaluates the
                right-hand form with dlog_bessel from the Bessel toolbox)
       [K]_ij = [U_int(r)]_ij B^pm_j / B^pm_i         coupling matrix
   bundled with its Born chain, L = d^2/dr^2 + P d/dr the left side,
       L h1 = K,   L h2 = K h1,   L h_geq3 = K (h2 + h_geq3)
   so h = h1 + h2 + h_geq3 EXACTLY: h1 and h2 are the exact first and second
   orders in U_int, h_geq3 = h - h1 - h2 holds every order >= 3 (it sits on
   both sides of its own equation; no series is summed), and h2 + h_geq3 is
   the part of order >= 2.  ONE first-order system of 8 N^2 components per
   branch.
3. _radial_grid, solve_branch_sampled / _solve_on_grid, _check_born_split:
   LSODA integrates that system from the zero state, h^- = h^-' = 0 at
   r_start (the regular start near r = 0) OUTWARD and h^+ = h^+' = 0 at r_max
   (exact for the stored background, which ends at r_max: U_int = 0 beyond
   it) INWARD (the one pm convention used
   everywhere in this pipeline: '-' = regular/outward, '+' = decaying/inward),
   both sampled on the shared radial grid; the BORN-SPLIT guard checks
   h = h1 + h2 + h_geq3 on every solve.
4. h_minus_inf, branch_objects: the constant Wronskian
   C, read off at r_max, where h^+ = h^+' = 0 exactly:
       C[a, b] = delta_ab + h^-[a, b](inf)
               = delta_ab + h^-[a, b](r_max)
                 + r_max I_nu(kappa_a r_max) K_nu(kappa_a r_max)
                   h^-'[a, b](r_max)
   (U_int = 0 beyond r_max, so h^- obeys the free h-ODE there and its value at
   infinity follows from its value and slope at r_max; the census, the band
   and the add-on all build C by this one rule);
       det C(s^2) = 0  <=>  omega^2 = -s^2 is an eigenvalue.
   From C^-1 and the two branches the coincident Green's-function bracket
   (code: bracket_outer; the '-' label b in the FIRST slot of C^-1)
       [G_bracket]_ik = sum_ab [1 + h^+]_ia [C^-1]_ba [1 + h^-]_kb ,
   so that [G_n(r, r)]_ik = K_nu(kappa_i r) I_nu(kappa_k r)/r^2 [G_bracket]_ik,
   and its split into Born orders (orders in U_int), EXACT (finite matrix
   algebra, no series summed): G_bracket - 1 = G_bracket_born_order_1
   + G_bracket_born_order_2 + G_bracket_geq3, each Born order assembled from
   products that start at its own order, never as a difference.
5. _fv_weight, _gbar_integrand, gbar_traces: each Born order traced with the
   analytic FV weight r I_nu K_nu(kappa_i r) (np.trapezoid on the radial grid),
       Gbar_minus_fv = deg Int dr r sum_i I_nu K_nu(kappa_i r) [G_bracket - 1]_ii
   (the -1 subtracts the FV pointwise), Gbar_1 (Born order 1, the tadpole),
   Gbar_2 (Born order 2, the fish) and delta_geq3 (Born orders >= 3), so that
   as numbers
       delta_geq3(n, s^2) = Gbar_minus_fv - Gbar_1 - Gbar_2 .
   Renormalization: the false-vacuum and tadpole (orders 0 and 1 in U_int)
   subtractions act componentwise on the Green's function, wave by wave --
   that is the renormalization of the coincident Green's function.  Of their
   two finite add-backs only the tadpole's reaches the determinant (the
   false-vacuum one cancels in Gbar - Gbar_FV): traced and s2-integrated it
   is the assembler's (T2) G1_addback_band + (T4) G1_addback_high_s2, whose
   closed form is A1_fin.  The fish acts as the determinant-level
   subtraction Gbar_2 inside the s^2 integral, its finite part -A2_fin/2
   (T7) restored once at assembly.
6. _run_slice: one s2 slice (every requested wave) -> one npz, each wave
   passed through the BAND-EIG screen (det C > 0: no eigenvalue in the band).
Side entries: state_minus_r_max (the census, with h_minus_inf), gbar_fv
(the FV trace alone, kept for inspection).
GUARDS (abort-only, never enter lnD): BESSEL-RATIO (_rhs), RADIAL-GRID
(_radial_grid), BORN-SPLIT (_check_born_split), BAND-EIG (_run_slice).
Precision: scaled Bessel functions (ive/kve from fv_analytic_toolbox_vgeneral_phi_v2)
and Born orders built from products that start at their own order (no
cancellation of large numbers) -> float64 throughout; LSODA rtol 1e-11 /
atol 1e-13 (LSODA_RTOL / LSODA_ATOL).
"""
import argparse
import os
import sys
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

sys.dont_write_bytecode = True
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# analytic false-vacuum Bessel kernels (the ONE implementation)
from fv_analytic_toolbox_vgeneral_phi_v2 import (IK_prod_fv,
                                 dlogI_and_logI, dlogK_and_logK)
from pipeline_helpers_vgeneral_phi_v2 import (atomic_savez, band_s2_text,
                                              bounce_fingerprint, coupled_pairs,
                                              load_background, potential_id,
                                              radius_first, bracket_outer,
                                              VGENERAL_PHI_ENGINE_VERSION)

# Ceiling on the exponent of the Bessel ratio B_j/B_i in the coupling K
# (guard in _rhs): a margin below ln(DBL_MAX) = 709.78, where exp() overflows.
BESSEL_LOG_RATIO_MAX = 500.0

# LSODA tolerances of the h-ODE solve, the same for every caller (band
# workers, sector stages, census, add-on)
LSODA_RTOL = 1e-11
LSODA_ATOL = 1e-13

# BORN-SPLIT guard: |h - (h1 + h2 + h_geq3)| relative to |h| (in _check_born_split)
BORN_SPLIT_TOL = 1e-8

# LSODA's step cap in units of the shortest false-vacuum length 1/m_max
# (m_max^2 = the largest m_i^2).  Where U_int ~ 0 on both sides of a wall
# (vacua of equal masses) the error control alone lets the steps grow, and
# one step could leap the whole wall.  The bounce relaxes into a vacuum as
# exp(-m r), so U_int rises towards a wall over lengths set by the vacuum
# masses; a step capped at this fraction of 1/m_max cannot jump that rise.
LSODA_MAX_STEP_PER_FV_LENGTH = 0.5

# The radial output grid rule (_radial_grid): at least RADIAL_POINTS_MIN
# points, RADIAL_POINTS_PER_INV_KAPPA per 1/kappa_max so the coincident-trace
# quadrature resolves the FV weight, and never more than RADIAL_POINTS_MAX
# (the RADIAL-GRID guard aborts instead of under-resolving)
RADIAL_POINTS_MIN = 4000
RADIAL_POINTS_PER_INV_KAPPA = 25
RADIAL_POINTS_MAX = 60000


# ---------------------------------------------------------------------------
class GreensFunction:
    """Solves the h-ODE on ONE bounce background and assembles the Gbar
    traces.  ONE CODE PATH FOR EVERY MODEL: always the general N x N matrix
    system, whether or not the potential happens to be diagonal, so a
    decoupled cross-check exercises exactly the code a coupled
    result relies on."""

    def __init__(self, R, U_int, m2):
        R = np.asarray(R, float)
        self.r_max = float(R[-1])
        # the inner end of every solve: half the first cell of the bounce mesh
        self.r_start = 0.5 * R[1]
        self.m2 = np.asarray(m2, float)
        self.N = int(self.m2.size)                  # channel count (= field number)
        # LSODA's step cap, LSODA_MAX_STEP_PER_FV_LENGTH / m_max
        self.max_step = LSODA_MAX_STEP_PER_FV_LENGTH / np.sqrt(self.m2.max())
        U_int = np.asarray(U_int, float)
        if U_int.shape[1:] != (self.N, self.N):
            raise RuntimeError(f'[ABORT] U_int has shape {U_int.shape}, expected '
                               f'(nR, {self.N}, {self.N}) for {self.N} channels.')
        if not np.all(np.isfinite(U_int)):
            raise RuntimeError(
                '[ABORT] non-finite entries in U_int, the insertion on the bounce '
                'background: the bounce npz is corrupt.')
        N = self.N
        # The channel pairs the insertion couples (coupled_pairs); every other
        # entry of U_int is round-off (e.g. the off-diagonal x-y entry of a
        # spectator y the bounce leaves at zero; its U_yy still counts).
        # U_int itself is used as computed.
        coupled = coupled_pairs(U_int)
        # same_block[i, j]: channels i and j lie in the same diagonal block of
        # U_int (after reordering the channels), coupled directly or through
        # other channels (x-y and y-z coupled put x, y and z in one block): the
        # N-th power of the 0/1 table of coupled pairs, with 1 on its diagonal,
        # is nonzero exactly there.  Two channels in different blocks (e.g. a
        # decoupled spectator) never meet the Bessel ratio B_j/B_i: their
        # entries of U_int, K and h stay at round-off.
        self.same_block = np.linalg.matrix_power(
            (coupled | np.eye(N, dtype=bool)).astype(float), N) > 0
        self.rtol, self.atol = LSODA_RTOL, LSODA_ATOL
        # ONE vector spline holds all N*N components of U_int (CubicSpline
        # fits each column independently); scipy evaluates it, one call per r.
        # Finite data on a strictly increasing R give a finite spline value at
        # every r inside [R[0], r_max], so U_int_at needs no per-call check.
        self._U_int_spline = CubicSpline(R, U_int.reshape(len(R), N * N),
                                       extrapolate=False)

    # -- the shifted masses kappa_i --------------------------------------------
    def kappa(self, s2):
        """kappa_i = sqrt(s2 + m_i^2), (N,): the ONE creation point of the
        shifted masses in the engine (every Bessel argument is kappa_i r)."""
        return np.sqrt(self.m2 + float(s2))

    # -- U_int(r): the insertion, evaluated from the spline --------------------
    def U_int_at(self, r):
        """U_int(r) as (N, N) at r in [r_start, r_max], the window every solve
        stays in (the bounce grid R starts at r = 0).  Beyond r_max the engine
        takes the background to be the false vacuum, U_int = 0 (the bounce's
        tail there enters lnD_ren at second order, except through the tadpole
        moment, which addback_tadpole completes in closed form): that is what
        makes the start h^+ = h^+' = 0 of the '+' branch at r_max exact."""
        return self._U_int_spline(r).reshape(self.N, self.N)

    # -- the h-ODE right-hand side -------------------------------------------
    def _rhs(self, branch, nu, kappa_i):
        """The bundled 8*N*N first-order system of the h-ODE and its Born
        chain.  P (friction) and K (coupling) depend on r ONLY, not on y; the
        implicit solver re-calls f at the SAME r (corrector / numerical-
        Jacobian evaluations), so both are memoized on the last r and reused
        there bit-identically."""
        # (d/dz log, log) of the branch's Bessel function, I_nu ('-') or K_nu ('+'),
        # at z = kappa_i r -- the bare function, without the 1/r of B = Bessel/r
        dlog_and_log_bessel = dlogI_and_logI if branch == '-' else dlogK_and_logK
        # ---- GUARD (aborts; does NOT enter lnD): Bessel-ratio overflow ------
        # K carries the ratio B_j/B_i = Bessel(kappa_j r)/Bessel(kappa_i r) for
        # every pair in the same block of U_int (same_block).  Its log grows
        # with r (|z d(log Bessel)/dz| grows with z = kappa r), so it is
        # largest at r_max, for the two most separated channels of one block,
        # about nu ln(kappa_j/kappa_i) at high waves.  It grows with the MASS
        # SEPARATION within a block and with the wave -- a limit of the
        # potential being asked for, not of the solver.  (i, j) is the pair
        # that sets it.
        log_bessel_r_max = np.array(
            [dlog_and_log_bessel(nu, k * self.r_max)[1] for k in kappa_i])
        log_bessel_ratio = np.where(self.same_block,
                                    np.abs(log_bessel_r_max[None, :]
                                           - log_bessel_r_max[:, None]), 0.0)
        i, j = np.unravel_index(np.argmax(log_bessel_ratio), log_bessel_ratio.shape)
        if log_bessel_ratio[i, j] > BESSEL_LOG_RATIO_MAX:
            raise RuntimeError(
                f'[ABORT BESSEL-RATIO] the h-ODE coupling K would have to '
                f'exponentiate a log-Bessel difference of {log_bessel_ratio[i, j]:.0f} '
                f'(nu={nu:g}, channels {i + 1} and {j + 1}, counted from 1: kappa {kappa_i[i]:.4g} and '
                f'{kappa_i[j]:.4g} at r_max={self.r_max:.4g}), over the '
                f'ceiling {BESSEL_LOG_RATIO_MAX:g} (exp() overflows at '
                f'709.78).  Cause: the masses^2 of these two channels, '
                f'{self.m2[i]:g} and {self.m2[j]:g}, which lie in the same '
                f'block of U_int, are too far apart for this wave.  Use a '
                f'potential whose channels in one block of U_int have closer '
                f'masses.')
        N = self.N
        I_N = np.eye(N)
        PK_at_last_r = {'r': None}                     # last r -> its P, K
        def f(r, y):
            if PK_at_last_r['r'] != r:
                dlog_bessel = np.empty(N); log_bessel = np.empty(N)
                for i in range(N):
                    dlog_bessel[i], log_bessel[i] = dlog_and_log_bessel(nu, kappa_i[i] * r)
                PK_at_last_r['r'] = r
                # P_i = 3/r + 2 (ln B_i)' = 1/r + 2 kappa_i (ln Bessel)'(kappa_i r),
                # as B_i = Bessel(kappa_i r)/r: the friction of the h-ODE
                PK_at_last_r['P'] = 1.0 / r + 2.0 * kappa_i * dlog_bessel
                # K[i,j] = [U_int]_ij B_j/B_i = [U_int]_ij exp(log Bessel_j - log Bessel_i)
                # (the 1/r of B cancels): the coupling matrix of the h-ODE (exponent
                # 0 for a pair in different blocks of U_int: U_int is at most
                # round-off there, and the true exponent could overflow exp)
                PK_at_last_r['K'] = self.U_int_at(r) * np.exp(
                    np.where(self.same_block, log_bessel[None, :] - log_bessel[:, None], 0.0))
            P, K = PK_at_last_r['P'], PK_at_last_r['K']
            # the four equations h, h1, h2, h_geq3, each as (value, slope);
            # all share L = d^2/dr^2 + P d/dr and differ only in the source
            state = y.reshape(4, 2, N, N)
            value, slope = state[:, 0], state[:, 1]
            dstate_dr = np.empty_like(state)
            dstate_dr[:, 0] = slope                         # value' = slope
            dstate_dr[:, 1] = -P[:, None] * slope           # slope' = -P slope + source:
            dstate_dr[0, 1] += K @ (I_N + value[0])         #   the h-ODE  L h = K (1 + h)
            dstate_dr[1, 1] += K                            #   Born 1     L h1 = K
            dstate_dr[2, 1] += K @ value[1]                 #   Born 2     L h2 = K h1
            dstate_dr[3, 1] += K @ (value[2] + value[3])    #   remainder  L h_geq3 = K (h2 + h_geq3)
            return dstate_dr.reshape(-1)
        return f

    # -- LSODA solve plumbing -------------------------------------------------
    def _radial_grid(self, s2):
        """The shared radial output grid r_start..r_max (identical for both
        branches) of gbar_traces and gbar_fv: the point count follows the
        RADIAL_POINTS_* rule, scaling with kappa_max so the coincident-trace
        quadrature resolves the FV weight."""
        kappa_max = self.kappa(s2).max()
        n_points_needed = (RADIAL_POINTS_PER_INV_KAPPA * kappa_max
                           * (self.r_max - self.r_start))
        # ---- GUARD (aborts; does NOT enter lnD): radial-grid resolution -----
        if n_points_needed > RADIAL_POINTS_MAX:
            raise RuntimeError(
                f'[ABORT RADIAL-GRID] the radial grid at s2={s2:g} needs '
                f'{n_points_needed:.0f} points to resolve the Bessel coincident trace but '
                f'is capped at {RADIAL_POINTS_MAX} -- it would be silently under-resolved.  '
                f'Raise the cap or lower s2_max for this heavier-mass / wider '
                f'potential (kappa_max={kappa_max:.3g}, r_max-r_start='
                f'{self.r_max - self.r_start:.3g}).')
        n_radial_points = int(max(RADIAL_POINTS_MIN, n_points_needed))
        return np.linspace(self.r_start, self.r_max, n_radial_points)

    def solve_branch_sampled(self, branch, n, s2, r_grid):
        """The bundled h-ODE of one branch ('-' outward from r_start, '+'
        inward from r_max), sampled on r_grid, Born-split certified, returned
        as the raw (8, N, N, npts) state stack (blocks h, h', h1, h1', h2, h2',
        h_geq3, h_geq3').  branch_objects assembles everything from it, so
        the production band and the add-on suite solve through the same
        code path."""
        state = self._solve_on_grid(branch, n, s2, r_grid)
        self._check_born_split(state, n, s2, branch)
        return state

    def _solve_on_grid(self, branch, n, s2, r_grid):
        """Integrate the bundled h-ODE of one branch at (n, s2) with LSODA from
        the zero state (h = h' = 0 in all eight blocks) at its start point --
        '-' at r_start, integrated outward; '+' at r_max, integrated inward on
        the reversed t_eval -- sampled ON r_grid via t_eval (LSODA's own
        interpolant at exactly these radii).  Returns the (8, N, N, npts)
        state stack on increasing r.  One errstate suppresses the per-RHS
        overflow warnings for the whole solve."""
        rhs = self._rhs(branch, n + 1.0, self.kappa(s2))
        t_eval = r_grid[::-1] if branch == '+' else r_grid
        with np.errstate(all='ignore'):
            sol = solve_ivp(rhs, (t_eval[0], t_eval[-1]), np.zeros(8 * self.N * self.N),
                            method='LSODA', rtol=self.rtol, atol=self.atol,
                            t_eval=t_eval, max_step=self.max_step)
        if not sol.success:
            raise RuntimeError(f'LSODA failed on the h-ODE solve: {sol.message}')
        y = sol.y[:, ::-1] if branch == '+' else sol.y     # back to increasing r
        return y.reshape(8, self.N, self.N, -1)

    @staticmethod
    def _check_born_split(y, n, s2, branch):
        """GUARD BORN-SPLIT: h == h1 + h2 + h_geq3 exactly (the three Born
        equations sum to the h equation, same zero initial condition), so a
        disagreement beyond integration error means the ODE integrated
        wrongly.  Tolerance BORN_SPLIT_TOL relative to max(1, max |h|)."""
        h = y[0]
        dev = float(np.max(np.abs(h - (y[2] + y[4] + y[6]))))
        scale = max(1.0, float(np.max(np.abs(h))))
        if dev > BORN_SPLIT_TOL * scale:
            raise RuntimeError(
                f'[ABORT BORN-SPLIT] n={n}, s2={s2:g}, branch {branch}: the '
                f'solved h does not equal h1 + h2 + h_geq3 '
                f'(max deviation {dev:.3e}, |h|max {scale:.3e}, relative '
                f'{dev / scale:.3e} > {BORN_SPLIT_TOL:g}).  These are algebraically identical '
                f'(the three Born equations sum to the equation for h, same zero '
                f'initial condition), so the Born hierarchy did NOT integrate '
                f'correctly and the Born-subtracted remainder delta_geq3 built from blocks 2/4/6 cannot '
                f'be trusted.')

    # -- h^-(inf): the regular branch carried from r_max to infinity ---------
    def h_minus_inf(self, n, s2, h_minus_r_max, dh_minus_r_max):
        """h^-(r -> inf), from the value and slope of the regular branch
        at r_max:
            h^-(inf) = h^-(r_max)
                       + diag[r_max I_nu(kappa_i r_max) K_nu(kappa_i r_max)]
                         h^-'(r_max).
        Exact: beyond r_max U_int = 0, so r I_nu^2(kappa_i r) h^-'(r) stays
        constant, and Int_{r_max}^inf dr/(r I_nu^2) = K_nu/I_nu at
        kappa_i r_max.  The constant Wronskian is C = 1 + h^-(inf);
        r^3 [(phi^+)^T phi^-' - (phi^+')^T phi^-] at r_max, with phi^+ = B^+
        (h^+ = h^+' = 0 there) and phi^- = B^-(1 + h^-), gives the same
        formula (the Bessel Wronskian K I' - K' I = 1/z turns the B factors
        into 1).  The map is linear, so it applies to the full h^- and to
        every Born order alike; the stacks may carry leading axes (the three
        Born orders) before the (N, N) matrix, whose ROW index (the
        component) is what the FV weight w_fv_r_max multiplies.  The one
        rule every C of this pipeline is built by (branch_objects, the
        census)."""
        w_fv_r_max = self.r_max * IK_prod_fv(float(n) + 1.0,
                                             self.kappa(s2) * self.r_max)
        return h_minus_r_max + w_fv_r_max[:, None] * dh_minus_r_max

    # -- THE ONE assembly of the branch objects at one (n, s2) ---------------
    def branch_objects(self, n, s2, r_grid):
        """Solve both branches of the bundled h-ODE at ONE (n, s2), sampled on
        r_grid, and assemble every object built from them.  This is the ONE
        place the assembly is spelled: gbar_traces (production) and the add-on
        toolbox (add_on_greensfct/greensfct_toolbox_v2) both call it.  Every
        stack is [radius, component i, basis function a], exactly as the solves
        store h[i, a] (radius_first only moves the radius axis to the front).
        With N = the channel count and npts = len(r_grid) the returned dict holds

          h_minus, h_plus      (npts, N, N)  the full h^-(r), h^+(r) (state block 0)
          one_plus_h_minus     (npts, N, N)  1 + h^-(r), as the Born sum
                                             1 + h1 + h2 + h_geq3
          one_plus_h_plus      (npts, N, N)  1 + h^+(r), the same Born sum (so
                                             that the Born orders below are an
                                             algebraic identity in the same
                                             blocks; BORN-SPLIT certifies the
                                             sum equals block 0)
          C                    (N, N)   the constant Wronskian C = 1 + h^-(inf)
                                        = 1 + C_born_order_1 + C_born_order_2
                                        + C_geq3, each Born order of h^-
                                        carried from r_max to infinity by
                                        h_minus_inf (the same blocks as
                                        one_plus_h_minus, so the Born orders of
                                        C^-1 below are exact in them; the census
                                        builds C from the full h^-, equal up
                                        to the BORN-SPLIT tolerance)
          C_inv                (N, N)   numpy.linalg.inv(C)
          kappa_i              (N,)     the shifted masses sqrt(s2 + m_i^2)
          nu                   ()       the Bessel order n + 1
          G_bracket            (npts, N, N)  the coincident bracket
                 [G_bracket]_ik = sum_ab [1+h^+]_ia [C^-1]_ba [1+h^-]_kb
                                        (the '-' label b in the FIRST slot of
                                        C^-1, the '+' label a in the SECOND:
                                        bracket_outer)
          G_bracket_born_order_1       (npts, N, N)  the O(U^1) part of G_bracket - 1,
                 [G_bracket_born_order_1]_ik = [h1^+]_ik + [h1^-]_ki - [C_born_order_1]_ki
          G_bracket_born_order_2       (npts, N, N)  the O(U^2) part of G_bracket - 1,
                 [G_bracket_born_order_2]_ik = [h2^+]_ik + [h2^-]_ki - [C2]_ki + [C1 C1]_ki
                     + sum_a ([h1^+]_ia - [C1]_ai) [h1^-]_ka - sum_a [h1^+]_ia [C1]_ka
          G_bracket_geq3               (npts, N, N)  all orders >= 3 of G_bracket - 1,
                 assembled from products that start at order 3 (never as
                 G_bracket - 1 minus the two lower Born orders)
        The Born orders come from C^-1 = 1 - C1 + (C1 C1 - C2) + [orders >= 3] with
        C1 = C_born_order_1, C2 = C_born_order_2 (state blocks 2 and 4 at
        r_max, value + slope through h_minus_inf),
        collecting every way to spend k powers of U among the three factors
        of the bracket."""
        N = self.N
        I_N = np.eye(N)
        nu = float(n) + 1.0                       # the Bessel order n + 1
        kappa_i = self.kappa(s2)                  # the shifted masses sqrt(s2 + m_i^2)
        # ---- the two branch solves (each ONE bundled h/h1/h2/h_geq3 system)
        state_minus = self.solve_branch_sampled('-', n, s2, r_grid)   # (8, N, N, npts)
        state_plus = self.solve_branch_sampled('+', n, s2, r_grid)
        # stacked (npts, N, N) views of the state blocks of each branch
        h_minus = radius_first(state_minus[0])
        h1_minus = radius_first(state_minus[2])
        h2_minus = radius_first(state_minus[4])
        h_geq3_minus = radius_first(state_minus[6])
        h_plus = radius_first(state_plus[0])
        h1_plus = radius_first(state_plus[2])
        h2_plus = radius_first(state_plus[4])
        h_geq3_plus = radius_first(state_plus[6])
        # the regular branch at r_max: all eight blocks (h, h', h1, h1', h2,
        # h2', h_geq3, h_geq3'), (8, N, N)
        state_minus_r_max = state_minus[..., -1]
        # the Born orders of h^-(inf), each from value + slope at r_max; their
        # sum is C - 1 (BORN-SPLIT certified the endpoint sum), and the first
        # two also feed the Born orders of C^-1
        C_born_order_1, C_born_order_2, C_geq3 = self.h_minus_inf(
            n, s2, state_minus_r_max[[2, 4, 6]], state_minus_r_max[[3, 5, 7]])
        C = I_N + C_born_order_1 + C_born_order_2 + C_geq3   # the constant Wronskian C = 1 + h^-(inf)
        C_inv = np.linalg.inv(C)
        one_plus_h_plus = I_N + h1_plus + h2_plus + h_geq3_plus
        one_plus_h_minus = I_N + h1_minus + h2_minus + h_geq3_minus
        # [G_bracket]_ik = sum_ab [1+h^+]_ia [C^-1]_ba [1+h^-]_kb, so that
        # [G_n(r, r)]_ik = K_nu(kappa_i r) I_nu(kappa_k r)/r^2 [G_bracket]_ik
        G_bracket = bracket_outer(one_plus_h_plus, C_inv, one_plus_h_minus)
        # Born orders of C^-1 = 1 - C1 + (C1 C1 - C2) + C_inv_geq3 (C1, C2 the
        # Born orders of C - 1 above).  The remainder is exact, no series:
        # C_inv_geq3 = -C^-1 [C (1 + C_inv_born_order_1 + C_inv_born_order_2) - 1],
        # whose bracket, written out below, has only terms of order >= 3
        C_inv_born_order_1 = -C_born_order_1
        C_inv_born_order_2 = C_born_order_1 @ C_born_order_1 - C_born_order_2
        C_inv_geq3 = -C_inv @ (
            C_geq3
            - (C_born_order_2 + C_geq3) @ C_born_order_1
            + (C - I_N) @ C_inv_born_order_2)
        # the decaying branch MATCHED to the regular basis, (1 + h^+) C^-T,
        # and its Born orders.  (G_bracket = (1 + h^+)(1 + h~^-)^T with the
        # recombined regular branch 1 + h~^- = (1 + h^-) C^-1, whose constant
        # Wronskian is 1; the code groups C^-1 with the decaying branch.)
        # Every product below starts at the order its name says, so no lower
        # Born order is ever subtracted off the full bracket (.T transposes the
        # (N, N) table, .swapaxes(-1, -2) the (N, N) matrix at every radius)
        matched_plus_born_order_1 = h1_plus + C_inv_born_order_1.T
        matched_plus_born_order_2 = (h2_plus + C_inv_born_order_2.T
                                     + h1_plus @ C_inv_born_order_1.T)
        matched_plus_geq3 = (
            h_geq3_plus @ C_inv.T
            + (I_N + h1_plus + h2_plus) @ C_inv_geq3.T
            + h1_plus @ C_inv_born_order_2.T
            + h2_plus @ (C_inv_born_order_1 + C_inv_born_order_2).T)
        # the bracket is the matched decaying branch times (1 + h^-)^T, so its
        # Born orders collect the products of the Born orders of its two factors:
        # G_bracket - 1 = G_bracket_born_order_1 + G_bracket_born_order_2 + G_bracket_geq3
        G_bracket_born_order_1 = (matched_plus_born_order_1                 # O(U^1) part of G_bracket-1
                                  + h1_minus.swapaxes(-1, -2))
        G_bracket_born_order_2 = (matched_plus_born_order_2                 # O(U^2) part
                                  + h2_minus.swapaxes(-1, -2)
                                  + matched_plus_born_order_1 @ h1_minus.swapaxes(-1, -2))
        G_bracket_geq3 = (                                                  # all orders >= 3, assembled directly
            matched_plus_geq3 @ one_plus_h_minus.swapaxes(-1, -2)
            + (I_N + matched_plus_born_order_1 + matched_plus_born_order_2)
              @ h_geq3_minus.swapaxes(-1, -2)
            + matched_plus_born_order_1 @ h2_minus.swapaxes(-1, -2)
            + matched_plus_born_order_2 @ (h1_minus + h2_minus).swapaxes(-1, -2))
        return dict(h_minus=h_minus, h_plus=h_plus,
                    one_plus_h_minus=one_plus_h_minus,
                    one_plus_h_plus=one_plus_h_plus,
                    kappa_i=kappa_i, nu=nu,
                    C=C, C_inv=C_inv, G_bracket=G_bracket,
                    G_bracket_born_order_1=G_bracket_born_order_1,
                    G_bracket_born_order_2=G_bracket_born_order_2,
                    G_bracket_geq3=G_bracket_geq3)

    # -- the Gbar trace quadrature -------------------------------------------
    def _fv_weight(self, nu, s2, r):
        """w_fv_i(r) = r I_nu(kappa_i r) K_nu(kappa_i r), kappa_i = sqrt(m_i^2 + s2):
        the FV weight of every coincident trace, (N, npts).
        It is the 4D measure's 2 pi^2 r^3 against the harmonic sum's
        1/(2 pi^2), leaving r^3, times the free channel product I K / r^2:
        r^3 times the diagonal of the add-on's FV weight
        K_nu(kappa_i r) I_nu(kappa_j r)/r^2 (greensfct_toolbox_v2
        .fv_weight_matrix), so every trace here is a traced add-on wave."""
        return r[None, :] * IK_prod_fv(nu, self.kappa(s2)[:, None] * r[None, :])

    @staticmethod
    def _gbar_integrand(w_fv, bracket_stack):
        """The per-r integrand sum_i w_fv_i(r) [A]_ii(r) of a coincident trace
        deg * Int dr sum_i w_fv_i [A]_ii (np.trapezoid in gbar_traces);
        w_fv is the (N, npts) weight, bracket_stack the (npts, N, N) bracket
        part A whose diagonal is traced."""
        return np.einsum('ir,rii->r', w_fv, bracket_stack)

    # -- THE production evaluation: all Gbar traces at one (n, s2) -----------
    def gbar_traces(self, n, s2):
        """All Gbar traces of wave n at ONE s2, from ONE solve of the two
        branches (branch_objects).  Each is deg * Int dr sum_i w_fv_i [A]_ii on
        the engine's radial grid (np.trapezoid; w_fv_i = r I_nu K_nu(kappa_i r),
        the analytic FV weight) of one part A of the coincident bracket:
            Gbar_1         A = G_bracket_born_order_1: the Born-order-1 (tadpole)
                           trace, the part the renormalization of the
                           coincident Green's function removes; O((n+1)^-1)
                           per wave, so its wave sum diverges already at
                           fixed s2
            Gbar_2         A = G_bracket_born_order_2: the Born-order-2 (fish)
                           trace, the determinant-level subtraction;
                           O((n+1)^-3) at fixed s2 (the add-on keeps it), but
                           its s2-integral is O((n+1)^-1) per wave
            delta_geq3     A = G_bracket_geq3: the >= O(U^3) remainder, traced
                           from that part itself (every product in it starts
                           at order 3), never formed as a difference; its
                           s2-integral I_n(inf) = O((n+1)^-3) is the
                           convergent sequence stage 4 fits
            Gbar_minus_fv  A = G_bracket - 1, the sum of these three parts:
                           Gbar_bounce - Gbar_FV, the FV subtracted POINTWISE
            Gbar_fv        A = 1: the analytic FV trace alone (= gbar_fv(n, s2))
        Also returned: det_C, whose sign is the BAND-EIG screen of _run_slice
        (det C > 0 for every band wave n >= 2; a sector wave below its
        crossing, e.g. n = 0 at s2 < |omega2_00|, has det C < 0), and
        'integrands', the per-r integrands deg sum_i w_fv_i [A]_ii on the grid
        'r' (kept for inspection; nothing in the pipeline reads them)."""
        nu = n + 1.0
        deg = nu * nu                          # (n+1)^2 angular degeneracy
        r = self._radial_grid(s2)
        branch_objects = self.branch_objects(n, s2, r)
        w_fv = self._fv_weight(nu, s2, r)         # analytic FV weight r I_nu K_nu, (N, npts)
        integrand_Gbar_1 = self._gbar_integrand(w_fv, branch_objects['G_bracket_born_order_1'])
        integrand_Gbar_2 = self._gbar_integrand(w_fv, branch_objects['G_bracket_born_order_2'])
        integrand_delta_geq3 = self._gbar_integrand(w_fv, branch_objects['G_bracket_geq3'])
        integrand_Gbar_minus_fv = integrand_Gbar_1 + integrand_Gbar_2 + integrand_delta_geq3
        integrand_Gbar_fv = w_fv.sum(axis=0)      # A = 1: the weight summed over channels
        return {
            'Gbar_minus_fv': deg * np.trapezoid(integrand_Gbar_minus_fv, r),
            'Gbar_1': deg * np.trapezoid(integrand_Gbar_1, r),
            'Gbar_2': deg * np.trapezoid(integrand_Gbar_2, r),
            'delta_geq3': deg * np.trapezoid(integrand_delta_geq3, r),
            'Gbar_fv': deg * np.trapezoid(integrand_Gbar_fv, r),
            'det_C': float(np.linalg.det(branch_objects['C'])),
            'integrands': {
                'r': r,
                'Gbar_fv': deg * integrand_Gbar_fv,
                'Gbar_minus_fv': deg * integrand_Gbar_minus_fv,
                'Gbar_1': deg * integrand_Gbar_1, 'Gbar_2': deg * integrand_Gbar_2,
                'delta_geq3': deg * integrand_delta_geq3},
        }

    # -- side entries: the census solve and the closed-form FV trace --
    def state_minus_r_max(self, n, s2):
        """The regular branch at the endpoint: the (8, N, N) state block
        (blocks h, h', h1, h1', h2, h2', h_geq3, h_geq3') at r_max from ONE
        two-point t_eval=[r_start, r_max] solve.  The eigenvalue census builds
        C = identity + h_minus_inf(h, h') from blocks 0 and 1, the full h
        and h' (mode_census constant_wronskian_and_floor); the Born-order blocks
        2/4/6 are the RENORMALIZATION split and play no part in it
        (branch_objects builds the same C from their sum, equal to blocks 0
        and 1 up to the BORN-SPLIT tolerance)."""
        return self._solve_on_grid(
            '-', n, s2, np.array([self.r_start, self.r_max]))[..., -1]

    def gbar_fv(self, n, s2):
        """The false-vacuum coincident trace of wave n at s2 alone,
            Gbar_FV(n, s2) = (n+1)^2 Int_{r_start}^{r_max} dr r
                             sum_i I_nu K_nu(kappa_i r),
        kappa_i = sqrt(m_i^2 + s2).  No h-ODE solve: the coincident FV bracket
        is the identity, so only the FV weight w_fv is integrated, on the
        engine's radial grid -- a box-dependent number, so anything drawn from
        it must state the box [r_start, r_max].  gbar_traces returns the same
        number as 'Gbar_fv'.  Kept for inspection: nothing in the pipeline
        calls it."""
        nu = float(n) + 1.0
        r = self._radial_grid(float(s2))
        w_fv = self._fv_weight(nu, s2, r)
        return float(nu * nu * np.trapezoid(w_fv.sum(axis=0), r))

# ---------------------------------------------------------------------------
def _run_slice(engine, args, provenance, s2):
    """Compute one s2 slice (all requested waves) and write its npz.
    `provenance` is the metadata dict main built once (m2, engine, rtol,
    atol, bounce_sha, potential_id, code_version): not physics, it lets the
    resume skip-guard reject a stale same-tag slice computed with different
    tolerances/grid/bounce."""
    n_values = np.arange(args.n_min, args.n_max + 1)
    traces_per_wave = {k: [] for k in ('Gbar_minus_fv', 'Gbar_1', 'Gbar_2', 'delta_geq3')}
    t0 = time.time()
    for n in n_values:
        traces = engine.gbar_traces(int(n), s2)
        # ---- GUARD (aborts; does NOT enter lnD): band eigenvalue screen -----
        # Coleman screen: the n >= 2 band waves have NO in-range eigenvalue,
        # so det C > 0 and delta_geq3 is finite.  det C <= 0 signals an
        # in-range eigenvalue in the (un-pole-subtracted) band, which would be
        # integrated through into a wrong lnD; fail loud instead.  (A screen,
        # not a count: an even number of in-range eigenvalues would also give
        # det C > 0; the mode count is the census of n = 0, 1.)
        if not (np.isfinite(traces['delta_geq3']) and traces['det_C'] > 0.0):
            raise RuntimeError(
                f'[ABORT BAND-EIG] band wave n={n}, s2={s2:g}: '
                f'delta_geq3={traces["delta_geq3"]}, det C={traces["det_C"]:.6g} '
                f'-- det C <= 0 signals an in-range eigenvalue in the '
                f'(un-pole-subtracted) band.  This bounce/potential is not '
                f'Coleman-valid.')
        for k in traces_per_wave:
            traces_per_wave[k].append(traces[k])
        print(f'  n={n:4d} delta_geq3={traces["delta_geq3"]:+.6e} [{time.time()-t0:.0f}s]',
              flush=True)
    # the exact n = 0, 1 traces on this s2 (base slices only): Gbar_minus_fv
    # for the plot layer, which must not solve, and delta_geq3 for the band
    # driver's WATCH SECTOR-CUTOFF.  Their poles at s2 = -omega2_00 /
    # -omega2_10 are NOT subtracted here (the sector stages own that): a
    # slice next to a pole stores a large value, a slice exactly ON one
    # stores NaN (the plot layer guards against both).
    traces_n0_n1 = {}
    if args.file_suffix == '':
        for nd in (0, 1):
            try:
                traces_nd = engine.gbar_traces(nd, s2)
            except np.linalg.LinAlgError:        # exactly ON the pole: det C = 0
                traces_nd = dict(Gbar_minus_fv=np.nan, delta_geq3=np.nan)
            traces_n0_n1[f'Gbar_minus_fv_n{nd}'] = float(traces_nd['Gbar_minus_fv'])
            traces_n0_n1[f'delta_geq3_n{nd}'] = float(traces_nd['delta_geq3'])
        print(f'  [diag traces] Gbar_minus_fv_n0={traces_n0_n1["Gbar_minus_fv_n0"]:+.6e} '
              f'n1={traces_n0_n1["Gbar_minus_fv_n1"]:+.6e}', flush=True)
    # the slice name carries s2 as the band ladder spells it (band_s2_text,
    # 7 significant digits), '.' -> 'p'
    s2_tag = band_s2_text(s2).replace('.', 'p')
    slice_npz = os.path.join(args.data_dir,
                             f'residual_band_{args.tag}_s2{s2_tag}{args.file_suffix}.npz')
    traces_per_wave = {k: np.asarray(v) for k, v in traces_per_wave.items()}   # the four per-wave arrays of this slice
    atomic_savez(slice_npz, n_values=n_values, s2=s2,
                 **traces_per_wave, **traces_n0_n1, **provenance)
    print(f'[OK] wrote {slice_npz}')


def main():
    ap = argparse.ArgumentParser(description='vgeneral_phi_v2 h-ODE -> '
                                             "Green's-function band engine")
    ap.add_argument('--s2-list', required=True,
                    help='comma-separated s2 values; one output npz per slice '
                         '(one process does one or several slices)')
    ap.add_argument('--n-min', type=int, required=True,
                    help='lowest wave n of this slice')
    ap.add_argument('--n-max', type=int, required=True,
                    help='highest wave n of this slice (the adaptive band '
                         'driver sets it from its own watcher rectangle)')
    ap.add_argument('--data-dir', default=os.environ.get('VGENERAL_PHI_DATA', '.'))
    ap.add_argument('--tag', default='vgeneral_phi_v2')
    ap.add_argument('--file-suffix', default='',
                    help='appended to the slice filename before .npz; the '
                         'adaptive driver uses _n<lo>to<hi> for wave-extension '
                         'blocks on existing slices (default: none)')
    args = ap.parse_args()
    s2_values = [float(x) for x in args.s2_list.split(',') if x.strip()]
    if not s2_values:
        ap.error('--s2-list holds no s2 value')
    # the bounce npz comes from the env var, never a CLI path, so every
    # parallel worker of the band driver sees the same background
    bounce_npz = os.environ.get('VGENERAL_PHI_BOUNCE_NPZ')
    if not bounce_npz:
        raise SystemExit('set VGENERAL_PHI_BOUNCE_NPZ to the bounce npz path')
    background = load_background(bounce_npz)
    engine = GreensFunction(background['R'], background['U_int'], background['m2'])
    print(f'[engine] general {engine.N}x{engine.N} matrix Born chain '
          f'(8*{engine.N}*{engine.N} = {8*engine.N*engine.N} ODE components); the same '
          f'path for every model, coupled or diagonal')
    # provenance metadata (not physics), formed ONCE for every slice of this
    # process; the stored keys are the frozen interface read_band_tables reads
    provenance = dict(
        m2=background['m2'],
        engine='greens_function_vgeneral_phi_v2 all-order float64',
        rtol=float(engine.rtol), atol=float(engine.atol),
        bounce_sha=bounce_fingerprint(bounce_npz),
        potential_id=potential_id(background['m2'], background['params'],
                                  background['fields']),
        code_version=VGENERAL_PHI_ENGINE_VERSION)
    for s2 in s2_values:
        _run_slice(engine, args, provenance, s2)


if __name__ == '__main__':
    main()
