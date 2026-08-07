#!/usr/bin/env python3
"""delta_g_bar_greater_equal_3_coupled_toy_model.py -- the residual-delta band engine (Green's-function
subtraction).  The class is still ResidualBand (it IS the band engine); the
file name names its OUTPUT: delta_n(s^2), the >= O(U^3) residual trace
difference.  N = 2 CHANNELS here (the two coupled scalar fields; the count is
still read from the background rather than written in, so every matrix below
is (N,N) and the bundled h-ODE state is 8*N*N).  A fully decoupled U (all
off-diagonals zero -- the bdet toy model) falls back to per-channel SCALAR
Born chains, which is the same code with one active channel.

WHAT IT COMPUTES (per O(4) wave n, deformation s^2; nu = n+1, deg = nu^2)
------------------------------------------------------------------------
The production object is delta_n(s^2): the Green's-function PIECEWISE
SUBTRACTION of three coincident resolvent-trace objects that come from ONE
h-ODE solve,
    delta_n(s^2) = [Gbar_bounce - Gbar_FV] - Gbar(1) - Gbar(2)
                 = GfullmFV - gbar1 - gbar2      (all orders >= Born 3).
It is computed piecewise per (n, s^2) for the band n_min..n_max, s^2 <= s2_max,
as requested per round by the adaptive driver (band_adaptive_coupled_toy_model), whose
watchers decide the cutoffs; this engine just computes what it is asked.
The s^2 -> inf completion (ratio; tail_s2_completion_coupled_toy_model), the
n -> inf completion (zeta; tail_high_n_zeta_coupled_toy_model) and the sectors
n = 0, 1 are SEPARATE downstream stages -- this engine owns only the band.

Radial fluctuation operator (handwritten notes, 'procedure' p.3):
    [M_n(s^2)]_ij = [-d^2/dr^2 - (3/r) d/dr + n(n+2)/r^2 + s^2 + m_i^2] d_ij + U_ij(r)
with U_ij(r) = [H'(phi'_b(r))]_ij - m_i^2 d_ij in the primed (rotated) frame.

Mode functions on Bessel backgrounds  B-_i = I_nu(kap_i r)/r,  B+_i = K_nu(kap_i r)/r,
kap_i = sqrt(s^2 + m_i^2):   psi^{pm}_{i,a} = B^{pm}_i [ d_{ia} + h^{pm}_{ia} ].
h-matrix ODE (notes eq (43)):
    h''_{ia} + P_i h'_{ia} = sum_j U_ij Q_ij [ d_{ja} + h_{ja} ],
    P_i = 1/r + 2 kap_i B'(kap_i r)/B(kap_i r),   Q_ij = B_j(kap_j r)/B_i(kap_i r).

Born chain, solved as ONE bundled linear system per branch (all 2x2 matrices):
    L h1 = S,   L h2 = S h1,   L hR = S (h2 + hR),   S_ij = U_ij Q_ij
so h = h1 + h2 + hR EXACTLY and hR is the all-orders >=3 residual.  The h1
(order 1) and h2 (order 2) solves also supply the Born trace pieces below.

Coincident deg-weighted trace difference (fundamental-matrix Wronskian
C = I + h^-(inf), exact at the Dirichlet-free boundary r -> Rmax):
    (Gfull - GFV)_n(s^2) = nu^2 Int dr r sum_i IK_i(r) [T(r) - I]_ii,
    T = P (I+M)^{-1} Q',  P = I+h^+,  Q' = (I+h^-)^T,  M = (h^-(inf))^T.
Grading T - I by U-order splits it EXACTLY (an algebraic identity, verified
pointwise) into the two Born trace pieces plus the >=3 residual:
    T - I = B1 + B2 + X            (per r, per channel; no approximation)
    B1 = p1 + q1 - m1
    B2 = p2 + q2 + p1 q1 - m2 - p1 m1 - m1 q1 + m1^2

DELTA IS THE PIECEWISE SUBTRACTION of the three integrated trace objects,
formed per (n, s^2) from ONE h-ODE solve:
    delta_n(s^2) = GfullmFV - gbar1 - gbar2      <- production key delta_geq3
    GfullmFV = nu^2 Int dr r sum_i IK_i [T-1]_ii     (Gbar_bounce - Gbar_FV)
    gbar1    = nu^2 Int dr r sum_i IK_i [B1]_ii      (first Born trace)
    gbar2    = nu^2 Int dr r sum_i IK_i [B2]_ii      (second Born trace)
Since T-I = B1 + B2 + X exactly, this subtraction IS the >= O(U^3) residual.
The three objects come from the SAME bundled h1/h2/hR solve, so the subtraction
adds no runtime; eval_point returns all of them and delta_geq3 is just their
difference.  (Deep-wave catastrophic cancellation is relative ~1e-8, absolute
~1e-13; verified to reproduce the cancellation-free residual to 1e-9 in lnD --
far below the tail/BubbleDet error.)

CROSSCHECK (crosschecks/graded_residual_crosscheck_coupled_toy_model.py): the SAME delta
built the cancellation-free way,
    X = sum_{a+b+c>=3} P_a Mid_b Q'_c - P M^3 (I+M)^{-1} Q'
    (Mid = graded parts of I - M + M^2, orders 0..6),
which never subtracts the O(1) Born pieces.  Request it with
eval_point(graded=True) (returned as 'delta_graded'); it agrees with the
production subtraction to float64.  The two Born trace pieces gbar1/gbar2 are
inspected standalone in crosschecks/g_bar_1_coupled_toy_model.py and g_bar_2_coupled_toy_model.py.

Endpoint determinant validator (written next to every I_n; costs nothing):
    I_n^{>=3}(L2) =exact= nu^2 [ J3p(0) - J3p(L2) ],
    J3p(s2) = ln det(I + h^-(inf)) - tr m1 - tr m2 + (1/2) tr m1^2 .
Any s^2-grid information loss is therefore MEASURED per wave, not assumed.

Precision: only scaled Bessel functions (ive/kve) and intrinsically >=3rd-order
products appear -> float64 throughout; no mpmath (see notes/asymptotic_limit.pdf
sec. 'mpmath').

ENGINE PACKAGING (same math, same formulas, same evaluation
order; only the *packaging* of the per-step scalar work changed (verified
bit-identical on all six output fields, bdet 1.99x / F2 3.44x faster):
  - fast scalar Bessel helpers (plain if/else on scipy's scalar ive/kve;
        the matrix path reuses one ive/kve value for dlog AND log B);
        warning suppression paid once per solve, not per RHS call.
  - U(r) spline: the same CubicSpline coefficients evaluated directly
        (bisect + PPoly's exact evaluation recurrence).
  - matrix-path radial post-loop vectorized ((npts,N,N) stacks, same
        term order); r-independent Mid_b, M^3, (I+M)^-1 built once.
--rtol/--atol expose the LSODA tolerances (production default 1e-11/1e-13
unchanged).

MODULE LAYOUT: the analytic FV Bessel kernels live in fv_analytic_coupled_toy_model
(imported below): IK_prod is the coincident FV Green-function weight,
dlogI_and_logI_fast / dlogK_and_logK_fast feed the h-ODE RHS, dlogI_fast/dlogK_fast are
imported for the scalar RHS; dlogI/dlogK/logB (array reference forms) live in
that module but are not imported here.  The provenance helpers bounce_sha256 /
potential_id live in pipeline_helpers_coupled_toy_model (stamped into every slice npz).
"""
import argparse
import math
import os
import sys
import time
from bisect import bisect_right
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

sys.dont_write_bytecode = True
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# analytic false-vacuum Bessel kernels (the ONE certified implementation)
from fv_analytic_coupled_toy_model import (IK_prod,                             # noqa: E402
                                 dlogI_fast, dlogK_fast,
                                 dlogI_and_logI_fast, dlogK_and_logK_fast)
from pipeline_helpers_coupled_toy_model import atomic_savez, bounce_sha256, potential_id  # noqa: E402

# provenance/robustness metadata (not physics): a module-level version string
# stamped into every slice npz so the resume skip-guard can reject a stale
# same-tag slice produced by a materially different engine build.
COUPLED_TOY_MODEL_ENGINE_VERSION = 'coupled_toy_model.1'

# ---------------------------------------------------------------------------
class ResidualBand:
    def __init__(self, R, Uij, m2, rtol=1e-11, atol=1e-13, max_step=0.25,
                 force_matrix=None):
        self.R = np.asarray(R, float)
        self.rmax = float(self.R[-1])
        self.r_start = max(1e-4, self.R[1] * 0.5)   # inner ODE start r0 (both branches)
        self.m2 = np.asarray(m2, float)
        self.N = int(self.m2.size)                  # channel count (= 2 fields)
        # force_matrix: run the GENERAL matrix path even when U is diagonal
        # (decoupled), so a decoupled model is computed by the IDENTICAL code a
        # coupled model uses -- the scalar fast path is only a verified-identical
        # (to ODE precision) optimization, never a different determinant.
        # Default None -> read $COUPLED_TOY_MODEL_FORCE_MATRIX (0/1); the certified
        # production default is OFF (fast path).  Honoured by BOTH eval_point
        # and h_inf, so band, sectors AND eigenvalue locators all obey it.
        if force_matrix is None:
            force_matrix = bool(int(os.environ.get(
                'COUPLED_TOY_MODEL_FORCE_MATRIX', '0')))
        self.force_matrix = bool(force_matrix)
        U = np.asarray(Uij, float)
        if U.shape[1:] != (self.N, self.N):
            raise RuntimeError(f'[ABORT] U has shape {U.shape}, expected '
                               f'(nR, {self.N}, {self.N}) for {self.N} channels.')
        N = self.N
        self._Uspl = [[CubicSpline(self.R, U[:, i, j], extrapolate=False)
                       for j in range(N)] for i in range(N)]
        self.rtol, self.atol, self.max_step = rtol, atol, max_step
        # ---- direct access to the SAME spline coefficients ----------
        # PPoly evaluates segment i at s = r - x[i] via the Horner recurrence
        #     v = c[0,i]; v = v*s + c[1,i]; v = v*s + c[2,i]; v = v*s + c[3,i]
        # We reproduce exactly that with a bisect interval lookup.
        self._knots = self._Uspl[0][0].x                 # same R for all N*N
        self._knots_list = [float(x) for x in self._knots]
        self._R0 = self.R[0]
        # stacked coefficients (4, nseg, N, N) for the matrix path
        self._Uc = np.stack([np.stack([self._Uspl[i][j].c for j in range(N)],
                                      axis=-1) for i in range(N)], axis=-2)
        # per-channel python-float coefficients for the scalar path
        self._Uc_scal = [[(float(c[0]), float(c[1]), float(c[2]), float(c[3]))
                          for c in self._Uspl[i][i].c.T] for i in range(N)]
        self._nseg = len(self._knots) - 1
        # ---- block-diagonal scalar fast path detection --------------------
        # If EVERY off-diagonal U_ij vanishes identically, the Born chains
        # decouple per channel: solve only the channels whose diagonal U_ii
        # is nonzero (an inert channel has h == 0 and contributes nothing).
        # This reproduces the historical two-field modes: single-channel
        # (only U11 nonzero -> [0]) and decoupled (both diagonals -> [0,1]).
        absU = np.abs(U)
        off_max = max((absU[:, i, j].max() for i in range(N)
                       for j in range(N) if i != j), default=0.0)
        diag_active = [i for i in range(N) if absU[:, i, i].max() != 0.0]
        self.scalar_single_channel = bool(off_max == 0.0
                                          and len(diag_active) <= 1)
        self.scalar_decoupled = bool(off_max == 0.0 and len(diag_active) > 1)
        if off_max == 0.0:
            self._scalar_channels = diag_active if diag_active else [0]
        else:
            self._scalar_channels = None

    # -- fast spline front-ends -----------------------------------------
    def _seg(self, rr):
        """PPoly's interval: largest i with x[i] <= rr (< x[i+1]), clamped."""
        i = bisect_right(self._knots_list, rr) - 1
        if i < 0:
            i = 0
        elif i > self._nseg - 1:
            i = self._nseg - 1
        return i

    def Umat(self, r):
        if r > self.rmax:
            return np.zeros((self.N, self.N))
        rr = min(max(r, self._R0), self.rmax)
        i = self._seg(rr)
        s = rr - self._knots[i]
        c = self._Uc[:, i]                                  # (4, N, N)
        # PPoly's evaluate_poly1 order: c3 + c2*s + c1*(s*s) + c0*((s*s)*s)
        z2 = s * s
        v = c[3] + c[2] * s
        v = v + c[1] * z2
        v = v + c[0] * (z2 * s)
        if not np.all(np.isfinite(v)):
            raise RuntimeError(
                '[ABORT] non-finite U(r) matrix from the resolvent background '
                'spline -- corrupt bounce / interpolation.  Silently zeroing it '
                'would falsify the background, so fail loud.')
        return v

    def Uscal(self, r, ich):
        if r > self.rmax:
            return 0.0
        rr = min(max(r, self._R0), self.rmax)
        i = self._seg(rr)
        s = rr - self._knots[i]
        c0, c1, c2, c3 = self._Uc_scal[ich][i]
        # PPoly's evaluate_poly1 order: c3 + c2*s + c1*(s*s) + c0*((s*s)*s)
        z2 = s * s
        v = c3 + c2 * s
        v = v + c1 * z2
        v = v + c0 * (z2 * s)
        if not math.isfinite(v):
            raise RuntimeError(
                '[ABORT] non-finite U(r) scalar channel from the resolvent '
                'background spline -- corrupt bounce.  Silently zeroing it '
                'would falsify the background, so fail loud.')
        return v

    # -- RHS closures ---------------------------------------------------------
    def _rhs(self, branch, nu, kap):
        """P (centrifugal + Bessel dlog) and S (U * Bessel ratio) depend on r
        ONLY, not on y; the implicit solver calls f repeatedly at the SAME r
        (corrector / numerical-Jacobian evaluations), so they are memoized on
        the last r and reused there -- bit-identical (the very same floats are
        returned; if r changes they are recomputed by the same formula)."""
        terms = dlogI_and_logI_fast if branch == '-' else dlogK_and_logK_fast
        kapf = [float(k) for k in kap]                        # N channel kappas
        N = self.N
        I_N = np.eye(N)
        cache = {'r': None}                     # last r -> its P, S
        def f(r, y):
            if cache['r'] != r:
                dl = np.empty(N); lb = np.empty(N)
                for i in range(N):
                    dl[i], lb[i] = terms(nu, kapf[i] * r)
                cache['r'] = r
                cache['P'] = 1.0 / r + 2.0 * np.asarray(kapf) * dl
                cache['S'] = self.Umat(r) * np.exp(lb[None, :] - lb[:, None])
            P, S = cache['P'], cache['S']               # S[i,j] = U_ij B_j/B_i
            H = y.reshape(8, N, N)
            h, dh, h1, dh1, h2, dh2, hR, dhR = H
            out = np.empty_like(H)
            out[0] = dh;  out[1] = -P[:, None] * dh  + S @ (I_N + h)
            out[2] = dh1; out[3] = -P[:, None] * dh1 + S
            out[4] = dh2; out[5] = -P[:, None] * dh2 + S @ h1
            out[6] = dhR; out[7] = -P[:, None] * dhR + S @ (h2 + hR)
            return out.reshape(-1)
        return f

    def _rhs_scalar(self, branch, nu, kap_i, ich):
        """P and S depend on r only; memoized on the last r (the implicit solver
        re-evaluates f at the same r across corrector iterations) -- reused
        bit-identical, recomputed by the same formula when r changes."""
        dlog = dlogI_fast if branch == '-' else dlogK_fast
        cache = {'r': None}                     # last r -> its P, S
        def f(r, y):
            if cache['r'] != r:
                cache['r'] = r
                cache['P'] = 1.0 / r + 2.0 * kap_i * dlog(nu, kap_i * r)
                cache['S'] = self.Uscal(r, ich)
            P, S = cache['P'], cache['S']
            h, dh, h1, dh1, h2, dh2, hR, dhR = y
            return (dh,  -P * dh  + S * (1.0 + h),
                    dh1, -P * dh1 + S,
                    dh2, -P * dh2 + S * h1,
                    dhR, -P * dhR + S * (h2 + hR))
        return f

    def _band_grid(self, s2, ngrid):
        """The shared radial output grid r_start..Rmax, identical for both
        branches.  npts scales with kappa_max so the coincident-trace quadrature
        resolves the analytic Bessel weight; r_start = R[1]/2 (floored at 1e-4)
        is the inner ODE start."""
        kap = np.sqrt(self.m2 + s2)
        need = 25 * kap.max() * (self.rmax - self.r_start)   # points to resolve
        # ---- GUARD (aborts; does NOT enter lnD): radial-grid resolution ------
        if need > 60000:
            raise RuntimeError(
                f'[ABORT RADIAL-GRID] the radial grid at s2={s2:g} needs '
                f'{need:.0f} points to resolve the Bessel coincident trace but '
                f'is capped at 60000 -- it would be silently under-resolved.  '
                f'Raise the cap or lower s2_max for this heavier-mass / wider '
                f'potential (kap_max={kap.max():.3g}, rmax-r_start='
                f'{self.rmax - self.r_start:.3g}).')
        npts = int(max(ngrid, need))
        return np.linspace(self.r_start, self.rmax, npts)

    def _solve_on_grid(self, rhs, r_grid, y0, backward):
        """Integrate the bundled h-ODE with LSODA and return its solution
        sampled ON r_grid (strictly increasing, r_start..Rmax), shape
        (len(y0), len(r_grid)), columns ordered by increasing r.  backward=True
        is the '+' branch: it integrates inward (Rmax -> r_start, via a reversed
        t_eval) and the returned columns are flipped back to increasing r.

        The solution is requested through t_eval (NOT dense_output), so scipy
        never builds the dense OdeSolution object.  That construction is what
        raised `ts must be strictly increasing` at large n / small s2, where the
        stiff h-ODE makes LSODA emit a degenerate step that corrupts the
        collected step list; sampling on t_eval sidesteps it entirely.  Verified
        bit-identical (0.00e+00) to the former dense output on every wave that
        solved (bdet scalar + non-stiff matrix), and it additionally solves the
        stiff high-n/low-s2 matrix waves the dense path could not.  One errstate
        for the whole solve suppresses the per-RHS overflow/invalid warnings."""
        t_eval = r_grid[::-1] if backward else r_grid
        with np.errstate(all='ignore'):
            sol = solve_ivp(rhs, (t_eval[0], t_eval[-1]), y0, method='LSODA',
                            rtol=self.rtol, atol=self.atol, t_eval=t_eval,
                            max_step=self.max_step)
        if not sol.success:
            raise RuntimeError(f'LSODA failed on the h-ODE solve: {sol.message}')
        return sol.y[:, ::-1] if backward else sol.y

    def h_inf(self, n, s2, branch='-'):
        """h^{branch}(Rmax) as the generic (8,N,N) block (rows h, dh, h1, dh1,
        h2, dh2, hR, dhR).  The sector eigenvalue locators consume M = h^-(inf)
        through det(I + h1+h2+hR)(inf); this returns that boundary block from ONE
        LSODA t_eval=[r_start, Rmax] solve, taking the r=Rmax endpoint column.
        Scalar potentials fill the (ich, ich) diagonal blocks (off-blocks stay 0)."""
        nu = n + 1.0
        N = self.N
        grid = np.array([self.r_start, self.rmax])
        backward = branch == '+'
        if self._scalar_channels is not None and not self.force_matrix:
            out = np.zeros((8, N, N))
            for i in self._scalar_channels:
                kap_i = float(np.sqrt(self.m2[i] + s2))
                y = self._solve_on_grid(self._rhs_scalar(branch, nu, kap_i, i),
                                        grid, np.zeros(8), backward)
                out[:, i, i] = y[:, -1]
            return out
        kap = np.sqrt(self.m2 + s2)
        y = self._solve_on_grid(self._rhs(branch, nu, kap), grid,
                                np.zeros(8 * N * N), backward)
        return y[:, -1].reshape(8, N, N)

    # -- graded residual X (CROSSCHECK-ONLY: reached via eval_point(graded=True);
    #    production delta is the subtraction GfullmFV - gbar1 - gbar2).  The
    #    scalar and vectorized (matrix) forms below are the two live builds. --
    @staticmethod
    def _graded_residual_scalar(p1, p2, p3, q1, q2, q3, m1, m2, m3):
        P = {0: 1.0, 1: p1, 2: p2, 3: p3}
        Q = {0: 1.0, 1: q1, 2: q2, 3: q3}
        Mid = {0: 1.0, 1: -m1, 2: -m2 + m1 * m1,
               3: -m3 + 2.0 * m1 * m2,
               4: 2.0 * m1 * m3 + m2 * m2,
               5: 2.0 * m2 * m3,
               6: m3 * m3}
        X = 0.0
        for a in P:
            for b in Mid:
                for c in Q:
                    if a + b + c >= 3:
                        X = X + P[a] * Mid[b] * Q[c]
        M = m1 + m2 + m3
        X = X - (1.0 + p1 + p2 + p3) * (M ** 3) / (1.0 + M) \
                * (1.0 + q1 + q2 + q3)
        return X

    # -- vectorized matrix-path grading over the radial grid ---------
    @staticmethod
    def _graded_residual_vec(p1, p2, p3, q1, q2, q3, m1, m2, m3):
        """Same terms, same order as _graded_residual_scalar, with p*/q* stacked
        as (npts,N,N).  The radius-independent Mid_b, M^3 and (I+M)^{-1} are
        built ONCE (they never depended on the radial point)."""
        npts, N = p1.shape[0], p1.shape[1]
        I_N = np.eye(N)
        Ist = np.broadcast_to(I_N, (npts, N, N))
        P = {0: Ist, 1: p1, 2: p2, 3: p3}
        Q = {0: Ist, 1: q1, 2: q2, 3: q3}
        Mid = {0: I_N, 1: -m1, 2: -m2 + m1 @ m1,
               3: -m3 + m1 @ m2 + m2 @ m1,
               4: m1 @ m3 + m3 @ m1 + m2 @ m2,
               5: m2 @ m3 + m3 @ m2,
               6: m3 @ m3}
        X = np.zeros((npts, N, N))
        for a in P:
            for b in Mid:
                for c in Q:
                    if a + b + c >= 3:
                        X = X + (P[a] @ Mid[b]) @ Q[c]
        M = m1 + m2 + m3
        M3 = M @ M @ M
        Cinv = np.linalg.inv(I_N + M)
        Pf = I_N + p1 + p2 + p3
        Qf = I_N + q1 + q2 + q3
        X = X - ((Pf @ M3) @ Cinv) @ Qf
        return X

    def _eval_point_scalar(self, n, s2, ngrid=4000, graded=False):
        nu = n + 1.0
        kap = np.sqrt(self.m2 + s2)
        chans = self._scalar_channels
        r = self._band_grid(s2, ngrid)
        npts = r.size
        # ---- 1. the two branch solves per channel (each a bundled h1/h2/hR
        #         system); the loop accumulates the three objects' integrands --
        d_full = np.zeros(npts); d_B1 = np.zeros(npts)
        d_B2 = np.zeros(npts); d_X = np.zeros(npts)
        J3p = 0.0; lndetC = 0.0
        for i in chans:
            kap_i = float(kap[i])
            ym = self._solve_on_grid(self._rhs_scalar('-', nu, kap_i, i),
                                     r, np.zeros(8), False)          # (8, npts)
            yp = self._solve_on_grid(self._rhs_scalar('+', nu, kap_i, i),
                                     r, np.zeros(8), True)
            yinf = ym[:, -1]                                         # at Rmax
            m1, m2_, m3 = float(yinf[2]), float(yinf[4]), float(yinf[6])
            Minf = m1 + m2_ + m3
            w = r * IK_prod(nu, kap[i] * r)
            p1, p2, p3 = yp[2], yp[4], yp[6]
            q1, q2, q3 = ym[2], ym[4], ym[6]
            T = (1.0 + p1 + p2 + p3) * (1.0 + q1 + q2 + q3) / (1.0 + Minf)
            B1 = p1 + q1 - m1
            B2 = (p2 + q2 + p1 * q1 - m2_ - p1 * m1 - m1 * q1 + m1 * m1)
            d_full += w * (T - 1.0)
            d_B1 += w * B1
            d_B2 += w * B2
            if graded:               # CROSSCHECK-ONLY cancellation-free residual
                d_X += w * self._graded_residual_scalar(
                    p1, p2, p3, q1, q2, q3, m1, m2_, m3)
            J3p += (np.log(1.0 + Minf) - m1 - m2_ + 0.5 * m1 * m1)
            lndetC += np.log(1.0 + Minf)
        deg = nu * nu
        # ---- 2. the three coincident-trace objects (summed over channels) ----
        Gfull = deg * np.trapezoid(d_full, r)      # Gbar_bounce - Gbar_FV
        Gb1 = deg * np.trapezoid(d_B1, r)          # Gbar(1)  (first Born trace)
        Gb2 = deg * np.trapezoid(d_B2, r)          # Gbar(2)  (second Born trace)
        # ---- 3. the production delta = piecewise subtraction of the three ----
        delta_geq3 = Gfull - Gb1 - Gb2                       # all orders >= Born 3
        out = {
            'n': n, 's2': s2,
            'GfullmFV': Gfull, 'G1': Gb1, 'G2': Gb2,
            'delta_geq3': delta_geq3,
            'gbar1': Gb1, 'gbar2': Gb2,
            'J3plus': J3p, 'lndetC': float(lndetC),
        }
        if graded:
            out['delta_graded'] = deg * np.trapezoid(d_X, r)
        return out

    @staticmethod
    def _deg_weighted_trace(w, diags, r, deg):
        """The coincident-trace quadrature shared by the matrix-path Gbar
        objects (the scalar path inlines the same trapezoid):
            deg * Int_r0^Rmax dr sum_i w_i*[A]_ii ,
        w_i(r) = r * I_nu(k_i r) K_nu(k_i r) (the analytic Bessel FV weight),
        [A] the per-point 2x2 (T-1 for full, B1 for gbar1, B2 for gbar2);
        w and diags are (N, npts) stacks.  Calling it three times is what
        makes the code read as 'build three objects, then subtract'."""
        acc = w[0] * diags[0]
        for i in range(1, len(diags)):
            acc = acc + w[i] * diags[i]
        return deg * np.trapezoid(acc, r)

    def eval_point(self, n, s2, ngrid=4000, force_matrix=False, graded=False):
        """One (n, s2) point in three explicit steps:
          1. solve the h-ODE as ONE bundled system per branch (the two branch
             solves each yield h1,h2,hR together, so all three objects do too);
          2. build the THREE coincident-trace objects from that single solve --
                 GfullmFV = Gbar_bounce - Gbar_FV   (FV subtracted POINTWISE via
                            T-1; it is not a separate finite object)
                 gbar1    = Gbar(1)                 (first-Born trace)
                 gbar2    = Gbar(2)                 (second-Born trace)
          3. SUBTRACT them:  delta_geq3 = GfullmFV - gbar1 - gbar2  (>= Born 3).
        graded=True also returns 'delta_graded' = the SAME delta built the
        cancellation-free way (X), for the crosscheck.
        """
        if (self._scalar_channels is not None
                and not (force_matrix or self.force_matrix)):
            return self._eval_point_scalar(n, s2, ngrid=ngrid, graded=graded)
        nu = n + 1.0
        deg = nu * nu
        N = self.N
        kap = np.sqrt(self.m2 + s2)
        # ---- 1. the two branch solves (each one bundled h1/h2/hR system) -----
        r = self._band_grid(s2, ngrid)
        ym = self._solve_on_grid(self._rhs('-', nu, kap), r,
                                 np.zeros(8 * N * N), False).reshape(8, N, N, -1)
        yp = self._solve_on_grid(self._rhs('+', nu, kap), r,
                                 np.zeros(8 * N * N), True).reshape(8, N, N, -1)
        yinf = ym[..., -1]                                   # h^-(inf) at Rmax
        m1, m2_, m3 = yinf[2].T.copy(), yinf[4].T.copy(), yinf[6].T.copy()
        Minf = m1 + m2_ + m3
        IK = np.array([IK_prod(nu, kap[i] * r) for i in range(N)])   # (N, npts)
        w = r[None, :] * IK                        # analytic Bessel FV weight r*IK
        I_N = np.eye(N)
        Cinv = np.linalg.inv(I_N + Minf)
        # stacked (npts, N, N) views of the Born orders of each branch
        p1 = np.ascontiguousarray(np.moveaxis(yp[2], -1, 0))
        p2 = np.ascontiguousarray(np.moveaxis(yp[4], -1, 0))
        p3 = np.ascontiguousarray(np.moveaxis(yp[6], -1, 0))
        q1 = np.ascontiguousarray(np.moveaxis(ym[2], -1, 0).transpose(0, 2, 1))
        q2 = np.ascontiguousarray(np.moveaxis(ym[4], -1, 0).transpose(0, 2, 1))
        q3 = np.ascontiguousarray(np.moveaxis(ym[6], -1, 0).transpose(0, 2, 1))
        # ---- 2. the THREE coincident-trace objects (all from the one solve) --
        Pf = I_N + p1 + p2 + p3
        Qf = I_N + q1 + q2 + q3
        T = (Pf @ Cinv) @ Qf                       # T-1 subtracts the FV pointwise
        B1 = p1 + q1 - m1                                    # O(U^1) grade of T-1
        B2 = p2 + q2 + p1 @ q1 - m2_ - p1 @ m1 - m1 @ q1 + m1 @ m1  # O(U^2) grade
        Gfull = self._deg_weighted_trace(
            w, [T[:, i, i] - 1.0 for i in range(N)], r, deg)  # Gbar_bounce-Gbar_FV
        Gb1 = self._deg_weighted_trace(
            w, [B1[:, i, i] for i in range(N)], r, deg)               # Gbar(1)
        Gb2 = self._deg_weighted_trace(
            w, [B2[:, i, i] for i in range(N)], r, deg)               # Gbar(2)
        # ---- 3. the production delta = piecewise subtraction of the three ----
        delta_geq3 = Gfull - Gb1 - Gb2                       # all orders >= Born 3
        J3p = (np.log(np.linalg.det(I_N + Minf)) - np.trace(m1)
               - np.trace(m2_) + 0.5 * np.trace(m1 @ m1))
        out = {
            'n': n, 's2': s2,
            'GfullmFV': Gfull, 'G1': Gb1, 'G2': Gb2,
            'delta_geq3': delta_geq3,
            'gbar1': Gb1, 'gbar2': Gb2,
            'J3plus': J3p,
            'lndetC': float(np.log(np.linalg.det(I_N + Minf))),
        }
        if graded:      # CROSSCHECK-ONLY: the SAME delta built cancellation-free
            X = self._graded_residual_vec(p1, p2, p3, q1, q2, q3, m1, m2_, m3)
            out['delta_graded'] = self._deg_weighted_trace(
                w, [X[:, i, i] for i in range(N)], r, deg)
        return out

# ---------------------------------------------------------------------------
def load_background():
    """Load bounce + primed-frame Hessian (the shared background every band
    wave is built on: same npz schema, same primed frame as
    pipeline_helpers_coupled_toy_model.load_bounce, via the ONE schema reader
    bounce_arrays -- the two-field schema with the legacy key fallback)."""
    from pipeline_helpers_coupled_toy_model import load_bounce
    bounce_env = os.environ.get('COUPLED_TOY_MODEL_BOUNCE_NPZ')
    if not bounce_env:
        raise SystemExit('set COUPLED_TOY_MODEL_BOUNCE_NPZ to the bounce npz path')
    bg = load_bounce(bounce_env)
    b = np.load(bounce_env, allow_pickle=True)
    return bg['R'], bg['DW'], bg['m2'], b


def selftest_fastpath(n_list=[10, 40, 80], s2_list=[0.0, 100.0],
                      rtol=1e-13, atol=1e-15):
    """Run scalar fast path AND generic matrix path on the loaded background;
    assert delta_geq3 agreement to <=1e-9 relative at every (n, s2).
    Tolerances are tightened two decades below production (1e-11/1e-13) so
    accumulated LSODA global error (~1e-9 at production settings, identical
    equations either way) sits safely below the assertion threshold."""
    R, U, m2, b = load_background()
    band = ResidualBand(R, U, m2, rtol=rtol, atol=atol)
    if band._scalar_channels is None:
        print('[selftest] U fully coupled -> scalar fast path inactive; nothing to test')
        return
    mode = (f'scalar_single_channel (active channel '
            f'{band._scalar_channels})' if band.scalar_single_channel
            else f'scalar_decoupled ({len(band._scalar_channels)} chains)')
    print(f'[selftest] mode = {mode}')
    worst = 0.0
    for n in n_list:
        for s2 in s2_list:
            t0 = time.time()
            of = band.eval_point(int(n), float(s2))
            tf = time.time() - t0
            t0 = time.time()
            om = band.eval_point(int(n), float(s2), force_matrix=True)
            tm = time.time() - t0
            rel = abs(of['delta_geq3'] - om['delta_geq3']) \
                / max(abs(om['delta_geq3']), 1e-300)
            worst = max(worst, rel)
            print(f'[selftest] n={n:3d} s2={s2:8.2f}  '
                  f'fast={of["delta_geq3"]:+.12e} ({tf:5.2f}s)  '
                  f'matrix={om["delta_geq3"]:+.12e} ({tm:5.2f}s)  '
                  f'rel={rel:.3e}', flush=True)
            assert rel <= 1e-9, (f'fast-path mismatch n={n} s2={s2}: '
                                 f'rel={rel:.3e} > 1e-9')
    print(f'[selftest] PASS  worst rel(delta_geq3) = {worst:.3e} <= 1e-9')


def _run_slice(band, b, m2, args, s2):
    """Compute one s2 slice and write one coupled_toy_model-tagged npz (unchanged
    physics fields; provenance/robustness metadata appended)."""
    if args.n_max is None:
        if args.r_wall:
            rW = args.r_wall
        else:
            from pipeline_helpers_coupled_toy_model import bounce_arrays, wall_radius
            R, Phi, _params, _fv, _fields = bounce_arrays(b)
            rW = wall_radius(R, Phi)
        n_max = int(np.ceil(1.25 * rW * np.sqrt(s2 + m2.min()))) + 20
    else:
        n_max = args.n_max
    ns = np.arange(args.n_min, n_max + 1)
    # J3plus: diagnostic only; not used by the Method-I s2-completion stage
    rows = {k: [] for k in ('GfullmFV', 'G1', 'G2', 'delta_geq3', 'J3plus')}
    t0 = time.time()
    for n in ns:
        o = band.eval_point(int(n), s2)
        # ---- GUARD (aborts; does NOT enter lnD): band eigenvalue screen ------
        # Coleman screen for the band: the n>=2 waves have NO in-range
        # eigenvalue, so det(I+Minf)>0 (finite lndetC) and delta_geq3 is finite.
        # A non-finite value means det(I+Minf)<=0 -- an in-range band eigenvalue
        # (an extra negative/near-zero mode Coleman puts only in n=0/1).  The
        # band is NOT pole-subtracted, so such a mode would be integrated
        # through into a wrong lnD; fail loud instead.
        if not (np.isfinite(o['delta_geq3']) and np.isfinite(o['lndetC'])):
            raise RuntimeError(
                f'[ABORT BAND-EIG] band wave n={n}, s2={s2:g}: '
                f'delta_geq3={o["delta_geq3"]}, lndetC={o["lndetC"]} non-finite '
                f'-- det(I+Minf)<=0 signals an in-range eigenvalue in the '
                f'(un-pole-subtracted) band.  This bounce/potential is not '
                f'Coleman-valid.')
        for k in rows:
            rows[k].append(o[k])
        print(f'  n={n:4d} delta_geq3={o["delta_geq3"]:+.6e} [{time.time()-t0:.0f}s]',
              flush=True)
    tagv = f"{s2:.6f}".replace('.', 'p')
    out = os.path.join(args.out_dir,
                       f'residual_band_{args.tag}_s2{tagv}{args.file_suffix}.npz')
    # provenance/robustness metadata (not physics): lets the resume skip-guard
    # reject a stale same-tag slice computed with different tolerances/grid/
    # bounce.  None of these fields is read by any downstream physics path.
    atomic_savez(out, n_values=ns, s2=s2,
             **{k: np.asarray(v) for k, v in rows.items()},
             m2=m2, engine='delta_g_bar_greater_equal_3_coupled_toy_model all-order float64',
             rtol=float(band.rtol), atol=float(band.atol),
             bounce_sha=bounce_sha256(os.environ.get('COUPLED_TOY_MODEL_BOUNCE_NPZ', '')),
             potential_id=potential_id(m2),
             code_version=COUPLED_TOY_MODEL_ENGINE_VERSION)
    print(f'[OK] wrote {out}')


def main():
    ap = argparse.ArgumentParser(description='coupled_toy_model all-order residual band')
    ap.add_argument('--s2', type=float, default=None)
    ap.add_argument('--s2-list', default=None,
                    help='comma-separated s2 values; one output npz per slice '
                         '(lets one process do several slices)')
    ap.add_argument('--n-min', type=int, default=15)
    ap.add_argument('--n-max', type=int, default=None,
                    help='default: edge-matched ceil(1.25*r_W*sqrt(s2+min m^2))+20')
    ap.add_argument('--r-wall', type=float, default=None)
    ap.add_argument('--out-dir', default=os.environ.get('G_PROJECT_DATA', '.'))
    ap.add_argument('--tag', default='coupled_toy_model')
    ap.add_argument('--file-suffix', default='',
                    help='appended to the slice filename before .npz; the '
                         'adaptive driver uses _n<lo>to<hi> for wave-extension '
                         'blocks on existing slices (default: none)')
    ap.add_argument('--rtol', type=float, default=1e-11,
                    help='LSODA relative tolerance (production default 1e-11; '
                         '1e-10 measured safe: per-wave |dI_n| <= 1e-8, '
                         'assembled lnD shift < 1e-4)')
    ap.add_argument('--atol', type=float, default=1e-13,
                    help='LSODA absolute tolerance (production default 1e-13; '
                         'pair 1e-12 with --rtol 1e-10)')
    ap.add_argument('--selftest', action='store_true',
                    help='run selftest_fastpath() and exit')
    args = ap.parse_args()
    if args.selftest:
        selftest_fastpath()
        return
    if args.s2_list:
        s2_values = [float(x) for x in args.s2_list.split(',') if x.strip()]
    elif args.s2 is not None:
        s2_values = [args.s2]
    else:
        ap.error('provide --s2 or --s2-list')
    R, U, m2, b = load_background()
    band = ResidualBand(R, U, m2, rtol=args.rtol, atol=args.atol)
    if (args.rtol, args.atol) != (1e-11, 1e-13):
        print(f'[tolerances] LSODA rtol={args.rtol:g} atol={args.atol:g} '
              f'(production certified default is 1e-11/1e-13)')
    if band.scalar_single_channel:
        print(f'[fastpath] all off-diagonal U_ij = 0 -> scalar 8-dim Born '
              f'chain (active channel(s) {band._scalar_channels} of '
              f'{band.N})')
    elif band.scalar_decoupled:
        print(f'[fastpath] all off-diagonal U_ij = 0 -> '
              f'{len(band._scalar_channels)} decoupled scalar Born chains')
    for s2 in s2_values:
        _run_slice(band, b, m2, args, float(s2))


if __name__ == '__main__':
    main()
