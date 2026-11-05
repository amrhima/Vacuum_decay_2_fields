#!/usr/bin/env python3
"""bounce_polish_vgeneral_phi_v2.py -- level 1, the solve library of stage 0
(bounce_vgeneral_phi_v2): THE production bounce phi_b(r) of one vacuum pair
on its radial grid R, as ONE boundary-value solve of the O(4) equation of
motion in the primed frame.  The CosmoTransitions profile is the SEED only;
this solve carries all accuracy, for every bounce -- one-field, decoupled,
coupled.

READS   R_seed, phi_seed   the seed profile, primed frame
                           (bounce_vgeneral_phi_v2.seed_profile)
        pot_prime          the PrimedPotential of the pair
                           (potential_vgeneral_phi_v2): V', dV', H' along the
                           profile and the false-vacuum channel masses m2
WRITES  bvp_polish(...) -> dict with R, phi_b, S_E, success and the solve
        meters, read by bounce_vgeneral_phi_v2.polish_profile (the kept
        bounce: R is the radial grid of every later stage, its end r_max the
        radius where the constant Wronskian C is evaluated) and by
        bounce_vgeneral_phi_v2.reference_profile (the tighter re-solve on the
        kept grid, eval_R = R); POLISH_TOL_REF, POLISH_MAX_NODES and
        POLISH_DECAY_LENGTHS_PAST_WALL are the constants those callers read.
        fv_mass_min / polish_r_max / action_O4 are its internal helpers --
        importable, but no other module calls them.

HOW  (all of it PRIMED: phi' = L^T (phi - phi_F), the frame the
     PrimedPotential handed in already defines; the prime marks the primed
     frame ONLY, radial derivatives are written d/dr)
    d^2 phi'_i/dr^2 + (3/r) d phi'_i/dr = dV'/dphi'_i(phi'(r)) ,  i = 1..N
    d phi'_i/dr (0) = 0      (regularity at the origin)
    d phi'_i/dr (r_max) = -m_i K_2(m_i r_max)/K_1(m_i r_max) phi'_i(r_max)
                             (at r_max the profile is, up to terms quadratic
                             in it, its decaying tail A_i K_1(m_i r)/r about
                             the primed false vacuum, the origin; _bvp_system)
  1. the solve radius r_max from the model's own scales: the wall radius
     r_W of the seed plus a fixed number of decay lengths 1/m_min of the
     slowest of the channels the bounce acts on (nonzero U_int row;
     fv_mass_min, polish_r_max; the constants block);
  2. the initial guess: the seed profile, padded with zeros (the false
     vacuum) up to r_max;
  3. scipy.integrate.solve_bvp: collocation on a mesh the solver refines
     itself, Newton on the resulting nonlinear system with the ANALYTIC
     gradient and Hessian of the PrimedPotential (_bvp_system), and no
     shooting parameter anywhere -- the whole profile is one unknown vector
     and the residual driven to zero is that of the full coupled system;
  4. the profile on the whole converged mesh (or on eval_R) and its O(4)
     action S_E (action_O4, Simpson).
"""
import os
import sys
import time

import numpy as np
from scipy.integrate import solve_bvp, simpson
from scipy.special import kve

sys.dont_write_bytecode = True
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from pipeline_helpers_vgeneral_phi_v2 import (channel_masses_and_mu, coupled_pairs,
                                              insertion, wall_radius)


# --------------------------------------------------------------------------- #
#  Accuracy constants                                                          #
# --------------------------------------------------------------------------- #
# The constants of the solve, each a RULE (r in the model's own units; m_min
# = the lightest false-vacuum mass of the channels the bounce acts on
# (nonzero U_int row), r_W = the wall radius):
POLISH_TOL = float(os.environ.get('VGENERAL_PHI_POLISH_TOL', '1e-10'))
#   solve_bvp's relative collocation tolerance (overridable, e.g. tighter
#   for a more accurate bounce; a thin wall that exhausts the mesh needs a
#   tighter SEED instead, bounce_vgeneral_phi_v2.SEED_PHITOL)
POLISH_TOL_REF = (POLISH_TOL / 10, POLISH_TOL / 3)
#   the tolerances the REFERENCE profile of the bounce sensitivity indicator
#   (bounce_vgeneral_phi_v2.reference_profile) is attempted at, tightest
#   first, both TIGHTER than POLISH_TOL (a re-solve at POLISH_TOL itself
#   starts on the kept mesh, which already meets that tolerance, so it is no
#   reference); no convergence at any -> stored NaN,
#   assemble_lnD_vgeneral_phi_v2 reports the indicator as not evaluated
POLISH_DECAY_LENGTHS_PAST_WALL = 10.0
#   K, the number of decay lengths 1/m_min kept past the wall: the SOLVE
#   radius r_max = r_W + K / m_min (never shorter than the seed's own
#   domain; polish_r_max).  Past the wall the solve does not depend on r_max,
#   up to terms quadratic in the tail (the tail condition holds there); but
#   the converged mesh is stored whole, so r_max is the end of the radial
#   grid of every later stage, where the tail has fallen to about e^{-K} of
#   the field at the wall.  The tail beyond r_max enters lnD_ren linearly
#   only through the tadpole moment, which addback_tadpole completes in
#   closed form.  Every other piece beyond r_max (fish, band, sectors,
#   n = 0, 1 eigenvalues, S_E, the second-order part of the tadpole) is of
#   second order in the tail, or falls like e^{-2K} in a channel the bounce
#   acts on without moving it, so it is of order e^{-2K} (~2e-9 at K = 10)
#   relative to the wall's own contribution and is dropped.
POLISH_MAX_NODES = int(os.environ.get('VGENERAL_PHI_POLISH_MAX_NODES',
                                      '200000'))
#   solve_bvp's mesh ceiling (overridable; reached by a very thin wall whose
#   seed wall lies too far from the true one -- SEED_PHITOL --, by a model
#   written in large units, and by a tolerance so tight that the collocation
#   residual meets round-off; the mesh-cap remedy of
#   bounce_vgeneral_phi_v2.polish_profile says what to do in each case)


# --------------------------------------------------------------------------- #
#  The boundary-value problem                                                  #
# --------------------------------------------------------------------------- #
def _bvp_system(pot_prime, r_max):
    """(fun, fun_jac, bc, bc_jac, S) for scipy.integrate.solve_bvp: the O(4)
    bounce equation d^2phi_i/dr^2 + (3/r) dphi_i/dr = dV/dphi_i in primed
    coordinates, as a first-order system for y = [phi, dphi], dphi = dphi/dr:
        y = [phi_1..phi_N, dphi_1..dphi_N],   dy/dr = f(r, y) + S y / r,
        f = [dphi, gradV(phi)],   S = diag(0_N, -3 I_N).
    S is handed to scipy separately so that r = 0 is a regular point of the
    discretization: scipy enforces dphi(0) = 0 through the projector
    I - S^+ S and uses d^2phi_i/dr^2 = dV/dphi_i(phi(0)) / 4 at r = 0.
    Boundary conditions: dphi(0) = 0 (regularity) and, at r_max, the tail
    condition dphi_i + tail_rate_i phi_i = 0 with
        tail_rate_i = m_i K_2(m_i r_max) / K_1(m_i r_max) :
    about the false vacuum, the origin of the primed frame, the bounce
    equation linearises to d^2phi_i/dr^2 + (3/r) dphi_i/dr = m_i^2 phi_i
    (the primed Hessian is diag(m2) there, so the channels decouple), whose
    decaying solution is K_1(m_i r)/r; every multiple of it has the
    log-derivative -tail_rate_i, so the condition fixes the shape of the
    tail and leaves its amplitude to the solve, and holds at any r_max past
    the wall up to terms quadratic in the tail.  The Bessel ratio is formed
    from the scaled kve, so it never underflows.
    fun_jac is the exact Jacobian from the analytic Hessian of the primed
    potential."""
    N = pot_prime.N
    S = np.zeros((2 * N, 2 * N))
    S[N:, N:] = -3.0 * np.eye(N)
    masses = np.sqrt(pot_prime.m2)
    tail_rate = masses * kve(2, masses * r_max) / kve(1, masses * r_max)

    def fun(r, y):
        """f(r, y) = [dphi, gradV(phi)], vectorised over the mesh."""
        phi, dphi = y[:N], y[N:]
        out = np.empty_like(y)
        out[:N] = dphi
        out[N:] = pot_prime.dV(phi.T).T   # layout: scipy state (2N, m) <-> point-axis-first (m, N)
        return out

    def fun_jac(r, y):
        """df/dy for f = [dphi, gradV(phi)]: the identity for d(dphi)/d(dphi)
        and the Hessian of V for d(gradV)/d(phi)."""
        m = y.shape[1]
        H = pot_prime.H(y[:N].T)   # (m, N, N); layout: scipy state (2N, m) <-> point-axis-first (m, N)
        J = np.zeros((2 * N, 2 * N, m))
        for i in range(N):
            J[i, N + i, :] = 1.0
        J[N:, :N, :] = np.moveaxis(H, 0, -1)   # layout: scipy wants (N, N, m)
        return J

    def bc(ya, yb):
        """Residuals of the two boundary conditions, ya = y(0), yb = y(r_max):
        dphi(0) = 0 and the Robin tail condition dphi + tail_rate phi = 0."""
        return np.concatenate([ya[N:], yb[N:] + tail_rate * yb[:N]])

    def bc_jac(ya, yb):
        dya = np.zeros((2 * N, 2 * N))
        dyb = np.zeros((2 * N, 2 * N))
        dya[:N, N:] = np.eye(N)            # d(dphi(0))/d(ya)
        dyb[N:, :N] = np.diag(tail_rate)   # d(dphi + tail_rate phi)(r_max)/d(yb):
        dyb[N:, N:] = np.eye(N)            #   its phi part and its dphi part
        return dya, dyb

    return fun, fun_jac, bc, bc_jac, S


# --------------------------------------------------------------------------- #
#  Domain, action, and the driver                                              #
# --------------------------------------------------------------------------- #
def fv_mass_min(pot_prime, phi_seed):
    """m_min = sqrt(min m_i^2), the lightest false-vacuum mass among the
    channels the BOUNCE ACTS ON: the channels i whose row of the insertion
    U_int = H'(phi_seed) - diag(m2) has an entry above round-off somewhere on
    the seed (coupled_pairs, the rule the h-ODE engine uses; every channel
    the bounce moves has one; m2 from channel_masses_and_mu, GUARD
    FV-MASS-POSITIVE).  It sets the slowest exponential fall-off past the
    wall, of the bounce and of G - G_FV, hence the solve radius r_max
    (polish_r_max), the end of the radial grid of every later stage.  A
    channel with a zero U_int row (e.g. the idle field of a decoupled
    potential) has G = G_FV and is left out, so a light idle field cannot
    inflate r_max; a spectator y that the bounce leaves at zero but whose
    mass it shifts in the wall (U_yy != 0) counts: its G - G_FV past the
    wall falls like e^{-2 m_y r}.  Not the m_min2 of load_background
    (pipeline_helpers), the lightest mass^2 of ALL channels: the two
    differ when the lightest channel is idle (then m_min^2 > m_min2)."""
    m2, _ = channel_masses_and_mu(pot_prime)
    acted_on = coupled_pairs(insertion(pot_prime, phi_seed)).any(axis=1)   # the channels the bounce acts on
    return float(np.sqrt(np.min(m2[acted_on])))


def polish_r_max(R_seed_max, m_min, r_W):
    """The solve radius r_max = r_W + POLISH_DECAY_LENGTHS_PAST_WALL / m_min:
    the wall plus a fixed number of decay lengths of the slowest channel
    the bounce acts on (the constants block says why), set by the
    model's own scales r_W and m_min alone.  It is never shorter than the
    seed's domain R_seed_max, the grid on which the seed enters as the
    initial guess."""
    return float(max(R_seed_max, r_W + POLISH_DECAY_LENGTHS_PAST_WALL / m_min))


def action_O4(R, phi_b, pot_prime, dphi_b):
    """The O(4) Euclidean action of a profile,
        S_E = 2 pi^2 Int dr r^3 [ |dphi_b/dr|^2 / 2 + V(phi_b(r)) - V(phi_F) ] ,
    by Simpson integration on the given grid (the same number in either frame:
    PrimedPotential.V already subtracts V(phi_F), and the rotation L is
    orthogonal); the action of the tail beyond the grid's end, quadratic in
    the profile there, is left out (order e^{-2K}, see the constants block).
    `dphi_b` is the derivative dphi_b/dr the boundary-value solver carries
    as half of its state vector."""
    R = np.asarray(R, float)
    phi_b = np.asarray(phi_b, float)
    V = np.asarray(pot_prime.V(phi_b), float)
    integ = R ** 3 * (0.5 * np.sum(np.asarray(dphi_b, float) ** 2, axis=1) + V)
    return float(2.0 * np.pi ** 2 * simpson(integ, x=R))


def bvp_polish(R_seed, phi_seed, pot_prime, tol=POLISH_TOL, eval_R=None):
    """Solve the bounce BVP with scipy.integrate.solve_bvp, seeded by the
    CosmoTransitions profile (R_seed, phi_seed) in the PRIMED frame.
    Returns a dict and never raises on a failed solve: the caller reads
    ['success'] and aborts on False (the seed is never a fallback).  It does
    raise on the two preconditions it cannot work around: a seed grid that
    does not start at r = 0 exactly (scipy's singular point is x[0]), and a
    false vacuum whose Hessian is not positive definite
    (channel_masses_and_mu, GUARD FV-MASS-POSITIVE, called by fv_mass_min).

    Arguments
      R_seed, phi_seed     the seed, primed frame
      pot_prime            the PrimedPotential (analytic gradient + Hessian)
      tol                  solve_bvp relative collocation tolerance
      eval_R               the reference solve: solve on the kept domain
                           [0, eval_R[-1]] and report the solution on eval_R
                           instead of the solver's own mesh
                           (bounce_vgeneral_phi_v2.reference_profile passes
                           the kept bounce's grid, which is also its seed
                           grid; both profiles then share the outer
                           radius and are tabulated identically)
    Returned keys
      R, phi_b, S_E              the polished bounce and its O(4) action
      success, status, message   solve_bvp's verdict
      niter, nodes, rms_max      mesh-refinement passes, final mesh size,
                                 worst relative collocation residual
      tol, m_min, r_W            tolerance, lightest FV mass of the
                                 channels the bounce acts on, wall radius
      r_max                      the solve radius, the end of the returned
                                 grid R
      solve_seconds              solve wall-clock time, seconds
    """
    R_seed = np.asarray(R_seed, float)
    phi_seed = np.asarray(phi_seed, float)
    N = pot_prime.N

    # solve_bvp puts the singular point of its term S y / (r - a) at a = x[0],
    # so a seed grid that started at a small POSITIVE radius would silently
    # solve d^2phi/dr^2 + 3/(r - a) dphi/dr = dV, a different equation.
    # CosmoTransitions sets its first radius to exactly 0, so this is a guard,
    # not a branch.
    if R_seed[0] != 0.0:
        raise RuntimeError(
            f'[ABORT BOUNCE-POLISH] the seed grid starts at '
            f'r = {R_seed[0]!r}, not exactly 0.  scipy.integrate.solve_bvp places '
            f'its singular point at x[0], so the polish would silently solve '
            f'the O(4) equation about r = {R_seed[0]!r} instead of the origin.')

    # ---- the solve radius r_max, where the tail condition is
    #      imposed and the grid R ends: the seed's wall radius r_W plus a
    #      fixed number of decay lengths 1/m_min of the lightest channel
    #      the bounce acts on (polish_r_max); the reference solve (eval_R
    #      given) reuses the kept profile's radius eval_R[-1] --------------
    m_min = fv_mass_min(pot_prime, phi_seed)
    r_W = wall_radius(R_seed, phi_seed)
    r_max = (polish_r_max(R_seed[-1], m_min, r_W) if eval_R is None
             else float(eval_R[-1]))

    # ---- initial guess: the seed profile, padded at the false vacuum over
    #      the extended stretch (the true profile there is exponentially
    #      small, so a flat pad is a good guess and solve_bvp refines it) ----
    r_guess = R_seed
    phi_guess = phi_seed
    if r_max > R_seed[-1]:
        # one guess node per 1/(4 m_min) of extension, at least two: ample for
        # a guess the solver refines, and proportional to the extension, so a
        # sliver of extension gets only a few nodes; the filter below drops a
        # node that rounds onto R_seed[-1] (solve_bvp raises from inside itself
        # on a guess mesh that is not strictly increasing)
        n_pad = max(2, int(np.ceil(4.0 * m_min * (r_max - R_seed[-1]))))
        pad = np.linspace(R_seed[-1], r_max, n_pad + 1)[1:]
        pad = pad[pad > R_seed[-1]]
        r_guess = np.concatenate([R_seed, pad])
        phi_guess = np.vstack([phi_seed, np.zeros((pad.size, N))])
    # the guess state y = [phi, dphi/dr], dphi/dr by finite differences of the
    # guess profile; scipy's layout (2N, m): rows = components, columns = nodes
    guess_state = np.vstack([phi_guess.T, np.gradient(phi_guess, r_guess, axis=0).T])

    # ---- THE SOLVE: solve_bvp refines the guess mesh and drives the
    #      collocation residual of the full coupled system below tol by
    #      Newton steps with the analytic Jacobian (_bvp_system); sol.sol
    #      is the converged profile, tabulated below -----------------------
    fun, fun_jac, bc, bc_jac, S = _bvp_system(pot_prime, r_max)
    t0 = time.perf_counter()
    sol = solve_bvp(fun, bc, r_guess, guess_state, S=S, fun_jac=fun_jac, bc_jac=bc_jac,
                    tol=tol, max_nodes=POLISH_MAX_NODES)
    solve_seconds = time.perf_counter() - t0

    # ---- the profile and its action: the production solve keeps its whole
    #      converged mesh on [0, r_max] -- the radial grid R of every later
    #      stage -- the reference solve reports on eval_R (see the argument
    #      list); S_E is the O(4) action of the tabulated profile -----------
    R_out = np.asarray(sol.x if eval_R is None else eval_R, float)
    y_out = sol.sol(R_out)
    # y = [phi_b, dphi_b/dr] from scipy's layout (2N, m) back to (m, N)
    phi_b, dphi_b = y_out[:N].T, y_out[N:].T
    return dict(
        R=R_out,
        phi_b=phi_b,
        S_E=action_O4(R_out, phi_b, pot_prime, dphi_b),
        success=bool(sol.success),
        status=int(sol.status),
        message=str(sol.message),
        niter=int(sol.niter),
        nodes=int(sol.x.size),
        rms_max=float(np.max(sol.rms_residuals)),
        tol=float(tol),
        m_min=m_min, r_W=r_W,
        r_max=r_max,
        solve_seconds=solve_seconds,
    )
