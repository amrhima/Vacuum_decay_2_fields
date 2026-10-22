#!/usr/bin/env python3
"""addback_fish_vgeneral_phi_v2.py -- level 2, stage 6: the fish add-back.
A2_fin is the MSbar finite part (mu = sum_i m_i) of the fish trace
Tr[(G_FV U_int)^2]; in
    ln det(1 + G_FV U_int) = sum_k (-1)^(k+1) Tr[(G_FV U_int)^k] / k
it is the k = 2 (Born-order-2, two-insertion bubble) term, so the assembler
adds it once, as (T7) = -A2_fin/2.

READS   --bounce-npz only, through load_background: R, U_int, m2 and mu.
WRITES  addback_fish_<tag>.npz, read by stage 7 (assemble_lnD):
    A2_fin              ()   enters lnD_ren as -A2_fin/2
    mu                  ()   renormalization scale sum_i m_i (record)
    q_momentum_max_ladder_steps,
    n_q_momentum_ladder_steps,
    A2_fin_ladder_steps (k,) the q_momentum_max, the momentum node count and
                             A2_fin at each rung of the two ladders; the LAST
                             rung is the converged grid, on which the
                             assembler evaluates the reference profile's fish
                             for eps_bounce
    bounce_sha, potential_id, code_version    metadata
        Utilde_matrix and Pi_fin_fv_q are defined here; the add-on's G1_ren
        (greensfct_coincident_renorm_v2) imports both, the assembler imports
        Utilde_matrix and A2_from_Utilde for eps_bounce.
HOW     The k = 1 (tadpole, stage 5) and k = 2 (fish) terms are the two
        UV-divergent ones.  At fixed s2 the wave-summed fish trace is finite,
        but its s2 integral diverges logarithmically, so the engine removes
        the fish trace Gbar_2 wave by wave (delta_geq3 starts at Born order 3)
        and this stage computes its dim-reg MSbar finite part in momentum
        space:

    A2_fin       = (1/(8 pi^2)) Int_0^{q_momentum_max} dq q^3
                   sum_ij Utilde_ij(q)^2 Pi_fin_fv_ij(q^2)       (A2_from_Utilde)
    Utilde_ij(q) = (2 pi)^2 Int dr r^3 [U_int]_ij(r) J1(qr)/(qr)  (Utilde_matrix)
    Pi_fin_fv_ij = -(1/(16 pi^2)) Int_0^1 dx ln(Delta_ij/mu^2)    (Pi_fin_fv_q)
    Delta_ij     = x(1-x) q^2 + x m_j^2 + (1-x) m_i^2

        Utilde is the 4D Fourier transform of the O(4)-symmetric insertion,
        Pi_fin_fv the MSbar finite part of the bubble of two free propagators
        of masses m_i, m_j, x the Feynman parameter.  So lnD_ren receives
            -A2_fin/2 = (1/(256 pi^4)) Int dq q^3 sum_ij Utilde_ij(q)^2
                        Int_0^1 dx ln(Delta_ij/mu^2) .
        compute_A2_fin converges the q integral in two ladders, the momentum
        cutoff and then the momentum spacing (constants below); a ladder that
        reaches its ceiling aborts this stage (guard FISH-CONVERGED).
"""
import argparse
import os
import sys
import numpy as np
from scipy.integrate import quad_vec, trapezoid
from scipy.interpolate import CubicSpline
from scipy.special import j1

sys.dont_write_bytecode = True
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from pipeline_helpers_vgeneral_phi_v2 import (add_standard_cli, atomic_savez,
                                 load_background,
                                 provenance_stamp, stage_paths)

# ---- the two ladders of compute_A2_fin: every knob lives HERE ------------ #
# (the FISH-CONVERGED abort names the one to raise)
A2_TOL = 1e-6              # the ONE stop rule (function settled): the
                           # change of A2, relative to max(|A2|, 1), is
                           # below it -- a relative change for |A2| >= 1,
                           # an absolute change for |A2| < 1
# q is a momentum, so its ladder is set in units of the model's own mass
# scale mu = sum_i m_i; these numbers only fix where the ladder starts, how
# it climbs and where it gives up -- convergence is tested, never assumed.
Q_FLOOR_IN_MU = 7.0        # first cutoff the ladder tries
Q_STEP_IN_MU = 2.0         # cutoff growth per ladder step
Q_CEILING_IN_MU = 46.0     # no rung above it; not converged below it -> abort
                           # (FISH-CONVERGED)
N_Q_MOMENTUM_START = 2000  # nodes on [0, Q_FLOOR_IN_MU mu]: sets the spacing dq
                           # the cutoff ladder holds fixed and the dq ladder halves
DQ_HALVINGS_MAX = 4        # abort after this many halvings (FISH-CONVERGED);
                           # each halving cuts the second-order q-trapezoid
                           # error by 4 and doubles the (n_q x n_r) kernel array
FEYNMAN_X_RTOL = 1e-10     # accuracy of the adaptive Feynman-x integral of
                           # the bubble (Pi_fin_fv_q), far below A2_TOL


# --------------------------------------------------------------------------- #
#  The fish add-back (mu = sum_i m_i, the FV Hessian masses)                  #
# --------------------------------------------------------------------------- #
def Utilde_matrix(q_grid, R, U_int, nodes_per_period=16):
    """The momentum-space insertion, the radial Fourier (Hankel) transform
    of U_int (the 4D Fourier transform of an O(4)-symmetric function reduces
    to it):
        Utilde_ij(q) = (2 pi)^2 Int dr r^3 J1(qr)/(qr) [U_int]_ij(r).
    The kernel oscillates in r with the period 2 pi / q, fastest at the
    largest q, so the radial integral is taken on a UNIFORM grid that puts
    nodes_per_period nodes into that period out to R[-1] (the bounce mesh is
    adaptive -- dense at the wall, coarse in the tail -- and where its tail
    spacing is not small against 2 pi / q it aliases the transform); U_int is
    resampled onto that grid by a cubic spline, the interpolation the h-ODE
    engine uses.  The q -> 0 kernel J1(x)/x -> 1/2 is taken exactly (so the q
    grid may start at 0, where the q^3 measure of A2 vanishes anyway); the
    r = 0 node needs no special case, its r^3 measure is zero.  Returns
    (nq, N, N) -- the ONE owner of this transform (the add-on's G1_ren
    imports it)."""
    R = np.asarray(R, float)
    n_uniform = int(np.ceil(nodes_per_period * float(np.max(q_grid)) * R[-1]
                            / (2.0 * np.pi))) + 1
    R_uniform = np.linspace(R[0], R[-1], n_uniform)
    U_uniform = CubicSpline(R, U_int)(R_uniform)          # (n_uniform, N, N)
    with np.errstate(invalid='ignore'):
        QR = np.outer(q_grid, R_uniform)
        J1_over_qr = np.where(QR > 1e-10, j1(QR) / QR, 0.5)
    # the trapezoid rule on the uniform grid as weights w_k (dr, halved at
    # the two ends): Int dr r^3 J1(qr)/(qr) U(r) = sum_k J1_over_qr[q, k]
    # w_k r_k^3 U(r_k) -- one contraction over the radial index k for every
    # q and all N x N components at once
    trapezoid_weight = np.full(n_uniform, R_uniform[1] - R_uniform[0])
    trapezoid_weight[[0, -1]] *= 0.5
    weighted_U = (trapezoid_weight * R_uniform**3)[:, None, None] * U_uniform
    return (2.0 * np.pi)**2 * np.tensordot(J1_over_qr, weighted_U, axes=1)


def Pi_fin_fv_q(q_grid, m2_i, m2_j, mu):
    """The MSbar-finite two-mass bubble at every q of the grid,
        Pi_fin_fv_ij(q^2) = -1/(16 pi^2) Int_0^1 dx ln(Delta_ij(x, q^2)/mu^2),
        Delta_ij = x(1-x) q^2 + x m_j^2 + (1-x) m_i^2 .
    The Feynman-parameter integral is scipy quad_vec (adaptive Gauss-Kronrod
    in x for all q of the grid at once, error below FEYNMAN_X_RTOL times the
    largest integral on the grid): near x = 0 and x = 1 ln Delta varies on
    the scale m^2/q^2, and the adaptive rule refines exactly there, so the
    bubble is converged for any q and any ratio of the channel masses.
    Delta > 0 for every x because both channel masses are positive (guard
    FV-MASS-POSITIVE where m2 is formed, pipeline_helpers.channel_masses_and_mu),
    so the log is always finite.  The ONE owner of the bubble (the add-on's
    G1_ren imports it with the shifted masses kappa_i^2 = m_i^2 + s2)."""
    q2 = np.asarray(q_grid, float)**2
    feynman_x_integral, _ = quad_vec(
        lambda x: np.log((x * (1.0 - x) * q2 + x * m2_j + (1.0 - x) * m2_i) / mu**2),
        0.0, 1.0, epsrel=FEYNMAN_X_RTOL, norm='max')
    return -feynman_x_integral / (16.0 * np.pi**2)


def A2_from_Utilde(q_grid, Utilde, m2, mu):
    """A2_fin on ONE momentum grid [0, q_max] from Utilde (Utilde_matrix):
        A2_fin = Int d^4q/(2 pi)^4 sum_ij Utilde_ij(q) Utilde_ji(-q) Pi_fin_fv_ij(q^2)
               = 1/(8 pi^2) Int dq q^3 sum_ij Utilde_ij(q)^2 Pi_fin_fv_ij(q^2) :
    the angles of d^4q give 2 pi^2 q^3 dq, and Utilde_ji(-q) = Utilde_ij(q)
    because U_int is real, symmetric and O(4)-symmetric.  The double sum runs
    over all N x N pairs (i, j); the q integral is the trapezoid rule."""
    N = Utilde.shape[-1]
    integrand_q = np.zeros(len(q_grid))
    for ii in range(N):
        for jj in range(N):
            Pi_fin_fv = Pi_fin_fv_q(q_grid, m2[ii], m2[jj], mu)
            integrand_q += q_grid**3 * Utilde[:, ii, jj]**2 * Pi_fin_fv
    return float(trapezoid(integrand_q, q_grid)) / (8.0 * np.pi**2)


def compute_A2_fin(R, U_int, m2, mu, ladder_record=None):
    """A2_fin(mu), its q integral converged in two directions, one ladder
    each, in this order (the starts, steps, ceilings and A2_TOL are the
    module constants above; the Feynman-x integral inside Pi_fin_fv_q is
    adaptive):
      1. the momentum CUTOFF q_momentum_max: start Q_FLOOR_IN_MU mu,
         + Q_STEP_IN_MU mu per step, no rung above Q_CEILING_IN_MU mu, at the
         fixed spacing dq = Q_FLOOR_IN_MU mu / (N_Q_MOMENTUM_START - 1);
      2. the momentum SPACING dq at the settled cutoff: halved per step, at
         most DQ_HALVINGS_MAX times (the cutoff ladder never tests dq);
    each ladder until settled() (A2_TOL), one transform Utilde_matrix per
    grid.  GUARD FISH-CONVERGED (aborts): a ladder reaches its ceiling
    before it settles.  `ladder_record` (optional list) records each
    (q_momentum_max, n_q_momentum, A2) step; its last step is the converged
    grid.  Returns A2_fin."""
    # The three momenta of the ladder, in the model's own mass units.
    q_floor = Q_FLOOR_IN_MU * mu
    q_step = Q_STEP_IN_MU * mu
    q_ceiling = Q_CEILING_IN_MU * mu
    dq = q_floor / (N_Q_MOMENTUM_START - 1)

    def momentum_grid(q_momentum_max, dq):
        """The grid [0, q_momentum_max] at the spacing dq (round(): to within half a node,
        the only sense in which an integer node count can hold dq fixed)."""
        return np.linspace(0.0, q_momentum_max, max(2, int(round(q_momentum_max / dq)) + 1))

    def A2_ladder_step(q_grid, Utilde):
        """One rung: A^(2)_fin on this grid, recorded and printed."""
        A2 = A2_from_Utilde(q_grid, Utilde, m2, mu)
        if ladder_record is not None:
            ladder_record.append((float(q_grid[-1]), int(len(q_grid)), A2))
        print(f"        [A2_fin ladder] q_momentum_max={q_grid[-1]:6.1f} "
              f"(={q_grid[-1] / mu:5.1f} mu)  n_q_momentum={len(q_grid):5d}  "
              f"A2={A2:+.10e}")
        return A2

    def settled(A2_new, A2_prev, direction):
        """The one stop rule of every ladder: the change of A2 relative to
        max(|A2|, 1) is below A2_TOL -- a relative change for |A2| >= 1, an
        absolute change for |A2| < 1 (A2 enters lnD as -A2/2, so near A2 = 0
        its absolute accuracy is what matters: A2_TOL in A2, A2_TOL/2 in
        lnD)."""
        change = abs(A2_new - A2_prev) / max(abs(A2_new), 1.0)
        print(f"          -> {direction} change={change:.2e} "
              f"(relative; absolute when |A2| < 1; tol {A2_TOL:g})")
        return change < A2_TOL

    # ---- LADDER 1: the momentum cutoff, at the starting dq --------------------
    q_momentum_max = q_floor
    q_grid = momentum_grid(q_momentum_max, dq)
    Utilde = Utilde_matrix(q_grid, R, U_int)
    A2_prev = A2_ladder_step(q_grid, Utilde)
    q_ok = False
    while q_momentum_max + q_step <= q_ceiling:
        q_momentum_max += q_step
        q_grid = momentum_grid(q_momentum_max, dq)
        Utilde = Utilde_matrix(q_grid, R, U_int)
        A2_new = A2_ladder_step(q_grid, Utilde)
        q_ok = settled(A2_new, A2_prev, 'q_momentum_max')
        A2_prev = A2_new
        if q_ok:
            break
    if not q_ok:
        raise RuntimeError(
            f"[ABORT FISH-CONVERGED] the q_momentum_max ladder did NOT settle "
            f"(A2_TOL={A2_TOL:g}) by the ceiling q_momentum_max={q_ceiling:g} "
            f"(= {Q_CEILING_IN_MU:g} mu, module constant Q_CEILING_IN_MU; last "
            f"value {A2_prev:+.6e}) -- raise the constant and rerun this stage.")

    # ---- LADDER 2: the momentum spacing, at the settled cutoff ---------------
    for _ in range(DQ_HALVINGS_MAX):
        dq *= 0.5
        q_grid = momentum_grid(q_momentum_max, dq)
        Utilde = Utilde_matrix(q_grid, R, U_int)
        A2_new = A2_ladder_step(q_grid, Utilde)
        dq_ok = settled(A2_new, A2_prev, 'dq')
        A2_prev = A2_new
        if dq_ok:
            return A2_prev
    raise RuntimeError(
        f"[ABORT FISH-CONVERGED] the momentum-spacing ladder did NOT settle "
        f"(A2_TOL={A2_TOL:g}) within {DQ_HALVINGS_MAX} halvings of dq (module "
        f"constant DQ_HALVINGS_MAX; last value {A2_prev:+.6e}) -- raise the "
        f"constant and rerun this stage.")


def main():
    ap = argparse.ArgumentParser(
        description='vgeneral_phi_v2 stage 6: fish add-back A2_fin, converged '
                    'in the momentum cutoff and the momentum spacing (two '
                    'ladders; the Feynman-x integral is adaptive)')
    add_standard_cli(ap)
    args = ap.parse_args()

    background = load_background(args.bounce_npz)
    U_int = background['U_int']   # built once, by ONE batched H call
    ladder_record = []
    A2_fin = compute_A2_fin(background['R'], U_int,
                            background['m2'], background['mu'],
                            ladder_record=ladder_record)

    out = stage_paths(args.data_dir, args.tag)['addback_fish']
    # ladder_record entries are (q_momentum_max, n_q_momentum, A2): one array
    # per column, each key naming what it actually holds.
    atomic_savez(out,
             A2_fin=A2_fin, mu=background['mu'],
             q_momentum_max_ladder_steps=np.array(
                 [t[0] for t in ladder_record]),
             n_q_momentum_ladder_steps=np.array([t[1] for t in ladder_record]),
             A2_fin_ladder_steps=np.array([t[2] for t in ladder_record]),
             **provenance_stamp(args.bounce_npz, background['m2'],
                                background['params'], background['fields']))
    print(f"[ADDBACK] SUMMARY: A2_fin = {A2_fin:+.6f}   "
          f"(mu = sum_i m_i = {background['mu']:.6f}, "
          f"q_momentum_max = {ladder_record[-1][0]:g} "
          f"= {ladder_record[-1][0] / background['mu']:.3g} mu, "
          f"{len(ladder_record)} ladder steps, both ladders converged)")
    print(f"[OK] wrote {out}")


if __name__ == '__main__':
    main()
