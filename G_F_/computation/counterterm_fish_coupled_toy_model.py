#!/usr/bin/env python3
"""counterterm_fish_coupled_toy_model.py -- stage 7: the fish (two-insertion bubble)
counterterm finite part A^(2)_fin(mu).

WHAT IT COMPUTES
----------------
The two-mass-bubble ("fish") diagram carries the UV divergence of the
one-loop determinant; in dim-reg MS-bar its divergent part cancels
against the determinant, leaving the mu-dependent FINITE part
(renormalization scale mu = sum_i m_i, the FV Hessian masses):

    A^(2)_fin = (1/(8 pi^2)) Int_0^{Q} dq q^3
                sum_{ij} |V~_ij(q)|^2 B0_fin(q; m_i, m_j; mu) ,

    V~_ij(q) = (2 pi)^2 Int dr r^3 V_ij(r) j1(qr)/(qr)   (radial Hankel
               transform of the potential insertion V = H(phi_b) - H_FV),
    B0_fin   = -(1/(16 pi^2)) Int_0^1 dx ln(Delta/mu^2),
    Delta    = x(1-x) q^2 + x m_j^2 + (1-x) m_i^2 .

ADAPTIVE Q-CUTOFF (the production path -- a fixed cutoff is
bounce-specific, so production always lets the ladder choose it): start
at q_max = 30, grow by 10 until the relative change of A^(2)_fin falls
below rtol = 1e-6, ceiling 200; the q-grid spacing dq is held fixed by
scaling n_q with q_max.  --q-max forces a fixed cutoff (diagnostic
only).  A2_fin enters the assembly as -A2_fin/2, and an unconverged
ladder (converged=False) makes assemble_lnD_coupled_toy_model abort there
(CT-QMAX guard; aborts, does NOT enter lnD).

INPUTS   --bounce-npz (the potential is rebuilt from the coupling vector
         stored in it); no other stage
         output is needed.
OUTPUT   ct_fish_<tag>.npz with keys
    A2_fin              ()   the finite fish counterterm (enters lnD_ren
                             as -A2_fin/2)
    mu                  ()   renormalization scale sum_i m_i
    q_max_used          ()   the Q-cutoff of the returned value
    adaptive_qmax_trace (k,) the q_max ladder actually evaluated
    adaptive_A2_trace   (k,) A^(2)_fin at each ladder step (the printed
                             adaptive ladder, stored)
    converged           ()   bool: ladder ended below rtol (False = the
                             ceiling was hit; assemble_lnD_coupled_toy_model
                             aborts on it, CT-QMAX guard)
    bounce_sha, potential_id, code_version    metadata
"""
import argparse
import os
import sys
import numpy as np
from scipy.integrate import trapezoid
from scipy.special import jv

sys.dont_write_bytecode = True
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from pipeline_helpers_coupled_toy_model import (add_standard_cli, atomic_savez,        # noqa: E402
                                 load_bounce, potential_insertion_V,
                                 provenance_stamp, stage_paths)

A2_RTOL = 1e-6                    # adaptive-ladder convergence tolerance


# --------------------------------------------------------------------------- #
#  Counterterm (mu = sum_i m_i, the FV Hessian masses)                             #
# --------------------------------------------------------------------------- #
def _compute_A2_fin_at(R, V_full, m_sq, mu, q_max, n_q, n_x):
    """Inner kernel: A^(2)_fin evaluated with an explicit upper Q-cutoff.
    The channel double sum runs over ALL 2x2 entries of the insertion."""
    N = V_full.shape[-1]
    mask_R = R > 1e-10
    x_grid = np.linspace(0.0, 1.0, n_x)
    q_grid = np.linspace(0.01, q_max, n_q)
    V_tilde = np.zeros((n_q, N, N))
    # Precompute the Hankel kernel jv(1,qR)/qR once on the full (q,R) grid;
    # where qR <= 1e-10 use the exact limit j1(x)/x -> 1/2, and zero the
    # R <= 1e-10 columns (the r^3 measure kills them anyway).
    with np.errstate(invalid='ignore'):
        QR = np.outer(q_grid, R)
        Kmat = np.where(QR > 1e-10, jv(1, QR) / QR, 0.5)
    Kmat = np.where(mask_R[None, :], Kmat, 0.0)
    R3 = (R**3)[None, :]
    for ii in range(N):
        for jj in range(N):
            integrand = R3 * V_full[:, ii, jj][None, :] * Kmat
            V_tilde[:, ii, jj] = (2.0 * np.pi)**2 * trapezoid(
                integrand, R, axis=1)
    full_int = np.zeros(n_q)
    for ii in range(N):
        for jj in range(N):
            mi2, mj2 = m_sq[ii], m_sq[jj]
            for iq, q in enumerate(q_grid):
                vt = V_tilde[iq, ii, jj]
                Delta = (x_grid * (1.0 - x_grid) * q**2
                         + x_grid * mj2 + (1.0 - x_grid) * mi2)
                Delta = np.maximum(Delta, 1e-30)
                B0_fin = -(1.0 / (16.0 * np.pi**2)) * float(
                    trapezoid(np.log(Delta / mu**2), x_grid))
                full_int[iq] += q**3 * vt**2 * B0_fin
    return float(trapezoid(full_int, q_grid)) / (8.0 * np.pi**2)


def compute_A2_fin(R, V_full, m_sq, mu, q_max=None, n_q=2000, n_x=200,
                   rtol=1e-6, q_max_floor=30.0, q_max_ceiling=200.0,
                   q_step=10.0, verbose=True, trace=None):
    """Two-mass-bubble counterterm A^(2)_fin(mu) with the ADAPTIVE Q-cutoff.

    If q_max is given, use it directly (fixed-cutoff behaviour --
    diagnostic only; NOT the production path).
    If q_max is None (default), choose the upper Q-cutoff adaptively:
    start at q_max_floor (=30) and grow by q_step (=10) until the
    relative change in A^(2)_fin falls below rtol, capped at
    q_max_ceiling (=200).  The Q-grid spacing dq is held fixed by
    scaling n_q proportionally with q_max so the low-Q region (where
    V_tilde is non-trivial) does not lose resolution.

    `trace` (optional list) RECORDS each evaluated (q_max, A2) ladder step
    -- pure metadata for the output npz; it never changes any arithmetic.
    """
    if q_max is not None:
        A = _compute_A2_fin_at(R, V_full, m_sq, mu, q_max, n_q, n_x)
        if trace is not None:
            trace.append((float(q_max), A))
        return A, True          # fixed cutoff: converged by definition

    dq_floor = (q_max_floor - 0.01) / max(n_q - 1, 1)

    def _scaled_n_q(q):
        return max(2, int((q - 0.01) / dq_floor) + 1)

    q = q_max_floor
    n_use = _scaled_n_q(q)
    A_prev = _compute_A2_fin_at(R, V_full, m_sq, mu, q, n_use, n_x)
    if trace is not None:
        trace.append((float(q), A_prev))
    if verbose:
        print(f"        [A2_fin adaptive] q_max={q:6.1f}  n_q={n_use:5d}  "
              f"A2={A_prev:+.10e}")
    while q < q_max_ceiling:
        q_new = q + q_step
        n_use = _scaled_n_q(q_new)
        A_new = _compute_A2_fin_at(R, V_full, m_sq, mu, q_new, n_use, n_x)
        if trace is not None:
            trace.append((float(q_new), A_new))
        rel_change = abs(A_new - A_prev) / max(abs(A_new), 1e-30)
        if verbose:
            print(f"        [A2_fin adaptive] q_max={q_new:6.1f}  "
                  f"n_q={n_use:5d}  A2={A_new:+.10e}  "
                  f"rel_change={rel_change:.2e}")
        if rel_change < rtol:
            return A_new, True
        A_prev = A_new
        q = q_new
    # Ceiling exhausted without convergence: warn UNCONDITIONALLY here, store
    # converged=False in the npz, and let the assembly abort on it (CT-QMAX
    # guard in assemble_lnD_coupled_toy_model) -- it must never flow into lnD_ren
    # silently.
    print(f"[WARN CT-QMAX] adaptive q_max did NOT reach rtol={rtol} by the "
          f"ceiling q_max={q_max_ceiling}; returning last (possibly "
          f"unconverged) value {A_prev:+.6e}.", file=sys.stderr)
    return A_prev, False


def main():
    ap = argparse.ArgumentParser(
        description='coupled_toy_model stage 7: fish counterterm A2_fin (adaptive q_max)')
    add_standard_cli(ap)
    ap.add_argument('--q-max', type=float, default=None,
                    help='FORCE a fixed Q-cutoff (diagnostic only).  '
                         'Default None = the production adaptive ladder '
                         '(floor 30, step 10, ceiling 200, rtol 1e-6).')
    args = ap.parse_args()

    bg = load_bounce(args.bounce_npz)
    V_full, _ = potential_insertion_V(bg['R'], bg['Phi'],
                                      bg['pot'], bg['Hfv'])
    trace = []
    A2_fin, converged = compute_A2_fin(bg['R'], V_full, bg['masses'],
                                       bg['mu'], q_max=args.q_max,
                                       rtol=A2_RTOL, trace=trace)

    out = stage_paths(args.data_dir, args.tag)['ct_fish']
    atomic_savez(out,
             A2_fin=A2_fin, mu=bg['mu'],
             q_max_used=float(trace[-1][0]),
             adaptive_qmax_trace=np.array([t[0] for t in trace]),
             adaptive_A2_trace=np.array([t[1] for t in trace]),
             converged=converged,
             **provenance_stamp(args.bounce_npz, bg['m2']))
    print(f"[CT] SUMMARY: A2_fin = {A2_fin:+.6f}   "
          f"(mu = sum_i m_i = {bg['mu']:.6f}, q_max = {trace[-1][0]:g}, "
          f"{len(trace)} ladder steps, converged = {converged})")
    print(f"[OK] wrote {out}")


if __name__ == '__main__':
    main()
