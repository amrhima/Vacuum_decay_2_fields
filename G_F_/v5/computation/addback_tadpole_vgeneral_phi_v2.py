#!/usr/bin/env python3
"""addback_tadpole_vgeneral_phi_v2.py -- level 2, stage 5: the tadpole
add-back A1_fin, the MSbar finite part (mu = sum_i m_i) of the Born-order-1
(tadpole) term of lnD_ren, in closed form.

READS   --bounce-npz only, through load_background: the radial grid R, the
        bounce phi_b, the insertion U_int = H'(phi_b) - diag(m2), the channel
        masses m2, the scale mu and the cubic couplings of the false vacuum,
        g_ijk = d^3V'/dphi'_i dphi'_j dphi'_k at phi' = 0
        (pot_prime.cubic_couplings, for the bounce tail beyond r_max).
WRITES  addback_tadpole_<tag>.npz, read by stage 7 (assemble_lnD):
    tadpole_moment  (N,)  the tadpole moments M_i (below); the assembler forms
                          the terms (T2), (T4) of lnD_ren from them
    A1_fin          ()    sum_i M_i tau_fin_fv(m_i^2) = (T2) + (T4); the
                          assembler reads it only for the bounce sensitivity
                          indicator eps_bounce
    mu, m2                the scale and the channel masses^2 of A1_fin (record)
    bounce_sha, potential_id, code_version    metadata
        tau_fin_fv is defined here and imported by the assembler, by the add-on
        (greensfct_fv_coincident_renorm_v2) and by the thin-wall test bed
        (thinwall_toolbox_v2); the assembler also imports tadpole_moment and
        compute_A1_fin for eps_bounce.
HOW     A1_fin is the MSbar finite part of the k = 1 (tadpole) term of
            ln det(1 + G_FV U_int) = sum_k (-1)^(k+1) Tr[(G_FV U_int)^k] / k
        and enters lnD_ren with sign +.  The coincident free propagator
        [G_FV(x, x)]_ii = Int d^4l/(2 pi)^4 1/(l^2 + m_i^2) is the same at every
        x and diagonal in the channels, so the trace factorizes:

    A1_fin = sum_i M_i tau_fin_fv(m_i^2)
           = (1/8) sum_i m_i^2 (ln(m_i^2/mu^2) - 1) Int_0^inf dr r^3 [U_int]_ii,
    M_i    = 2 pi^2 Int_0^inf dr r^3 [U_int]_ii(r)              (tadpole_moment),
    tau_fin_fv(W^2) = W^2/(16 pi^2) (ln(W^2/mu^2) - 1), the MSbar finite part
                      of the loop Int d^4l/(2 pi)^4 1/(l^2 + W^2).

        The same number is minus the s2 integral (in dim reg), over all waves
        n, of the traced Born-order-1 add-back of the renormalized Green's
        function (the add-on's G1_ren_addback); the assembler adds A1_fin to
        lnD_ren in that form, split at s2_max into (T2) and (T4).
"""
import argparse
import os
import sys
import numpy as np
from scipy.integrate import trapezoid
from scipy.special import kve

sys.dont_write_bytecode = True
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from pipeline_helpers_vgeneral_phi_v2 import (add_standard_cli, atomic_savez,
                                 load_background,
                                 provenance_stamp, stage_paths)


# --------------------------------------------------------------------------- #
#  The tadpole add-back (mu = sum_i m_i, the FV Hessian masses)               #
# --------------------------------------------------------------------------- #
def tadpole_moment(R, U_int, phi_b, pot_prime):
    """M_i = 2 pi^2 Int_0^inf dr r^3 [U_int]_ii(r), shape (N,): the 4D volume
    integral of the diagonal of the insertion, by the trapezoid rule on the
    bounce grid up to r_max = R[-1] plus the tail beyond it in closed form.
    Beyond r_max the bounce is its decaying tail
    phi_k(r) = phi_k(r_max) (r_max/r) K_1(m_k r)/K_1(m_k r_max) (the outer
    condition of the bounce solve) and the insertion is linear in it,
    [U_int]_ii -> sum_k g_iik phi_k with g_ijk = d^3V'/dphi'_i dphi'_j dphi'_k
    at the false vacuum (PrimedPotential.cubic_couplings), so with
    Int_R^inf dr r^2 K_1(m r) = R^2 K_2(m R)/m the tail is
        2 pi^2 sum_k g_iik phi_k(r_max) r_max^3 K_2(m_k r_max) / (m_k K_1(m_k r_max))
    (the Bessel ratio from the scaled kve, so it never underflows).  The
    terms of second order in the tail (the part of U_int quadratic in phi,
    the nonlinear correction to the tail) are left out: relative to the wall
    they are of order exp(-2K), with K = m_min (r_max - r_W) the number of
    decay lengths between the wall r_W and r_max of the slowest channel the
    bounce acts on (mass m_min); bounce_polish makes K at least
    POLISH_DECAY_LENGTHS_PAST_WALL."""
    N = U_int.shape[1]
    m = np.sqrt(pot_prime.m2)
    r_max = R[-1]
    on_grid = np.array([float(trapezoid(R**3 * U_int[:, i, i], R)) for i in range(N)])
    # Int_{r_max}^inf dr r^3 phi_k(r) of the bounce tail, per channel k
    phi_b_tail_moment = (phi_b[-1] * r_max**3 * kve(2, m * r_max)
                         / (m * kve(1, m * r_max)))
    beyond_grid = np.einsum('iik,k->i', pot_prime.cubic_couplings, phi_b_tail_moment)
    return 2.0 * np.pi**2 * (on_grid + beyond_grid)


def tau_fin_fv(W2, mu):
    """tau_fin_fv(W^2) = W^2/(16 pi^2) (ln(W^2/mu^2) - 1): the MSbar finite part
    of the free one-propagator loop Int d^4l/(2pi)^4 1/(l^2+W^2)
    at the mass^2 W^2 (numpy, elementwise over an array of W^2).  The ONE
    definition: compute_A1_fin evaluates it at the channel masses m_i^2, the
    assembler's add-back pieces (T2), (T4) at m_i^2 and m_i^2 + s2_max, the
    add-on's renormalised FV matrix at the shifted masses kappa_i^2 = m_i^2 + s^2."""
    return W2 / (16.0 * np.pi**2) * (np.log(W2 / mu**2) - 1.0)


def compute_A1_fin(M_i, m2, mu):
    """A1_fin(mu) = sum_i M_i tau_fin_fv(m_i^2) over all N channels (the log is
    finite: m_i^2 > 0 is enforced where m2 is formed, guard FV-MASS-POSITIVE
    of pipeline_helpers.channel_masses_and_mu)."""
    return float(M_i @ tau_fin_fv(m2, mu))


def main():
    ap = argparse.ArgumentParser(
        description='vgeneral_phi_v2 stage 5: A1_fin = sum_i M_i tau_fin_fv(m_i^2), '
                    'the closed form of the traced s2-integrated Born-order-1 '
                    'add-back of the renormalized Green\'s function (the '
                    'finite mu-dependent tadpole part)')
    add_standard_cli(ap)
    args = ap.parse_args()

    background = load_background(args.bounce_npz)
    # the insertion U_int is built once by load_background; its diagonal's
    # r^3 moment is integrated once here (grid plus tail) and stored
    M_i = tadpole_moment(background['R'], background['U_int'],
                         background['phi_b'], background['pot_prime'])
    A1_fin = compute_A1_fin(M_i, background['m2'], background['mu'])

    out = stage_paths(args.data_dir, args.tag)['addback_tadpole']
    atomic_savez(out,
             A1_fin=A1_fin, tadpole_moment=M_i, mu=background['mu'],
             m2=np.asarray(background['m2'], float),
             **provenance_stamp(args.bounce_npz, background['m2'],
                                background['params'], background['fields']))
    print(f"[ADDBACK] SUMMARY: A1_fin = sum_i M_i tau_fin_fv(m_i^2) = {A1_fin:+.6f}   "
          f"(mu = sum_i m_i = {background['mu']:.6f}, "
          f"{len(background['m2'])} channels; the closed form of the traced "
          f"Born-order-1 add-back -- the assembler adds its two pieces "
          f"(T2)+(T4))")
    print(f"[OK] wrote {out}")


if __name__ == '__main__':
    main()
