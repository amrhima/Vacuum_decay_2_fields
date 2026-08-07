#!/usr/bin/env python3
"""counterterm_tadpole_coupled_toy_model.py -- stage 6: the tadpole counterterm
finite part A^(1)_fin(mu).

WHAT IT COMPUTES
----------------
The renormalization is dim-reg MS-bar; the divergent tadpole (one
potential insertion) cancels pointwise between bounce and false vacuum,
leaving the mu-dependent FINITE part (renormalization scale mu = sum_i m_i,
the FV Hessian masses; = m1 + m2 for two fields):

    A^(1)_fin = -(1/8) sum_{i=1..N} m_i^2 [1 - ln(m_i^2/mu^2)]
                        * Int dr r^3 V_ii(r) ,

with V(r) = H(phi_b(r)) - H_FV the potential insertion (diagonal entries
only; pipeline_helpers_coupled_toy_model.potential_insertion_V) and the radial
integral done
with scipy trapezoid on the bounce grid.  It enters the assembly as
+A1_fin (see assemble_lnD_coupled_toy_model).

INPUTS   --bounce-npz (the potential is rebuilt from the coupling vector
         stored in it); no other stage
         output is needed.
OUTPUT   ct_tadpole_<tag>.npz with keys
    A1_fin  ()   the finite tadpole counterterm (enters lnD_ren)
    mu      ()   renormalization scale sum_i m_i
    m_sq    (2,) the FV Hessian masses^2 (both channels)
    bounce_sha, potential_id, code_version    metadata
"""
import argparse
import os
import sys
import numpy as np
from scipy.integrate import trapezoid

sys.dont_write_bytecode = True
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from pipeline_helpers_coupled_toy_model import (add_standard_cli, atomic_savez,        # noqa: E402
                                 load_bounce, potential_insertion_V,
                                 provenance_stamp, stage_paths)


# --------------------------------------------------------------------------- #
#  Counterterm (mu = sum_i m_i, the FV Hessian masses)                             #
# --------------------------------------------------------------------------- #
def compute_A1_fin(R, V_diag, m_sq, mu):
    A1 = 0.0
    for i in range(len(m_sq)):              # sum over BOTH channels
        I_R = float(trapezoid(R**3 * V_diag[i], R))
        log_factor = 1.0 - np.log(m_sq[i] / mu**2)
        A1 += m_sq[i] * log_factor * I_R
    return -A1 / 8.0


def main():
    ap = argparse.ArgumentParser(
        description='coupled_toy_model stage 6: tadpole counterterm A1_fin')
    add_standard_cli(ap)
    args = ap.parse_args()

    bg = load_bounce(args.bounce_npz)
    _, V_diag = potential_insertion_V(bg['R'], bg['Phi'],
                                      bg['pot'], bg['Hfv'])
    A1_fin = compute_A1_fin(bg['R'], V_diag, bg['masses'], bg['mu'])

    out = stage_paths(args.data_dir, args.tag)['ct_tadpole']
    atomic_savez(out,
             A1_fin=A1_fin, mu=bg['mu'],
             m_sq=np.asarray(bg['masses'], float),
             **provenance_stamp(args.bounce_npz, bg['m2']))
    print(f"[CT] SUMMARY: A1_fin = {A1_fin:+.6f}   "
          f"(mu = sum_i m_i = {bg['mu']:.6f}, {len(bg['masses'])} channels)")
    print(f"[OK] wrote {out}")


if __name__ == '__main__':
    main()
