#!/usr/bin/env python3
"""greensfct_fv_coincident_renorm_v2.py -- the RENORMALISED coincident
false-vacuum matrix [G0_fv_ren_addback]_ij (dim reg + MSbar) at ONE s^2
(add-on README tree: level 2, branch (g): it needs only the bounce):

    python3 greensfct_fv_coincident_renorm_v2.py --s2 <s2>
    python3 greensfct_fv_coincident_renorm_v2.py --s2 <s2> --mu <scale>

READS   bounce_data_<tag>.npz (level 1): m2, mu = sum_i m_i and the field
        names (load_background) -- no engine, no h-ODE; tau_fin_fv from the
        tadpole add-back stage (addback_tadpole_vgeneral_phi_v2).
WRITES  greensfct_fv_coincident_renorm_v2_<tag>_s2<v>.npz:
        G0_fv_ren_addback (N, N), kappa_i (N,), s2, mu (the thin-wall
        crosschecks read its mu, --run-npz).  The end product takes the
        function, not the file: greensfct_coincident_renorm_v2.py imports
        fv_coincident_renorm_matrix below as its Dyson-order-0 add-back
        G0_fv_ren_addback.

THE OBJECT.  The divergent coincident FV sum (the exhibit of
greensfct_fv_coincident_v2.py) is, in momentum space, the free
one-propagator loop per channel, Int d^4l/(2pi)^4 1/(l^2 + kappa_i^2); what
dim reg + MSbar leaves of it is the closed form

    tau_fin_fv(W^2) = W^2/(16 pi^2) [ln(W^2/mu^2) - 1],

and the renormalised coincident FV matrix is its diagonal at the SHIFTED
channel masses:

    [G0_fv_ren_addback]_ij = delta_ij tau_fin_fv(kappa_i^2),
                             kappa_i^2 = m_i^2 + s^2.

This is the Dyson-order-0 add-back of G_ren: greensfct_coincident_renorm_v2.py
leaves the free waves out of its wave sum one by one (divergent and finite
part together), and this matrix restores the finite part once, in closed
form.

It never enters lnD_ren: lnD is the log of det M / det M_FV, in which this
free term cancels (and, constant in x, its spacetime integral would be
proportional to the volume).  The tadpole add-back of the determinant,
A1_fin = sum_i M_i tau_fin_fv(m_i^2), uses the same function at the
unshifted masses, weighted by the tadpole moments M_i: that is the traced,
s2-integrated Born-order-1 add-back, not this matrix.
"""
import argparse

import numpy as np

from greensfct_toolbox_v2 import (add_greensfct_cli, resolve_bounce,
                                  check_cli, save_greensfct_npz)
from pipeline_helpers_vgeneral_phi_v2 import load_background, s2_filetag
# tau_fin_fv(W2, mu), defined with the tadpole add-back; here it is
# evaluated at the shifted masses kappa_i^2
from addback_tadpole_vgeneral_phi_v2 import tau_fin_fv


def fv_coincident_renorm_matrix(kappa_i, mu):
    """The renormalised coincident FV matrix (N, N):
    delta_ij tau_fin_fv(kappa_i^2) -- the Dyson-order-0 add-back,
    diagonal in the field channels and r-independent."""
    kappa_i = np.asarray(kappa_i, float)
    return np.diag(tau_fin_fv(kappa_i**2, mu))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_greensfct_cli(ap, need_r_grid=False)      # the matrix is r-independent
    ap.add_argument('--mu', type=float, default=None,
                    help='MSbar scale (default: the pipeline scale '
                         'mu = sum_i m_i from the background)')
    args = ap.parse_args()

    bounce_npz = resolve_bounce(args)
    # no engine: the object is a closed form of (m_i^2, s2, mu) alone
    background = load_background(bounce_npz)
    check_cli(args, background)
    mu = float(args.mu) if args.mu is not None else float(background['mu'])
    kappa_i = np.sqrt(background['m2'] + args.s2)      # sqrt(m_i^2 + s2)

    G0_fv_ren_addback = fv_coincident_renorm_matrix(kappa_i, mu)
    print(f'[GF-ADDON] RENORMALISED coincident FV matrix:  s2={args.s2:g}  '
          f'mu={mu:g}  diag tau_fin_fv(kappa_i^2) = '
          f'{np.array2string(np.diag(G0_fv_ren_addback), precision=6)}')
    print('[NOTE] this term never enters lnD_ren: the determinant is the '
          'RATIO to the FV determinant (tau_fin_fv cancels identically) and '
          'the constant would multiply the infinite volume.')

    out = (f'greensfct_fv_coincident_renorm_v2_{args.tag}'
           f'_s2{s2_filetag(args.s2)}.npz')
    save_greensfct_npz(f'{args.data_dir}/{out}', bounce_npz, background,
                G0_fv_ren_addback=G0_fv_ren_addback,
                kappa_i=kappa_i, s2=args.s2, mu=mu)


if __name__ == '__main__':
    main()
