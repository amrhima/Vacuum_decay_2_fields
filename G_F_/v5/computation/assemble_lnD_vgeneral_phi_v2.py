#!/usr/bin/env python3
"""assemble_lnD_vgeneral_phi_v2.py -- level 5, stage 7: the end product
    lnD_ren = ln|det' M_b / det M_FV|_ren     (MSbar, mu = sum_i m_i),
the renormalised log of the fluctuation-determinant ratio, which enters the
decay rate as Gamma/V = (S_E/2 pi)^2 exp(-S_E - lnD_ren/2).  det' omits the
four translation zero modes and keeps the negative mode as ln|omega2_00|;
det M_FV is the full false-vacuum determinant (no mode is removed from it;
five fictitious modes make up for the five modes removed from M_b).  It
rests on d/ds2 ln[det(M_b + s2)/det(M_FV + s2)] = Tr[G(s2) - G_FV(s2)],
G = (M_b + s2)^-1.  The Born orders >= 3 of this log vanish as s2 -> inf, so
    lnD|_(Born >= 3) = -Int_0^inf ds2 Tr[G(s2) - G_FV(s2)]|_(Born >= 3)
with the dangerous modes projected out; the divergent Born orders 1 and 2
enter as their MSbar finite parts.  The seven terms (T1)-(T7) below
assemble these pieces from the six stage npz.

READS   (every stage npz through pipeline_helpers.load_stage_npz: it must
        carry this bounce's fingerprint)
    sector_n0_<tag>.npz, sector_n1_<tag>.npz  (stages 2a/2b) each: the scalar
        Delta_n, s2_max and the arrays s2_sector, delta_geq3 (for
        the guard SECTOR-CUTOFF only); sector_n0 also the pole-subtraction
        eigenvalue omega2_00, sector_n1 omega2_10, the crossing s2_c and the
        degeneracy deg
    band_integrals_<tag>.npz  (stage 3) band_n, I_n_s2_max, T_n_s2_max,
        I_n_inf, n_max, s2_max, high_s2_tail_method
    tail_high_n_<tag>.npz     (stage 4) Zcal, fit_a/b/c, fit_lo, n_max, A3
    addback_tadpole_<tag>.npz (stage 5) tadpole_moment M_i, A1_fin
    addback_fish_<tag>.npz    (stage 6) A2_fin and the last ladder rung
        (q_momentum_max_ladder_steps, n_q_momentum_ladder_steps)
    --bounce-npz  through load_background: the channel masses m2, the scale
        mu = sum_i m_i, m_min2 = m2fict, and R, phi_b, pot_prime (the
        eps_bounce terms); read directly: the census verdict
        (census_accept, census_s2_c, census_zero_mode_trust, bounce_method;
        reported at the end when the npz stores them, never re-derived) and
        the reference profile phi_b_ref with its tolerance
        bounce_ref_tol.
WRITES  lnD_ren_<tag>.npz (the filename and keys plot_computation/ and the
        thin-wall test read): lnD_ren, Delta_0/1, omega2_00,
        omega2_10, m2fict, deg1, band_n, I_n_s2_max, T_n_s2_max,
        I_n_inf, G1_addback_band, G1_addback_high_s2,
        fit_a/b/c, n_max, Zcal, A2_fin, s2_max, mu (the thin-wall
        test's default scale) -- plot_common.lnD_from_parts rebuilds lnD_ren
        term by term from these (the sector ARRAYS are read from the sector
        npz, never copied here); record only, no reader: A1_fin,
        Sigma_dang, s2_c, fit_lo, fict_block_invariant,
        high_s2_tail_method, A3, eps_bounce and its parts eps_bounce_dA1,
        eps_bounce_dA2_half, eps_bounce_dA3_rel, eps_bounce_dA1_quad,
        bounce_ref_tol, assembly; plus the three provenance keys every stage
        stamps (bounce_sha, potential_id, code_version).  fict_block_terms is
        imported by plot_computation/plot_common.

HOW     lnD_ren is the sum of seven terms:
       (T1) Sigma_dang          the sectors n = 0, 1
       (T2) G1_addback_band                 tadpole add-back, all n, s2 <= s2_max
       (T3) sum_{n=2..n_max} I_n(s2_max)    band integrals, s2 <= s2_max
       (T4) G1_addback_high_s2              tadpole add-back, all n, s2 > s2_max
       (T5) sum_{n=2..n_max} T_n(s2_max)    high-s2 tail of the band waves
       (T6) Zcal                            high-n tail, the waves n > n_max
       (T7) -A2_fin/2                       fish add-back
   I_n(s2_max) = -Int_0^{s2_max} ds2 delta_geq3(n, s2), and T_n(s2_max) the
   same integral beyond s2_max (from the large-s2 shape of delta_geq3);
   delta_geq3(n, s2) is the traced Green's function of wave n (degeneracy
   (n+1)^2 included) with its Born orders 0 (false vacuum), 1 (tadpole) and
   2 (fish) removed, whose MSbar finite parts of orders 1 and 2 return as
   A1_fin = (T2) + (T4) and (T7).  Grouped, the same number reads
       lnD_ren = Sigma_dang + sum_{n=2..n_max} I_n(inf) + Zcal
                 + A1_fin - A2_fin/2,
       I_n(inf) = I_n(s2_max) + T_n(s2_max),   Zcal = sum_{n > n_max} I_n(inf).

   (T1) THE DANGEROUS WAVES Sigma_dang: the waves n = 0 and n = 1
   with their dangerous modes, ONE negative mode (eigenvalue omega2_00 < 0,
   wave n = 0) and ONE radial translation zero mode (omega2_10 ~ 0, wave
   n = 1, 4-fold degenerate in the angular index); both are the lowest
   eigenvalue (radial number 0) of their wave.  The census selection gate
   refuses every other bounce before the band.
       Sigma_dang = Delta_0 + Delta_1
           + fict_block + ln|omega2_00| ,
       Delta_n = -Int_0^{s2_max} ds2 [delta_geq3(n, s2)
                                                 - deg_n/(omega2 + s2)]
   (stages 2a, 2b; omega2 = omega2_00 for n = 0, omega2_10 for n = 1;
   deg_n = (n+1)^2).  Subtracting the five poles leaves the trace of M_b five
   modes short of that of M_FV, and the s2 integral would diverge like
   5 ln s2; five fictitious modes of mass^2 m2fict restore the count.
   fict_block (fict_block_terms) holds in closed form their terms, the pole
   terms beyond s2_max and the constant -5 ln(m2fict); m2fict drops out
   exactly:
       fict_block = -ln(omega2_00 + s2_max) - deg1 ln(omega2_10 + s2_max).
   The negative mode is restored as +ln|omega2_00|; the translation modes are
   not (they give the factor (S_E/2 pi)^2 of the decay rate).  The located
   n = 1 crossing s2_c = -omega2_10 is only reported.
   (T2), (T4) THE TADPOLE ADD-BACK A1_fin = sum_i M_i tau_fin_fv(m_i^2),
   split at s2_max (formulas at the code); stage 5's stored A1_fin is the
   same number in one piece, used here only by eps_bounce.
   (T3), (T5) loaded from stage 3 and summed over the band waves n = 2..n_max.
   (T6) THE HIGH-N TAIL of stage 4: the fit a/nu^3 + b/nu^5 + c/nu^7
   (nu = n + 1) of I_n(inf) on [fit_lo, n_max], summed over n > n_max in
   closed form (Hurwitz zeta).
   (T7) THE FISH ADD-BACK -A2_fin/2 of stage 6, the one add-back that exists
   only at the determinant level.
   Beyond s2_max the band waves are completed by (T5); of the sectors only the
   pole and fictitious-mode terms are continued (fict_block), their trace
   delta_geq3 there is dropped within the SECTOR-CUTOFF bound.

GUARDS hosted here (each aborts; does NOT enter lnD):
  SECTOR-CUTOFF  the sectors' raw trace delta_geq3 beyond s2_max (dropped),
                 extrapolated by its power-law decay, must contribute less than
                 SECTOR_REMAINDER_TOL to lnD (else the band must be run to a
                 larger s2_max) -- the bound the band's SECTOR-CUTOFF
                 watcher (stage 1) stopped on, re-checked on the sector
                 grids;
  LND-FINITE     abort on a non-finite lnD_ren before writing anything;
  FICT-BLOCK     (fict_block_terms) every log argument of the fictitious-mode
                 block, m2fict and omega2 + s2_max, must be positive (s2_max
                 above every subtracted pole).
CROSS-CONSISTENCY ABORTS: (i) the two sector npz and the band must share
the identical s2_max (each sector MAY use its own grid); (ii) the bounce
fingerprint of every input npz must match the --bounce-npz file; (iii) the
band integrals and the tail must share one n_max (the band sum runs over
n = 2..n_max, the zeta tail over n > n_max).

BOUNCE SENSITIVITY INDICATOR eps_bounce (printed and stored; does not enter
lnD_ren).  The bounce npz holds a reference profile phi_b_ref, the
same boundary-value solve at the tighter tolerance bounce_ref_tol on the same
grid R (NaN when it did not converge).  A1_fin, A2_fin and A3 are recomputed
on it by the production routines (the fish on the momentum grid of stage 6's
last ladder rung, so dA2 is a pure solution difference), and A1_fin also on
the midpoint-refined grid of the kept profile:
       eps_bounce = |dA1_fin| + |dA2_fin|/2 + |band_sum * dA3/A3| + |dA1_quad|
The band term is a heuristic (A3 is one moment of tr U_int^3 and does not fix
how every wave responds); the sectors, the fish radial mesh and the ODE
tolerances are not measured.  So eps_bounce shows how sensitive lnD_ren is to
the bounce; it is not a bound on its error.  Below EPS_BOUNCE_NEGLIGIBLE it is
reported as small.
"""
import argparse
import os
import sys
import numpy as np

sys.dont_write_bytecode = True
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from scipy.interpolate import CubicSpline
from pipeline_helpers_vgeneral_phi_v2 import (add_standard_cli, atomic_savez,
                                 insertion, load_background, load_stage_npz,
                                 provenance_stamp, stage_paths,
                                 trace_U_int_cubed)

# the routines that turn a profile into A1_fin, A2_fin, A3 -- the SAME code
# that produced the kept profile's stored numbers (this is the last stage,
# so the flow stays downward)
from addback_tadpole_vgeneral_phi_v2 import (compute_A1_fin,
                                                 tadpole_moment, tau_fin_fv)
from addback_fish_vgeneral_phi_v2 import A2_from_Utilde, Utilde_matrix
from watchers_guards_vgeneral_phi_v2 import (SECTOR_REMAINDER_TOL, analytic_moments,
                                             sector_high_s2_remainder_bound)
from mode_census_vgeneral_phi_v2 import stored_census


# The level (lnD units) below which the bounce sensitivity indicator
# eps_bounce is reported as small rather than as a number.
EPS_BOUNCE_NEGLIGIBLE = 1e-4

# --------------------------------------------------------------------------- #
#  fict_block: the fictitious-mode part of Sigma_dang (closed form)           #
# --------------------------------------------------------------------------- #
def fict_block_terms(m2fict, s2_max, omega2_00, omega2_10, deg1):
    """The fictitious-mode part fict_block of Sigma_dang, in closed form.
    S = s2_max; omega2_00 (< 0) and omega2_10 (~ 0) are the subtracted
    eigenvalues of the waves n = 0 (the negative mode) and n = 1 (the
    translation zero mode, deg1 = 4 angular copies).  One fictitious mode of
    mass^2 m2fict replaces each removed mode, nfict = 1 + deg1 = 5:
        I_fict_tot = -Int_0^S ds2 nfict/(m2fict + s2)
                   = -nfict ln((m2fict + S)/m2fict)
        T_fict     = -Int_S^inf ds2 [nfict/(m2fict + s2) - 1/(omega2_00 + s2)
                                     - deg1/(omega2_10 + s2)]
                   = ln((m2fict + S)/(omega2_00 + S))
                     + deg1 ln((m2fict + S)/(omega2_10 + S))
        fict_block = I_fict_tot + T_fict - nfict ln(m2fict)
                   = -ln(omega2_00 + S) - deg1 ln(omega2_10 + S)
    exactly, for every m2fict > 0 (T_fict converges because nfict = 1 + deg1);
    the right-hand side is returned as fict_block_invariant.  Returns
    (fict_block, fict_block_invariant).  Guard FICT-BLOCK: every log argument
    (m2fict, omega2_00 + S, omega2_10 + S) must be positive, i.e. S must lie
    above every subtracted pole s2 = -omega2; a non-positive one aborts with
    its name."""
    for name, val in (('m2fict', m2fict),
                      ('omega2_00 + s2_max', omega2_00 + s2_max),
                      ('omega2_10 + s2_max', omega2_10 + s2_max)):
        if not (float(val) > 0.0):
            raise RuntimeError(
                f'[ABORT FICT-BLOCK] the fictitious-mode bookkeeping needs '
                f'{name} > 0 but it is {float(val):+.6e} (m2fict={m2fict:g}, '
                f's2_max={s2_max:g}, omega2_00={omega2_00:+.6e}, '
                f'omega2_10={omega2_10:+.6e}): the log of a non-positive '
                f'number is NaN, not a number this assembly can carry.  The '
                f'cutoff s2_max must lie ABOVE the pole s2 = -omega2 of '
                f'every subtracted mode -- run the band, and the sectors '
                f'after it, to a larger s2_max.')
    nfict = 1.0 + deg1                       # one fict mode per removed mode
    ln_ratio_fict = np.log((m2fict + s2_max) / m2fict)
    I_fict_tot = -nfict * ln_ratio_fict
    T_fict = (np.log((m2fict + s2_max) / (omega2_00 + s2_max))
              + deg1 * np.log((m2fict + s2_max) / (omega2_10 + s2_max)))
    fict_block = I_fict_tot + T_fict - nfict * np.log(m2fict)
    fict_block_invariant = (-np.log(omega2_00 + s2_max)
                            - deg1 * np.log(omega2_10 + s2_max))
    return fict_block, fict_block_invariant


def main():
    ap = argparse.ArgumentParser(
        description='vgeneral_phi_v2 stage 7: lnD_ren, assembled from the '
                    'seven terms (T1)-(T7) of the stage npz')
    add_standard_cli(ap)
    args = ap.parse_args()

    # ---- inputs: the six stage npz + the bounce ------------------------------
    # load_stage_npz proves every input belongs to THIS bounce (its stored
    # bounce fingerprint, npz key bounce_sha, must equal the checksum of the
    # --bounce-npz file; a missing stamp aborts as loudly as a wrong one).
    paths = stage_paths(args.data_dir, args.tag)
    sector_n0_npz = load_stage_npz(paths['sector_n0'], args.bounce_npz,
                        'gbar_sector_vgeneral_phi_v2 --sector-n 0')
    sector_n1_npz = load_stage_npz(paths['sector_n1'], args.bounce_npz,
                        'gbar_sector_vgeneral_phi_v2 --sector-n 1')
    band_integrals_npz = load_stage_npz(paths['band_integrals'], args.bounce_npz,
                        'band_integrals_vgeneral_phi_v2')
    tail_high_n_npz = load_stage_npz(paths['tail_high_n'], args.bounce_npz,
                        'tail_high_n_zeta_vgeneral_phi_v2')
    addback_tadpole_npz = load_stage_npz(paths['addback_tadpole'], args.bounce_npz,
                         'addback_tadpole_vgeneral_phi_v2')
    addback_fish_npz = load_stage_npz(paths['addback_fish'], args.bounce_npz,
                         'addback_fish_vgeneral_phi_v2')
    background = load_background(args.bounce_npz)
    m2, mu, m_min2 = background['m2'], background['mu'], background['m_min2']
    # the bounce npz itself: the census verdict of bounce_vgeneral_phi_v2
    # (reported at the end, never re-derived here) and the reference profile
    # of the bounce sensitivity indicator (keys phi_b_ref, bounce_ref_tol:
    # a tighter solve_bvp solution on the SAME grid R; NaN when none
    # converged)
    with np.load(args.bounce_npz, allow_pickle=True) as bnpz:
        # None for a bounce npz without the census keys (e.g. one from an
        # external solver): there is no stored verdict to report (stages
        # 0a / 0b, eigenvalue_neg / eigenvalue_zero, census such a bounce
        # themselves)
        census = stored_census(bnpz)
        bounce_method = (str(bnpz['bounce_method']) if 'bounce_method' in bnpz.files
                         else 'bounce method not stored')
        # a bounce npz without a reference profile is treated like one whose
        # reference solve did not converge (NaN): the indicator is not evaluated
        has_ref = 'phi_b_ref' in bnpz.files
        phi_b_ref = (np.asarray(bnpz['phi_b_ref'], float) if has_ref
                     else np.full(background['phi_b'].shape, np.nan))
        bounce_ref_tol = float(bnpz['bounce_ref_tol']) if has_ref else float('nan')

    # ---- cross-consistency aborts (metadata, before any physics) -------------
    s2_max = float(band_integrals_npz['s2_max'])
    if not (float(sector_n0_npz['s2_max']) == float(sector_n1_npz['s2_max']) == s2_max):
        raise RuntimeError('[ABORT] the two sector npz and the band npz do '
                           'not share one s2_max -- the assembly combines the '
                           'SCALARS Delta_0 + Delta_1 '
                           'and the band at a common cutoff (each sector MAY '
                           'converge on its own grid; only the cutoff must '
                           'match).')

    # ---- unpack the sector scalars (no arithmetic on load) -------------------
    omega2_00 = float(sector_n0_npz['omega2_00'])
    omega2_10 = float(sector_n1_npz['omega2_10'])
    s2_c = float(sector_n1_npz['s2_c'])
    deg1 = float(sector_n1_npz['deg'])
    Delta_0 = float(sector_n0_npz['Delta_n'])
    Delta_1 = float(sector_n1_npz['Delta_n'])

    # ---- GUARD SECTOR-CUTOFF (aborts; does NOT enter lnD): is s2_max high
    # enough for the sectors?  Beyond s2_max fict_block (its part T_fict)
    # continues only the subtracted poles; the raw trace delta_geq3 beyond the
    # cutoff is dropped, so the residual truncation is Int_{s2_max}^inf
    # delta_geq3(n, s2) ds2 per wave (d(lnD)/d(s2_max) = -delta_geq3(n, s2_max)).
    # sector_high_s2_remainder_bound extrapolates the stored trace's
    # power-law decay and returns (remainder bound, decay power,
    # |delta_geq3(n, s2_max)|, max log-residual of the fit);
    # max_log_resid_n0/n1 are printed so a poor fit is visible.
    (remainder_bound_n0, decay_power_n0, abs_delta_n0_at_s2_max,
     max_log_resid_n0) = sector_high_s2_remainder_bound(
        sector_n0_npz['s2_sector'], sector_n0_npz['delta_geq3'])
    (remainder_bound_n1, decay_power_n1, abs_delta_n1_at_s2_max,
     max_log_resid_n1) = sector_high_s2_remainder_bound(
        sector_n1_npz['s2_sector'], sector_n1_npz['delta_geq3'])
    sector_remainder_bound = remainder_bound_n0 + remainder_bound_n1
    print(f"[SECTOR-CUTOFF] raw-trace cutoff at s2_max = {s2_max:g}: "
          f"|delta_geq3(0, s2_max)|={abs_delta_n0_at_s2_max:.2e} "
          f"(decay s2^{decay_power_n0:.2f}, "
          f"bound {remainder_bound_n0:.2e}, fit "
          f"residual {max_log_resid_n0:.2f}), "
          f"|delta_geq3(1, s2_max)|={abs_delta_n1_at_s2_max:.2e} "
          f"(decay s2^{decay_power_n1:.2f}, "
          f"bound {remainder_bound_n1:.2e}, fit "
          f"residual {max_log_resid_n1:.2f}); "
          f"dropped high-s2 remainder of the sectors, bound "
          f"{sector_remainder_bound:.2e} "
          f"(limit {SECTOR_REMAINDER_TOL:g})")
    if sector_remainder_bound > SECTOR_REMAINDER_TOL:
        # project the cutoff that meets the tolerance from the fitted
        # remainder scaling  bound ~ s2_max^(decay_power+1),  using the
        # slower-decaying wave
        p_slow = max(decay_power_n0, decay_power_n1)
        if np.isfinite(sector_remainder_bound) and p_slow < -1.0:
            s2_need = s2_max * (sector_remainder_bound
                      / SECTOR_REMAINDER_TOL) ** (-1.0 / (p_slow + 1.0))
            hint = (f'run the band (--pin-s2-max {s2_need:.3g} or larger, '
                    f'projected from the fitted decay s2^{p_slow:.2f}) and '
                    f'the sectors after it')
        else:
            hint = ('the remainder fit cannot project a cutoff -- run the band '
                    'to twice the s2_max and the sectors after it')
        raise RuntimeError(
            f'[ABORT SECTOR-CUTOFF] the cutoff s2_max = {s2_max:g} is too '
            f'low for the sectors of this potential: their raw trace '
            f'delta_geq3 has not decayed (estimated lnD truncation '
            f'{sector_remainder_bound:.2e} > '
            f'{SECTOR_REMAINDER_TOL:g}).  {hint} (beyond s2_max fict_block '
            f'continues only the poles, not the trace).')

    # ---- (T1) Sigma_dang (module docstring) ----------------------------------
    # ln|omega2_00|: the negative-mode add-back, at the n = 0 eigenvalue whose
    # pole sector_n0 subtracted, once for the one negative mode.  The absolute
    # value is the convention: the continued Gaussian integral of the negative
    # mode gives (i/2)|omega2_00|^(-1/2), and the factor i/2 goes into the
    # decay rate (Gamma = -2 Im E of the false vacuum), not into lnD_ren.
    ln_abs_omega2_00 = float(np.log(abs(omega2_00)))
    m2fict = m_min2        # the fictitious mass^2: every m2fict > 0 gives the same block
    fict_block, fict_block_invariant = fict_block_terms(
        m2fict, s2_max, omega2_00, omega2_10, deg1)
    Sigma_dang = float(Delta_0 + Delta_1
                                   + fict_block + ln_abs_omega2_00)

    # ---- the band pieces (T3), (T5), Zcal (T6) and A2_fin of (T7)
    # (loaded, no arithmetic on load) ------------------------------------------
    band_n = np.asarray(band_integrals_npz['band_n'], int)
    # (T3) per wave: I_n(s2_max)
    I_n_s2_max = np.asarray(band_integrals_npz['I_n_s2_max'], float)
    # (T5) per wave: T_n(s2_max), the high-s2 tail of I_n beyond s2_max
    T_n_s2_max = np.asarray(band_integrals_npz['T_n_s2_max'], float)
    # stage 3's sum I_n(inf) per wave
    I_n_inf = np.asarray(band_integrals_npz['I_n_inf'], float)
    fit_lo, n_max = (int(tail_high_n_npz['fit_lo']),
                     int(tail_high_n_npz['n_max']))
    fit_a, fit_b, fit_c = (float(tail_high_n_npz['fit_a']),
                           float(tail_high_n_npz['fit_b']),
                           float(tail_high_n_npz['fit_c']))
    Zcal = float(tail_high_n_npz['Zcal'])                              # (T6)
    A3 = float(tail_high_n_npz['A3'])
    A2_fin = float(addback_fish_npz['A2_fin'])   # (T7) enters as -A2_fin/2
    # stages 3 and 4 must have read the same band_cutoffs: the band sum runs
    # over n = 2..n_max and the zeta tail over n > n_max, so one n_max joins
    # them without a gap or an overlap
    if n_max != int(band_integrals_npz['n_max']):
        raise RuntimeError(
            f"[ABORT] tail_high_n n_max={n_max} != band_integrals n_max="
            f"{int(band_integrals_npz['n_max'])}: stages 3 and 4 ran on "
            f"different band cutoffs -- rerun them.")
    band_sum_s2_max = float(I_n_s2_max.sum())    # (T3) sum_{n=2..n_max} I_n(s2_max)
    T_sum = float(T_n_s2_max.sum())              # (T5) sum_{n=2..n_max} T_n(s2_max)
    band_sum = float(I_n_inf.sum())              # sum_n I_n(inf) = (T3)+(T5)

    # ---- (T2), (T4): the tadpole add-back A1_fin, split at s2_max.  Summed
    # over ALL waves n, the traced Born-order-1 add-back of the renormalized
    # Green's function (the add-on's G1_ren_addback) is
    #     Int d^4x sum_i [G^(1),ren]_ii(s2) = d/ds2 sum_i M_i tau_fin_fv(m_i^2 + s2)
    #         = sum_i (1/8) Int_0^inf dr r^3 [U_int]_ii ln(kappa_i^2/mu^2),
    # M_i = 2 pi^2 Int_0^inf dr r^3 [U_int]_ii (stage 5), kappa_i^2 = m_i^2 + s2,
    # tau_fin_fv(W^2) = W^2/(16 pi^2) (ln(W^2/mu^2) - 1) (addback_tadpole), and
    # minus its s2 integral is
    #     (T2) on [0, s2_max]:  sum_i M_i [tau_fin_fv(m_i^2) - tau_fin_fv(m_i^2 + s2_max)]
    #     (T4) on s2 > s2_max:  sum_i M_i tau_fin_fv(m_i^2 + s2_max)
    # (T4) is taken in dim reg: in d = 4 the s2 integral diverges at large s2,
    # in dim reg its upper end gives nothing.  (T2) + (T4) = A1_fin.
    M_i = np.asarray(addback_tadpole_npz['tadpole_moment'], float)
    G1_addback_band = -float(M_i @ (tau_fin_fv(m2 + s2_max, mu) - tau_fin_fv(m2, mu)))
    G1_addback_high_s2 = float(M_i @ tau_fin_fv(m2 + s2_max, mu))

    # ---- THE ASSEMBLY: the seven terms (T1)-(T7) of the module docstring ----
    lnD_ren = (Sigma_dang    # (T1) the sectors n = 0, 1
               + G1_addback_band         # (T2) tadpole add-back on [0, s2_max]
               + band_sum_s2_max         # (T3) sum_n I_n(s2_max), n = 2..n_max
               + G1_addback_high_s2      # (T4) tadpole add-back beyond s2_max
               + T_sum                   # (T5) high-s2 tail of the band waves
               + Zcal                    # (T6) high-n tail, n > n_max
               - 0.5 * A2_fin)           # (T7) fish add-back
    # GUARD LND-FINITE (aborts; does NOT enter lnD): the assembled number
    # must be finite before anything is printed or written.
    if not np.isfinite(lnD_ren):
        raise RuntimeError('[ABORT LND-FINITE] non-finite lnD_ren.')

    # ---- final table: term by term, then the grouped form ---------------------
    print("=" * 72)
    print(f"  det M_FV in full (five fictitious modes); sectors: pole-subtracted s2 integrals; "
          f"band high-s2 tail: {str(band_integrals_npz['high_s2_tail_method'])}")
    print(f"  (T1) Sigma_dang  = Delta_0 {Delta_0:+.6f} "
          f"+ Delta_1 {Delta_1:+.6f} (deg {deg1:g})")
    print(f"       + fict_block {fict_block:+.6f} + ln|omega2_00| {ln_abs_omega2_00:+.6f}"
          f"   = {Sigma_dang:+.6f}")
    print(f"       (the subtracted poles: omega2_00 = {omega2_00:+.10e}, "
          f"omega2_10 = {omega2_10:+.6e}; fict_block at m2fict = {m2fict:.6g})")
    print(f"  (T2) Born-order-1 add-back, all n, -Int_0^s2max    = {G1_addback_band:+.6f}")
    print(f"  (T3) sum_n I_n(s2_max), n=2..{n_max:<4d}              = {band_sum_s2_max:+.6f}")
    print(f"  (T4) Born-order-1 add-back beyond s2_max  = {G1_addback_high_s2:+.6f}")
    print(f"  (T5) sum_n T_n(s2_max) (high-s2 tail), n=2..{n_max:<4d} = {T_sum:+.6f}")
    print(f"  (T6) Zcal (n > {n_max}: zeta sum of the fit of I_n(inf)) = {Zcal:+.6f}"
          f"   (fit_a {fit_a:+.5g}, A3 {A3:+.5g})")
    print(f"  (T7) -A2_fin/2 (fish add-back)                       = {-0.5*A2_fin:+.6f}")
    print(f"  lnD_ren = (T1)+(T2)+(T3)+(T4)+(T5)+(T6)+(T7)          = {lnD_ren:+.6f}")
    print(f"  grouped: sum_n I_n(inf) = (T3)+(T5) = {band_sum:+.6f};"
          f"   A1_fin = (T2)+(T4) = {G1_addback_band + G1_addback_high_s2:+.6f}"
          f"   (mu = {mu:g})")

    # ---- eps_bounce (module docstring); its four terms are stored as
    # eps_bounce_dA1, eps_bounce_dA2_half, eps_bounce_dA3_rel = |dA3/A3| and
    # eps_bounce_dA1_quad = |A1_fin(midpoint-refined grid) - A1_fin|.
    R, phi_b, pot_prime = (background['R'], background['phi_b'],
                           background['pot_prime'])
    # the kept profile's tadpole add-back, stage 5's A1_fin = (T2) + (T4)
    A1_fin = float(addback_tadpole_npz['A1_fin'])
    # the quadrature term, from the KEPT profile alone: A1_fin on the
    # midpoint-refined grid R_mid (profile by scipy CubicSpline) vs stored
    R_mid = np.sort(np.concatenate([R, 0.5 * (R[:-1] + R[1:])]))
    phi_b_mid = CubicSpline(R, phi_b)(R_mid)              # (2 nR - 1, N)
    A1_mid = compute_A1_fin(tadpole_moment(R_mid, insertion(pot_prime, phi_b_mid),
                                           phi_b_mid, pot_prime), m2, mu)
    eps_bounce_dA1_quad = abs(A1_mid - A1_fin)
    if np.all(np.isfinite(phi_b_ref)):          # a converged reference is stored
        # the solution terms: the reference profile through the same routines;
        # its fish on the momentum grid [0, q_momentum_max] the kept profile's
        # two ladders converged on (the last rung stage 6 recorded), so dA2
        # carries no quadrature difference
        U_int_ref = insertion(pot_prime, phi_b_ref)
        A1_ref = compute_A1_fin(tadpole_moment(R, U_int_ref, phi_b_ref, pot_prime),
                                m2, mu)
        q_grid = np.linspace(0.0,
                             float(addback_fish_npz['q_momentum_max_ladder_steps'][-1]),
                             int(addback_fish_npz['n_q_momentum_ladder_steps'][-1]))
        A2_ref = A2_from_Utilde(q_grid, Utilde_matrix(q_grid, R, U_int_ref), m2, mu)
        A3_ref = analytic_moments(R, trace_U_int_cubed(U_int_ref))['A3']
        eps_bounce_dA1 = abs(A1_fin - A1_ref)
        # the fish term enters lnD as -A2_fin/2
        eps_bounce_dA2_half = abs(A2_fin - A2_ref) / 2.0
        eps_bounce_dA3_rel = (abs(A3 - A3_ref) / abs(A3) if A3 != 0.0
                              else float('nan'))
        eps_bounce_band = abs(band_sum * eps_bounce_dA3_rel)
        eps_bounce = (eps_bounce_dA1 + eps_bounce_dA2_half + eps_bounce_band
                      + eps_bounce_dA1_quad)
        if eps_bounce < EPS_BOUNCE_NEGLIGIBLE:
            print(f"bounce sensitivity indicator small ({eps_bounce:.2e} < "
                  f"{EPS_BOUNCE_NEGLIGIBLE:g}; vs the tol={bounce_ref_tol:g} "
                  f"reference profile on the same grid)")
        else:
            print(f"bounce sensitivity indicator: {eps_bounce:.6f} (vs the "
                  f"tol={bounce_ref_tol:g} reference profile on the same grid)")
        print(f"   sensitivity terms: tadpole |dA1_fin| = "
              f"{eps_bounce_dA1:.3e}, fish |dA2_fin|/2 = "
              f"{eps_bounce_dA2_half:.3e}, band |sum I_n * dA3/A3| = "
              f"{eps_bounce_band:.3e}, mesh |dA1_quad| = "
              f"{eps_bounce_dA1_quad:.3e}   "
              f"(indicator only: the band response is approximated by dA3/A3; "
              f"sector and non-tadpole mesh errors are not measured)")
    else:
        # no usable reference: bounce_vgeneral_phi_v2 built no reference
        # profile -- say so rather than print a NaN or a zero as the indicator
        eps_bounce_dA1 = eps_bounce_dA2_half = float('nan')
        eps_bounce_dA3_rel = eps_bounce = float('nan')
        why = ('the bounce npz carries no reference profile (phi_b_ref '
               + ('is NaN: no reference solve converged)' if has_ref else
                  'absent: written before bounce_vgeneral_phi_v2 stored one -- re-run it)'))
        print(f"bounce sensitivity indicator could not be evaluated: {why}; "
              f"the mesh term alone is |dA1_quad| = "
              f"{eps_bounce_dA1_quad:.3e}")
    # the census verdict of bounce_vgeneral_phi_v2 (key census_accept: one n=0
    # crossing and one dangerous n=1 crossing, one mode on each, no unresolved
    # region), reported, not re-derived
    if census is None:
        print(f"zero crossing condition: the bounce npz stores no census, so "
              f"there is no stored verdict to report here (eigenvalue_neg / "
              f"eigenvalue_zero, stages 0a / 0b, census such a bounce "
              f"themselves; {bounce_method})")
    else:
        print(f"zero crossing condition satisfied: "
              f"{'yes' if census['census_accept'] else 'no'}"
              f"  (the stored census verdict census_accept of "
              f"bounce_vgeneral_phi_v2; census_s2_c = {census['census_s2_c']:+.3e}, "
              f"zero-mode trust level {census['census_zero_mode_trust']:g}; "
              f"{bounce_method})")
    print(f"zero crossing in n=1 sector is: s2_c = {s2_c:+.3e}  (= -omega2_10, "
          f"the located root of eig_n1 carried by sector_n1)")
    print("=" * 72)

    out = paths['lnD_ren']
    atomic_savez(out,
             lnD_ren=lnD_ren,
             Sigma_dang=Sigma_dang,
             Delta_0=Delta_0,
             Delta_1=Delta_1,
             omega2_00=omega2_00, omega2_10=omega2_10,
             m2fict=m2fict,
             s2_c=s2_c,
             fict_block_invariant=fict_block_invariant,
             deg1=deg1,
             band_n=band_n, I_n_inf=I_n_inf,
             I_n_s2_max=I_n_s2_max, T_n_s2_max=T_n_s2_max,
             high_s2_tail_method=str(band_integrals_npz['high_s2_tail_method']),
             fit_a=fit_a, fit_b=fit_b, fit_c=fit_c, Zcal=Zcal,
             A3=A3,
             fit_lo=fit_lo, n_max=n_max,
             A1_fin=A1_fin, A2_fin=A2_fin, eps_bounce=eps_bounce,
             # the eps_bounce parts (record only, no reader)
             eps_bounce_dA1=eps_bounce_dA1,
             eps_bounce_dA2_half=eps_bounce_dA2_half,
             eps_bounce_dA3_rel=eps_bounce_dA3_rel,
             eps_bounce_dA1_quad=eps_bounce_dA1_quad,
             bounce_ref_tol=bounce_ref_tol,
             G1_addback_band=G1_addback_band, G1_addback_high_s2=G1_addback_high_s2,
             mu=mu, s2_max=s2_max,
             assembly='lnD_ren = (T1) Sigma_dang + (T2) G1_addback_band '
                      '+ (T3) sum I_n(s2_max) + (T4) G1_addback_high_s2 '
                      '+ (T5) sum T_n(s2_max) + (T6) Zcal + (T7) (-A2_fin/2)',
             **provenance_stamp(args.bounce_npz, background['m2'],
                                background['params'], background['fields']))
    print(f"[OK] wrote {out}")


if __name__ == '__main__':
    main()
