#!/usr/bin/env python3
"""assemble_lnD_coupled_toy_model.py -- stage 8: the final assembly of lnD^ren from
the per-stage npz outputs (Carosi FULL-FV convention -- the standard
full-FV assembly).

METHOD MANDATE (user, standing): production = RESOLVENT (Green's-function)
method ONLY; the fixed-s2 determinant method is a CROSSCHECK, never the
production path.  Carosi (arXiv:2601.08686): the two are the same object,
so the numbers must not change.

WHAT IT COMPUTES
----------------
1. PRODUCTION sector block (from the sector npz + closed forms):
       sector_block_resolvent = I8res_0 + I8res_1
                              + [fict block](m2fict) + ln|w_neg2|
   with the Carosi fictitious-mode pieces carried in CLOSED FORM
   (fict_block_terms):
       I8fict_n = -deg_n ln((m2fict+s2_hi)/m2fict)  (fict mode inside the
                  integral), -deg_n ln(m2fict) outside per removed mode,
       T_sec8   = exact-log tail beyond s2_sector_max,
   so the m2fict dependence collapses exactly.
2. THE ASSEMBLY (term order is part of the certified float path):
       lnD_ren = sector_block_resolvent
               + sum_{n=2..fit_hi} I_n(inf)     [band_sum, stage-4 npz]
               + tail                            [stage-5 npz]
               + A1_fin - (1/2) A2_fin           [stage-6/7 npz]

GUARDS hosted here (each aborts; does NOT enter lnD):
  SECTOR-CUTOFF  the raw resolvent trace beyond s2_sector_max, extrapolated
                 by its measured power-law decay, must contribute less than
                 --sector-tail-tol to lnD (else raise --s2-sector-max);
  FICT-IDENTITY  the m2fict-collapse identity (exact, any m2fict), checked
                 at the production m2fict AND at every --fict-sweep multiple
                 (limit 1e-9 each; the target is m2fict-independent, so this
                 single guard subsumes any spread test):
                 I8fict_0 + I8fict_1 + T_sec8 - 5 ln(m2fict)
                   = -ln(w_neg2+s2_hi) - 4 ln(w_zero2+s2_hi)   to 1e-9;
  CT-QMAX        refuse an unconverged fish counterterm (adaptive q_max
                 ladder hit its ceiling without reaching rtol);
  LND-FINITE     abort on a non-finite lnD_ren before writing anything.

CROSS-CONSISTENCY ABORTS: (i) the two sector npz must share the identical
s2_sector_max (each sector MAY use its own grid); (ii) the bounce_sha of every input npz must
match the --bounce-npz file; (iii) mu of the two counterterm npz must
agree.

INPUTS   sector_n0/sector_n1/band_integrals/tail_highn/ct_tadpole/ct_fish
         _<tag>.npz + --bounce-npz (mbar2 for the default m2fict).
OUTPUT   D_integral_<tag>.npz (the filename and keys plots/
         read): lnD_ren, sector_block_resolvent, I8res_0/1,
         s2_sector_n0/n1, delta_sector_[sub_]n0/n1, sector_conv,
         sector_runtime_s, lam_neg/zm_cont, m2fict, fict_identity_max_dev,
         w_zero2_cont, fict_block_invariant, degeneracy_n1, band_n,
         band_I_inf, completion, band_T, band_R_shape,
         fit_a/c/e, tail, A3_analytic, baacke_c, fit_window,
         qmeter_max, A1_fin, A2_fin, eps_bounce, mu, s2_max, s2_sector_max,
         assembly.

BOUNCE ERROR: the stage-0 BOUNCE-ACCURACY watcher measures the error of the
bounce it kept against ANOTHER CosmoTransitions solve at different settings
(read that module on what the number does and does not cover), ALWAYS --
passing the zero-crossing condition does not exempt a bounce, because |s2_c| was
measured to under-report a CosmoTransitions-shaped profile error by ~10^3 (a
bounce can pass |s2_c| < 1e-8 by four decades and still carry 1.6e-3).
Stage 0 stores two terms already in lnD units and one relative:
       eps_bounce = |dA1_fin| + |dA2_fin|/2 + |band_sum * dA3/A3|
the tadpole and fish terms exact (no proportionality assumption), the band term
scaled here because band_sum is not known until this stage.  The n=0,1 sector
block is omitted: measured to move by <~1e-3, which is the sector quadrature's
own noise floor rather than a resolved profile effect.  Below
EPS_BOUNCE_NEGLIGIBLE the result is reported as negligible; otherwise the number
is printed next to lnD_ren, together with the zero-crossing verdict.
"""
import argparse
import os
import sys
import numpy as np

sys.dont_write_bytecode = True
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from pipeline_helpers_coupled_toy_model import (add_standard_cli, atomic_savez,        # noqa: E402
                                 bounce_sha256, load_bounce, stage_paths)
from pipeline_quadrature_coupled_toy_model import sector_tail_error           # noqa: E402


# The level below which the MEASURED bounce error is reported as negligible
# rather than as a number.  It is a threshold on the error ITSELF, in lnD
# units -- not on the |s2_c| proxy, which was measured to under-report a
# CosmoTransitions-shaped profile error by ~10^3 (see the stage-0 watcher).
# 1e-4 sits an order below the pipeline's smallest other budgeted term
# (SECTOR-CUTOFF ~ 1e-5..3e-4) and two below the band/tail terms (~7e-3).
EPS_BOUNCE_NEGLIGIBLE = 1e-4


# --------------------------------------------------------------------------- #
#  m2fict-collapse identity (Carosi fictitious-mode bookkeeping, closed form)  #
# --------------------------------------------------------------------------- #
def fict_block_terms(m2fict, s2_hi, lam_neg, lam_zm, deg1):
    """Closed-form m2fict pieces of the Carosi bookkeeping and their collapse.

    Arguments are the POLE-SUBTRACTION eigenvalues: lam_neg = w_neg2 (the
    n=0 negative-mode det crossing) and lam_zm = -w_zero2_cont (the n=1
    zero-mode eigenvalue, ~0).

    I8fict_n = -Int_0^{s2_hi} deg_n/(m2fict+s2) ds2
             = -deg_n * ln((m2fict+s2_hi)/m2fict)
    T_sec8   = ln((m2fict+s2_hi)/(lam_neg+s2_hi))
             + deg1*ln((m2fict+s2_hi)/(lam_zm+s2_hi))
    collapse identity (exact, any m2fict, any s2_hi):
      I8fict_0 + I8fict_1 + T_sec8 - 5 ln(m2fict)
             = -ln(lam_neg+s2_hi) - deg1 ln(lam_zm+s2_hi)    [the invariant]

    The invariant is the ONLY surviving constant (in the determinant
    picture it sits inside the reduced determinants).  Returns
    (fict_block, fict_block_target); the caller checks |diff| <= 1e-9
    (FICT-IDENTITY)."""
    nfict = 1.0 + deg1                       # n0 + nneg = 5 fictitious modes
    lgrat = np.log((m2fict + s2_hi) / m2fict)
    I8fict_tot = -nfict * lgrat
    T_sec8 = (np.log((m2fict + s2_hi) / (lam_neg + s2_hi))
              + deg1 * np.log((m2fict + s2_hi) / (lam_zm + s2_hi)))
    fict_block = I8fict_tot + T_sec8 - nfict * np.log(m2fict)
    fict_block_target = (-np.log(lam_neg + s2_hi)
                         - deg1 * np.log(lam_zm + s2_hi))
    return fict_block, fict_block_target


def _load_stage(path, stage_name):
    if not os.path.isfile(path):
        raise RuntimeError(f'[ABORT] stage output missing: {path} -- run '
                           f'{stage_name} first.')
    return np.load(path, allow_pickle=True)


def main():
    ap = argparse.ArgumentParser(
        description='coupled_toy_model stage 8: assemble lnD_ren from the stage npz')
    add_standard_cli(ap)
    ap.add_argument('--m2-fict', type=float, default=None,
                    help='fictitious mode mass^2 of the Carosi bookkeeping '
                         '(default: mbar2).  Never survives in lnD_ren: the '
                         'closed-form fict pieces collapse (guard '
                         'FICT-IDENTITY).')
    ap.add_argument('--fict-sweep', default='0.25,4.0,25.0',
                    help='comma-separated MULTIPLIERS of --m2-fict at which '
                         'the FICT-IDENTITY m2fict-collapse identity is '
                         'checked (in addition to --m2-fict itself)')
    ap.add_argument('--sector-tail-tol', type=float, default=1e-3,
                    help='SECTOR-CUTOFF guard: the estimated lnD truncation error '
                         'from cutting the raw resolvent trace at s2_sector_max '
                         '(sum over n=0,1 of Int_{s2_max}^inf delta_n ds2) must '
                         'not exceed this -- otherwise the cutoff is too low for '
                         'this potential; raise --s2-sector-max.  This is the '
                         'automatic replacement for a manual s2-cutoff sweep.')
    ap.add_argument('--out-npz', default=None,
                    help='output npz path (default: <data-dir>/'
                         'D_integral_<tag>.npz).  Point it elsewhere '
                         'for validation runs so production outputs are never '
                         'overwritten.')
    args = ap.parse_args()

    # ---- inputs: the six stage npz + the bounce ------------------------------
    paths = stage_paths(args.data_dir, args.tag)
    d0 = _load_stage(paths['sector_n0'], 'gbar_n0_coupled_toy_model')
    d1 = _load_stage(paths['sector_n1'], 'gbar_n1_coupled_toy_model')
    db = _load_stage(paths['band_integrals'], 'tail_s2_completion_coupled_toy_model')
    dt = _load_stage(paths['tail_highn'], 'tail_high_n_zeta_coupled_toy_model')
    dA1 = _load_stage(paths['ct_tadpole'], 'counterterm_tadpole_coupled_toy_model')
    dA2 = _load_stage(paths['ct_fish'], 'counterterm_fish_coupled_toy_model')
    bg = load_bounce(args.bounce_npz)
    mbar2 = bg['mbar2']

    # ---- cross-consistency aborts (metadata, before any physics) -------------
    cur_sha = bounce_sha256(args.bounce_npz)
    for name, d in (('sector_n0', d0), ('sector_n1', d1),
                    ('band_integrals', db), ('tail_highn', dt),
                    ('ct_tadpole', dA1), ('ct_fish', dA2)):
        sha = str(d['bounce_sha']) if 'bounce_sha' in d.files else ''
        if cur_sha and sha and sha != cur_sha:
            raise RuntimeError(
                f'[ABORT] {name} npz was computed against a DIFFERENT bounce '
                f'(bounce_sha {sha[:8]} != {cur_sha[:8]}) -- recompute the '
                f'stage or point --bounce-npz at the right file.')
    if float(d0['s2_sector_max']) != float(d1['s2_sector_max']):
        raise RuntimeError('[ABORT] the n=0 and n=1 sector npz use a different '
                           's2_sector_max -- the assembly combines the SCALAR '
                           'I8res_0 + I8res_1 at a common cutoff (fict/T_sec8 use '
                           's2_sector_max, not the grid), so only the cutoff must '
                           'match.  Each sector MAY converge on its own grid: the '
                           'n=0 negative-mode pole is often mid-range and uses a '
                           'pole-clustered grid, n=1 (pole ~ 0) a plain one.')
    if float(dA1['mu']) != float(dA2['mu']):
        raise RuntimeError(f"[ABORT] counterterm mu mismatch: tadpole "
                           f"{float(dA1['mu'])!r} vs fish {float(dA2['mu'])!r}"
                           f" -- recompute the counterterm stages.")

    # ---- unpack the sector scalars (no arithmetic on load) -------------------
    s2_sec = np.asarray(d0['s2_sector'], float)
    s2_sec_max = float(d0['s2_sector_max'])
    lam_neg_cont = float(d0['lam_neg_cont'])
    lam_zm_cont = float(d1['lam_zm_cont'])
    w_zero2_cont = float(d1['w_zero2_cont'])
    deg1 = float(d1['deg'])
    I8res = {0: float(d0['I8res']), 1: float(d1['I8res'])}
    sec_conv = {0: float(d0['conv_meter']), 1: float(d1['conv_meter'])}
    sector_runtime_s = float(d0['runtime_s']) + float(d1['runtime_s'])

    # ---- GUARD SECTOR-CUTOFF (aborts; does NOT enter lnD): is the sector -----
    # cutoff s2_sector_max high enough?
    # The closed-form fict + T_sec8 continue only the subtracted pole to
    # infinity; the physical raw trace delta_n beyond the cutoff is dropped, so
    # the residual lnD truncation is Int_{s2_max}^inf delta_n ds2 per wave
    # (the exact d(lnD)/d(s2_hi) = -delta_n(s2_hi)).  sector_tail_error reads the
    # stored raw trace of THIS potential and extrapolates its power-law tail --
    # the automatic, model-independent replacement for a manual s2-cutoff sweep.
    tail0, p0, eps0 = sector_tail_error(d0['s2_sector'], d0['delta_raw'])
    tail1, p1, eps1 = sector_tail_error(d1['s2_sector'], d1['delta_raw'])
    sec_tail_err = tail0 + tail1
    print(f"[SECTOR-CUTOFF] raw-trace cutoff at s2_sector_max = {s2_sec_max:g}: "
          f"|delta_0|={eps0:.2e} (decay s2^{p0:.2f}, tail {tail0:.2e}), "
          f"|delta_1|={eps1:.2e} (decay s2^{p1:.2f}, tail {tail1:.2e}); "
          f"lnD tail error {sec_tail_err:.2e} (limit {args.sector_tail_tol:g})")
    if sec_tail_err > args.sector_tail_tol:
        raise RuntimeError(
            f'[ABORT SECTOR-CUTOFF] the sector cutoff s2_sector_max = {s2_sec_max:g} '
            f'is too low for this potential: the raw resolvent trace has not '
            f'decayed (estimated lnD truncation {sec_tail_err:.2e} > '
            f'{args.sector_tail_tol:g}) -- rerun the sectors with a larger '
            f'--s2-sector-max (the fict/T_sec8 tail only carries the pole, not '
            f'the physical trace).')

    # ln|w_neg2| add-back at the CONTINUUM det-zero crossing -- the SAME
    # eigenvalue the production pole subtraction uses (one convention,
    # one continuum tier).
    lnw = float(np.log(abs(lam_neg_cont)))

    # ---- PRODUCTION sector block: closed-form fict pieces --------------------
    m2fict_prod = float(args.m2_fict) if args.m2_fict is not None else mbar2
    if m2fict_prod <= 0.0:
        raise RuntimeError('[ABORT] --m2-fict must be > 0.')
    fb_prod, fict_block_target = fict_block_terms(
        m2fict_prod, s2_sec_max, lam_neg_cont, lam_zm_cont, deg1)
    sector_block_resolvent = float(I8res[0] + I8res[1] + fb_prod + lnw)
    print(f"[SECTOR] PRODUCTION sector block (continuum RESOLVENT "
          f"s2-integrals, Carosi pole subtraction):")
    print(f"       grid: geometric [{float(d0['sector_s2_lo']):g}, "
          f"{s2_sec_max:g}], {int(d0['sector_nodes'])} nodes x 2 waves; "
          f"sector runtime {sector_runtime_s:.1f} s")
    print(f"       lam_neg_cont = {lam_neg_cont:+.10e}   "
          f"lam_zm_cont = {lam_zm_cont:+.6e}   (pole-subtraction "
          f"eigenvalues, continuum crossings)")
    print(f"       I8res_0 = {I8res[0]:+.6f} (SECTOR-STEP meter "
          f"{sec_conv[0]:.2e})   I8res_1 = {I8res[1]:+.6f} (SECTOR-STEP meter "
          f"{sec_conv[1]:.2e}; limit {float(d0['conv_tol']):g})")
    print(f"       fict block(m2fict={m2fict_prod:.6g}) = "
          f"{fb_prod:+.9f}   ln|w_neg2| = {lnw:+.6f}")
    print(f"       sector_block_resolvent = I8res_0 + I8res_1 + fict + "
          f"ln|w_neg2| = {sector_block_resolvent:+.6f}")

    # ---- GUARD FICT-IDENTITY (aborts; does NOT enter lnD): m2fict-collapse ---
    # identity (exact, any m2fict), checked at the production m2fict AND at
    # every --fict-sweep multiple.  The target is m2fict-independent, so this
    # ONE guard subsumes any spread test across the sweep: nothing
    # m2fict-dependent can reach lnD_ren.
    sweep_mults = [float(x) for x in args.fict_sweep.split(',') if x.strip()]
    sweep_vals = [m2fict_prod] + [m2fict_prod * m for m in sweep_mults]
    nfict = 1.0 + deg1
    fict_identity_max_dev = max(
        abs(fict_block_terms(mf, s2_sec_max, lam_neg_cont, lam_zm_cont,
                             deg1)[0] - fict_block_target)
        for mf in sweep_vals)
    print(f'[FICT-IDENTITY] m2fict-collapse identity at m2fict = '
          f'{[f"{v:.4g}" for v in sweep_vals]}: max |I8fict_tot + T_sec8 - '
          f'{nfict:g} ln(m2fict) - target| = {fict_identity_max_dev:.3e}   '
          f'target -ln(lam_neg+s2_sec_max) - {deg1:g} ln(lam_zm+s2_sec_max) = '
          f'{fict_block_target:+.9f}   (limit 1e-9)')
    if fict_identity_max_dev > 1e-9:
        raise RuntimeError(
            f'[ABORT FICT-IDENTITY] m2fict-collapse identity off by '
            f'{fict_identity_max_dev:.3e} > 1e-9 -- the closed-form '
            f'fictitious-mode bookkeeping is broken.')

    # ---- band + tail + counterterms (loaded, no arithmetic on load) ----------
    ns = np.asarray(db['band_n'])
    I_inf = np.asarray(db['I_inf'], float)
    fit_window = np.asarray(dt['fit_window'], int)
    fit_hi = int(fit_window[1])
    a, c, e = float(dt['fit_a']), float(dt['fit_c']), float(dt['fit_e'])
    tail = float(dt['tail'])
    A3 = float(dt['A3_analytic'])
    baacke_c = float(dt['baacke_c'])
    A1_fin = float(dA1['A1_fin'])
    A2_fin = float(dA2['A2_fin'])
    mu = float(dA1['mu'])
    # The fit window must lie inside the computed band: otherwise waves in
    # (band_top, fit_hi] fall between band_sum (n <= fit_hi) and the zeta tail
    # (n > fit_hi) and are silently dropped.
    if fit_hi > int(ns.max()):
        raise RuntimeError(
            f'[ABORT] fit_hi={fit_hi} exceeds the band top n={int(ns.max())}: '
            f'waves ({int(ns.max())}, {fit_hi}] would be dropped between the '
            f'band sum and the zeta tail.')
    # GUARD CT-QMAX (aborts; does NOT enter lnD): refuse an UNCONVERGED fish
    # counterterm (A2 hit its q_max ceiling without reaching rtol) -- it must
    # not flow into lnD_ren silently.
    if 'converged' in dA2.files and not bool(dA2['converged']):
        raise RuntimeError(
            '[ABORT CT-QMAX] A2_fin (fish counterterm) did not converge '
            '(q_max ceiling reached without rtol) -- rerun counterterm_fish '
            'with a larger q_max ceiling or a looser tol.')
    band_sum = I_inf[ns <= fit_hi].sum()

    # ---- assembly: continuum RESOLVENT sectors + band + tail + counterterms --
    lnD_ren = sector_block_resolvent + band_sum + tail + A1_fin - 0.5 * A2_fin
    # GUARD LND-FINITE (aborts; does NOT enter lnD): the assembled number
    # must be finite before anything is printed or written.
    if not np.isfinite(lnD_ren):
        raise RuntimeError('[ABORT LND-FINITE] non-finite lnD_ren.')

    # ---- final table ----------------------------------------------------------
    print("=" * 72)
    print(f"  completion                 = {str(db['completion'])} "
          f"(pure Method I band)")
    print(f"  convention                 = Carosi FULL-FV (standard "
          f"full-FV assembly)")
    print(f"  sectors                    = CONTINUUM RESOLVENT s2-integrals "
          f"(production; e_b = 0)")
    print(f"  I8res_0                    = {I8res[0]:+.6f}")
    print(f"  I8res_1 (deg {deg1:g})          = {I8res[1]:+.6f}")
    print(f"  sector_block_resolvent     = {sector_block_resolvent:+.6f}")
    print(f"  sum I_n(inf), n=2..{fit_hi:<4d}   = {band_sum:+.4f}")
    print(f"  tail (odd family)          = {tail:+.4f}")
    print(f"  A1_fin                     = {A1_fin:+.4f}")
    print(f"  -A2_fin/2                  = {-0.5*A2_fin:+.4f}")
    print(f"  lnD_ren (RESOLVENT sectors)= {lnD_ren:+.4f}")

    # ---- the stage-0 BOUNCE-ACCURACY verdict, reported next to lnD_ren -------
    # TWO OUTCOMES, and only one of them carries a number.
    #
    # (a) THE ZERO-CROSSING CONDITION IS SATISFIED (|s2_c| < bounce_trust).  The
    #     four n=1 modes are the translation modes d_mu phi_b, exact zero modes
    #     of the fluctuation operator for any true solution of the bounce
    #     equation, so |s2_c| is the bounce error measured against a KNOWN exact
    #     answer.  Below the trust level that error does not move lnD_ren
    #     measurably, so there is nothing to quote: stage 0 does not even
    #     compute the moment changes (it stores 0), and the line printed here is
    #     just "bounce error negligible".
    #
    # (b) THE BOUNCE COULD NOT BE IMPROVED PAST THE CONDITION.  Then stage 0 has
    #     already exhausted its only route -- the CosmoTransitions escalation
    #     ladder; this pipeline runs the purely-CosmoTransitions bounce and does
    #     not refine it further -- and stored the self-convergence change of the
    #     two profile moments a bounce error can actually reach lnD_ren through:
    #       M1 -- the tadpole moment.  A1_fin is a FIXED-WEIGHT LINEAR
    #             functional of the insertion U, so dA1_fin/A1_fin = dM1/M1
    #             exactly, and the tadpole dominates the bounce error;
    #       A3 -- the band's leading 1/nu^3 coefficient, so d(band) =
    #             band_sum * dA3/A3.
    #     eps_bounce = |A1_fin * dM1_rel| + |band_sum * dA3_rel| .
    #     The fish term |(A2_fin/2) * dA2_rel| is OMITTED on purpose: A2_fin
    #     moved by 5e-6 RELATIVE in the measured ladder re-solve, far below every
    #     other error term of the assembly.
    eps_bounce = float('nan')
    with np.load(args.bounce_npz, allow_pickle=True) as bnpz:
        measured = 'bounce_dA1_abs' in bnpz.files
        if measured:
            bounce_s2c = float(bnpz['bounce_s2c'])
            bounce_ok = bool(bnpz['bounce_zero_crossing_ok'])
            bounce_trust = float(bnpz['bounce_trust'])
            bounce_method = str(bnpz['bounce_method'])
            dA1_abs = float(bnpz['bounce_dA1_abs'])
            dA2_abs = float(bnpz['bounce_dA2_abs'])
            dA3_rel = float(bnpz['bounce_dA3_rel'])
    if not measured:
        print("bounce error was not measured: this bounce npz carries no "
              "BOUNCE-ACCURACY verdict (it predates the current verdict "
              "schema, so it records no trust level and no pass/fail can be "
              "reconstructed from it) -- re-run stage 0 on it to get one")
    elif not (np.isfinite(dA1_abs) and np.isfinite(dA2_abs)
              and np.isfinite(dA3_rel)):
        # stage 0 could not build a reference profile, so there is no number to
        # quote -- say that, rather than printing a NaN or, worse, a zero.
        print("bounce error could not be quantified: the stage-0 reference "
              "solve failed for this bounce")
        print(f"zero crossing condition satisfied: "
              f"{'yes' if bounce_ok else 'no'}")
        print(f"zero crossing in n=1 sector is: {bounce_s2c:.3e}  "
              f"(condition |s2_c| < {bounce_trust:g}; {bounce_method})")
    else:
        band_term = abs(band_sum * dA3_rel)
        eps_bounce = dA1_abs + dA2_abs + band_term
        if eps_bounce < EPS_BOUNCE_NEGLIGIBLE:
            print(f"bounce error negligible ({eps_bounce:.2e} < "
                  f"{EPS_BOUNCE_NEGLIGIBLE:g})")
        else:
            print(f"bounce error is: {eps_bounce:.6f}")
        print(f"zero crossing condition satisfied: "
              f"{'yes' if bounce_ok else 'no'}")
        print(f"zero crossing in n=1 sector is: {bounce_s2c:.3e}  "
              f"(condition |s2_c| < {bounce_trust:g}; {bounce_method})")
        print(f"   bounce error terms: tadpole |dA1_fin| = {dA1_abs:.3e}, "
              f"fish |dA2_fin|/2 = {dA2_abs:.3e}, band |sum I_n * dA3/A3| = "
              f"{band_term:.3e}   (n=0,1 sector block omitted: measured to move "
              f"by <~1e-3, at the sector quadrature's own noise floor)")
    print("=" * 72)

    out = args.out_npz or paths['D_integral']
    atomic_savez(out,
             lnD_ren=lnD_ren,
             sector_block_resolvent=sector_block_resolvent,
             I8res_0=I8res[0], I8res_1=I8res[1],
             s2_sector_n0=np.asarray(d0['s2_sector'], float),
             s2_sector_n1=np.asarray(d1['s2_sector'], float),
             delta_sector_n0=np.asarray(d0['delta_raw'], float),
             delta_sector_n1=np.asarray(d1['delta_raw'], float),
             delta_sector_sub_n0=np.asarray(d0['delta_sub'], float),
             delta_sector_sub_n1=np.asarray(d1['delta_sub'], float),
             sector_conv=np.array([sec_conv[0], sec_conv[1]]),
             sector_runtime_s=sector_runtime_s,
             lam_neg_cont=lam_neg_cont, lam_zm_cont=lam_zm_cont,
             m2fict=m2fict_prod,
             fict_identity_max_dev=fict_identity_max_dev,
             w_zero2_cont=w_zero2_cont,
             fict_block_invariant=fict_block_target,
             degeneracy_n1=deg1,
             band_n=ns, band_I_inf=I_inf,
             completion=str(db['completion']),
             band_T=np.asarray(db['T'], float),
             band_R_shape=np.asarray(db['R_shape'], float),
             fit_a=a, fit_c=c, fit_e=e, tail=tail, A3_analytic=A3, baacke_c=baacke_c,
             fit_window=fit_window,
             qmeter_max=float(db['qmeter_max']),
             A1_fin=A1_fin, A2_fin=A2_fin, eps_bounce=eps_bounce,
             mu=mu, s2_max=float(db['s2_max']), s2_sector_max=s2_sec_max,
             assembly='coupled_toy_model modular: PRODUCTION n=0,1 sectors from '
                      'continuum RESOLVENT s2-integrals (gbar_n0/gbar_n1_'
                      'coupled_toy_model: Carosi FULL-FV, pole subtraction at the '
                      'numerical continuum eigenvalues; analytic Bessel FV '
                      'via fv_analytic_coupled_toy_model (analytic FV, no lattice); '
                      'log-Simpson s2-quadrature in pipeline_quadrature_coupled_toy_model) '
                      '+ Method-I calibrated band completion (tail_s2_'
                      'completion_coupled_toy_model) + odd zeta tail (tail_high_n_'
                      'zeta_coupled_toy_model) + A1/A2 counterterms (counterterm_'
                      'tadpole/fish_coupled_toy_model), assembled with the closed-'
                      'form fict/T_sec8 pieces + ln|w_neg2| here '
                      '(assemble_lnD_coupled_toy_model)')
    print(f"[OK] wrote {out}")


if __name__ == '__main__':
    main()
