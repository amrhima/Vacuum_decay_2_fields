#!/usr/bin/env python3
"""plot_bounces_coupled_toy_model.py -- plot the bounce solutions of the
hardcoded toy potential and MARK THE ONE THE PIPELINE ACTUALLY USES.

WHY THIS FILE EXISTS
--------------------
The general pipeline had an interactive gallery: it solved every downward
vacuum pair it had discovered, plotted them all, and asked the user which one
to run.  This pipeline has no search and no choice -- the bounce is hardcoded
in potential_coupled_toy_model.MODELS -- so the selection machinery is gone and
only the PICTURE is kept.  Its job is to show what the other bounce solutions
of the same potential look like and to make unmistakably clear which one the
production run computes ln D_ren for.

WHAT IT DRAWS
-------------
For the chosen model (default the coupled F2_T0 toy potential):

  overview_<model>.png    one figure with three panels --
      (1) the field components x(r), y(r) of EVERY downward pair.  The pair the
          pipeline uses is drawn thick and in colour and carries the
          "USED BY THE PIPELINE" tag; the others are thin and grey.
      (2) the potential landscape as a filled contour map, with every bounce
          path drawn on it and every vacuum labelled M0..M3 (the false and true
          endpoints of the USED bounce in red and blue).
      (3) the potential surface V(x,y) in 3-D with the used bounce path lifted
          onto it.
  bounce_F<i>_T<j>.png    one figure per solution: its own components-vs-r
          panel and its own path over the landscape, titled with the action S
          and stating in the title whether it is the used bounce.

WHERE THE PROFILES COME FROM
----------------------------
For each downward pair, in order: the canonical profile shipped next to this
file (bounce_data_F<i>_T<j>.npz), else a cached solve in the output directory,
else a fresh CosmoTransitions solve through the SAME stage-0 entry point
(bounce_coupled_toy_model.solve_bounce) -- so a plotted bounce is solved by
exactly the production code, with the production settings, coupled or
decoupled.  --no-solve draws only what already exists.

NOTHING HERE ENTERS ln D.  This is presentation only.

USAGE (from the pipeline root)
  ../env_G_project/bin/python computation/plot_bounces_coupled_toy_model.py
  ... --model bdet                # the decoupled companion instead
  ... --out-dir some/where        # default: G_project/pictures, else ./plots
  ... --no-solve                  # only plot profiles that already exist

A profile solved for a picture is cached in <pipeline>/plots/bounce_cache, not
in the output directory (which may be the shared G_project/pictures folder).
"""
import argparse
import os
import sys

import numpy as np

sys.dont_write_bytecode = True
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

os.environ.setdefault('MPLBACKEND', 'Agg')

from potential_coupled_toy_model import (                                # noqa: E402
    FIELDS,
    MODEL_DEFAULT,
    CTShiftedLiftedPotential,
    V_numeric,
    guard_vacua,
    model_names,
    tunneling_fields,
)
from bounce_coupled_toy_model import (bounce_npz_payload, solve_bounce)  # noqa: E402
from pipeline_helpers_coupled_toy_model import atomic_savez              # noqa: E402


def default_out_dir():
    """Where the PNGs go: the project-wide G_project/pictures folder when it
    exists (the standing convention for plots in this project), otherwise a
    local plots/ folder next to the pipeline."""
    root = os.path.dirname(os.path.dirname(_HERE))          # .../G_project
    pics = os.path.join(root, 'pictures')
    if os.path.isdir(pics):
        return pics
    return os.path.join(os.path.dirname(_HERE), 'plots')


def cache_dir():
    """Where a solved-for-the-picture profile is cached.  Deliberately NOT the
    output directory: that may be the shared G_project/pictures folder, which
    holds pictures and must not collect npz.  The cache lives with the pipeline
    (plots/bounce_cache), so a second run of this script redraws instantly
    instead of re-solving."""
    return os.path.join(os.path.dirname(_HERE), 'plots', 'bounce_cache')


def load_or_solve(model, pair, allow_solve=True):
    """The (R, Phi_orig, S) of one downward pair: the shipped canonical npz if
    it belongs to THIS model, else a cached solve in cache_dir(), else a fresh
    CosmoTransitions solve through the production stage-0 entry point.
    Returns None when nothing exists and solving is switched off."""
    iF, iT = pair
    params = model['params']

    shipped = os.path.join(_HERE, f'bounce_data_F{iF}_T{iT}.npz')
    cached = os.path.join(cache_dir(),
                          f'bounce_cache_{model["name"]}_F{iF}_T{iT}.npz')
    for path in (shipped, cached):
        if not os.path.isfile(path):
            continue
        d = np.load(path, allow_pickle=True)
        if not np.allclose(np.asarray(d['params'], float), params,
                           rtol=0, atol=1e-12):
            d.close()
            continue                       # solved for the other model
        R = np.asarray(d['R'], float)
        Phi = (np.asarray(d['Phi_bounce_orig'], float)
               if 'Phi_bounce_orig' in d.files else
               np.stack([np.asarray(d['X_bounce_orig'], float),
                         np.asarray(d['Y_bounce_orig'], float)], axis=1))
        S = float(d['S_CT'])
        d.close()
        print(f'  F{iF}_T{iT}: read {os.path.basename(path)}  (S = {S:.4f})')
        return R, Phi, S

    if not allow_solve:
        print(f'  F{iF}_T{iT}: no stored profile and --no-solve -- skipped')
        return None

    false_vac, true_vac = model['minima'][iF], model['minima'][iT]
    print(f'  F{iF}_T{iT}: solving with CosmoTransitions '
          f'({"decoupled 1-D instanton" if model["decoupled"] else "path deformation"})')
    pot_prime = CTShiftedLiftedPotential(params, false_vac)
    try:
        (tv, fv, R, Phi_prime, Phi_orig, S, solver) = solve_bounce(
            pot_prime, false_vac, true_vac, params, f'F{iF}_T{iT}')
    except Exception as exc:
        # A pair CosmoTransitions cannot connect is a picture missing from the
        # gallery, not a reason to lose the rest of it.
        print(f'  F{iF}_T{iT}: SOLVE FAILED ({type(exc).__name__}: {exc}) '
              f'-- omitted from the gallery')
        return None
    os.makedirs(cache_dir(), exist_ok=True)
    atomic_savez(cached, **bounce_npz_payload(
        R, Phi_prime, Phi_orig, params, fv, tv, S, iF, iT,
        f'plot_cache_F{iF}_T{iT}', solver=solver, mass_basis_L=pot_prime.L))
    print(f'  F{iF}_T{iT}: solved (S = {S:.4f}), cached as '
          f'{os.path.basename(cached)}')
    return R, Phi_orig, S


def _landscape(model, paths, npts=180):
    """A (GX, GY, GV) mesh of the potential covering every vacuum and every
    plotted path, with a margin."""
    pts = [model['minima']] + [p for p in paths]
    allpts = np.vstack(pts)
    lo, hi = allpts.min(axis=0), allpts.max(axis=0)
    # generous margin: the vacuum labels are drawn as offset text and would be
    # clipped at the frame for a vacuum sitting near the edge of the data
    pad = 0.32 * (hi - lo + 1e-12)
    gx = np.linspace(lo[0] - pad[0], hi[0] + pad[0], npts)
    gy = np.linspace(lo[1] - pad[1], hi[1] + pad[1], npts)
    GX, GY = np.meshgrid(gx, gy)
    GV = V_numeric(np.stack([GX, GY], axis=-1), model['params'])
    return GX, GY, GV


def _mark_vacua(ax, model, iF, iT, colour_other='white', fontsize=8):
    """Label every hardcoded vacuum: the USED bounce's false endpoint in red,
    its true endpoint in blue, the rest as plain markers.  The labels are the
    vacuum INDICES M0..M<n-1> -- the same indices the bounce file names
    F<i>_T<j> use, so a plot and a file name always refer to the same vacuum."""
    box = dict(boxstyle='round,pad=0.18', fc='black', ec='none', alpha=0.45)
    # place each label on the side that points INTO the plot, so a vacuum near
    # the frame does not have its text run off the edge
    mid = model['minima'].mean(axis=0)

    def _off(pt):
        return (9 if pt[0] <= mid[0] else -9, 7 if pt[1] <= mid[1] else -7)

    def _ha(pt):
        return 'left' if pt[0] <= mid[0] else 'right'

    for k, pt in enumerate(model['minima']):
        if k == iF:
            ax.scatter(*pt, c='red', marker='o', s=55, zorder=6,
                       edgecolors='k', linewidths=0.5)
            ax.annotate(f'M{k} FALSE', pt, textcoords='offset points',
                        xytext=_off(pt), ha=_ha(pt), color='red', fontsize=9,
                        weight='bold', zorder=7, bbox=box)
        elif k == iT:
            ax.scatter(*pt, c='deepskyblue', marker='X', s=70, zorder=6,
                       edgecolors='k', linewidths=0.5)
            ax.annotate(f'M{k} TRUE', pt, textcoords='offset points',
                        xytext=_off(pt), ha=_ha(pt), color='deepskyblue',
                        fontsize=9, weight='bold', zorder=7, bbox=box)
        else:
            ax.scatter(*pt, c=colour_other, marker='.', s=30, zorder=5)
            ax.annotate(f'M{k}', pt, textcoords='offset points',
                        xytext=_off(pt), ha=_ha(pt), color=colour_other,
                        fontsize=fontsize, zorder=7)


def plot_gallery(model, sols, out_dir):
    """The overview figure: every solved bounce of this potential, with the one
    the pipeline uses drawn thick, coloured and tagged."""
    import matplotlib.pyplot as plt

    used = model['pair']
    GX, GY, GV = _landscape(model, [s[1] for s in sols.values()])

    fig = plt.figure(figsize=(17.5, 5.0))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')

    # -- panel 1: field components vs r -------------------------------------
    for pair, (R, Phi, S) in sorted(sols.items()):
        is_used = (pair == used)
        lab = f'F{pair[0]}_T{pair[1]}  S={S:.2f}'
        if is_used:
            ax1.plot(R, Phi[:, 0], '-', color='crimson', lw=2.4,
                     label=f'{lab}   <-- USED  [{FIELDS[0]}]')
            ax1.plot(R, Phi[:, 1], '--', color='darkorange', lw=2.4,
                     label=f'{lab}   <-- USED  [{FIELDS[1]}]')
        else:
            ax1.plot(R, Phi[:, 0], '-', color='0.62', lw=1.0)
            ax1.plot(R, Phi[:, 1], '--', color='0.62', lw=1.0,
                     label=f'{lab}   (not used)')
    ax1.set_xlabel(r'Euclidean radius $r$')
    ax1.set_ylabel('field value (original frame)')
    ax1.set_title(f'all bounce solutions of the {model["name"]} potential\n'
                  f'(solid = {FIELDS[0]}, dashed = {FIELDS[1]})')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=7, loc='best')

    # -- panel 2: the landscape with every path -----------------------------
    cf = ax2.contourf(GX, GY, GV, levels=30, cmap='viridis', alpha=0.8)
    ax2.contour(GX, GY, GV, levels=30, colors='k', linewidths=0.3, alpha=0.35)
    fig.colorbar(cf, ax=ax2, label=rf'$V({FIELDS[0]},{FIELDS[1]})$')
    for pair, (R, Phi, S) in sorted(sols.items()):
        if pair == used:
            ax2.plot(Phi[:, 0], Phi[:, 1], '-', color='crimson', lw=3.0,
                     zorder=4, label=f'F{pair[0]}_T{pair[1]}  <-- USED')
        else:
            ax2.plot(Phi[:, 0], Phi[:, 1], '-', color='white', lw=1.1,
                     alpha=0.75, zorder=3,
                     label=f'F{pair[0]}_T{pair[1]}')
    _mark_vacua(ax2, model, used[0], used[1])
    ax2.set_xlabel(f'field {FIELDS[0]}')
    ax2.set_ylabel(f'field {FIELDS[1]}')
    ax2.set_title('bounce paths over the potential landscape')
    ax2.legend(fontsize=7, loc='best', framealpha=0.6)

    # -- panel 3: the surface with the used path lifted onto it -------------
    ax3.plot_surface(GX, GY, GV, cmap='viridis', alpha=0.5, linewidth=0,
                     antialiased=True)
    if used in sols:
        R, Phi, S = sols[used]
        Vpath = np.asarray(V_numeric(Phi, model['params']), float)
        ax3.plot(Phi[:, 0], Phi[:, 1], Vpath, '-', color='crimson', lw=2.6)
    for k, pt in enumerate(model['minima']):
        Vv = float(model['V_min'][k])
        col = ('red' if k == used[0] else
               'deepskyblue' if k == used[1] else 'gray')
        ax3.scatter(pt[0], pt[1], Vv, c=col, marker='o')
        ax3.text(pt[0], pt[1], Vv, f'  M{k}', color=col, fontsize=8)
    ax3.set_xlabel(f'field {FIELDS[0]}')
    ax3.set_ylabel(f'field {FIELDS[1]}')
    ax3.set_zlabel(rf'$V({FIELDS[0]},{FIELDS[1]})$')
    ax3.set_title('potential surface + the USED bounce path (red)')

    iF, iT = used
    kind = 'DECOUPLED' if model['decoupled'] else 'COUPLED'
    moving = [FIELDS[i] for i in tunneling_fields(model['minima'][iF],
                                                  model['minima'][iT])]
    fig.suptitle(
        f'coupled_toy_model -- potential "{model["name"]}" ({kind}).   '
        f'THE PIPELINE COMPUTES ln D_ren FOR  F{iF}_T{iT}  '
        f'(M{iF} -> M{iT}, moving field(s) {moving}, '
        f'S = {sols[used][2]:.4f})' if used in sols else
        f'coupled_toy_model -- potential "{model["name"]}" ({kind})',
        fontsize=12, weight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    png = os.path.join(out_dir, f'overview_{model["name"]}.png')
    fig.savefig(png, dpi=120)
    plt.close(fig)
    print(f'[plot] wrote {png}')


def plot_one(model, pair, sol, out_dir):
    """One figure for one bounce solution: components vs r, and its path over
    the landscape.  The title says whether this is the bounce the pipeline
    uses."""
    import matplotlib.pyplot as plt

    R, Phi, S = sol
    iF, iT = pair
    is_used = (pair == model['pair'])
    GX, GY, GV = _landscape(model, [Phi])

    fig = plt.figure(figsize=(11.5, 4.6))
    ax1 = fig.add_subplot(1, 2, 1)
    for i, f in enumerate(FIELDS):
        ax1.plot(R, Phi[:, i], label=rf'${f}(r)$')
    ax1.set_xlabel(r'Euclidean radius $r$')
    ax1.set_ylabel('field value (original frame)')
    ax1.set_title(f'F{iF}_T{iT}:  S = {S:.4f}')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = fig.add_subplot(1, 2, 2)
    cf = ax2.contourf(GX, GY, GV, levels=30, cmap='viridis', alpha=0.8)
    ax2.contour(GX, GY, GV, levels=30, colors='k', linewidths=0.3, alpha=0.35)
    fig.colorbar(cf, ax=ax2, label=rf'$V({FIELDS[0]},{FIELDS[1]})$')
    ax2.plot(Phi[:, 0], Phi[:, 1], '-',
             color='crimson' if is_used else 'white', lw=2.4)
    _mark_vacua(ax2, model, iF, iT)
    ax2.set_xlabel(f'field {FIELDS[0]}')
    ax2.set_ylabel(f'field {FIELDS[1]}')
    ax2.set_title('bounce path over the potential landscape')

    tag = ('THIS IS THE BOUNCE THE PIPELINE USES'
           if is_used else 'not used by the pipeline')
    fig.suptitle(f'{model["name"]}  F{iF}_T{iT}  --  {tag}', fontsize=12,
                 weight='bold', color='crimson' if is_used else '0.35')
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    png = os.path.join(out_dir, f'bounce_{model["name"]}_F{iF}_T{iT}.png')
    fig.savefig(png, dpi=120)
    plt.close(fig)
    print(f'[plot] wrote {png}')


def main():
    ap = argparse.ArgumentParser(
        description='plot the bounce solutions of the hardcoded toy potential '
                    'and mark the one the pipeline uses')
    ap.add_argument('--model', default=MODEL_DEFAULT, choices=model_names(),
                    help='which hardcoded potential to draw')
    ap.add_argument('--out-dir', default=None,
                    help='where the PNGs go (default: G_project/pictures if '
                         'it exists, else <pipeline>/plots)')
    ap.add_argument('--no-solve', action='store_true',
                    help='plot only bounces that already exist as npz; do not '
                         'call CosmoTransitions for the missing ones')
    args = ap.parse_args()

    model = guard_vacua(args.model)
    out_dir = args.out_dir or default_out_dir()
    os.makedirs(out_dir, exist_ok=True)

    iF, iT = model['pair']
    print(f'model {model["name"]} '
          f'({"DECOUPLED" if model["decoupled"] else "COUPLED"}); '
          f'downward pairs {["F%d_T%d" % p for p in model["pairs"]]}; '
          f'the pipeline uses F{iF}_T{iT}')
    print(f'output directory: {out_dir}')

    sols = {}
    for pair in model['pairs']:
        got = load_or_solve(model, pair, allow_solve=not args.no_solve)
        if got is not None:
            sols[pair] = got
    if not sols:
        raise SystemExit('[ABORT] no bounce solution could be obtained -- '
                         'nothing to plot.')
    if model['pair'] not in sols:
        print(f'[plot] WARNING: the bounce the pipeline uses (F{iF}_T{iT}) '
              f'could not be obtained, so it cannot be marked in the overview')

    for pair, sol in sorted(sols.items()):
        plot_one(model, pair, sol, out_dir)
    plot_gallery(model, sols, out_dir)
    print(f'[plot] done: {len(sols)} bounce solution(s) drawn; '
          f'F{iF}_T{iT} is the one the pipeline computes ln D_ren for')


if __name__ == '__main__':
    main()
