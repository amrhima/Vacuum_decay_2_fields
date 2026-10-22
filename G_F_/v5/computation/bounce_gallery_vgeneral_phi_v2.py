#!/usr/bin/env python3
"""bounce_gallery_vgeneral_phi_v2.py -- level 1, part of stage 0
(bounce_vgeneral_phi_v2): ALL bounce plotting + the interactive selection
menu.  PRESENTATION only; the physics lives in bounce_vgeneral_phi_v2,
nothing here enters lnD.

READS   every bounce_data_F<i>_T<j>.npz of <data-dir>/bounce_gen (keys R,
        phi_b, phi_b_orig, S_E, mass_basis_L, false_vac,
        true_vac and the census_* keys), and `minima`, the
        vacuum catalogue of the run (bounce_vgeneral_phi_v2.minima_catalogue:
        main's one Newton sweep, or the plot stage's own call; handed in,
        never recomputed here)
WRITES  bounce_gallery: the pictures of every solved bounce in
        <data-dir>/gallery/ (bounce_<name>_profiles.png, plus
        bounce_<name>_potential.png for N = 2, 3), and the menu rows (name,
        S_E, components changed, census note); select_bounce_interactive
        prints those rows and returns the chosen (iF, iT) to
        bounce_vgeneral_phi_v2.main.  plot_bounce, THE ONE RENDERER, is also
        what the plot stage plot_computation/plot_bounce_gallery_vgeneral_phi_v2
        imports (no twin).

HOW (plot_bounce, two pictures per bounce):
  * PROFILES picture: the bounce profiles phi'(r) of the bounce from the
    true vacuum to the false vacuum, in the primed fieldspace frame -- every
    component decays towards 0 = the false vacuum at the origin (the frame
    the bounce is solved and consumed in); the y axis is the primed
    field-space vector phibar'(r).  The legend sits OUTSIDE the axes (to
    the right, at the top), so the curves stay clear; it holds only the
    tree-level action S_E (its left-aligned title) and one colour per primed
    component phi'_i(r), named by its channel number (phi'_1 = the
    heaviest false-vacuum mass channel).
  * POTENTIAL picture (N=2, N=3): the bounce path (red) from the true vacuum to
    the false vacuum, in the original fieldspace frame -- that is where the
    vacua and their labels live; the declared fields are named phi_1 .. phi_N
    in their declared order, and every vacuum is drawn in blue (the bounce's
    false and true one 'o', the other minima '.').  N=2: the potential SURFACE
    V with the path (red) lifted onto it, next to the LANDSCAPE, the inverted
    potential -V (every minimum of V a hilltop) as a contour map with the path
    (red) seen top-down.  N=3: the path in field space, V as its colour.  No
    panel titles.  V is drawn without its constant term c (params_no_const),
    the zero of energy of the catalogue's vacuum values.
  * N=1: profiles + the two vacua as primed levels; N>3: profiles only.

Vacuum labels: every vacuum is named by its catalogue label TV0 / M<i> / FV0
(bounce_vgeneral_phi_v2.vacuum_label), the same label in every picture of
every bounce, resolved from the stored vacuum COORDINATES against the current
catalogue (bounce_vgeneral_phi_v2.catalogue_indices -- one owner).

CENSUS: the census eigenvalues omega^2 = -s2 (the negative modes at n=0,
the translation zero mode and every bound state at n=1) are printed on the
terminal of every run (mode_census_vgeneral_phi_v2.print_census_inventory),
not in the pictures.  The pictures carry the census VERDICT (census_verdict,
from the bounce npz): a SPECIAL bounce (census not the accepted index-1 one)
is drawn DASHED with a [SPECIAL] title mark (the only title of both
pictures) and a crimson legend frame; it
is plotted and listed, but selecting it for lnD aborts in the selection gate
of bounce_vgeneral_phi_v2 (assert_census_accepted).
"""
import glob
import os
import shutil
import subprocess
import sys

import numpy as np

sys.dont_write_bytecode = True
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from potential_vgeneral_phi_v2 import (
    V_numeric,
    params_no_const,
)
from mode_census_vgeneral_phi_v2 import census_violations, stored_census

from bounce_vgeneral_phi_v2 import (
    bounce_display_name,
    catalogue_indices,
    pair_from_name,
    vacuum_label,
)


def census_verdict(b):
    """The census verdict of a bounce npz dict, from its STORED census
    (mode_census_vgeneral_phi_v2.stored_census reads it): 'ACCEPT',
    'SPECIAL', or 'OLDER SCHEMA' for a file written before the current
    census schema (its pictures are drawn; re-solving the pair refreshes
    it)."""
    census = stored_census(b)
    if census is None:
        return 'OLDER SCHEMA'
    return 'ACCEPT' if census['census_accept'] else 'SPECIAL'


def _census_note(b, verdict):
    """Short census tag for the menu row, from the verdict plot_bounce
    returned for this bounce: '' for an accepted index-1 bounce, else the
    evidence lines of the selection gate (mode_census.census_violations --
    the one owner of what makes a bounce SPECIAL)."""
    if verdict == 'OLDER SCHEMA':
        return 'older census schema'
    if verdict == 'ACCEPT':
        return ''
    # census_violations names what the census carries BEYOND the accepted
    # one; when it names nothing, one of the two required crossings is
    # missing (the accept rule of bounce_census)
    reasons = census_violations(b) or [
        'no n=0 negative mode or no n=1 translation zero mode was located']
    return ('SPECIAL: ' + '; '.join(reason.strip() for reason in reasons)
            + ' -- selecting it for lnD aborts (open problem)')


# --------------------------------------------------------------------------- #
#  THE one figure renderer (the bounce_vgeneral_phi_v2 gallery AND the plot   #
#  stage plot_bounce_gallery_vgeneral_phi_v2 use it)                          #
# --------------------------------------------------------------------------- #
#: marker areas (points^2) of the vacua in the potential pictures, large
#: enough to read: the bounce's false and true vacuum ('o'),
#: every other minimum ('.', drawn smaller by matplotlib, so larger) -- every
#: vacuum a dot of the same size
VACUUM_MARKER_AREA = {'o': 60.0, '.': 300.0}


def plot_bounce(b, minima, iF, iT, fields, params, out_dir,
                png_prefix='bounce_'):
    """Render ONE bounce npz dict as two pictures in out_dir:
    <png_prefix><name>_profiles.png, the PRIMED profiles (N=1: + the primed
    vacuum levels) with the legend outside at the top right (S_E + one colour
    per component), and, for N=2 and N=3, <png_prefix><name>_potential.png
    (N=2: the surface V next to the landscape -V, both with the path; N=3:
    the field-space path).  Returns (name, S_E, verdict, png_paths)."""
    os.environ.setdefault('MPLBACKEND', 'Agg')
    # deferred: the compute stages import this module and must not pull matplotlib in
    import matplotlib.pyplot as plt

    # V without its constant monomial c: the same zero of energy as the
    # catalogue's vacuum values m['V'] (find_all_minima), so the vacuum
    # markers of the surface panel sit ON the drawn surface
    params = params_no_const(params, fields)
    N = len(fields)
    n_minima = len(minima)
    name = bounce_display_name(iF, iT, n_minima)
    R = np.asarray(b['R'], float)
    phi_b = np.asarray(b['phi_b'], float)              # profiles picture
    phi_b_orig = np.asarray(b['phi_b_orig'], float)    # potential picture
    S_E = float(b['S_E'])
    verdict = census_verdict(b)
    special = (verdict == 'SPECIAL')
    ls = '--' if special else '-'          # SPECIAL bounces are drawn dashed
    # the declared fields in the pictures: field fields[i] is phi_{i+1}
    phi = [rf'\phi_{{{i + 1}}}' for i in range(N)]
    V_of_phi = rf"V({', '.join(phi)})"

    # ---- picture 1, the profiles: PRIMED frame -- every component decays
    #      towards 0 (= the false vacuum at the origin; at r_max it is on
    #      its tail) ------------------------------------------------------
    # column i of phi_b is the false-vacuum mass eigenchannel in column i of
    # L = mass_basis_L (columns ordered by m^2, heaviest first), named
    # phi'_{i+1} in the legend:
    #     phi_b[:, i] = sum_j L[j, i] (phi_j - phi_F,j),
    # a combination of the declared fields, not the declared field of the
    # same position
    # the picture size (<= 8 in wide with its legend; text and line sizes:
    # plot_common, the one set of sizes)
    fig_profiles, ax1 = plt.subplots(figsize=(5.6, 4.2))
    for i in range(N):
        ax1.plot(R, phi_b[:, i], ls, label=rf"${phi[i]}'(r)$")
    # dotted lines thicker than the others (plot_common.DOTTED_LINE_WIDTH)
    ax1.axhline(0.0, color='gray', ls=':', lw=2.2)
    if N == 1:                     # the two vacua as PRIMED levels
        # to_prime: (x - x_F) @ L (potential_vgeneral_phi_v2.PrimedPotential.to_prime)
        L = np.asarray(b['mass_basis_L'], float)
        lev_true = float(((np.asarray(b['true_vac'], float)
                           - np.asarray(b['false_vac'], float)) @ L)[0])
        for lev, k in ((0.0, iF), (lev_true, iT)):
            ax1.axhline(lev, color='gray', ls=':')
            ax1.annotate(vacuum_label(k, n_minima), (R[-1], lev), ha='right',
                         va='bottom', color='gray')
    ax1.set_xlabel(r'$r$')
    ax1.set_ylabel(r"Bounce profiles $\overline{\phi}'(r)$")
    if special:                    # no title otherwise: the name is in the file name
        ax1.set_title('[SPECIAL]', color='crimson')
    ax1.grid(True)
    # laid out BEFORE the legend exists, so the axes keep their size and the
    # legend hangs off them (the saved bbox grows to hold it)
    fig_profiles.tight_layout()
    # the legend OUTSIDE the axes, to the right at the top (the placement of
    # plot_common.legend_outside), so the curves stay clear: S_E as its
    # left-aligned title, then one colour per primed component
    ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0,
               title=rf'$S_E$ = {S_E:.4f}',
               alignment='left', frameon=True,
               edgecolor=('crimson' if special else 'gray'))
    pictures = [('profiles', fig_profiles)]

    if N == 2:
        # grid over the path + all minima, padded (ORIGINAL frame)
        pts = np.vstack([phi_b_orig] + [np.asarray(m['point'], float)[None, :]
                                 for m in minima])
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        # a quarter of the WIDEST extent on every side, so a field that no
        # vacuum and no path point moves (a spectator) still gets a visible axis
        pad = 0.25 * np.max(hi - lo)
        gx = np.linspace(lo[0] - pad, hi[0] + pad, 160)
        gy = np.linspace(lo[1] - pad, hi[1] + pad, 160)
        GX, GY = np.meshgrid(gx, gy)
        GV = V_numeric(np.stack([GX, GY], axis=-1), params, fields)
        # a picture <= 8 in wide (text and line sizes: plot_common)
        fig_potential = plt.figure(figsize=(8.0, 3.9))
        # -- left: the potential SURFACE with the path lifted onto it -------
        ax2 = fig_potential.add_subplot(1, 2, 1, projection='3d')
        # drawn in zorder, not sorted by depth: the path under the translucent
        # surface (the potential's colours in the foreground), the vacua over
        # it, so their labels keep their colours (path 1, surface 2, vacua 3)
        ax2.computed_zorder = False
        ax2.plot_surface(GX, GY, GV, cmap='viridis', alpha=0.55, linewidth=0,
                         antialiased=True, zorder=2)
        Vpath = np.asarray(V_numeric(phi_b_orig, params, fields), float)
        ax2.plot(phi_b_orig[:, 0], phi_b_orig[:, 1], Vpath, 'r' + ls, zorder=1)
        # in both panels: the path in red, every vacuum in blue -- the
        # bounce's false and true vacuum ('o'), the other minima ('.')
        for k, m in enumerate(minima):
            pt = np.asarray(m['point'], float)
            Vv = float(m['V'])
            marker = 'o' if k in (iF, iT) else '.'
            ax2.scatter(pt[0], pt[1], Vv, c='blue', marker=marker,
                        s=VACUUM_MARKER_AREA[marker], zorder=3)
            ax2.text(pt[0], pt[1], Vv, f'  {vacuum_label(k, n_minima)}', color='blue')
        ax2.set_xlabel(f'${phi[0]}$')
        ax2.set_ylabel(f'${phi[1]}$')
        ax2.set_zlabel(f'${V_of_phi}$')
        # -- right: the LANDSCAPE, the inverted potential -V (every minimum
        #    of V is a hilltop), as a contour map + the path (this IS the
        #    field-space path -- the surface path seen top-down) ------------
        ax3 = fig_potential.add_subplot(1, 2, 2)
        cf = ax3.contourf(GX, GY, -GV, levels=30, cmap='viridis', alpha=0.75)
        ax3.contour(GX, GY, -GV, levels=30, colors='k', linewidths=0.3,
                    alpha=0.35)
        fig_potential.colorbar(cf, ax=ax3, label=f'$-{V_of_phi}$')
        ax3.plot(phi_b_orig[:, 0], phi_b_orig[:, 1], 'r' + ls, label='bounce path')
        for k, m in enumerate(minima):
            pt = np.asarray(m['point'], float)
            marker = 'o' if k in (iF, iT) else '.'
            ax3.scatter(*pt, c='blue', marker=marker, s=VACUUM_MARKER_AREA[marker],
                        zorder=5)
            ax3.annotate(vacuum_label(k, n_minima), pt, textcoords='offset points',
                         xytext=(6, 6), color='blue')
        ax3.set_xlabel(f'${phi[0]}$')
        ax3.set_ylabel(f'${phi[1]}$')
    elif N == 3:
        # field space (ORIGINAL frame); V along the path is the color -- the
        # unplottable 4th dimension.
        fig_potential = plt.figure(figsize=(6.8, 5.2))
        ax2 = fig_potential.add_subplot(1, 1, 1, projection='3d')
        Vpath = np.asarray(V_numeric(phi_b_orig, params, fields), float)
        sc = ax2.scatter(phi_b_orig[:, 0], phi_b_orig[:, 1], phi_b_orig[:, 2], c=Vpath,
                         cmap='viridis', s=6)
        ax2.plot(phi_b_orig[:, 0], phi_b_orig[:, 1], phi_b_orig[:, 2], ls, color='gray', lw=0.8,
                 alpha=0.7)
        fig_potential.colorbar(sc, ax=ax2, shrink=0.7, pad=0.12,
                               label=r'$V$ along the path (the unplottable 4th axis)')
        for k, m in enumerate(minima):
            pt = np.asarray(m['point'], float)
            marker = 'o' if k in (iF, iT) else '.'
            ax2.scatter(*pt, c='blue', marker=marker)
            ax2.text(*pt, f'  {vacuum_label(k, n_minima)}', color='blue', fontsize=7)
        ax2.set_xlabel(f'${phi[0]}$')
        ax2.set_ylabel(f'${phi[1]}$')
        ax2.set_zlabel(f'${phi[2]}$')
    if N in (2, 3):
        if special:                # no title otherwise, as the profiles picture
            ax2.set_title('[SPECIAL]', color='crimson')
        fig_potential.tight_layout()
        if N == 2:
            # tight_layout leaves no room for the 3-D box's V label: more space
            # between the two panels, so it stays clear of the landscape's
            fig_potential.subplots_adjust(wspace=0.58)
        pictures.append(('potential', fig_potential))

    os.makedirs(out_dir, exist_ok=True)
    pngs = []
    for picture, fig in pictures:
        png = os.path.join(out_dir, f'{png_prefix}{name}_{picture}.png')
        # bbox_inches='tight' keeps the outside legend inside the saved figure
        fig.savefig(png, dpi=140, bbox_inches='tight')
        plt.close(fig)
        print(f'[OK] wrote {png}')
        pngs.append(png)
    return name, S_E, verdict, pngs


def bounce_gallery(gen_dir, params, fields, gallery_dir, minima, show=False):
    """Render EVERY solved bounce npz in gen_dir through plot_bounce, labels
    resolved from the stored vacuum COORDINATES against `minima`, the vacuum
    catalogue of the run (bounce_vgeneral_phi_v2's ONE Newton sweep, passed
    down -- never recomputed here).  PNGs go to gallery_dir (stale PNGs
    cleared first); show=True additionally opens the folder.  Returns the
    menu rows [(display_name, S_E, n_tunneling_components, census_note), ...]
    sorted by S_E."""
    os.makedirs(gallery_dir, exist_ok=True)
    # ONE gallery per data dir, always reflecting the CURRENT potential: the
    # stale pictures go first, then one PNG and one menu row per solved bounce
    for stale in glob.glob(os.path.join(gallery_dir, 'bounce_*.png')):
        os.remove(stale)
    rows = []
    for f in sorted(glob.glob(os.path.join(gen_dir, 'bounce_data_F*_T*.npz'))):
        with np.load(f, allow_pickle=True) as bounce_npz_data:
            b = {k: bounce_npz_data[k] for k in bounce_npz_data.files}
        # ONE schema guard: the keys the renderer reads that a pre-schema
        # bounce npz lacks
        missing = [k for k in ('phi_b', 'phi_b_orig')
                   if k not in b]
        if missing:
            raise RuntimeError(
                f'[ABORT BOUNCE-SCHEMA] {os.path.basename(f)} stores no '
                f'{missing} key(s) -- not a current-schema bounce npz; delete '
                f'it from bounce_gen and let bounce_vgeneral_phi_v2 re-solve '
                f'the pair.')
        false_vac = np.asarray(b['false_vac'], float)
        true_vac = np.asarray(b['true_vac'], float)
        # labels by COORDINATES, never by the stored indices (a different
        # --seed-range renumbers the catalogue)
        iF, iT = catalogue_indices(minima, false_vac, true_vac, f)
        name, S_E, verdict, _pngs = plot_bounce(b, minima, iF, iT, fields,
                                                params, gallery_dir)
        # how many field components differ between the two vacua (display
        # only): those that move by more than 1e-6 of the vacuum separation
        vacuum_step = np.abs(true_vac - false_vac)
        n_tunneling_components = int(np.count_nonzero(
            vacuum_step > 1e-6 * np.linalg.norm(vacuum_step)))
        rows.append((name, S_E, n_tunneling_components, _census_note(b, verdict)))
    rows.sort(key=lambda r: r[1])
    if show and rows:
        # best-effort: open the gallery folder in the file browser; the path
        # is passed as one argument, never through a shell
        opener = ('open' if sys.platform == 'darwin'
                  else 'xdg-open' if sys.platform.startswith('linux') else None)
        if opener and shutil.which(opener):
            subprocess.run([opener, gallery_dir], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, check=False)
    return rows


def select_bounce_interactive(rows, minima, gallery_dir):
    """Print the menu of ALL solved bounce solutions (`rows`, the return of
    bounce_gallery, which already rendered the pictures into gallery_dir) and
    ask the user to type the NAME of the one the pipeline should use (a
    display name from the menu, or the raw file form F<i>_T<j>).  `minima`
    is the vacuum catalogue of the run.  Returns (iF, iT).  Re-prompts on a
    typo; only an unreadable stdin (non-interactive, EOF) aborts."""
    if not rows:
        raise SystemExit('[ABORT] no solved bounce solutions to choose from.')
    n_minima = len(minima)
    print('\n' + '=' * 68)
    print('  ALL SOLVED BOUNCE SOLUTIONS  (plots in ' + gallery_dir + ')')
    print('=' * 68)
    for name, S_E, n_tunneling_components, note in rows:
        extra = f'   [{note}]' if note else ''
        print(f'   {name:<12s}  S_E = {S_E:12.4f}   components changed: '
              f'{n_tunneling_components}{extra}')
    print('  (The census eigenvalues of every bounce are printed above; a')
    print('   SPECIAL bounce is drawn dashed and can be plotted but not')
    print('   selected for lnD.)')
    print('=' * 68)
    valid = {r[0] for r in rows}
    while True:
        try:
            choice = input('\nWrite the name of the bounce solution to use '
                           '(e.g. FV0_TV0): ').strip()
        except EOFError:
            raise SystemExit(
                '[ABORT] no interactive answer available -- rerun with '
                '--pair <name> instead, e.g. --pair ' + rows[0][0] +
                ' (any name from the menu above).')
        try:
            iF, iT = pair_from_name(choice, n_minima=n_minima)
            name = bounce_display_name(iF, iT, n_minima)
        except ValueError as e:
            print(f'  {e}  Choose one of: {sorted(valid)}')
            continue
        if name not in valid:
            print(f'  {name} is not one of the solved bounces -- choose one '
                  f'of: {sorted(valid)}')
            continue
        print(f'[bounce] selected {name}')
        return iF, iT
