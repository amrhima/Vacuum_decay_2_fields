#!/usr/bin/env python3
"""mode_census_vgeneral_phi_v2.py -- level 1 of the README tree, the census
library of stage 0 (bounce_vgeneral_phi_v2.bounce_census) and of stages 0a/0b
(eigenvalue_neg / eigenvalue_zero): THE MODE CENSUS of the dangerous sectors
n = 0 and n = 1 -- the constant Wronskian C whose eigenvalues locate the
modes, the four-step census that finds every crossing of them, the reader of
the census a bounce npz stores, and the printer and messages that report it.

A CROSSING is one s2 at which an eigenvalue of C(n, s2) is zero, i.e. one
eigenvalue omega^2 = -s2 of the radial fluctuation operator M_n: there a
solution regular at r = 0 also decays at large r (the eigenfunction), so the
regular and the decaying basis functions are linearly dependent, their
constant Wronskian C is singular (det C = 0), and the radial Green's
function, which carries C^-1, has its pole at s2 = -omega^2.

READS   the h-ODE engine its caller builds on the bounce
        (greens_function.GreensFunction on R, U_int, m2): one solve per
        (n, s2) gives C; U_int and m2 again for the scan windows; the census
        keys census_* a bounce npz stores (stored_census, census_from_stored).
WRITES  no file.  The sector censuses return the located crossings, the
        modes sharing each, |det C| at the crossing and the unresolved count:
        bounce_census stores them as the census_* keys of the bounce npz, and
        eigenvalue_neg / eigenvalue_zero turn the crossings into
        omega2_00 = -s2 (n=0) and omega2_10 = -s2_c (n=1): the poles
        deg/(omega2 + s2) the sector stages subtract and, for n=0, the
        ln|omega2_00| the assembly restores.
HOW     per sector, on a window derived from the bounce (n=0: a geometric
        scan from an accuracy floor x the variational ceiling up to that
        ceiling; n=1: a linear scan straddling zero, down to a margin short
        of the continuum threshold): coarse scan of the eigenvalues of C,
        refinement wherever one of them changes sign or approaches zero,
        brentq on each confirmed sign change, verification at the root --
        THE FOUR STEPS of the census section below.  The census is a COUNT
        of the dangerous modes -- the n=0 negative modes and the n=1 zero
        and negative modes (dangerous_n1_mask; a positive n=1 bound state
        stays inside the resolvent and is not dangerous) -- and that count
        is what the selection gate acts on.

CONTENTS
  THE CONSTANT WRONSKIAN
    constant_wronskian_and_floor: ONE h-ODE solve (none when C is given) ->
        (C, its eigenvalues, the accuracy floor of each eigenvalue), with
        C = identity + h_minus_inf, the constant Wronskian r^3 W built
        from h^- and h^-' at r_max
    det_C_scan: det C and the N eigenvalues of C on a fixed scan (what
        both eig stages store for the det-crossing figure)
  THE CENSUS
    census_sector(engine, n, grid): every s2 at which an eigenvalue of C is
        zero -- coarse scan, mark sign changes and approaches to zero,
        refine the s2 grid there, brentq
    modes_at_root: how many REAL eigenvalues of C sit AT their floor at a
        located root = how many modes share it
  THE STORED CENSUS (the census keys a bounce npz carries)
    CENSUS_SCHEMA_KEYS: every key the census writes
    stored_census(npz): the census an OPEN bounce npz carries, or None when
        one of those keys is missing
    census_from_stored(b): the census_* keys of an OPEN bounce npz (or its
        dict) as the typed census dict -- the ONE reader of the stored census
  REPORTING
    print_census_inventory: the mode inventory (n <= 1), one line per
        located crossing
    dangerous_n1_mask: which located n=1 crossings are DANGEROUS (zero or
        negative modes; a positive bound state is not) -- the ONE owner of
        the rule
    n1_crossing_labels: what each located n=1 crossing is, by that rule (the
        labels of both census printers)
    census_violations(b): the evidence lines of the selection-gate abort
    OPEN_PROBLEM_MSG: what a non-index-1 bounce is refused with
    LEG_POINTER_MSG: the extra sentence for a DECOUPLED potential
  THE SECTOR CENSUSES (called by bounce_vgeneral_phi_v2.bounce_census)
    variational_s2_ceiling: the s2 above which no crossing of any sector
        can lie
    negative_mode_window: the derived n=0 window (floor x ceiling,
        ceiling)
    zero_mode_window_lo: the low end of the n=1 window (one rule for the
        census and the n=1 det-crossing scan)
    zero_mode_levels: the two zero-mode levels (tol, trust) of a bounce,
        fractions of that window's reach
    negative_mode_census: the n=0 census: every negative mode
    zero_mode_census: the n=1 census: the translation zero mode and every
        other n=1 crossing

bounce_vgeneral_phi_v2, eigenvalue_neg_vgeneral_phi_v2,
eigenvalue_zero_vgeneral_phi_v2, bounce_gallery_vgeneral_phi_v2 and
assemble_lnD_vgeneral_phi_v2 (the stored census: the gallery's legend and
menu, the verdict the assembly reports), add_on_greensfct/
greensfct_dangerous_modes_v2 (VERIFY_FRAC, its level for a reproduced root)
and add_on_greensfct/greensfct_toolbox_v2 (constant_wronskian_and_floor, for
its crossing guard) import from here; nothing here imports from any of them.
"""
import numpy as np
from scipy.optimize import brentq

from pipeline_helpers_vgeneral_phi_v2 import require_finite


# --------------------------------------------------------------------------- #
#  THE CONSTANT WRONSKIAN C(n, s2) = identity + h_minus_inf                   #
# --------------------------------------------------------------------------- #
def constant_wronskian_and_floor(engine, n, s2, C=None):
    """(C, lam_C, floor) at one (n, s2): the constant Wronskian, its N
    eigenvalues lam_C and the ACCURACY FLOOR of each eigenvalue -- all the
    census reads at a scan point.  The census passes no C: C is then built
    by ONE h-ODE solve at this s2.  The add-on's crossing guard passes the
    C that GreensFunction.branch_objects has already built at this (n, s2):
    then nothing is solved, h_minus_inf is C minus the identity, and
    lam_C and floor are read off it.

    C is the constant Wronskian r^3 W of the two branches,
    C = 1 + h^-(inf), built by engine.h_minus_inf from the FULL
    h^- and h^-' at r_max (where h^+ = h^+' = 0):
        C[a, b] = delta_ab + h^-[a, b](r_max)
                  + r_max I_nu K_nu(kappa_a r_max) h^-'[a, b](r_max) .
    det C and the eigenvalues of C do not change under transposition, so
    the census needs no index convention for C.

    floor[k] = how far eigenvalue k can move within the accuracy of the
    solve, to first order.  The error of every solved entry of C is
    estimated from the engine's LSODA tolerances,
        |delta C| = rtol |h_minus_inf| + atol   (the 1 in C is exact);
    LSODA keeps only the local error of each step within these tolerances,
    so this is an estimate, not a bound.  In the eigenbasis of C
    (C X = X diag(lam_C), X the eigenvectors as columns) an error delta C
    moves eigenvalue k by (X^-1 delta C X)_kk; bounding every term of that
    sum by its size gives
        floor[k] = (|X^-1| |delta C| |X|)_kk .
    Every eigenvalue gets its OWN floor because the entries of C can differ
    by many orders of magnitude (a heavy channel's basis function, fed into
    a lighter channel through the wall, grows against that channel's Bessel
    function): the eigenvalue that crosses zero is then known far better
    than rtol times the largest entry, and one common floor would mask it.
    The floor is dimensionless and decides (a) when to stop refining (a
    whole sub-window below it is round-off), (b) when an imaginary part is
    round-off (|Im lambda| below it means real) and (c) the lower end of
    the n=0 scan (negative_mode_window).  It is a first-order estimate.
    A channel much heavier than the tunnelling one and coupled to it is a
    limit: the entries of C then differ so much that the floor hides the
    eigenvalue next to its crossing."""
    if C is None:
        h_minus_r_max, dh_minus_r_max = (
            engine.state_minus_r_max(n, float(s2))[:2])
        h_minus_inf = engine.h_minus_inf(n, float(s2), h_minus_r_max,
                                         dh_minus_r_max)
        C = np.eye(h_minus_inf.shape[0]) + h_minus_inf
    else:
        h_minus_inf = C - np.eye(C.shape[0])
    lam_C, X = np.linalg.eig(C)
    delta_C = engine.rtol * np.abs(h_minus_inf) + engine.atol
    floor = np.diag(np.abs(np.linalg.inv(X)) @ delta_C @ np.abs(X))
    return C, lam_C, floor


# Point count of the STORED det-crossing curve (the s2_scan/det_scan/
# lambda_scan npz keys the det-crossing figure reads).  A plot density, not an
# accuracy knob: the census locates its roots on its own scan-and-refine, and
# nothing here enters lnD.
DET_SCAN_PTS = 129


def det_C_scan(engine, n, scan):
    """det C(s2) and the N eigenvalues of C on a fixed plot-density scan --
    the stored curves behind the det-crossing figure
    (plot_det_C_vgeneral_phi_v2); both eig stages store them, so no plot
    re-solves the h-ODE.  det is the product of ALL N eigenvalues (real for
    the real matrix C); the stored eigenvalue curves are their real parts in
    ascending order, the lambda_k the census follows (_lam_at_node).
    Also returns the largest |Im lambda| on the scan: C is not symmetric, so
    a complex conjugate pair is legitimate (it carries no mode), and then
    det_scan != prod(lambda_scan) -- the figure needs that number to tell
    such a pair from a mismatched npz."""
    det = np.empty(len(scan))
    lambda_scan = []
    max_abs_imag = 0.0
    for i, s2 in enumerate(scan):
        _, lam_C, _ = constant_wronskian_and_floor(engine, n, float(s2))
        det[i] = float(np.prod(lam_C).real)
        max_abs_imag = max(max_abs_imag, float(np.max(np.abs(lam_C.imag))))
        lambda_scan.append(np.sort(lam_C.real))
    return det, np.array(lambda_scan), max_abs_imag


# --------------------------------------------------------------------------- #
#  THE MODE CENSUS: every s2 at which an eigenvalue of C(n, s2) is zero        #
# --------------------------------------------------------------------------- #
# For a sector n, an eigenvalue omega^2 of the fluctuation operator exists at
# s2 = -omega^2 exactly where det C(n, s2) = 0.  det C is the PRODUCT of the
# eigenvalues of C, so det C = 0 means one EIGENVALUE of C is zero.  The census
# therefore locates with the eigenvalues, never with the determinant: at every
# s2 it sorts the N eigenvalues of C by their real parts, so that the k-th of
# them, lambda_k, is a continuous curve of s2, and it finds every s2 at which
# one lambda_k is zero.  Each such s2 is one mode, with omega^2 = -s2.
#
# THE FOUR STEPS
#   1. COARSE SCAN.  On a coarse s2 grid across the sector's window, solve the
#      h-ODE and compute the EIGENVALUES of C at every node.
#   2. SPOT.  Mark a region where a real lambda_k CHANGES SIGN, or where a
#      real lambda_k COMES CLOSE TO ZERO without changing sign.
#   3. REFINE.  Around each marked region, lay a finer s2 grid -- which means
#      new h-ODE solves -- and look again.  Repeat while the refinement keeps
#      changing the picture and the accuracy floor is not reached.
#   4. LOCATE.  brentq on each confirmed sign change gives the exact s2; the
#      eigenvalue is omega^2 = -s2.
#
# WHY BOTH THE SIGN TEST AND THE NEAR-ZERO TEST ARE NEEDED.  Because the
# eigenvalues are sorted, two crossings of OPPOSITE slope inside ONE interval
# cancel: the two sorted curves swap roles and no sign change is left at the
# ends of the interval.  The near-zero test spots that region; the refinement
# then separates the two crossings into their own intervals, where the sign
# changes reappear and brentq works normally.  So: near-zero decides WHERE to
# refine, refinement restores the sign changes, brentq locates each crossing
# exactly.  A finite scan is not a proof of completeness: two crossings of
# opposite slope inside one interval, closer together than the refined grid
# resolves and leaving no dip at the interval's end nodes, are not reported.
#
# COMPLEX EIGENVALUES.  C is not symmetric, so its eigenvalues may be
# complex.  The census follows the REAL PARTS of all N eigenvalues in
# ascending order: a pair that collides and goes complex keeps two adjacent
# places with one common real part, so no lambda_k jumps when that happens.
# A zero needs Im = 0 as well (det C = 0 requires an eigenvalue to BE zero,
# and zero is real): a sign change that brentq locates on a complex pair is
# a real part passing zero off the real axis, not a mode, and is dropped.
# "Complex" means |Im lambda| ABOVE that eigenvalue's accuracy floor: a
# genuinely real eigenvalue may compute with a round-off imaginary part.
# Detection, verification and the mode count all read these same
# eigenvalues, so no separate rank test of C is needed.
#
# THE NUMBERS OF THE CENSUS, and what kind of number each one is.  One is an
# ACCURACY of the computation (VERIFY_FRAC; the accuracy of h itself is the
# engine's own LSODA rtol, engine.rtol), two are the REFINEMENT's own shape
# (REFINE_PTS, DIP_FRAC), and two are RUNAWAY GUARDS that only ever fail loud
# (REFINE_MAX_ROUNDS, SOLVE_BUDGET).  None of them is a statement about any
# particular potential, and none is an absolute level in s2 or in lambda.
# (The two MARGINS on the window ends live with the windows, below.)
REFINE_PTS = 33         # nodes of one refined sub-scan: the s2 step shrinks by
                        # REFINE_PTS - 1 = 32 per round.
DIP_FRAC = 0.5          # a local minimum of |lambda_k| counts as an APPROACH
                        # TO ZERO once lambda_k has at least halved from its
                        # neighbours.  A lambda_k that merely bends (an
                        # ordinary minimum at a finite value) stops halving
                        # after a round or two and the refinement ends there
                        # by itself, while one heading for zero halves at
                        # every round and is followed all the way down.  So
                        # this fraction decides nothing on its own -- it only
                        # says which minima are worth one more look.
VERIFY_FRAC = 1e-6      # a located root must have collapsed its lambda_k to
                        # this fraction of its size at the ends of the
                        # bracket (or all the way to the accuracy floor); a
                        # bracket that never held a zero leaves the ratio ~1.
REFINE_MAX_ROUNDS = 12  # runaway guard only: 32^12 = 1e18 is far more than the
                        # ratio between the widest scan window and the accuracy
                        # floor, so on a genuine approach to zero the FLOOR
                        # always stops the recursion first.
SOLVE_BUDGET = 20000    # h-ODE solves per sector census before failing loud.
                        # A settling census costs the coarse scan plus a few
                        # sub-scans of REFINE_PTS nodes and one brentq per
                        # crossing, so this only catches a refinement that
                        # will not terminate.


def modes_at_root(real_eigenvalues, accuracy_floor):
    """How many modes sit at ONE located crossing: how many REAL eigenvalues of
    C are AT their accuracy floor there (the two arrays in the same sorted
    order).  The count comes from the EIGENVALUES, never from two located
    crossings landing near each other, and the gallery legend repeats a
    shared eigenvalue once per mode.

    The two populations are far apart at a root (the located lambda_k at its
    floor, every other eigenvalue at its own finite size unless it is zero
    at the same s2), so the count is unambiguous; it catches a symmetric
    pair (two eigenvalues zero at the same s2), a bit-identical pair and two
    roots split far below the floor alike.  Never reported below 1 -- the
    located lambda_k is always there."""
    # the caller passes the non-real eigenvalues as NaN
    return max(1, int(np.count_nonzero(
        np.abs(real_eigenvalues) <= accuracy_floor)))


def _lam_at_node(engine, n, s2, lam_by_s2):
    """The sorted eigenvalues lambda_k of C at ONE s2 node
    -> (lam, floor, imag, det C).

    lam holds the REAL PARTS of the N eigenvalues in ascending order, floor
    the accuracy floor (constant_wronskian_and_floor) and imag the |Im| of
    each, in the same order.  Sorting destroys which eigenvalue is which but
    preserves CONTINUITY in s2, which is all a sign test and brentq need; a
    complex pair keeps two adjacent places (block comment above).  A
    general eigensolver, not a symmetric one: C is not symmetric.

    det C is the product of ALL N eigenvalues (a complex pair included); the
    census reads it only at a located root, as the reported crossing
    residual |det C|.

    `lam_by_s2` = {s2: (lam, floor, imag, det C)} for ONE
    census: refined windows share their end nodes with the level above, so
    those h-ODE solves happen once."""
    s2 = float(s2)
    if s2 not in lam_by_s2:
        if len(lam_by_s2) >= SOLVE_BUDGET:
            raise RuntimeError(
                f'[ABORT CENSUS] the n={n} census asked for more than '
                f'{SOLVE_BUDGET} h-ODE solves without settling (a settling '
                f'census needs a few sub-scans of {REFINE_PTS} nodes per '
                f'crossing); something in this sector is not converging and '
                f'no mode count should be read from it.')
        _, lam_C, floor = constant_wronskian_and_floor(engine, n, s2)
        require_finite(lam_C, f'[ABORT CENSUS] non-finite eigenvalue of the '
                              f'n={n} constant Wronskian at s2={s2:.6e}')
        order = np.argsort(lam_C.real)      # ascending real parts
        lam_by_s2[s2] = (lam_C.real[order], floor[order],
                         np.abs(lam_C.imag[order]), float(np.prod(lam_C).real))
    return lam_by_s2[s2]


def _lam_on_grid(engine, n, grid, lam_by_s2):
    """_lam_at_node over a grid -> (lam, floor, imag), each (P, N)."""
    pts = [_lam_at_node(engine, n, s2, lam_by_s2) for s2 in grid]
    return tuple(np.array([p[j] for p in pts]) for j in range(3))


def _crossings_and_dips(lam):
    """STEP 2, applied at every level -> (crossings, dips).

    crossings: [(i, k)] -- lambda_k changes sign across interval
        (grid[i], grid[i+1]).  Both ends must be finite; a NaN end (a lambda_k
        that _locate masked as round-off) carries no information and is
        skipped.
    dips: {i} -- intervals touching a node where a lambda_k COMES CLOSE TO
        ZERO without changing sign there.  "Close" is judged against that
        lambda_k's OWN neighbouring values: node j is an approach to zero when
        |lambda_k| is a local minimum there and has at least halved from the
        LARGER of its two neighbours (DIP_FRAC).  Against the LARGER one,
        because a symmetric approach to zero puts two comparably small values
        side by side -- two crossings inside one interval leave both of its
        ends near zero -- and comparing with the smaller shoulder would then
        see no descent at all.  The interval that carries the sign change of
        THAT lambda_k is not marked again -- the crossing explains the small
        value there by itself -- but the interval on the OTHER side of the
        minimum is, because it can hide a pair: a lambda_k with three zeros
        close together shows one sign change in one interval and two more,
        invisible, in the next.

    The two ENDS of the scan are NOT dips: refining an interval whose lambda_k
    is still descending as it leaves the window reveals nothing, because the
    structure is OUTSIDE the window.  That case is reported instead, as
    near_zero_at_low_end (what the CALLER does with it differs by sector:
    _descending_at_low_end)."""
    n_scan, N = lam.shape
    crossings, dips = [], set()
    for k in range(N):
        lam_k = lam[:, k]
        unmasked = np.isfinite(lam_k)
        sign_change_at = set()
        for i in range(n_scan - 1):
            if unmasked[i] and unmasked[i + 1] and lam_k[i] * lam_k[i + 1] <= 0.0:
                crossings.append((i, k))
                sign_change_at.add(i)
        abs_lam_k = np.abs(lam_k)
        for j in range(1, n_scan - 1):
            if not (unmasked[j] and unmasked[j - 1] and unmasked[j + 1]):
                continue
            smaller_neighbour, larger_neighbour = sorted(
                (abs_lam_k[j - 1], abs_lam_k[j + 1]))
            if abs_lam_k[j] > smaller_neighbour:
                continue                      # not a local minimum
            if abs_lam_k[j] > DIP_FRAC * larger_neighbour:
                continue                      # a flat bend, not a descent
            dips |= {j - 1, j} - sign_change_at
    return crossings, dips


def _descending_at_low_end(lam):
    """True if some real lambda_k is heading for zero as it leaves the window at
    the LOW end (the same DIP_FRAC descent test): a mode may lie below the
    window.  The two callers act on it differently, because their low ends are
    different objects.  In the ZERO sector (n=1) the low end is a margin short
    of the continuum threshold s2 = -min_i m_i^2 (zero_mode_window_lo), an
    arbitrary place to stop, so the sector IS re-censused on a wider window.
    In the NEGATIVE sector (n=0) the low end is the derived cutoff s2_lo of
    negative_mode_window (an accuracy floor times the variational ceiling);
    a crossing below it is not resolved from s2 = 0 by this census, so the
    window is not widened and the flag is instead the ONLY signal that an
    extra negative mode may sit just under the cutoff (a near index
    bifurcation), turning the verdict SPECIAL instead of a silent index-1
    ACCEPT.  It never moves or creates a root in either sector."""
    for k in range(lam.shape[1]):
        lowest_two = np.abs(lam[:2, k])   # |lambda_k| at the two lowest nodes
        if np.all(np.isfinite(lowest_two)) and lowest_two[0] <= DIP_FRAC * lowest_two[1]:
            return True
    return False


def _crossing_in_bracket(engine, n, k, lo, hi, lam_by_s2):
    """STEP 4: the exact s2 of ONE bracketed sign change of lambda_k, VERIFIED
    (brentq on the sorted lambda_k).

    brentq needs nothing but the bracket and continuity, both of which a
    sorted lambda_k with a sign change has.  The root is then VERIFIED by
    re-reading lambda_k at it: it must be real and have COLLAPSED there --
    either to its accuracy floor, or to VERIFY_FRAC of its size at the ends
    of the bracket.  Two levels because the root finder cannot always reach
    the floor: it stops when the BRACKET is narrower than xtol, and a steep
    lambda_k still has |lambda_k| = slope x xtol there.

    What the check must exclude is a sign change that is NOT a zero of C:
    the real part of a complex-conjugate pair can pass through zero while
    the pair stays complex, and a complex eigenvalue is never zero
    (|lambda| >= |Im lambda| > 0).  brentq then lands at a point where
    lambda_k is complex -- its imaginary part is above the accuracy floor --
    and the candidate is discarded.  Only a REAL eigenvalue that has
    collapsed to zero is a mode, so the root search cannot invent one.

    (A located s2 is pinned no better than floor / |dlambda_k/ds2|: closer
    to the root the computed lambda_k is below its accuracy floor.)
    -> (s2, None) for a mode, (None, None) for a complex eigenvalue (never a
    mode), or (None, why) when lambda_k did not collapse."""
    def lam_k_at(s):
        """lambda_k of C at s2 = s (one memoised h-ODE solve)."""
        return _lam_at_node(engine, n, s, lam_by_s2)[0][k]

    b_lo, b_hi = lam_k_at(lo), lam_k_at(hi)
    shoulder = max(abs(b_lo), abs(b_hi))
    # relative tolerance dimensionless; the absolute one in s2 units, set
    # by the bracket itself
    root_rtol = 4.0 * np.finfo(float).eps
    s2 = float(brentq(lam_k_at, lo, hi,
                      xtol=root_rtol * max(abs(lo), abs(hi)), rtol=root_rtol))
    lam, floor, imag, _ = _lam_at_node(engine, n, s2, lam_by_s2)
    if imag[k] > floor[k]:
        return None, None       # lambda_k is complex here: never zero, no mode
    if abs(lam[k]) <= max(floor[k], VERIFY_FRAC * shoulder):
        return s2, None
    return None, (f'a sign change of lambda_{k} bracketed s2 = {s2:.6e}, but '
                  f'it did not collapse there (|lambda| = '
                  f'{abs(lam[k]):.2e}, against its accuracy floor '
                  f'{floor[k]:.2e} and {VERIFY_FRAC:g} x {shoulder:.2e} at '
                  f'the bracket ends)')


def _locate(engine, n, a, b, expect, depth, lam_by_s2):
    """STEP 3 + STEP 4 inside [a, b] -> (roots [(s2, k)], unresolved
    [(a, b, why)]).

    `expect` is how many sign changes the level ABOVE saw in this window.  THE
    PICTURE HAS STOPPED CHANGING when the finer grid shows the same number of
    sign changes and no unexplained approach to zero; then each sign change is
    one bracket around one root and brentq finishes the job.  If the finer grid
    shows MORE sign changes (several roots were hiding inside one interval), or
    an approach to zero with no sign change at all (two crossings of opposite
    slope), the picture DID change and every marked interval is refined again.

    A near-zero interval NEXT TO a sign change is explained by that crossing,
    so it does not stop the window from settling -- otherwise nothing ever
    settles, since a root always leaves its neighbour near zero -- but it is
    still looked at once, on its own finer grid, because that is exactly where
    a further pair can hide.  A sign change INSIDE such an interval (another
    lambda_k crossing zero there) is located by that finer look, not also
    here, so every crossing is located exactly once.

    ROUND-OFF.  A lambda_k whose |lambda_k| is below its accuracy floor
    across the WHOLE window carries no information left: it is masked out
    and reported as an unresolved near-zero (it reached zero within the
    accuracy of C without ever changing sign).  This is what stops the
    refinement from zooming into round-off (an unbounded zoom returns
    spurious roots on a near-multiple root).

    The other way out is REFINE_MAX_ROUNDS of refinement without settling,
    reported the same loud way.  Both are UNRESOLVED, never a silent drop."""
    grid = np.linspace(float(a), float(b), REFINE_PTS)
    lam, floor, imag = _lam_on_grid(engine, n, grid, lam_by_s2)
    unresolved, live = [], np.zeros(lam.shape[1], bool)
    for k in range(lam.shape[1]):
        if np.any(np.hypot(lam[:, k], imag[:, k]) > floor[:, k]):   # |lambda| above its floor
            live[k] = True
        else:
            unresolved.append((float(a), float(b), (
                f'eigenvalue lambda_{k} of C is at its accuracy floor '
                f'(<= {np.nanmax(floor[:, k]):.2e}) across this whole window: it '
                f'is zero to within the accuracy of C, with no sign change to '
                f'locate')))
    if not live.any():
        return [], unresolved
    lam = np.where(live, lam, np.nan)           # round-off lambda_k masked out
    crossings, dips = _crossings_and_dips(lam)
    cross_at = set(i for i, _ in crossings)
    next_to_crossing = {j for j in dips if {j - 1, j + 1} & cross_at}
    roots = []
    if len(crossings) == expect and not (dips - next_to_crossing):
        for i, k in crossings:
            if i in next_to_crossing:
                continue                        # located by the finer look below
            s2, why = _crossing_in_bracket(engine, n, k, float(grid[i]),
                                           float(grid[i + 1]), lam_by_s2)
            if s2 is not None:
                roots.append((s2, k))
            elif why is not None:
                unresolved.append((float(grid[i]), float(grid[i + 1]), why))
        look_at = sorted(next_to_crossing)      # one look next door
    elif depth >= REFINE_MAX_ROUNDS:
        return [], unresolved + [(float(a), float(b), (
            f'{REFINE_MAX_ROUNDS} rounds of refinement did not settle: '
            f'{len(crossings)} sign change(s) where the level above saw '
            f'{expect}, and {len(dips)} approach(es) to zero without one'))]
    else:
        look_at = sorted(cross_at | dips)       # the picture is still changing
    for i in look_at:
        r, u = _locate(engine, n, grid[i], grid[i + 1],
                       sum(1 for j, _ in crossings if j == i), depth + 1, lam_by_s2)
        roots += r
        unresolved += u
    return roots, unresolved


def census_sector(engine, n, grid):
    """THE CENSUS of sector n on the coarse scan `grid` (STEP 1): every s2 at
    which an eigenvalue of C(n, s2) (constant_wronskian_and_floor) is zero.  Each
    one is a mode of the sector, with omega^2 = -s2.  negative_mode_census and
    zero_mode_census (below) locate with this.

    ONE ENTRY PER ZERO OF A lambda_k: the lambda_k are followed separately, so
    two of them zero at the SAME s2 return TWO entries with the same number --
    that pair IS a second mode -- and how many modes share one located s2 is
    counted independently AT it by modes_at_root.

    FAIL-LOUD (a broken computation, never a mode-count verdict): a non-finite
    eigenvalue of C anywhere on the scan, and a refinement that will not settle
    inside SOLVE_BUDGET h-ODE solves.

    Returns (s2_roots, census_report).  s2_roots = the located crossings,
    ascending.  census_report carries
      'n_scan', 'N' the coarse-scan point count and the matrix dimension
                    (printed only),
      'max_abs_imag'  the largest |Im lambda| seen on the coarse scan,
      'n_complex'   how many (point, lambda_k) samples across ALL evaluated
                    points were complex (their real parts are followed; a
                    zero needs Im = 0, so a complex pair carries no mode),
      'modes_at_root'  how many modes share each located crossing,
      'abs_det_C_at_root'  |det C| at each located crossing, read off the
                    same eigenvalues that verified it (the reported
                    crossing residual),
      'unresolved'  (a, b, why) for every region that came close to zero and
                    could not be settled -- printed, stored, and forcing the
                    SPECIAL verdict; never silently dropped,
      'near_zero_at_low_end'  a lambda_k descending toward zero as it leaves the
                    window at the low end: re-census on a wider window where
                    that is possible (n=1), and where it is not (n=0, the low
                    end is the scaled scan cutoff) a flag that forces SPECIAL,
      'n_solves'    how many h-ODE solves the census cost.
    """
    grid = np.asarray(grid, float)
    lam_by_s2 = {}
    lam, floor, imag = _lam_on_grid(engine, n, grid, lam_by_s2)
    crossings, dips = _crossings_and_dips(lam)
    roots, unresolved = [], []
    for i in sorted(set(i for i, _ in crossings) | dips):
        r, u = _locate(engine, n, grid[i], grid[i + 1],
                       sum(1 for j, _ in crossings if j == i), 1, lam_by_s2)
        roots += r
        unresolved += u
    # ONE entry per (lambda_k, root): a lambda_k that is exactly 0.0 at a node
    # is bracketed from both sides and would otherwise be located twice.
    roots = sorted(dict.fromkeys(roots), key=lambda rk: rk[0])
    s2_roots = [s2 for s2, _ in roots]
    n_modes_at_root, abs_det_C_at_root = [], []
    for s2 in s2_roots:
        # a lam_by_s2 hit: _crossing_in_bracket already verified the root
        # at this exact s2
        eigenvalue_re, accuracy_floor, eigenvalue_im, det_C = _lam_at_node(
            engine, n, s2, lam_by_s2)
        real_eigenvalues = np.where(eigenvalue_im <= accuracy_floor, eigenvalue_re, np.nan)
        n_modes_at_root.append(modes_at_root(real_eigenvalues, accuracy_floor))
        abs_det_C_at_root.append(abs(det_C))
    return s2_roots, dict(
        n_scan=len(grid), N=int(lam.shape[1]), max_abs_imag=float(np.max(imag)),
        modes_at_root=n_modes_at_root,
        abs_det_C_at_root=abs_det_C_at_root,
        unresolved=unresolved,
        near_zero_at_low_end=_descending_at_low_end(np.where(imag <= floor, lam, np.nan)),
        n_solves=len(lam_by_s2),
        n_complex=int(sum(np.count_nonzero(imag_node > floor_node)
                          for _, floor_node, imag_node, _ in lam_by_s2.values())))


# --------------------------------------------------------------------------- #
#  THE STORED CENSUS: the census keys a bounce npz carries                     #
# --------------------------------------------------------------------------- #
# Every key the census (bounce_vgeneral_phi_v2.bounce_census) writes into a
# bounce npz -- the schema test of stored_census below; a file that lacks one
# of them is never read with guessed values.
CENSUS_SCHEMA_KEYS = ('census_accept', 'census_n0_s2', 'census_n1_s2',
                      'census_n0_modes', 'census_n1_modes',
                      'census_n0_resid', 'census_n1_resid',
                      'census_n0_unresolved', 'census_n1_unresolved',
                      'census_symmetry',
                      'census_s2_c', 'census_zero_mode_tol',
                      'census_zero_mode_trust')


def stored_census(npz):
    """The census an OPEN bounce npz (or its plain dict) carries
    (census_from_stored), or None when it lacks one of CENSUS_SCHEMA_KEYS.
    THE ONE key test; what its callers do with None: the [SKIP] resume arm
    of bounce_vgeneral_phi_v2 re-solves the file, the selection gate there
    (assert_census_accepted) passes it untouched, eigenvalue_neg /
    eigenvalue_zero census it again with the same locator, assemble_lnD
    reports that no stored verdict exists, and the gallery menu marks it
    'older census schema' (its pictures are drawn like an accepted
    bounce's).  The file itself is never changed."""
    if any(k not in npz for k in CENSUS_SCHEMA_KEYS):
        return None
    return census_from_stored(npz)


def census_from_stored(b):
    """The census_* keys of a stored bounce npz (an OPEN npz or its plain
    dict) as the typed census dict bounce_census returns, under the SAME key
    names -- the ONE reader of the stored census (called by stored_census
    and census_violations).  A pure conversion: a missing key fails loud
    here (KeyError) -- delete such a file and let bounce_vgeneral_phi_v2
    re-solve the pair."""
    return dict(
        census_accept=bool(np.asarray(b['census_accept'])),
        census_n0_s2=np.atleast_1d(np.asarray(b['census_n0_s2'], float)),
        census_n1_s2=np.atleast_1d(np.asarray(b['census_n1_s2'], float)),
        census_n0_modes=np.atleast_1d(np.asarray(b['census_n0_modes'], int)),
        census_n1_modes=np.atleast_1d(np.asarray(b['census_n1_modes'], int)),
        census_n0_resid=float(b['census_n0_resid']),
        census_n1_resid=float(b['census_n1_resid']),
        census_n0_unresolved=int(np.asarray(b['census_n0_unresolved'])),
        census_n1_unresolved=int(np.asarray(b['census_n1_unresolved'])),
        census_symmetry=str(np.asarray(b['census_symmetry'])),
        census_s2_c=float(b['census_s2_c']),
        census_zero_mode_tol=float(b['census_zero_mode_tol']),
        census_zero_mode_trust=float(b['census_zero_mode_trust']),
    )


# --------------------------------------------------------------------------- #
#  REPORTING: the inventory printer, the violation lines, the gate messages    #
# --------------------------------------------------------------------------- #
def _shared_note(n_modes, census):
    """The sentence for a crossing SHARED by more than one mode: how many,
    with the field symmetry bounce_vgeneral_phi_v2.profile_symmetry detected
    (measured separately from the multiplicity), when it found one."""
    if int(n_modes) <= 1:
        return ''
    sym = str(census['census_symmetry'])
    return (f"  [{int(n_modes)} modes at this same s2, "
            + (f"detected symmetry: {sym}]" if sym else "degenerate]"))


def print_census_inventory(census, name, phi_b):
    """Render the mode inventory (n <= 1) of a census dict (keys
    census_*): one line per located crossing, its s2 and its eigenvalue
    omega^2 = -s2.  ONE printer (solve time, the [SKIP] resume arm,
    eigenvalue_neg / eigenvalue_zero), so the log never depends on which arm
    printed it.  `phi_b` (the primed profile) adds the
    per-component displacement fractions |Delta phi_i|/|Delta phi| IN THE
    PRIMED FRAME (the false vacuum's mass eigenbasis -- the basis the census
    matrices live in, rotated away from the declared field names), so a
    near-spectator channel (~0) is visible next to the eigenvalues."""
    print(f"[CENSUS {name}] mode inventory (n <= 1); each crossing IS an "
          f"eigenvalue omega^2 = -s2:")
    displacement = np.abs(np.asarray(phi_b, float)[0])   # the false vacuum is the primed origin
    if np.linalg.norm(displacement) > 0.0:
        print("    per-component displacement |Delta phi_i|/|Delta phi| = "
              + ", ".join(f"{v:.3f}" for v in displacement / np.linalg.norm(displacement))
              + "  (PRIMED frame = the false vacuum's mass eigenbasis, the "
                "basis the C matrices live in -- NOT the declared field "
                "names; ~0 = near-spectator channel)")
    s2_roots_n0 = np.atleast_1d(census['census_n0_s2'])
    for s2v, n_modes in zip(s2_roots_n0, np.atleast_1d(census['census_n0_modes'])):
        print(f"    n=0: omega^2 = {-s2v:+.10e}  (s2 = {s2v:+.6e}), "
              f"degeneracy (0+1)^2 = 1"
              + ("  -- negative mode" if s2v > 0 else "")
              + _shared_note(n_modes, census))
    s2_roots_n1 = np.atleast_1d(census['census_n1_s2'])
    n1_labels = n1_crossing_labels(s2_roots_n1, census['census_s2_c'],
                                   census['census_zero_mode_tol'])
    for s2v, n_modes, what in zip(s2_roots_n1,
                                  np.atleast_1d(census['census_n1_modes']),
                                  n1_labels):
        print(f"    n=1: omega^2 = {-s2v:+.10e}  (s2 = {s2v:+.6e}), "
              f"degeneracy (1+1)^2 = 4 -- {what}" + _shared_note(n_modes, census))
    for sector in (0, 1):
        u = int(census[f'census_n{sector}_unresolved'])
        if u:
            print(f"    n={sector}: {u} UNRESOLVED near-zero region(s) -- an "
                  f"eigenvalue came close to zero and no crossing could be "
                  f"located there; counted as no mode, and the bounce is "
                  f"SPECIAL because of it")
    print("    sectors n >= 2 were NOT scanned: this list is the n <= 1 "
          "inventory, not a total Morse index.")


def dangerous_n1_mask(s2_roots, s2_c, zero_mode_tol):
    """Which located n=1 crossings are DANGEROUS -- the ONE owner of the rule
    the verdict (bounce_vgeneral_phi_v2.bounce_census), the backstop
    (eigenvalue_zero_vgeneral_phi_v2) and the evidence lines below apply.
    A crossing is an eigenvalue omega^2 = -s2 of the n=1 sector:
      s2 >= -zero_mode_tol   a zero mode within the resolution (the
                             translation family, or a second one) or a
                             NEGATIVE mode -- a pole deg/(omega^2 + s^2) on
                             the s^2 integration path: dangerous;
      s2 <  -zero_mode_tol   an ordinary POSITIVE bound state of the sector.
                             It sits inside the resolvent like every positive
                             eigenvalue, contributes no pole on s^2 >= 0 and
                             needs no subtraction: not dangerous.
    The translation crossing s2_c always counts, wherever bounce error put it."""
    roots = np.asarray(s2_roots, float)
    return (roots >= -float(zero_mode_tol)) | (roots == float(s2_c))


def n1_crossing_labels(s2_roots, s2_c, zero_mode_tol):
    """What each located n=1 crossing is -- the ONE labelling both census
    printers use (zero_mode_census, print_census_inventory), by the rule of
    dangerous_n1_mask: the translation zero mode (exactly ONE entry, picked
    BY INDEX as the crossing nearest s2 = 0, so a second crossing at the same
    s2 is labelled as the extra crossing it is), an extra DANGEROUS crossing,
    or a positive bound state."""
    roots = np.atleast_1d(np.asarray(s2_roots, float))
    dangerous = dangerous_n1_mask(roots, s2_c, zero_mode_tol)
    i_zm = int(np.argmin(np.abs(roots))) if roots.size else -1
    labels = []
    for i, s2v in enumerate(roots):
        if i == i_zm:
            labels.append('the translation zero mode (exactly 0 for the exact '
                          'bounce; displaced by bounce error)')
        elif not dangerous[i]:
            labels.append('a positive bound state (omega^2 > 0; stays inside '
                          'the resolvent)')
        else:
            labels.append('an extra DANGEROUS n=1 crossing'
                          + (' (at the same s2 as the translation crossing)'
                             if s2v == float(s2_c) else ''))
    return labels


def census_violations(b):
    """Human-readable lines naming everything a stored bounce npz dict `b`
    carries beyond the accepted census (ONE negative mode in n=0, ONE
    translation zero mode in n=1; positive n=1 bound states are no
    violation) -- the evidence in the selection-gate abort message: extra
    dangerous crossings, modes SHARING one crossing, and any near-zero region
    the locator could not settle."""
    census = census_from_stored(b)
    lines = []
    n0_s2, n1_s2 = census['census_n0_s2'], census['census_n1_s2']
    sym = census['census_symmetry']
    n1_dangerous = dangerous_n1_mask(n1_s2, census['census_s2_c'],
                                     census['census_zero_mode_tol'])
    for sector, s2s, modes in (('n=0', n0_s2, census['census_n0_modes']),
                               ('n=1', n1_s2[n1_dangerous],
                                census['census_n1_modes'][n1_dangerous])):
        for s2v, n_modes in zip(s2s, modes):
            if int(n_modes) > 1:
                lines.append(f"    {sector} crossing at omega^2 = {-s2v:+.6e} "
                             f"is shared by {int(n_modes)} modes"
                             + (f" (symmetry: {sym})" if sym else ""))
    for s2v in n0_s2[1:]:
        lines.append(f"    extra n=0 crossing (negative mode): omega^2 = "
                     f"{-s2v:+.6e}")
    # skip the INDEX of the translation crossing: a second dangerous n=1
    # crossing at the same s2 is a violation and must appear in the evidence
    i_zm = int(np.argmin(np.abs(n1_s2))) if n1_s2.size else -1
    for i, s2v in enumerate(n1_s2):
        if i != i_zm and n1_dangerous[i]:
            lines.append(f"    extra dangerous n=1 crossing: omega^2 = "
                         f"{-s2v:+.6e}")
    for sector in (0, 1):
        u = int(census[f'census_n{sector}_unresolved'])
        if u:
            lines.append(f"    {u} UNRESOLVED near-zero region(s) in n={sector}"
                         f" -- an eigenvalue came close to zero there and the "
                         f"refinement could not settle it into a crossing "
                         f"before reaching C's accuracy floor, so a mode may "
                         f"be hiding in it")
    return lines


# THE SELECTION-GATE MESSAGES (one owner; bounce_vgeneral_phi_v2,
# eigenvalue_neg and eigenvalue_zero all print these, so a refused bounce is
# refused in the same words wherever it is caught).
# the resolvent pole sits at s^2 = -omega^2, hence the subtraction
# deg/(omega^2 + s^2)
OPEN_PROBLEM_MSG = (
    "This bounce is a genuine stationary point of the Euclidean action, but "
    "the lnD assembly of this\npipeline is written for the index-1 spectrum "
    "only: ONE negative mode in n=0 and ONE translation\nzero mode in n=1 "
    "(four angular copies).  Every further DANGEROUS crossing -- a second "
    "n=0 crossing,\nor an n=1 crossing at omega^2 = -s2 <= 0 -- is one more "
    "resolvent pole deg/(omega^2 + s^2) on the\ns^2 integration path that "
    "needs its own subtraction and closed-form restoration inside the\n"
    "dangerous-sector quadratures.  Moreover the rate of a bounce with N_- "
    "negative modes, counted with\ntheir degeneracy (N_- = sum_n (n+1)^2 x "
    "the negative modes of wave n), carries the factor\n1 - (-1)^N_-: it "
    "vanishes for even N_-, and for odd N_- > 1 its steepest-descent weight "
    "is\nundetermined.  Positive bound states (omega^2 > 0) are not "
    "dangerous: they stay inside the\nresolvent.  Not implemented -- an OPEN "
    "PROBLEM.  Select an index-1 bounce from the gallery instead.")
LEG_POINTER_MSG = (
    "\nThis potential is DECOUPLED (no cross-term couplings).  If several of "
    "its fields tunnel in\nthis bounce, the decay of this false vacuum is the "
    "SUM of the per-leg rates, and the\none-flip legs are their own gallery "
    "entries -- select a leg instead.  (A bounce that\nmoves one field "
    "already is such a leg; this hint does not apply to it.)")


# --------------------------------------------------------------------------- #
#  The two sector censuses (n = 0 negative mode, n = 1 zero mode) and their   #
#  DERIVED scan windows -- called by bounce_vgeneral_phi_v2 for every solved  #
#  bounce (THE census; eigenvalue_neg/eigenvalue_zero read it from the npz    #
#  and call these only for a bounce npz without the census keys)              #
# --------------------------------------------------------------------------- #
CENSUS_SCAN_PTS = 65        # coarse-scan points; every sign change or approach
                            # to zero is then refined on its own finer grid
S2_CEILING_MARGIN = 1.1     # margin on the variational ceiling: lambda_min of
                            # H_b is sampled on the finite radial grid
CONTINUUM_MARGIN = 0.9      # the n=1 window stops this fraction of the way to
                            # the continuum threshold s2 = -min_i m_i^2
CONTINUUM_WIDEN_ROUNDS = 3  # watcher: halve the remaining distance to the
                            # threshold and re-census, at most this often
# the two ZERO-MODE quality levels, fractions of the n=1 window's reach
# below zero (zero_mode_levels): tol = ZERO_MODE_TOL_FRAC x reach,
# trust = ZERO_MODE_TRUST_FRAC x reach
ZERO_MODE_TOL_FRAC = 1e-3
ZERO_MODE_TRUST_FRAC = 1e-8


def variational_s2_ceiling(U_int, m2):
    """The largest s2 = -omega^2 at which any det C crossing of any sector can
    sit -- the ONE owner of this bound.
        M_n = -d^2/dr^2 - (3/r) d/dr + n(n+2)/r^2 + H_b(r),
        H_b = U_int + diag(m2) = H'(phi_b(r)), the full Hessian of V,
    the kinetic and centrifugal terms are >= 0, so for every normalized mode
        omega^2 = <phi|M_n|phi> >= min_r lambda_min(H_b(r)) =: lam_min,
    i.e. every crossing obeys s2 = -omega^2 <= -lam_min.  Returns
    S2_CEILING_MARGIN * |lam_min|; aborts if lam_min >= 0 (no negative
    eigenvalue at all: the profile is not a bounce)."""
    H_b = np.asarray(U_int, float) + np.diag(np.asarray(m2, float))
    lam_min = float(np.linalg.eigvalsh(H_b).min())
    if lam_min >= 0.0:
        raise RuntimeError(
            f'[ABORT EIG-WINDOW] the field-space Hessian along this profile is '
            f'positive semidefinite everywhere (min over r of its smallest '
            f'eigenvalue = {lam_min:+.6e} >= 0), so the fluctuation operator '
            f'has NO negative eigenvalue and no det C crossing at s2 > 0 '
            f'exists.  This profile is not a bounce.')
    return float(S2_CEILING_MARGIN * abs(lam_min))


def negative_mode_window(engine, U_int, m2):
    """The n=0 scan window (s2_lo, s2_hi), both ends DERIVED:
    s2_hi = the variational ceiling (variational_s2_ceiling); s2_lo = the
    dimensionless accuracy floor, at s2 = 0, of the eigenvalue of C nearest
    zero (the lambda_k a negative mode crosses on) times s2_hi -- the lower end
    of the geometric scan in s2 units, not a root-error estimate."""
    s2_hi = variational_s2_ceiling(U_int, m2)
    _, lam_C, floor = constant_wronskian_and_floor(engine, 0, 0.0)
    s2_lo = floor[np.argmin(np.abs(lam_C))] * s2_hi
    if s2_lo >= s2_hi:
        raise RuntimeError(
            f'[ABORT EIG-WINDOW] the derived n=0 scan window is empty: the '
            f'scaled lower cutoff is {s2_lo:.6e}, at or above the '
            f'variational ceiling {s2_hi:.6e}.  Inspect the bounce (radial '
            f'range, grid) before reading any census from it.')
    return s2_lo, s2_hi


def zero_mode_window_lo(s2_ceiling, m_min2, widening):
    """The low end of the n=1 census window after `widening` widenings,
    -min(s2_ceiling, margin x m_min2): the margin starts at CONTINUUM_MARGIN
    of the way to the continuum threshold s2 = -m_min2 (there kappa_i =
    sqrt(m_i^2 + s2) vanishes and no bound state is left), and each widening
    halves the distance left, margin = 1 - (1 - CONTINUUM_MARGIN)/2^widening.
    The ONE rule of zero_mode_census's window and of the n=1 det-crossing
    scan (eigenvalue_zero_vgeneral_phi_v2, at the deepest reach
    CONTINUUM_WIDEN_ROUNDS)."""
    margin = 1.0 - (1.0 - CONTINUUM_MARGIN) / 2**widening
    return -min(s2_ceiling, margin * m_min2)


def zero_mode_levels(s2_ceiling, m_min2):
    """The two ZERO-MODE levels of a bounce, (tol, trust) -- the ONE rule
    bounce_census and eigenvalue_zero derive them by.  Both are fractions of
    the reach of the n=1 window below zero,
        reach = min(s2_ceiling, CONTINUUM_MARGIN x m_min2),
    the s2 range in which a positive n=1 bound state (0 < omega^2 < m_min2,
    below the continuum threshold) can sit.  tol is the boundary of
    dangerous_n1_mask and the ZERO-MODE abort/warn level; trust is the
    level below which |s2_c| counts as converged.  Scaling both with the
    reach keeps tol a small fraction of the bound-state range for every
    model, however light its lightest channel."""
    reach = -zero_mode_window_lo(s2_ceiling, m_min2, 0)
    return ZERO_MODE_TOL_FRAC * reach, ZERO_MODE_TRUST_FRAC * reach


def _print_census_lines(n, s2_roots, n_modes_at_root, census_report, lo,
                        labels):
    """The per-crossing lines every sector census prints: one line per
    located crossing (with its entry of `labels`), one per unresolved region,
    the low-end flag (`lo` = the window's low end), and the scan summary."""
    for i, (s2_root, n_modes, label) in enumerate(zip(s2_roots, n_modes_at_root,
                                                      labels)):
        print(f"    crossing {i+1}: s2 = {s2_root:+.10e}  "
              f"omega^2 = {-s2_root:+.10e}  "
              + (f"({n_modes} modes share it)" if n_modes > 1 else "(1 mode)")
              + f" -- {label}")
    for a, b, why in census_report['unresolved']:
        print(f"    [UNRESOLVED n={n}] near-zero region s2 = [{a:.10g}, "
              f"{b:.10g}]: {why} -- reported, never counted as a mode; it "
              f"makes the bounce SPECIAL where a dangerous crossing could "
              f"hide in it (every n=0 region; an n=1 region only where it "
              f"reaches s2 >= -tol, the zero-mode level)")
    if census_report['near_zero_at_low_end']:
        print(f"    [UNRESOLVED n={n}] an eigenvalue of C is still "
              f"descending toward zero as it leaves the window at the low "
              f"end {lo:.6e} -- reported, not answered")
    print(f"    census scan: {census_report['n_scan']} coarse points, "
          f"{census_report['N']} eigenvalues of C (the "
          f"matrix dimension N), {census_report['n_solves']} h-ODE solves, "
          f"{len(census_report['unresolved'])} unresolved region(s), "
          f"max |Im lambda| {census_report['max_abs_imag']:.2e}"
          + (f"; {census_report['n_complex']} complex eigenvalue "
             f"sample(s) (real parts followed, no zero there)"
             if census_report['n_complex'] else ""))


def negative_mode_census(engine, s2_lo, s2_hi):
    """The n=0 mode census on the DERIVED window [s2_lo, s2_hi] (geometric
    coarse scan: the window spans many decades): locate every zero crossing
    of an eigenvalue of C, print them with their eigenvalues omega^2 = -s2,
    and return (s2_roots, n_modes_at_root, crossing_resid, n_unresolved) --
    what bounce_census stores (keys census_n0_*); crossing_resid = |det C|
    at the first located crossing.  omega2_00 = -s2_roots[0] is formed only
    for an index-1 bounce, from its ONE crossing (eigenvalue_neg_vgeneral_phi_v2,
    after its backstop).  This function counts and reports; the CALLER
    (bounce_census) decides.  No crossing at all aborts: not a bounce, or
    UNRESOLVED when the census left a near-zero region unsettled."""
    s2_roots, census_report = census_sector(
        engine, 0, np.geomspace(s2_lo, s2_hi, CENSUS_SCAN_PTS))
    if not s2_roots:
        raise RuntimeError(
            f'[ABORT COLEMAN-N0] no n=0 eigenvalue of the constant Wronskian was '
            f'found zero anywhere on the derived scan window '
            f'[{s2_lo:.6e}, {s2_hi:.6e}] (the variational ceiling above, '
            f'the scaled lower cutoff below): '
            + ('no negative mode found, so this profile is not a bounce.'
               if not census_report['unresolved'] else
               f'UNRESOLVED, not a verdict -- '
               f'{len(census_report["unresolved"])} near-zero region(s) left '
               f'unsettled.  A known cause is an eigenvalue that stays at its '
               f'accuracy floor, a limit of this census when a much heavier '
               f'channel is coupled to the tunnelling one.'))
    n_modes_at_root = [int(m) for m in census_report['modes_at_root']]
    shared = [m for m in n_modes_at_root if m > 1]
    if len(s2_roots) == 1 and not shared:
        print(f"[COLEMAN-N0] 1 crossing, index 1 (the one negative mode of a "
              f"valid decay bounce) on the scan [{s2_lo:.6e}, {s2_hi:.6e}]")
    else:
        print(f"[COLEMAN-N0] {len(s2_roots)} crossing(s) on the scan "
              f"[{s2_lo:.6e}, {s2_hi:.6e}]"
              + (f", {len(shared)} of them shared by more than one mode"
                 if shared else "")
              + ": NOT the index-1 single-wall bounce; the caller acts on it.")
    _print_census_lines(0, s2_roots, n_modes_at_root, census_report, s2_lo,
                        ['a negative mode'] * len(s2_roots))
    n_unresolved = (len(census_report['unresolved'])
                    + int(bool(census_report['near_zero_at_low_end'])))
    # |det C| at the first located crossing -- THE crossing of an index-1
    # bounce: its reported residual, stored by bounce_census as
    # census_n0_resid
    crossing_resid = census_report['abs_det_C_at_root'][0]
    return s2_roots, n_modes_at_root, crossing_resid, n_unresolved


def zero_mode_census(engine, s2_ceiling, zero_mode_tol):
    """The n=1 mode census on a LINEAR window STRADDLING zero: the upper end
    is the variational ceiling `s2_ceiling` (variational_s2_ceiling, computed
    by the caller), the lower end zero_mode_window_lo (CONTINUUM_MARGIN of
    the way to the continuum threshold s2 = -min_i m_i^2, or -s2_ceiling when
    that is closer to zero).  The window straddles zero because the
    translation zero mode (omega^2 = 0 exactly for a true bounce) lands
    numerically on either side.  WATCHER: a lambda_k still descending at the
    low end widens the window (halving the remaining distance to the
    threshold, at most CONTINUUM_WIDEN_ROUNDS times).  The
    translation crossing is the located root NEAREST s2 = 0.
    `zero_mode_tol` is the zero-mode level of THIS bounce, derived by the
    caller (zero_mode_levels; bounce_census stores it as
    census_zero_mode_tol): the boundary of dangerous_n1_mask, used here to
    label the crossings and to decide which unresolved regions count.
    Returns (s2_c, s2_roots, n_modes_at_root, crossing_resid, n_unresolved)
    -- what bounce_census stores (keys census_s2_c, census_n1_*);
    crossing_resid = |det C| at s2_c."""
    m_min2 = float(np.min(engine.m2))
    for widening in range(CONTINUUM_WIDEN_ROUNDS + 1):
        lo = zero_mode_window_lo(s2_ceiling, m_min2, widening)
        print(f"[EIG-WINDOW] n=1 census on [{lo:.6e}, {s2_ceiling:.6e}] "
              f"(linear scan straddling zero; the continuum threshold is "
              f"s2 = -min_i m_i^2 = {-m_min2:.6e})")
        s2_roots, census_report = census_sector(
            engine, 1, np.linspace(lo, s2_ceiling, CENSUS_SCAN_PTS))
        if (not census_report['near_zero_at_low_end']
                or widening == CONTINUUM_WIDEN_ROUNDS):
            break
        print(f"[EIG-WINDOW] an eigenvalue of C is still descending toward "
              f"zero at the low end {lo:.6e}: halving the distance to the "
              f"threshold and re-censusing (round {widening + 1} of "
              f"{CONTINUUM_WIDEN_ROUNDS})")
    if not s2_roots:
        raise RuntimeError(
            f'[ABORT COLEMAN-N1] no n=1 eigenvalue of the constant '
            f'Wronskian crosses zero anywhere on [{lo:.6e}, {s2_ceiling:.6e}], '
            f'a window straddling s2 = 0: no translation zero mode, so not a '
            f'bounce of this potential (a genuine bounce has the four '
            f'translation zero modes at s2_c = 0).')
    i_zm = int(np.argmin([abs(s2_root) for s2_root in s2_roots]))
    s2_c = s2_roots[i_zm]
    if len(s2_roots) == 1:
        print(f"[COLEMAN-N1] 1 crossing on [{lo:.6e}, {s2_ceiling:.6e}]: the "
              f"translation zero mode at s2={s2_c:.3e} ((1+1)^2 = 4 "
              f"translation zero modes), 0 extra n=1 crossings")
    else:
        print(f"[COLEMAN-N1] {len(s2_roots)} crossing(s) on "
              f"[{lo:.6e}, {s2_ceiling:.6e}]; the translation zero mode is "
              f"the one nearest s2 = 0, at s2={s2_c:.3e}.  The caller decides "
              f"what the extra crossing(s) mean.")
    n_modes_at_root = [int(m) for m in census_report['modes_at_root']]

    _print_census_lines(1, s2_roots, n_modes_at_root, census_report, lo,
                        n1_crossing_labels(s2_roots, s2_c, zero_mode_tol))
    # an unresolved near-zero region counts only where it could hide a
    # DANGEROUS crossing, s2 >= -zero_mode_tol (dangerous_n1_mask); below
    # that it could only hide a positive bound state.  The low end of this
    # window, lo <= -min(ceiling, CONTINUUM_MARGIN x m_min2) = -reach, lies
    # 1/ZERO_MODE_TOL_FRAC times farther below zero than -zero_mode_tol
    # (zero_mode_levels), so a lambda_k still descending there points at a
    # positive bound state, never at a dangerous one.
    n_unresolved = sum(1 for a, b, why in census_report['unresolved']
                       if b >= -zero_mode_tol)
    n_below = len(census_report['unresolved']) - n_unresolved
    if n_below:
        print(f"    {n_below} unresolved n=1 region(s) lie entirely below "
              f"s2 = -zero_mode_tol = {-zero_mode_tol:.4e}: they could only "
              f"hide a positive bound state and do not count toward the verdict")
    # |det C| at the translation crossing: its reported residual, stored by
    # bounce_census as census_n1_resid
    crossing_resid = census_report['abs_det_C_at_root'][i_zm]
    return s2_c, s2_roots, n_modes_at_root, crossing_resid, n_unresolved
