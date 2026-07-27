"""potential_coupled_toy_model.py -- THE potential of the coupled toy model:
TWO scalar fields (x, y), a HARDCODED degree-4 polynomial, and the HARDCODED
vacua of that potential.

WHAT THIS FILE IS (and what it deliberately is not)
---------------------------------------------------
This is the simplified, single-model descendant of the general N-field
potential module of pipeline_rk_vgeneral_phi.  Everything that made that file
GENERAL is gone: there is no field-number freedom (it is always the two fields
x and y), no user --fields/--couplings interface, and -- the big one -- NO
VACUUM SEARCH.  The general pipeline Newton-solved grad V = 0 from a seed grid
to discover the minima and then enumerated every downward (false, true) pair.
Here the minima and the pair the pipeline runs are simply WRITTEN DOWN, because
the potential is fixed and its vacua are known numbers.

Two models are shipped, and they differ ONLY in their coupling dict:

  'F2_T0'  the COUPLED toy model (the default, and the reason this pipeline
           exists).  Cross terms x*y^2 and x^2*y^2 are present, so the Hessian
           H_ij(x,y) is NOT diagonal: the two fields genuinely mix, the bounce
           path CURVES in field space, and stage 0 must use the two-field
           CosmoTransitions path deformation.
               V = -0.8 + (x^2-1)^2 + (y^2-1)^2 + x*y^2 + 0.15*x^2*y^2
  'bdet'   the DECOUPLED companion (same quartic wells, cross terms switched
           OFF, cubic terms +0.5x^3 + 0.5y^3 added to split the four minima).
           V = sum_i V_i(phi_i) exactly, so the Hessian is diagonal EVERYWHERE,
           the bounce path is a STRAIGHT LINE along the tunneling field's axis,
           and stage 0 uses the 1-D CosmoTransitions instanton instead.
               V = -0.8 + (x^2-1)^2 + (y^2-1)^2 + 0.5*x^3 + 0.5*y^3
           This is the single-field-equivalent model the BubbleDet package can
           be run on, which is what it is kept here for.

Nothing else in the pipeline changes between the two: the SAME band engine,
the SAME sectors, counterterms, tails and assembly run on either bounce.  The
coupled/decoupled distinction is confined to ONE decision -- which
CosmoTransitions solver stage 0 calls (bounce_coupled_toy_model.solve_bounce).

THE COEFFICIENT VECTOR `params`
-------------------------------
Both models are expressed in the SAME fixed basis of the 15 monomials of total
degree <= 4 in (x, y), in the order MONOMIAL_ORDER below.  `params` is that
ordered coefficient vector; it travels with the bounce (written into the bounce
npz), and pipeline_helpers_coupled_toy_model.load_bounce rebuilds this
potential from the STORED vector, so a stage never re-reads MODELS -- it uses
exactly the potential the bounce was solved with.  The layout is byte-identical
to the vgeneral_phi one, so the shipped bounce npz drop straight in.

The constant monomial 'c' is irrelevant to ln D (the primed frame subtracts V
at the false vacuum), and CTShiftedLiftedPotential strips it analytically.
"""
from __future__ import annotations
import numpy as np
import sympy as sp

# --------------------------------------------------------------------------- #
#  1. The FIXED two-field degree-<=4 basis                                     #
# --------------------------------------------------------------------------- #
FIELDS = ('x', 'y')                     # the two coupled scalar fields
N_FIELDS = 2                            # never a variable in this pipeline

# The 15 monomials of total degree <= 4 in (x, y), ordered by total degree and
# then DESCENDING lexicographically in the exponents.  This IS the order of the
# `params` vector stored in every bounce npz -- do not reorder it.
MONOMIAL_ORDER: tuple[str, ...] = (
    'c',                                     # degree 0
    'x', 'y',                                # degree 1
    'x2', 'xy', 'y2',                        # degree 2
    'x3', 'x2y', 'xy2', 'y3',                # degree 3
    'x4', 'x3y', 'x2y2', 'xy3', 'y4',        # degree 4
)
N_PARAMS = len(MONOMIAL_ORDER)               # 15

# The monomials that couple the two fields.  A coefficient vector whose entries
# on THESE monomials all vanish has a DIAGONAL Hessian everywhere, i.e. the
# potential separates as V(x,y) = V_1(x) + V_2(y) -- that is what is_decoupled
# tests, and it is the single switch that sends stage 0 to the 1-D solver.
CROSS_TERMS: tuple[str, ...] = ('xy', 'x2y', 'xy2', 'x3y', 'x2y2', 'xy3')
_CROSS_MASK = np.array([m in CROSS_TERMS for m in MONOMIAL_ORDER], dtype=bool)

# --- symbolic V, gradient and 2x2 Hessian, built ONCE at import -------------
# sympy differentiates the generic degree-<=4 polynomial analytically and
# lambdify turns V, (dV/dx, dV/dy) and the 2x2 Hessian into numpy callables of
# (x, y, c_c, c_x, ..., c_y4).  Doing it symbolically (rather than typing the
# derivatives out) is what makes a coupling change impossible to get wrong.
_X, _Y = sp.symbols('x y', real=True)
_SYMS = (_X, _Y)
_MONOMIALS = {
    'c': sp.Integer(1),
    'x': _X,           'y': _Y,
    'x2': _X**2,       'xy': _X*_Y,       'y2': _Y**2,
    'x3': _X**3,       'x2y': _X**2*_Y,   'xy2': _X*_Y**2,   'y3': _Y**3,
    'x4': _X**4,       'x3y': _X**3*_Y,   'x2y2': _X**2*_Y**2,
    'xy3': _X*_Y**3,   'y4': _Y**4,
}
_PSYMS = sp.symbols(' '.join(f'c_{m}' for m in MONOMIAL_ORDER), real=True)
V_EXPR = sum(cs * _MONOMIALS[m] for cs, m in zip(_PSYMS, MONOMIAL_ORDER))
_GRAD_EXPR = tuple(sp.diff(V_EXPR, s) for s in _SYMS)
H_EXPR = sp.hessian(V_EXPR, _SYMS)                      # 2x2
_ARGS = _SYMS + tuple(_PSYMS)
_V_FUNC = sp.lambdify(_ARGS, V_EXPR, 'numpy')
_GRAD_FUNC = sp.lambdify(_ARGS, _GRAD_EXPR, 'numpy')
_H_FUNC = sp.lambdify(_ARGS, H_EXPR, 'numpy')


def couplings_to_params(couplings: dict) -> np.ndarray:
    """The ordered 15-vector from a {monomial_name: value} dict; unlisted
    monomials are 0.  An unknown name raises, so a typo like 'x5' fails loud
    rather than being silently ignored."""
    bad = set(couplings) - set(MONOMIAL_ORDER)
    if bad:
        raise RuntimeError(
            f'[ABORT] unknown potential monomial(s) {sorted(bad)} -- the valid '
            f'degree-<=4 names in the two fields {FIELDS} are '
            f'{list(MONOMIAL_ORDER)}.')
    return np.array([float(couplings.get(m, 0.0)) for m in MONOMIAL_ORDER],
                    dtype=float)


# --------------------------------------------------------------------------- #
#  2. THE TWO HARDCODED MODELS (couplings + vacua + the pair that is run)      #
# --------------------------------------------------------------------------- #
# Each entry gives, for one fixed potential:
#   couplings  the degree-<=4 monomial coefficients;
#   minima     ALL local minima of that potential, as (x, y) pairs, sorted by
#              V ASCENDING with coordinate tie-break -- exactly the order the
#              general pipeline's seed-grid search produced, so the index i in
#              the bounce file name bounce_data_F<iF>_T<iT>.npz means the same
#              vacuum here as it did there (and as it does in the shipped npz);
#   pair       (false_index, true_index): THE bounce this pipeline computes;
#   why        one line on why that pair.
# The coordinates are the 8-decimal values the shipped bounce npz were solved
# with -- they are reproduced here EXACTLY, so the from-scratch solve and the
# shipped profile describe the same physical bounce.  guard_vacua() below
# re-verifies that each one really is a minimum of the couplings next to it.
MODELS = {
    # ---- the coupled toy model: cross terms x*y^2 and x^2*y^2 present -------
    'F2_T0': dict(
        couplings={'c': 1.2,        # = -0.8 + 1 + 1  (cancels in ln D)
                   'x2': -2.0, 'y2': -2.0,
                   'xy2': 1.0, 'x2y2': 0.15,
                   'x4': 1.0, 'y4': 1.0},
        minima=((-1.10497208, -1.20868267),      # M0  V = -1.885444  (deepest)
                (-1.10497208, +1.20868267),      # M1  V = -1.885444
                (+0.91207670, -0.69395271),      # M2  V = -0.003647
                (+0.91207670, +0.69395271)),     # M3  V = -0.003647  (highest)
        pair=(2, 0),
        why=('the certified F2_T0 bounce: from the shallow minimum M2 down to '
             'the global minimum M0.  It is the CHEAPEST of the four downward '
             'pairs (S = 282.86 against 2750.83 for the y-flipping ones), so '
             'it is the one that dominates the decay rate.'),
    ),
    # ---- the decoupled companion: no cross terms, cubics split the minima ---
    'bdet': dict(
        couplings={'c': 1.2,
                   'x2': -2.0, 'y2': -2.0,
                   'x3': 0.5, 'y3': 0.5,
                   'x4': 1.0, 'y4': 1.0},
        minima=((-1.20492629, -1.20492629),      # M0  V = -2.141037  (deepest)
                (-1.20492629, +0.82992629),      # M1  V = -1.087842
                (+0.82992629, -1.20492629),      # M2  V = -1.087842
                (+0.82992629, +0.82992629)),     # M3  V = -0.034647  (highest)
        pair=(3, 1),
        why=('the BubbleDet reference bounce: from the global false vacuum M3 '
             'down to M1, flipping the x field ONLY (m_x^2 = 6.7551) while y '
             'sits frozen as a spectator.  Exactly ONE field moves, which is '
             'what makes it a genuine index-1 bounce and the input the '
             'single-field BubbleDet package consumes.  Flipping BOTH fields '
             '(M3 -> M0) is a product saddle with two negative modes and is '
             'rejected downstream by the COLEMAN guards.'),
    ),
}
MODEL_DEFAULT = 'F2_T0'


def model_names():
    """The names of the hardcoded models, default first."""
    return [MODEL_DEFAULT] + [k for k in MODELS if k != MODEL_DEFAULT]


def get_model(name=None):
    """The hardcoded model `name` (default MODEL_DEFAULT) as a dict with
        params   (15,) the ordered coefficient vector
        minima   (n_min, 2) all local minima, V ascending
        V_min    (n_min,) their potential values
        pair     (iF, iT) the bounce this pipeline computes
        pairs    every downward (false, true) pair, cheapest bounce first is
                 NOT implied -- they are listed in (iF, iT) order
        decoupled  bool: does the Hessian separate?
    Everything is derived from the WRITTEN-DOWN numbers; nothing is searched
    for.  The vacua are checked (guard_vacua) the first time a model is
    requested."""
    name = MODEL_DEFAULT if name is None else str(name)
    if name not in MODELS:
        raise RuntimeError(f'[ABORT] unknown model {name!r} -- this pipeline '
                           f'ships exactly {model_names()}.')
    spec = MODELS[name]
    params = couplings_to_params(spec['couplings'])
    minima = np.asarray(spec['minima'], float)
    V_min = np.array([float(V_numeric(pt, params)) for pt in minima])
    pairs = [(iF, iT) for iF in range(len(minima)) for iT in range(len(minima))
             if iF != iT and V_min[iF] > V_min[iT]]
    if tuple(spec['pair']) not in pairs:
        raise RuntimeError(
            f'[ABORT VACUA] model {name}: the hardcoded pair '
            f'F{spec["pair"][0]}_T{spec["pair"][1]} is not a DOWNWARD '
            f'(V_false > V_true) pair of its own vacua {pairs}.')
    return dict(name=name, params=params, minima=minima, V_min=V_min,
                pair=tuple(int(v) for v in spec['pair']), pairs=pairs,
                why=spec['why'], couplings=dict(spec['couplings']),
                decoupled=is_decoupled(params))


# --------------------------------------------------------------------------- #
#  3. Numeric V / gradient / Hessian                                           #
# --------------------------------------------------------------------------- #
def _unpack(params) -> tuple:
    p = tuple(np.asarray(params, float).ravel())
    if len(p) != N_PARAMS:
        raise RuntimeError(
            f'[ABORT] potential params has {len(p)} entries, expected '
            f'{N_PARAMS} (the degree-<=4 monomial coefficients of the two '
            f'fields {FIELDS}, in MONOMIAL_ORDER).')
    return p


def V_numeric(X, params):
    """V at one point or at a whole array of points (last axis = the two
    field components)."""
    p = _unpack(params)
    X = np.asarray(X, dtype=float)
    return _V_FUNC(X[..., 0], X[..., 1], *p)


def gradV_numeric(X, params) -> np.ndarray:
    """(dV/dx, dV/dy), same shape as X."""
    p = _unpack(params)
    X = np.asarray(X, dtype=float)
    dVx, dVy = _GRAD_FUNC(X[..., 0], X[..., 1], *p)
    g = np.empty_like(X, dtype=float)
    g[..., 0] = dVx
    g[..., 1] = dVy
    return g


def H_numeric(point, params) -> np.ndarray:
    """The 2x2 Hessian [[V_xx, V_xy], [V_xy, V_yy]] at ONE field point."""
    p = _unpack(params)
    return np.array(_H_FUNC(float(point[0]), float(point[1]), *p), dtype=float)


def H_numeric_batch(X, params) -> np.ndarray:
    """The Hessian at a whole array of points: ONE lambdify call, broadcast,
    stacked into a (..., 2, 2) tensor."""
    p = _unpack(params)
    X = np.asarray(X, dtype=float)
    xs, ys = X[..., 0], X[..., 1]
    H = _H_FUNC(xs, ys, *p)
    out = np.empty(np.shape(xs) + (2, 2), dtype=float)
    for i in range(2):
        for j in range(2):
            out[..., i, j] = np.broadcast_to(np.asarray(H[i][j], float),
                                             np.shape(xs))
    return out


def is_decoupled(params, tol: float = 0.0) -> bool:
    """True iff the potential has NO cross terms, i.e. every coefficient on
    CROSS_TERMS vanishes to within `tol`.  Then H is diagonal everywhere and
    V(x,y) = V_1(x) + V_2(y), so a pair whose two vacua differ in exactly ONE
    field has a genuinely 1-D bounce (the tunneling field moves, the other is
    a frozen spectator) -- and THAT is the case stage 0 hands to the 1-D
    CosmoTransitions instanton instead of the path-deformation solver."""
    p = np.asarray(params, float).ravel()
    if p.size != N_PARAMS:
        raise RuntimeError(f'[ABORT] is_decoupled needs the {N_PARAMS}-entry '
                           f'coefficient vector, got {p.size}.')
    return bool(np.all(np.abs(p[_CROSS_MASK]) <= tol))


def tunneling_fields(false_vac, true_vac, vac_tol: float = 1e-6):
    """The indices of the field components that actually MOVE between the two
    vacua (|phi_F,i - phi_T,i| > vac_tol).  For a decoupled potential the
    count matters physically: exactly one moving field is an index-1 bounce,
    two moving fields is a product saddle with two negative modes."""
    d = np.abs(np.asarray(true_vac, float) - np.asarray(false_vac, float))
    return [int(i) for i in np.nonzero(d > vac_tol)[0]]


# --------------------------------------------------------------------------- #
#  4. GUARD VACUA (aborts; does NOT enter lnD)                                 #
# --------------------------------------------------------------------------- #
# The vacua are hardcoded rather than searched for, so nothing in the code
# would notice if a coupling were edited and the written-down minima went
# stale.  This guard closes exactly that hole: every declared vacuum must
# still solve grad V = 0 and still have a positive-definite Hessian for the
# couplings it is declared next to, and the list must still be sorted the way
# the vacuum indices assume.  It runs once per model, from get_model's first
# caller (stage 0) and from this file's self-test.
def guard_vacua(model_name=None, grad_tol: float = 1e-6):
    """Verify the hardcoded vacua of a model against its hardcoded couplings.
    Raises on the first failure; returns the checked model dict."""
    m = get_model(model_name)
    params, minima, V_min = m['params'], m['minima'], m['V_min']
    for i, pt in enumerate(minima):
        g = float(np.max(np.abs(gradV_numeric(pt, params))))
        if g > grad_tol:
            raise RuntimeError(
                f'[ABORT VACUA] model {m["name"]}: the hardcoded vacuum M{i} = '
                f'{np.round(pt, 8)} is NOT a critical point of the hardcoded '
                f'couplings (|grad V| = {g:.2e} > {grad_tol:g}).  The couplings '
                f'and the vacua in MODELS have drifted apart -- fix them '
                f'together.')
        ev = np.linalg.eigvalsh(H_numeric(pt, params))
        if not np.all(ev > 0):
            raise RuntimeError(
                f'[ABORT VACUA] model {m["name"]}: the hardcoded vacuum M{i} = '
                f'{np.round(pt, 8)} is not a MINIMUM -- Hessian eigenvalues '
                f'{np.round(ev, 8)} are not all positive.')
    order = sorted(range(len(minima)),
                   key=lambda i: (V_min[i], tuple(minima[i])))
    if order != list(range(len(minima))):
        raise RuntimeError(
            f'[ABORT VACUA] model {m["name"]}: the hardcoded minima are not in '
            f'the V-ascending (coordinate tie-broken) order the vacuum indices '
            f'assume -- correct order would be {order}.')
    return m


# --------------------------------------------------------------------------- #
#  5. Shifted + rotated potential (the primed frame the bounce is solved in)   #
# --------------------------------------------------------------------------- #
class CTShiftedLiftedPotential:
    """V in the PRIMED frame phi' = L^T (phi - phi_F): the potential shifted so
    the chosen false vacuum phi_F is the origin and rotated (by the orthogonal
    2x2 L that diagonalizes the false-vacuum Hessian H_F = L Lambda L^T) into
    its mass eigenbasis.  V(0) is subtracted, so the false vacuum sits at
    V' = 0.  Exposes V, dV, H (and a batched H) in primed coordinates -- the
    interface the bounce solver and the fluctuation background use."""

    def __init__(self, params, false_vac, fields=None):
        # `fields` is accepted and ignored: this pipeline has exactly the two
        # fields FIELDS, and the argument only exists so a caller written
        # against the general (N-field) signature keeps working.
        self.fields = FIELDS
        self.N = N_FIELDS
        self.params = np.asarray(params, dtype=float)
        self.false_vac = np.asarray(false_vac, dtype=float)
        if self.false_vac.shape != (N_FIELDS,):
            raise RuntimeError(
                f'[ABORT] false_vac has shape {self.false_vac.shape}, expected '
                f'({N_FIELDS},) for the two fields {FIELDS}.')

        # Everything this class exposes is CONSTANT-BLIND: V is always the
        # difference V - V(false_vac), and dV/H are derivatives.  So the
        # constant monomial 'c' is stripped ANALYTICALLY before any numeric
        # evaluation (physics unchanged).  Without this, a huge constant makes
        # V and V_false agree in every stored float digit and the physical
        # difference is lost to catastrophic cancellation -- while dV stays
        # exact, handing the bounce solver an inconsistent V/dV pair.
        self._params_num = self.params.copy()
        self._params_num[MONOMIAL_ORDER.index('c')] = 0.0

        self.V_false = float(V_numeric(self.false_vac, self._params_num))
        H_F = H_numeric(self.false_vac, self._params_num)

        eigvals, L = np.linalg.eigh(H_F)     # columns of L are the eigenvectors
        self.L = L
        self.LT = L.T
        self.eigvals_false = eigvals

        print("\n[CTShiftedLiftedPotential]")
        print("  fields                  =", self.fields)
        print("  false_vac (orig coords) =", self.false_vac)
        print("  V_original(false_vac)   =", self.V_false)
        print("  rotation L (columns = eigenvectors) =")
        print(self.L)
        print("  eigenvalues(H_F) =", eigvals)

    # ---------- coordinate maps (vectorised over the last axis) ----------
    def to_original(self, X_prime):
        X_prime = np.asarray(X_prime, dtype=float)
        return self.false_vac + np.einsum("ij,...j->...i", self.L, X_prime)

    def to_prime(self, X_orig):
        X_orig = np.asarray(X_orig, dtype=float)
        return np.einsum("ij,...j->...i", self.LT, X_orig - self.false_vac)

    # ---------- potential, gradient, Hessian in primed coords ----------
    def V(self, X_prime):
        return (V_numeric(self.to_original(X_prime), self._params_num)
                - self.V_false)

    def dV(self, X_prime):
        grad_orig = gradV_numeric(self.to_original(X_prime), self._params_num)
        return np.einsum("ij,...j->...i", self.LT, grad_orig)   # g' = L^T g

    def H(self, X_prime):
        H_orig = H_numeric(self.to_original(X_prime), self._params_num)
        return self.LT @ H_orig @ self.L                        # H' = L^T H L

    def H_batch(self, Xp):
        """The original-frame Hessian at a whole array of primed points,
        rotated into primed coords by L^T H L in a single einsum."""
        return np.einsum('ai,...ij,jb->...ab', self.LT,
                         H_numeric_batch(self.to_original(Xp),
                                         self._params_num),
                         self.L)


# --------------------------------------------------------------------------- #
#  6. Self-test                                                                #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("fields         =", FIELDS)
    print("MONOMIAL_ORDER =", MONOMIAL_ORDER)
    print("symbolic V     =", V_EXPR)
    for nm in model_names():
        m = guard_vacua(nm)
        print(f"\n=== model {nm}"
              f"{'  (DEFAULT)' if nm == MODEL_DEFAULT else ''} ===")
        print("  couplings =", m['couplings'])
        print("  params    =", m['params'])
        print("  Hessian is",
              "DIAGONAL everywhere -> 1-D CosmoTransitions instanton"
              if m['decoupled'] else
              "NON-diagonal -> CosmoTransitions path deformation")
        for i, (pt, Vv) in enumerate(zip(m['minima'], m['V_min'])):
            ev = np.linalg.eigvalsh(H_numeric(pt, m['params']))
            print(f"   M{i}: phi = {np.array2string(pt, precision=8)}  "
                  f"V = {Vv:+.8f}  m^2 = {np.round(ev, 6)}")
        iF, iT = m['pair']
        print(f"  downward pairs      : "
              f"{['F%d_T%d' % p for p in m['pairs']]}")
        print(f"  THE PAIR THIS RUNS  : F{iF}_T{iT}  "
              f"(fields that move: "
              f"{[FIELDS[i] for i in tunneling_fields(m['minima'][iF], m['minima'][iT])]})")
        print(f"  why                 : {m['why']}")
        pot = CTShiftedLiftedPotential(m['params'], m['minima'][iF])
        print("  H'(0) =\n", pot.H(np.zeros(2)))
