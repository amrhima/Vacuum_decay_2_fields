"""potential_vgeneral_phi_v2.py -- Level 0 of the workflow tree (README
section 2): the N-field scalar potential V(phi_1..phi_N), a GENERAL
polynomial of total degree <= 4 in ANY number of fields, its vacua, and the
primed frame of a chosen false vacuum.  A library with no driver stage of
its own: stage 0 (bounce_vgeneral_phi_v2) builds the model from it, and
every later stage rebuilds the same model through
pipeline_helpers_vgeneral_phi_v2.load_background.

READS   the model (fields, couplings): from the CLI of stage 0 (--fields /
        --couplings; worked example in README section 1), or from the bounce
        npz bounce_data_<tag>.npz (keys `fields`, `params`, and `false_vac`
        for the primed frame) through load_background.  fields = distinct
        single lowercase letters ('c' is reserved for the constant monomial);
        couplings = the coefficient dict over the C(N+4,4) monomials of total
        degree <= 4 (15 for N=2, 35 for N=3, 70 for N=4, ...), V = sum_m c_m m,
        an unlisted monomial has coefficient 0.  Monomial names concatenate
        field and power, power 1 omitted: 'x2' = x^2, 'xy2' = x*y^2,
        'xyz' = x*y*z, 'x2z' = x^2*z.  There is no built-in model, and no
        field number is hardcoded.
WRITES  nothing to disk.  Handed down:
        params           the ORDERED coefficient vector (monomial_order);
                         stage 0 stores it with `fields` in every bounce npz
        the vacua        find_all_minima -> minima_catalogue (stage 0); the
                         catalogue index i names the bounce files
                         bounce_data_F<i>_T<j>.npz and the labels TV0/M<i>/FV0
        PrimedPotential  V', grad' V', H' -> the bounce solve
                         (bounce_polish_vgeneral_phi_v2; m2 also sets its
                         radius and its tail condition); m2 ->
                         load_background (channel_masses_and_mu: m2,
                         mu = sum_i m_i); H'(phi_b) - diag(m2) -> insertion
                         = U_int; cubic_couplings, the cubic couplings of
                         the false vacuum -> the tail of the tadpole moment
                         (addback_tadpole)
HOW     1. monomial basis  fields -> monomial_order; sympy forms V, grad V,
                           the N x N Hessian H and its N x N x N derivative
                           d^3V once per fields tuple and lambdifies them
        2. evaluation      V_numeric, gradV_numeric, H_numeric: V, dV/dphi_i,
                           d^2V/dphi_i dphi_j at any array of points;
                           cubic_numeric: d^3V/dphi_i dphi_j dphi_k at one
                           point (original field coordinates)
        3. vacua           find_all_minima: the local minima {'point', 'V'}
                           from a root search of grad V = 0 over a seed grid
                           (the box: --seed-range, or the model's own Cauchy
                           root bound of grad V), in the canonical catalogue
                           order
        4. primed frame    PrimedPotential(params, phi_F, fields):
                           phi = phi_F + L phi', V' = V - V(phi_F),
                           H' = L^T H L, H'(0) = diag(m2) with m2
                           descending (channel 1 the heaviest),
                           cubic_couplings = d^3V(phi_F) rotated by L^T
                           on each index
"""
from __future__ import annotations
import itertools
import string
import sympy as sp
import numpy as np
from scipy.optimize import root
from typing import Sequence, Tuple, List, Dict

MAX_DEGREE = 4          # total polynomial degree of the potential


# ---------------------------------------------------------------------------
# 1. The monomial basis of V for a fields tuple (sympy, built once per tuple)
#    and the coefficient vector params over it -- read by every numeric
#    evaluation below; stage 0 stores (fields, params) in every bounce npz
# ---------------------------------------------------------------------------
def validate_fields(fields) -> Tuple[str, ...]:
    """Normalize + validate a fields spec: a sequence of DISTINCT single
    lowercase letters, none equal to the reserved constant name 'c'.  The
    single-letter rule keeps the concatenated monomial names ('xy2', 'x2z')
    unambiguous.  Returns the tuple; raises on anything else (fail-loud)."""
    fields = tuple(str(f).strip() for f in fields)
    if len(fields) < 1:
        raise RuntimeError('[ABORT] FIELDS needs at least one field.')
    for f in fields:
        if len(f) != 1 or f not in string.ascii_lowercase:
            raise RuntimeError(
                f'[ABORT] field name {f!r} invalid -- fields must be single '
                f'lowercase letters (monomial names concatenate them).')
        if f == 'c':
            raise RuntimeError("[ABORT] field name 'c' is reserved for the "
                               "constant monomial.")
    if len(set(fields)) != len(fields):
        raise RuntimeError(f'[ABORT] duplicate field names in {fields}.')
    return fields


def _exponent_tuples(n_fields: int) -> List[Tuple[int, ...]]:
    """All exponent tuples alpha with |alpha| <= MAX_DEGREE, in a FIXED
    canonical order (total degree, then descending-lex within each degree) --
    for two fields:
    c; x, y; x2, xy, y2; x3, x2y, xy2, y3; x4, x3y, x2y2, xy3, y4."""
    out = []
    for deg in range(MAX_DEGREE + 1):
        tier = [a for a in itertools.product(range(deg + 1), repeat=n_fields)
                if sum(a) == deg]
        tier.sort(reverse=True)
        out.extend(tier)
    return out


def monomial_name(alpha: Tuple[int, ...], fields: Tuple[str, ...]) -> str:
    """'c' for the constant, else field+power concatenation (power 1
    omitted): (1,2) -> 'xy2', (2,0,1) -> 'x2z', (1,1,1) -> 'xyz'."""
    if sum(alpha) == 0:
        return 'c'
    return ''.join(f + ('' if e == 1 else str(e))
                   for f, e in zip(fields, alpha) if e > 0)


class _MonomialBasis:
    """The degree-<=4 monomial basis of V for ONE fields tuple: the exponent
    tuples, the monomial names in their fixed order, and the numpy functions
    V_func, gradV_func, H_func, cubic_func lambdified from the sympy V with
    one coefficient symbol per monomial.  Built once per fields tuple (see
    monomial_basis); everything numeric downstream evaluates these four
    functions."""

    def __init__(self, fields: Tuple[str, ...]):
        self.fields = fields
        self.N = len(fields)
        self.exponents = _exponent_tuples(self.N)
        self.monomial_order: Tuple[str, ...] = tuple(
            monomial_name(a, fields) for a in self.exponents)
        field_symbols = sp.symbols(fields, real=True)        # a tuple for every N
        # one coefficient symbol per monomial, in monomial_order
        coeff_symbols = sp.symbols(tuple(f'c_{m}' for m in self.monomial_order),
                                   real=True)
        V_expr = sum(c * sp.Mul(*[s**e for s, e in zip(field_symbols, a)])
                     for c, a in zip(coeff_symbols, self.exponents))
        grad_exprs = tuple(sp.diff(V_expr, s) for s in field_symbols)
        H_expr = sp.hessian(V_expr, field_symbols)              # NxN matrix
        cubic_exprs = [[[sp.diff(H_expr[i, j], s) for s in field_symbols]
                        for j in range(self.N)] for i in range(self.N)]   # d^3 V, N x N x N
        args = field_symbols + coeff_symbols
        self.V_func = sp.lambdify(args, V_expr, 'numpy')
        self.gradV_func = sp.lambdify(args, grad_exprs, 'numpy')
        self.H_func = sp.lambdify(args, H_expr, 'numpy')
        self.cubic_func = sp.lambdify(args, cubic_exprs, 'numpy')


_MONOMIAL_BASIS_CACHE: Dict[Tuple[str, ...], _MonomialBasis] = {}


def monomial_basis(fields) -> _MonomialBasis:
    """The symbolic monomial basis of V for a fields tuple (CLI --fields, or
    the `fields` key of a bounce npz), built once per tuple and cached."""
    fields = validate_fields(fields)
    if fields not in _MONOMIAL_BASIS_CACHE:
        _MONOMIAL_BASIS_CACHE[fields] = _MonomialBasis(fields)
    return _MONOMIAL_BASIS_CACHE[fields]


def _unpack_params(params: Sequence[float],
                   basis: _MonomialBasis) -> Tuple[float, ...]:
    """The coefficient tuple (couplings_to_params, or the `params` key of a
    bounce npz), checked to hold one entry per monomial of the basis."""
    p = tuple(params)
    if len(p) != len(basis.monomial_order):
        raise RuntimeError(
            f'[ABORT] potential params has {len(p)} entries, expected '
            f'{len(basis.monomial_order)} (the degree-<={MAX_DEGREE} monomial '
            f'coefficients of fields {basis.fields} in monomial_order).')
    return p


def monomial_order(fields) -> Tuple[str, ...]:
    """The FIXED monomial order of a fields tuple (= the order of the
    coefficient vector `params` stored in every bounce npz)."""
    return monomial_basis(fields).monomial_order


def n_params(fields) -> int:
    """C(N+4,4): the length of the params vector for this fields tuple."""
    return len(monomial_order(fields))


def is_decoupled(params, fields) -> bool:
    """True iff V has NO cross terms: every monomial with two or more
    different fields ('xy', 'xy2', 'x2z', ...; exactly the monomials that
    make the Hessian non-diagonal) has coefficient 0.  Then V = sum_i
    V_i(phi_i), the Hessian is diagonal everywhere and the O(4) bounce
    separates field by field: a pair whose vacua differ in one field has a
    genuinely one-field bounce (spectators frozen).  Read by the selection
    gates (bounce, eigenvalue_neg, eigenvalue_zero): a refused bounce of a
    decoupled potential gets LEG_POINTER_MSG, which points to those
    one-flip legs."""
    basis = monomial_basis(fields)
    p = np.asarray(_unpack_params(params, basis), float)
    cross_term = np.array([sum(e > 0 for e in a) >= 2 for a in basis.exponents])
    return bool(np.all(p[cross_term] == 0.0))


def couplings_to_params(couplings: Dict[str, float], fields) -> np.ndarray:
    """The ordered coefficient vector (monomial_order(fields)) from a coupling
    dict; unlisted monomials are 0.  Raises on any key that is not a
    degree-<=4 monomial name OF THESE FIELDS, so a typo like 'x5' -- or a
    'z2' with fields (x,y) -- fails loud rather than being ignored."""
    order = monomial_order(fields)
    bad = set(couplings) - set(order)
    if bad:
        raise RuntimeError(
            f'[ABORT] unknown potential monomial(s) {sorted(bad)} for fields '
            f'{monomial_basis(fields).fields} -- valid degree-<={MAX_DEGREE} names '
            f'are {list(order)}.')
    return np.array([float(couplings.get(m, 0.0)) for m in order], dtype=float)


def params_no_const(params, fields) -> np.ndarray:
    """The coefficient vector with the constant monomial 'c' set to 0: the
    zero of energy is irrelevant (dV and H never see c, and every V the
    pipeline uses is a difference), so it is removed BEFORE any numeric
    evaluation -- vacuum ordering and energy differences are then invariant
    under V -> V + constant, and a huge constant cannot swallow the physical
    difference in catastrophic cancellation.  Returns a new array."""
    basis = monomial_basis(fields)
    p = np.asarray(_unpack_params(params, basis), float)
    p[basis.monomial_order.index('c')] = 0.0
    return p


# ---------------------------------------------------------------------------
# 2. V, grad V and the Hessian H at any points, and the third derivatives
#    d^3V at one point (original field coordinates), from the lambdified
#    basis -- read by the vacuum finder (3), the primed frame (4; d^3V only
#    there), and (V_numeric) by stage 0's profile_symmetry and the gallery
# ---------------------------------------------------------------------------


def V_numeric(X, params: Sequence[float], fields):
    """V(X) at the point(s) X of shape (..., N), for the coefficient vector
    params in the monomial order of `fields`.  The lambdified functions take
    the N field components phi_1..phi_N as separate arguments:
    np.moveaxis(X, -1, 0) puts the field axis first and * unpacks it (one
    number per field for a single point, one array per field otherwise)."""
    basis = monomial_basis(fields)
    p = _unpack_params(params, basis)
    X = np.asarray(X, dtype=float)
    return basis.V_func(*np.moveaxis(X, -1, 0), *p)


def gradV_numeric(X, params: Sequence[float], fields) -> np.ndarray:
    """dV/dphi_i at X, shape (..., N): the N components stacked on the last
    axis."""
    basis = monomial_basis(fields)
    p = _unpack_params(params, basis)
    X = np.asarray(X, dtype=float)
    return np.stack(basis.gradV_func(*np.moveaxis(X, -1, 0), *p), axis=-1)


def H_numeric(X, params: Sequence[float], fields) -> np.ndarray:
    """The Hessian d^2V/dphi_i dphi_j at every point of X (..., N), shape
    (..., N, N); one point (N,) gives (N, N).  H_func is evaluated ONCE on
    the whole array and returns the N x N entries with the point axes LAST
    (N, N, ...); they are moved to the front."""
    basis = monomial_basis(fields)
    p = _unpack_params(params, basis)
    X = np.asarray(X, dtype=float)
    H = np.asarray(basis.H_func(*np.moveaxis(X, -1, 0), *p),
                   dtype=float)                                  # (N, N, ...)
    return np.moveaxis(H, (0, 1), (-2, -1))                      # (..., N, N)


def cubic_numeric(X, params: Sequence[float], fields) -> np.ndarray:
    """The third derivatives d^3V/dphi_i dphi_j dphi_k at ONE point X (N,),
    shape (N, N, N)."""
    basis = monomial_basis(fields)
    p = _unpack_params(params, basis)
    return np.asarray(basis.cubic_func(*np.asarray(X, dtype=float), *p),
                      dtype=float)


# ---------------------------------------------------------------------------
# 3. The vacua: the local minima of V from a root search of grad V = 0 over
#    a seed grid (original field coordinates), in the canonical catalogue
#    order -- read by bounce_vgeneral_phi_v2.minima_catalogue (stage 0)
# ---------------------------------------------------------------------------

def gradient_root_bound(params: Sequence[float], fields) -> float:
    """The half-width rho of the default vacuum-search box [-rho, rho]^N
    (the 'Cauchy root bound of grad V'): every critical point of V -- every
    point with grad V = 0, so every vacuum -- has |phi_i| <= rho for all i.
    build_guesses searches this box when no --seed-range is given.  rho is
    computed from the couplings alone, so it scales with the fields
    (phi -> s phi, with the couplings that keep V the same, gives s rho).

    Idea: far out, the highest power of phi_k in dV/dphi_k outgrows all the
    other terms together, so dV/dphi_k cannot vanish there.

    For each field k (the loop below), at a point whose LARGEST field value
    t = max_i |phi_i| is reached by phi_k (|phi_k| = t, every |phi_i| <= t):
      a, c    the powers (a_1..a_N) and the coefficient of one monomial of V;
              if it contains phi_k (a_k > 0) it gives dV/dphi_k the term
              a_k c phi^a / phi_k, of degree |a| - 1 = a_1+..+a_N - 1 and
              size at most |a_k c| t^(|a|-1)
      d       the highest degree in dV/dphi_k (3 when V holds phi_k^4, 1 for
              a spectator with only a mass term; at least 1)
      A       |(d+1) c| of the pure power c phi_k^(d+1): the leading term of
              dV/dphi_k has size exactly A t^d
      B_j     the sum of |a_k c| over all OTHER terms of degree j (j = 0..d)
      q_k(t)  = (A - B_d) t^d - B_(d-1) t^(d-1) - ... - B_0, a lower bound
              on |dV/dphi_k| there (the leading term minus all the others at
              their largest)
      t_k     the positive root of q_k (0 if none): q_k(t) > 0 for t > t_k
    rho = max_k t_k: beyond it the field with the largest value has
    dV/dphi_k != 0, so no critical point lies outside the box.

    Example, V = x^4 + x^3 - 2 x^2 + y^2 (x tunnels, y a massive spectator):
      k = x:  dV/dx = 4x^3 + 3x^2 - 4x:  d = 3, A = 4, B_3 = 0, B_2 = 3,
              B_1 = 4, B_0 = 0;  q_x = 4t^3 - 3t^2 - 4t,  t_x = 1.443
      k = y:  dV/dy = 2y:  d = 1, A = 2, B_1 = B_0 = 0;  q_y = 2t,  t_y = 0
      rho = 1.443; the vacuum x = -1.443, y = 0 lies on the edge of the box.

    RAISES [ABORT SEED-BOX] when A <= B_d for some field: q_k is then never
    positive and this bound gives no box, although the vacua may well lie in
    one.  Examples: V = x^2 y^2 (no pure power, A = 0) and (x^2 + y^2)^2 =
    x^4 + 2 x^2 y^2 + y^4 (dV/dx = 4x^3 + 4x y^2, A = B_3 = 4).  Such a model
    needs its box by hand: --seed-range LO,HI."""
    basis = monomial_basis(fields)
    p = np.asarray(_unpack_params(params, basis), float)
    root_bound_radius = 0.0                  # rho = max_k t_k
    for k in range(basis.N):
        # d = the highest degree in dV/dphi_k, at least 1 (a field entering V
        # only as c phi_k then has A = 0 and meets the abort below)
        d = 1
        for a, c in zip(basis.exponents, p):
            if a[k] > 0 and c != 0.0:
                d = max(d, sum(a) - 1)
        A = 0.0
        B = [0.0] * (d + 1)                  # B[j] = B_j, j = 0..d
        for a, c in zip(basis.exponents, p):
            if a[k] > 0 and c != 0.0:
                # no monomial containing phi_k has total degree above d + 1
                # (the definition of d), so a_k = d + 1 leaves every other
                # exponent 0: this is the pure power phi_k^(d+1)
                if a[k] == d + 1:
                    A = abs(a[k] * c)
                else:
                    B[sum(a) - 1] += abs(a[k] * c)
        if A <= B[d]:
            raise RuntimeError(
                f'[ABORT SEED-BOX] no Cauchy root bound for field '
                f'{basis.fields[k]!r}: in dV/d{basis.fields[k]} the pure '
                f'degree-{d} term ({A:.6g}) does not exceed the other '
                f'degree-{d} terms ({B[d]:.6g}), so this simple bound gives no '
                f'default search box (e.g. for (x^2+y^2)^2-type quartics).  '
                f'Search this model over an explicit box instead '
                f'(--seed-range LO,HI).')
        # the coefficients of q_k, highest power first: A - B_d > 0, then
        # -B_(d-1) .. -B_0 <= 0.  One sign change, so by Descartes' rule of
        # signs q_k has exactly one positive root when some B_j (j < d) is
        # > 0, and none when all vanish (q_k = (A - B_d) t^d, t_k = 0).
        # numpy.roots returns its d complex roots; none lies farther from 0
        # than t_k, so t_k is the largest real part among them
        q_k = [A - B[d]] + [-B[j] for j in reversed(range(d))]
        t_k = float(np.max(np.roots(q_k).real))
        root_bound_radius = max(root_bound_radius, t_k)
    return root_bound_radius


def build_guesses(params: Sequence[float], fields,
                  seed_range: Tuple[float, float] | None = None,
                  ) -> List[Tuple[float, ...]]:
    """The seed grid of the vacuum search: nodes_per_dim points per field
    axis over the box [LO, HI]^N, one scipy.optimize.root start per node.

    The box is seed_range = (LO, HI) (the pipeline CLI --seed-range LO,HI),
    or, when seed_range is None, the box DERIVED FROM THE MODEL: half-width
    gradient_root_bound(params, fields), the Cauchy root bound of grad V,
    which is PROVEN to contain every critical point of this potential.

    Nodes per field axis: 7 for N <= 3 (7^N seeds), then round(2200^(1/N))
    so the grid stays at a few thousand seeds (7^4, 5^5, 4^6, 3^7), but never
    fewer than 3 per axis -- from N = 8 on the grid holds 3^N seeds."""
    basis = monomial_basis(fields)
    if seed_range is None:
        root_bound_radius = gradient_root_bound(params, basis.fields)
        box_lo, box_hi = -root_bound_radius, root_bound_radius
    else:
        box_lo, box_hi = seed_range
        # An explicit box NARROWER than the proven one can miss vacua, and the
        # search cannot notice that by itself -- so say it here, where both
        # numbers are in hand.  A model with NO root bound is exactly the case
        # an explicit box exists for, so the note is skipped rather than
        # allowed to abort the search.
        try:
            root_bound_radius = gradient_root_bound(params, basis.fields)
        except RuntimeError:
            root_bound_radius = None
        if root_bound_radius is not None and (box_lo > -root_bound_radius
                                              or box_hi < root_bound_radius):
            print(f"  [vacua] NOTE: the requested seed box "
                  f"[{box_lo:g}, {box_hi:g}]^N is NARROWER than "
                  f"[{-root_bound_radius:.4g}, {root_bound_radius:.4g}]^N, "
                  f"the Cauchy root bound of "
                  f"grad V that is proven to contain every critical point of "
                  f"this potential -- vacua outside the requested box are "
                  f"missed.")
    N = basis.N
    nodes_per_dim = 7 if N <= 3 else max(3, int(round(2200.0 ** (1.0 / N))))
    axes = [np.linspace(box_lo, box_hi, nodes_per_dim)] * N
    return [tuple(float(v) for v in pt) for pt in itertools.product(*axes)]


def find_critical_points(
    params: Sequence[float],
    fields,
    seed_range: Tuple[float, float] | None = None,
    tol: float = 1e-8,
) -> List[Tuple[float, ...]]:
    """Solve grad V = 0 by scipy.optimize.root (default 'hybr', Powell
    hybrid) from every seed of build_guesses(params, fields, seed_range);
    return the DISTINCT roots at FULL PRECISION (the potential's critical
    points).  DISTINCT = further apart (max-norm) than merge_tol =
    max(1e-5 * span, 10 * tol * x_scale), a radius in the model's own field
    scale.  Each accepted root gets ONE
    final Newton step phi <- phi - H^-1 gradV (H and grad V at the root are
    already in hand; the step takes |grad V| from scipy's stopping residual
    down to the double-precision floor) -- these points ARE the vacua: the
    false vacuum is the primed-frame origin and the U_int subtraction point,
    so its residual tilts the whole background."""
    basis = monomial_basis(fields)
    p = _unpack_params(params, basis)
    guesses = build_guesses(p, basis.fields, seed_range)
    # the FIELD SCALE of the search: the largest |coordinate| of the seed box
    # (the model's root-bound radius, or max(|LO|, |HI|) of an explicit
    # --seed-range).  The residual gate and the merge radius below are
    # measured in it, so they follow any rescaling of the fields.
    x_scale = float(np.max(np.abs(guesses)))

    def grad_fun(vec):
        return gradV_numeric(vec, p, basis.fields)

    # PASS 1: Newton-solve from every seed, keeping every converged root --
    # duplicates included, because the merge radius below is MEASURED from
    # this cloud and so cannot be applied while it is still being built.
    # Each root is kept with its gradient (sol.fun = grad V at sol.x) and its
    # Hessian: the gate reads them here and the Newton step of PASS 2 reuses
    # them.
    roots_raw: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []   # (root, gradV, H)
    for guess in guesses:
        sol = root(grad_fun, x0=np.array(guess, dtype=float), tol=tol)
        if not sol.success:
            continue
        # gradient-residual GATE: scipy's `success` can accept a stalled
        # iterate.  A genuine root carries at most the gradient of a field
        # step at scipy's RELATIVE x-tolerance tol on the field scale,
        # |grad V| <= tol * x_scale * max|H|: a ceiling that follows any
        # rescaling of the couplings or of the fields.
        H_root = H_numeric(sol.x, p, basis.fields)
        H_scale = float(np.max(np.abs(H_root)))
        if not np.isfinite(H_scale):
            continue
        gradV_root = np.atleast_1d(np.asarray(sol.fun, dtype=float))
        if float(np.max(np.abs(gradV_root))) > tol * x_scale * H_scale:
            continue
        roots_raw.append((np.asarray(sol.x, dtype=float), gradV_root, H_root))
    if not roots_raw:
        return []

    # THE DISTINCTNESS RADIUS, in the model's own field scale:
    #     merge_tol = max(1e-5 * span, 10 * tol * x_scale),
    # span = widest coordinate spread of the root cloud (the model's own
    # structure, not where it sits).  Duplicate roots of one critical point
    # land on top of each other at machine precision while genuinely distinct
    # pairs sit at a finite fraction of the span, so the radius keeps many
    # orders of headroom on both sides.  10 * tol * x_scale = the solver's
    # own resolution (tol is scipy's RELATIVE x-tolerance): separations below
    # it are not resolved, and it keeps duplicates merged when all roots
    # coincide (span = 0).  Neither term is an absolute length: rescaling the
    # fields rescales both.
    cloud = np.asarray([pt for pt, _gradV, _H in roots_raw], dtype=float)
    span = float(np.max(cloud.max(axis=0) - cloud.min(axis=0)))
    merge_tol = max(1e-5 * span, 10.0 * tol * x_scale)

    # PASS 2: polish each root and drop the duplicates.
    crit_points: List[Tuple[float, ...]] = []
    worst_merge = 0.0
    for pt, gradV_pt, H_pt in roots_raw:
        # ONE Newton step on top of scipy's root: pt <- pt - H^-1 gradV.  H is
        # nonsingular at a nondegenerate critical point, and at a degenerate one
        # (or if the step is so large that the quadratic model does not apply,
        # i.e. bigger than the merge tolerance) the step is DROPPED and scipy's
        # own root is kept -- the polish must never move a root to a different
        # critical point.
        try:
            step = np.linalg.solve(H_pt, gradV_pt)
            if np.all(np.isfinite(step)) and np.max(np.abs(step)) < merge_tol:
                pt = pt - step
        except np.linalg.LinAlgError:
            pass
        pt_full = tuple(float(v) for v in pt)
        near = [max(abs(a - b) for a, b in zip(pt_full, q)) for q in crit_points]
        if near and min(near) <= merge_tol:
            worst_merge = max(worst_merge, min(near))
            continue
        crit_points.append(pt_full)
    # Loud when the merge radius did REAL work: duplicates land on top of
    # each other (machine precision), so a merge from a tenth of the radius
    # away may be a second critical point being swallowed.
    if worst_merge > 0.1 * merge_tol:
        print(f"  [vacua] WARNING: merged two roots {worst_merge:.3g} apart, "
              f"close to the merge radius {merge_tol:.3g} = max(1e-5 * span, "
              f"10 * tol * x_scale) with span {span:.3g}, tol {tol:.3g} and "
              f"x_scale {x_scale:.3g} -- if they "
              f"are DIFFERENT critical points then they sit closer than that "
              f"radius, finer than this finder resolves, and the vacuum list "
              f"is incomplete.")
    return crit_points


# ---------------------------------------------------------------------------
#   The CANONICAL vacuum numbering
# ---------------------------------------------------------------------------
# The catalogue index i of a minimum is its NAME downstream: it names the
# bounce files bounce_data_F<iF>_T<iT>.npz and the labels TV0 (i = 0) / M<i>
# / FV0 (i = last) (bounce_vgeneral_phi_v2.vacuum_label), and --pair and a
# resumed run look bounces up by these indices.  So the order must be a
# property of the POTENTIAL alone -- not of which seed converged first or of
# one interpreter's last bits: a raw sort on (V, coordinates) is NOT that (a
# last-bit difference in V between two interpreters would name a DIFFERENT
# false vacuum), so BOTH keys are compared through a tolerance taken from the
# model's own spread.  The tie-break is lexicographic in the model's own
# FIELD basis (part of the model's definition; the mass eigenbasis is not
# unique under degenerate masses, and rotation-invariant scalars cannot
# separate reflection-related vacua).

def _tolerance_levels(values: Sequence[float], tol: float) -> List[int]:
    """Rank `values` by size with a blind spot of width `tol`: sorted
    ascending, a new level opens only where consecutive values differ by more
    than tol (single linkage), so values agreeing within tol share a level.
    Returns one integer per value, in input order -- an ordering that no
    last-bit float difference can decide."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    levels = [0] * len(values)
    for k in range(1, len(order)):
        prev, cur = order[k - 1], order[k]
        levels[cur] = levels[prev] + (1 if values[cur] - values[prev] > tol
                                      else 0)
    return levels


def _canonical_vacuum_order(
    minima: Sequence[Dict[str, np.ndarray]],
) -> List[Dict[str, np.ndarray]]:
    """The catalogue order (index i -> file F<i>/T<i>, label TV0/M<i>/FV0):
    by V ascending, energy ties broken by the coordinates, BOTH compared through
    _tolerance_levels so no last-bit difference can move a label.  Returns a
    reordered list; the input is not modified.
    The two tolerances are fractions of the model's OWN spreads (rescale the
    potential or the fields and both follow):
      energy  1e-12 * V_scale, V_scale = max(vacuum-energy spread, max |V|)
              -- the SAME rule enumerate_bounce_pairs uses for "genuinely
              below", so vacua that tie here get no bounce there (decades
              above last-bit interpreter disagreement, decades below any
              real vacuum-energy gap);
      field   1e-9 * widest coordinate spread (decades above one-ulp noise,
              decades below merge_tol, so it can never merge two distinct
              vacua)."""
    minima = list(minima)
    if len(minima) < 2:
        return minima
    V = [float(m["V"]) for m in minima]
    vacuum_points = [np.asarray(m["point"], float).ravel() for m in minima]
    V_scale = max(max(V) - min(V), max(abs(v) for v in V))
    V_levels = _tolerance_levels(V, 1e-12 * V_scale)
    columns = [[float(p[j]) for p in vacuum_points] for j in range(len(vacuum_points[0]))]
    x_span = max(max(col) - min(col) for col in columns)
    x_levels = [_tolerance_levels(col, 1e-9 * x_span) for col in columns]
    keys = [(V_levels[i], tuple(lev[i] for lev in x_levels))
            for i in range(len(minima))]
    return [m for _key, m in sorted(zip(keys, minima), key=lambda t: t[0])]


def find_all_minima(
    params: Sequence[float],
    fields,
    seed_range: Tuple[float, float] | None = None,
) -> List[Dict[str, np.ndarray]]:
    """All local minima of V FOUND FROM THE SEED GRID (the candidate vacua), in
    the CANONICAL catalogue order whose index i names the bounce files
    F<i>/T<i> and the labels TV0/M<i>/FV0: by V ascending, energy ties broken
    by the coordinates, both compared at a tolerance derived from the model's
    own spread (_canonical_vacuum_order).  Each entry is {'point': (N,) array,
    'V': value of V-c}; subtracting the constant before evaluation keeps
    vacuum ordering and energy differences invariant under V -> V + constant.

    The search is a root solve of grad V = 0 from the seed grid of
    build_guesses over the box seed_range = (LO, HI) (CLI --seed-range) or,
    when None, over the model's own Cauchy root-bound box: that box CONTAINS
    every critical point but is sampled at finitely many seeds, so a minimum
    whose basin holds no seed is missed.  Finding NO minimum at all prints a
    hint here (the case the pipeline cannot continue from)."""
    basis = monomial_basis(fields)
    p = params_no_const(params, basis.fields)   # the irrelevant zero of energy removed
    crits = find_critical_points(p, basis.fields, seed_range)
    minima: List[Dict[str, np.ndarray]] = []
    for pt in crits:
        # a local MINIMUM: every eigenvalue of the N x N Hessian is positive
        if np.all(np.linalg.eigvalsh(H_numeric(pt, p, basis.fields)) > 0):
            minima.append({"point": np.array(pt, dtype=float),
                           "V": float(V_numeric(pt, p, basis.fields))})
    minima = _canonical_vacuum_order(minima)
    # no minimum at all: nothing downstream has a vacuum to work with -- say why
    if not minima:
        try:
            root_bound_radius = gradient_root_bound(p, basis.fields)
            root_bound_note = (
                f"every critical point of this potential lies inside "
                f"[{-root_bound_radius:.4g}, {root_bound_radius:.4g}]^N (the "
                f"Cauchy root bound of grad V), so a complete search needs a "
                f"box at least that wide and enough seeds inside it")
        except RuntimeError:
            root_bound_note = (
                "this potential has no Cauchy root bound (some field's pure "
                "power does not dominate -- see gradient_root_bound), so "
                "the box has to be chosen by hand")
        print(f"  [vacua] NOTE: the seed grid found {len(crits)} critical "
              f"point(s) and NO local minimum -- {root_bound_note}.  Widen or refine the "
              f"search with --seed-range LO,HI.")
    return minima


# ---------------------------------------------------------------------------
# 4. The primed frame of a chosen false vacuum: V', grad' V', H', m2, L and
#    the cubic couplings -- read by the bounce solve (stage 0), by
#    load_background (m2, mu, U_int) and by the tadpole add-back
#    (cubic_couplings)
# ---------------------------------------------------------------------------

class PrimedPotential:
    """The potential in the PRIMED frame of a chosen false vacuum phi_F:
        phi = phi_F + L phi',              H(phi_F) = L diag(m2) L^T,
        V'(phi') = V(phi_F + L phi') - V(phi_F),
        grad' V'(phi') = L^T grad V,       H'(phi') = L^T H L,
    so phi' = 0 is the false vacuum, V'(0) = 0 and H'(0) = diag(m2).  L and
    m2 come from numpy.linalg.eigh of the symmetric FV Hessian H(phi_F): m2
    holds its eigenvalues in DESCENDING order (eigh's ascending order
    reversed), the FV channel masses^2 m_i^2 (channel 1 the heaviest), and
    column i of the orthogonal L is the eigenvector of channel i (its sign
    is arbitrary).  cubic_couplings holds
    g_ijk = d^3 V'/dphi'_i dphi'_j dphi'_k at phi' = 0, a derivative and not a
    monomial coefficient: V' contains sum_ijk g_ijk phi'_i phi'_j phi'_k / 6,
    i.e. g_111 phi'_1^3/6 + g_112 phi'_1^2 phi'_2/2 + ...
    Every method takes one point (N,) or an array of points (..., N), point
    axes first, for ANY number of fields N = len(fields)."""

    def __init__(self, params, false_vac, fields):
        self.fields    = monomial_basis(fields).fields
        self.N         = len(self.fields)
        self.false_vac = np.asarray(false_vac, dtype=float)
        if self.false_vac.shape != (self.N,):
            raise RuntimeError(
                f'[ABORT] false_vac has shape {self.false_vac.shape}, '
                f'expected ({self.N},) for fields {self.fields}.')

        # the constant monomial c removed (params_no_const says why): every V
        # this class returns is a difference V - V(false_vac), and dV and H
        # never see c
        self.params_no_const = params_no_const(params, self.fields)

        self.V_false = float(V_numeric(self.false_vac, self.params_no_const,
                                       self.fields))
        H_F_orig = H_numeric(self.false_vac, self.params_no_const, self.fields)

        # numpy.linalg.eigh: H_F_orig = L diag(m2) L^T with m2 ascending;
        # m2 and the columns of L are reversed, so m2 is DESCENDING
        # (channel 1 the heaviest), and H'(0) = L^T H_F_orig L = diag(m2)
        m2, L = np.linalg.eigh(H_F_orig)
        m2, L = m2[::-1], L[:, ::-1]
        self.m2 = m2               # THE FV channel masses^2 m_i^2
        self.L = L                 # column i = channel i's eigenvector; L^-1 = L^T
        # g_ijk = sum_abc [d^3 V(phi_F)]_abc L_ai L_bj L_ck (L^T on every
        # index); far from the wall U_int is linear in the bounce,
        # [U_int]_ij(r) -> sum_k g_ijk phi'_k(r)
        self.cubic_couplings = np.einsum('abc,ai,bj,ck->ijk',
                                         cubic_numeric(self.false_vac, self.params_no_const,
                                                       self.fields), L, L, L)

        print("\n[PrimedPotential]")
        print("  fields                  =", self.fields)
        print("  false_vac (orig coords) =", self.false_vac)
        print("  V(false_vac) - c        =", self.V_false)
        print("  rotation L (columns = eigenvectors) =")
        print(self.L)
        print("  m2 (FV channel masses^2, heaviest first) =", self.m2)

    # ---------- coordinate maps (vectorised over the leading axes) ----------
    def to_original(self, X_prime):
        """phi = phi_F + L phi'; with the point axis first, L phi' is the
        product phi' @ L^T."""
        return self.false_vac + np.asarray(X_prime, dtype=float) @ self.L.T

    def to_prime(self, X_orig):
        """phi' = L^T (phi - phi_F); with the point axis first, the product
        (phi - phi_F) @ L."""
        return (np.asarray(X_orig, dtype=float) - self.false_vac) @ self.L

    # ---------- potential, gradient, Hessian in primed coords ----------
    def V(self, X_prime):
        """V'(phi') = V(phi_F + L phi') - V(phi_F): the potential measured
        from the false vacuum, V'(0) = 0."""
        return V_numeric(self.to_original(X_prime), self.params_no_const,
                         self.fields) - self.V_false

    def dV(self, X_prime):
        """grad' V'(phi') = L^T grad V(phi_F + L phi'); with the point axis
        first, the product (grad V) @ L."""
        g = gradV_numeric(self.to_original(X_prime), self.params_no_const,
                          self.fields)
        return g @ self.L

    def H(self, X_prime):
        """H'(phi') = L^T H(phi_F + L phi') L at one primed point (N,) or a
        whole array of them (..., N) (numpy's @ acts on the last two axes and
        broadcasts over the point axes).  L is fixed at the false vacuum, so
        H'(0) = diag(m2) but along the bounce H' is in general not diagonal:
        its off-diagonal entries (those of U_int) couple the channels."""
        return self.L.T @ H_numeric(self.to_original(X_prime),
                                    self.params_no_const, self.fields) @ self.L
