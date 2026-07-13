from __future__ import annotations

import sympy as sp
import numpy as np
from scipy.optimize import root
from typing import Sequence, Tuple, List, Dict

# ---------------------------------------------------------------------------
# 1. Symbols and parameterised potential
# ---------------------------------------------------------------------------

x, y = sp.symbols("x y", real=True)

a0, a_xy2, a_y4, a_x2y2, a_x4 = sp.symbols(
    "a0 a_xy2 a_y4 a_x2y2 a_x4", real=True
)

PARAM_SYMBOLS = (a0, a_xy2, a_y4, a_x2y2, a_x4)

V_expr = (
    a0
    + a_xy2 * x * y**2
    + a_y4 * (y**2 - 1.0) ** 2
    + a_x2y2 * x**2 * y**2
    + a_x4 * (x**2 - 1.0) ** 2
)

PARAMS_DEFAULT = np.array([-0.8, 1.0, 1.0, 0.15, 1.0], dtype=float)

# ---------------------------------------------------------------------------
# 2. Symbolic gradient / Hessian → numeric
# ---------------------------------------------------------------------------

dV_dx = sp.diff(V_expr, x)
dV_dy = sp.diff(V_expr, y)
H_expr = sp.hessian(V_expr, (x, y))

ARGS = (x, y) + PARAM_SYMBOLS

V_func = sp.lambdify(ARGS, V_expr, "numpy")
gradV_func = sp.lambdify(ARGS, (dV_dx, dV_dy), "numpy")
H_func = sp.lambdify(ARGS, H_expr, "numpy")


def _unpack_params(params: Sequence[float] | None) -> Tuple[float, ...]:
    if params is None:
        return tuple(PARAMS_DEFAULT)
    return tuple(params)


def V_numeric(X, params: Sequence[float] | None = None):
    p = _unpack_params(params)
    X = np.asarray(X, dtype=float)
    x_val = X[..., 0]
    y_val = X[..., 1]
    return V_func(x_val, y_val, *p)


def gradV_numeric(X, params: Sequence[float] | None = None) -> np.ndarray:
    p = _unpack_params(params)
    X = np.asarray(X, dtype=float)
    x_val = X[..., 0]
    y_val = X[..., 1]

    dVx, dVy = gradV_func(x_val, y_val, *p)
    g = np.empty_like(X, dtype=float)
    g[..., 0] = dVx
    g[..., 1] = dVy
    return g


def H_numeric(point: Sequence[float], params: Sequence[float] | None = None) -> np.ndarray:
    p = _unpack_params(params)
    x_val, y_val = float(point[0]), float(point[1])
    return np.array(H_func(x_val, y_val, *p), dtype=float)


# ---------------------------------------------------------------------------
# 3. Critical-point and vacuum finder
# ---------------------------------------------------------------------------

def build_guesses(
    params: Sequence[float] | None = None,
    x_min: float = -2.0,
    x_max: float = 2.0,
    y_min: float = -2.0,
    y_max: float = 2.0,
    Nx: int = 7,
    Ny: int = 7,
) -> List[Tuple[float, float]]:
    xs = np.linspace(x_min, x_max, Nx)
    ys = np.linspace(y_min, y_max, Ny)
    return [(float(xv), float(yv)) for xv in xs for yv in ys]


def find_critical_points(
    params: Sequence[float] | None = None,
    guesses: Sequence[Tuple[float, float]] | None = None,
    tol: float = 1e-8,
) -> List[Tuple[float, float]]:
    p = _unpack_params(params)
    if guesses is None:
        guesses = build_guesses(p)
    else:
        guesses = list(map(tuple, guesses))

    crit_points: List[Tuple[float, float]] = []

    def grad_fun(vec):
        return gradV_numeric(vec, p)

    for guess in guesses:
        sol = root(grad_fun, x0=np.array(guess, dtype=float))
        if not sol.success:
            continue
        pt = sol.x
        n_round = max(3, int(-np.log10(tol)))
        pt_round = (round(float(pt[0]), n_round), round(float(pt[1]), n_round))
        if pt_round not in crit_points:
            crit_points.append(pt_round)

    return crit_points


def classify_point(
    point: Tuple[float, float],
    params: Sequence[float] | None = None,
) -> Tuple[str, np.ndarray, float]:
    p = _unpack_params(params)
    H_num = H_numeric(point, p)
    eigvals = np.linalg.eigvals(H_num)

    if np.all(eigvals > 0):
        kind = "minimum"
    elif np.all(eigvals < 0):
        kind = "maximum"
    else:
        kind = "saddle"

    V_val = float(V_numeric(point, p))
    return kind, eigvals, V_val


def find_all_minima(
    params: Sequence[float] | None = None,
    guesses: Sequence[Tuple[float, float]] | None = None,
) -> List[Dict[str, np.ndarray]]:
    p = _unpack_params(params)
    if guesses is None:
        guesses = build_guesses(p)

    crits = find_critical_points(p, guesses)

    minima: List[Dict[str, np.ndarray]] = []
    for pt in crits:
        kind, eigvals, V_val = classify_point(pt, p)
        if kind == "minimum":
            minima.append({
                "point": np.array(pt, dtype=float),
                "V": float(V_val),
            })

    minima.sort(key=lambda d: d["V"])
    return minima


def find_vacua_from_potential(
    params: Sequence[float] | None = None,
    guesses: Sequence[Tuple[float, float]] | None = None,
) -> Tuple[Tuple[float, float], float, Tuple[float, float], float]:
    minima = find_all_minima(params, guesses)
    if len(minima) < 2:
        raise RuntimeError(
            "Expected at least two local minima (false and true vacua), "
            f"but found {len(minima)}."
        )

    tv = minima[0]
    fv = minima[-1]
    return (
        tuple(fv["point"]),
        float(fv["V"]),
        tuple(tv["point"]),
        float(tv["V"]),
    )


# ---------------------------------------------------------------------------
# 4. Linear shifted + lifted potential (PRIMED coordinates)
# ---------------------------------------------------------------------------

class CTShiftedLiftedPotential:
    def __init__(self, params, false_vac):
        self.params = np.asarray(params, dtype=float)
        self.false_vac = np.asarray(false_vac, dtype=float)

        # value & Hessian at false vacuum in ORIGINAL coords
        self.V_false = float(V_numeric(self.false_vac, self.params))
        H_F = H_numeric(self.false_vac, self.params)

        # orthogonal diagonalisation H_F = L Λ L^T
        eigvals, L = np.linalg.eigh(H_F)
        self.L = L
        self.LT = L.T
        self.eigvals_false = eigvals

        print("\n[CTLinearShiftedPotential]")
        print("  false_vac (orig coords) =", self.false_vac)
        print("  V_original(false_vac)   =", self.V_false)
        print("  rotation L (columns = eigenvectors) =")
        print(self.L)
        print("  eigenvalues(H_F) =", eigvals)

    def to_original(self, X_prime):
        X_prime = np.asarray(X_prime, dtype=float)
        phi_shift = np.einsum("ij,...j->...i", self.L, X_prime)
        return self.false_vac + phi_shift

    def to_prime(self, X_orig):
        X_orig = np.asarray(X_orig, dtype=float)
        diff = X_orig - self.false_vac
        return np.einsum("ij,...j->...i", self.LT, diff)

    def V(self, X_prime):
        X_prime = np.asarray(X_prime, dtype=float)
        X_orig = self.to_original(X_prime)
        return V_numeric(X_orig, self.params) - self.V_false

    def dV(self, X_prime):
        X_prime = np.asarray(X_prime, dtype=float)
        X_orig = self.to_original(X_prime)
        grad_orig = gradV_numeric(X_orig, self.params)
        return np.einsum("ij,...j->...i", self.LT, grad_orig)

    def H(self, X_prime):
        X_prime = np.asarray(X_prime, dtype=float)
        X_orig = self.to_original(X_prime)
        H_orig_here = H_numeric(X_orig, self.params)
        return self.LT @ H_orig_here @ self.L


__all__ = [
    "PARAM_SYMBOLS",
    "PARAMS_DEFAULT",
    "V_expr",
    "V_func",
    "V_numeric",
    "gradV_func",
    "gradV_numeric",
    "H_func",
    "H_numeric",
    "build_guesses",
    "find_critical_points",
    "classify_point",
    "find_all_minima",
    "find_vacua_from_potential",
    "CTShiftedLiftedPotential",
]
