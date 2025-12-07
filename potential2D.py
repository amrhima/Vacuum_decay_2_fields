# potential.py


from __future__ import annotations

import sympy as sp
import numpy as np
from scipy.optimize import root
from typing import Sequence, Tuple, List


# ---------------------------------------------------------------------------
# 1. Symbols and potential definition
# ---------------------------------------------------------------------------

# Field variables
phi1, phi2 = sp.symbols("phi1 phi2", real=True)

# Parameters
mu1, mu2 = sp.symbols("mu1 mu2", real=True)              # mass scales
lam1, lam2, lam12 = sp.symbols("lam1 lam2 lam12", real=True)  # quartics
kappa = sp.symbols("kappa", real=True)                  # cubic tilt in phi1

# Collect them in a fixed order for lambdify
PARAM_SYMBOLS = (mu1, mu2, lam1, lam2, lam12, kappa)

# --- Define the potential V(phi1, phi2; params) ---

V = (
    -sp.Rational(1, 2) * mu1**2 * phi1**2 + (lam1 / 4) * phi1**4
    -sp.Rational(1, 2) * mu2**2 * phi2**2 + (lam2 / 4) * phi2**4
    + lam12 * phi1**2 * phi2**2
    + kappa * phi1**3
)

# ---------------------------------------------------------------------------
# 2. Symbolic gradient and Hessian
# ---------------------------------------------------------------------------

# First derivatives
dV_phi1 = sp.diff(V, phi1)
dV_phi2 = sp.diff(V, phi2)

# Hessian matrix (2x2)
H = sp.hessian(V, (phi1, phi2))


print("V")
print(V)
print("dV_phi1")
print(dV_phi1)
print("dV_phi2")
print(dV_phi2)
print("H")
print(H)

# Lambdify: (phi1, phi2, *params) -> numeric values
ARGS = (phi1, phi2) + PARAM_SYMBOLS

gradV_func = sp.lambdify(ARGS, (dV_phi1, dV_phi2), "numpy")
H_func = sp.lambdify(ARGS, H, "numpy")
V_func = sp.lambdify(ARGS, V, "numpy")


# ---------------------------------------------------------------------------
# 3. Numerical wrappers
# ---------------------------------------------------------------------------

def gradV_numeric(x: Sequence[float], params: Sequence[float]) -> np.ndarray:
    phi1_val, phi2_val = x
    return np.array(
        gradV_func(phi1_val, phi2_val, *params),
        dtype=float,
    )


def V_numeric(x: Sequence[float], params: Sequence[float]) -> float:
    """Numeric V(phi1, phi2) for given parameters."""
    phi1_val, phi2_val = x
    return float(V_func(phi1_val, phi2_val, *params))


def H_numeric(x: Sequence[float], params: Sequence[float]) -> np.ndarray:
    """Numeric Hessian H_ij at (phi1, phi2) for given parameters."""
    phi1_val, phi2_val = x
    return np.array(
        H_func(phi1_val, phi2_val, *params),
        dtype=float,
    )


# ---------------------------------------------------------------------------
# 4. Finding and classifying critical points
# ---------------------------------------------------------------------------

def find_critical_points(
    params: Sequence[float],
    guesses: Sequence[Tuple[float, float]],
    tol: float = 1e-8,
) -> List[Tuple[float, float]]:
    crit_points: List[Tuple[float, float]] = []

    for guess in guesses:
        sol = root(lambda x: gradV_numeric(x, params), x0=np.array(guess, dtype=float))

        if not sol.success:
            continue

        p = sol.x
        p_rounded = tuple(np.round(p, int(-np.log10(tol))))

        if p_rounded not in crit_points:
            crit_points.append(p_rounded)

    return crit_points


def classify_point(
    point: Tuple[float, float],
    params: Sequence[float],
) -> Tuple[str, np.ndarray, float]:
    H_num = H_numeric(point, params)
    eigvals = np.linalg.eigvals(H_num)

    if np.all(eigvals > 0):
        kind = "minimum"
    elif np.all(eigvals < 0):
        kind = "maximum"
    else:
        kind = "saddle"

    V_val = V_numeric(point, params)
    return kind, eigvals, V_val


# ---------------------------------------------------------------------------
# 5. Small demo / sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Example parameter choice:
    params_example = (mu1, mu2, lam1, lam2, lam12, kappa) = (1.0, 1.2, 0.5, 0.6, 0.2, 0)


    v1 = mu1/np.sqrt(lam1)
    v2 = mu2/np.sqrt(lam2)
    print(v1)
    print(v2)
    guesses = [
        (0.0, 0.0),
        ( v1, 0.0), (-v1, 0.0),
        (0.0,  v2), (0.0, -v2),
        ( v1,  v2), (-v1,  v2),
        ( v1, -v2), (-v1, -v2),
    ]

    crits = find_critical_points(params_example, guesses)
    print("Critical points (phi1, phi2):")
    for p in crits:
        kind, eigvals, V_val = classify_point(p, params_example)
        print(f"  {p}: {kind}, eigenvalues={eigvals}, V={V_val}")
