# classical_bounce1D.py

import numpy as np
from dataclasses import dataclass
from typing import Callable, Sequence, Tuple, List, Optional, Dict

FloatFunc = Callable[[float, Sequence[float]], float]

@dataclass
class BounceSolution1D:
    rho: np.ndarray
    phi: np.ndarray
    dphi: np.ndarray
    action: Dict[str, float]
    a_shoot: float
    phi_FV: float
    phi_TV: float


def rk4_step(
    f: Callable[[float, np.ndarray], np.ndarray],
    r: float,
    y: np.ndarray,
    h: float,
) -> np.ndarray:
    """One step of fixed-step fourth-order Runge–Kutta."""
    k1 = f(r, y)
    k2 = f(r + 0.5 * h, y + 0.5 * h * k1)
    k3 = f(r + 0.5 * h, y + 0.5 * h * k2)
    k4 = f(r + h, y + h * k3)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _bounce_ode_factory_1d(
    dV_dphi: FloatFunc,
    params: Sequence[float],
    eps: float = 1e-8,
) -> Callable[[float, np.ndarray], np.ndarray]:
    def ode(r: float, y: np.ndarray) -> np.ndarray:
        phi, pi = y

        if abs(r) < eps:
            friction = 0.0
        else:
            friction = 3.0 / r

        # Clip field value to avoid overflow in the polynomial potential
        phi_clipped = np.clip(phi, -10.0, 10.0)

        dpi = dV_dphi(phi_clipped, params) - friction * pi

        # Replace any NaNs/Infs by large but finite numbers to keep the integrator running
        dpi = np.nan_to_num(dpi, nan=0.0, posinf=1e6, neginf=-1e6)

        return np.array([pi, dpi], dtype=float)

    return ode


def _integrate_profile_1d(
    a: float,
    dV_dphi: FloatFunc,
    V: FloatFunc,
    phi_FV: float,
    params: Sequence[float],
    rho_max: float,
    n_steps: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    ode = _bounce_ode_factory_1d(dV_dphi, params)

    rho = np.linspace(0.0, rho_max, n_steps + 1)
    h = rho[1] - rho[0]

    # State vector y = (phi, pi) with pi = phi'
    y = np.array([a, 0.0], dtype=float)

    phi_vals = np.empty_like(rho)
    pi_vals = np.empty_like(rho)

    action_kin = 0.0
    action_pot = 0.0
    V_FV = V(phi_FV, params)

    for i, r in enumerate(rho):
        phi_vals[i] = y[0]
        pi_vals[i] = y[1]

        # Approximate kinetic contribution to the action (i.e. surface term)
        if i > 0:
            pi_mid = 0.5 * (pi_vals[i] + pi_vals[i - 1])
            r_mid = 0.5 * (rho[i] + rho[i - 1])
            action_kin += 0.5 * pi_mid**2 * r_mid**3 * h
            phi_mid = 0.5 * (phi_vals[i] + phi_vals[i - 1])
            action_pot += (V(phi_mid, params) - V_FV) * r_mid**3 * h

        if i < n_steps:
            y = rk4_step(ode, r, y, h)

    return rho, phi_vals, pi_vals, action_kin, action_pot

def _end_value_1d(
    a: float,
    phi_FV: float,
    dV_dphi: FloatFunc,
    V: FloatFunc,
    params: Sequence[float],
    rho_max: float,
    n_steps: int,
) -> float:
    """Boundary mismatch F(a) = phi(rho_max; a) - phi_FV for the 1D bounce."""
    _, phi, _, _, _ = _integrate_profile_1d(a, dV_dphi, V, phi_FV, params, rho_max, n_steps)
    return float(phi[-1] - phi_FV)


def shoot_bounce_1d(
    params: Sequence[float],
    dV_dphi: FloatFunc,
    V: FloatFunc,
    phi_FV: float,
    phi_TV: float,
    rho_max: float,
    n_steps: int = 2000,
    a0: Optional[float] = None,
    a1: Optional[float] = None,
    tol: float = 1e-8,
    max_iter: int = 50,
) -> BounceSolution1D:
    # Crude initial guesses for a if not provided:
    if a0 is None:
        a0 = phi_TV  # start near the true vacuum
    if a1 is None:
        # second guess somewhere between TV and FV (or slightly beyond)
        a1 = phi_TV + 0.5 * (phi_FV - phi_TV)

    # Initial mismatches at rho_max
    F0 = _end_value_1d(a0, phi_FV, dV_dphi, V, params, rho_max, n_steps)
    F1 = _end_value_1d(a1, phi_FV, dV_dphi, V, params, rho_max, n_steps)

    for _ in range(max_iter):
        if abs(F1 - F0) < 1e-14:
            # Secant degeneracy; break and use the current best guess
            break

        # Secant update for a
        a2 = a1 - F1 * (a1 - a0) / (F1 - F0)
        F2 = _end_value_1d(a2, phi_FV, dV_dphi, V, params, rho_max, n_steps)

        if abs(F2) < tol:
            a1, F1 = a2, F2
            break

        a0, F0, a1, F1 = a1, F1, a2, F2

    # Final integration with the best a
    rho, phi, pi, action_kin, action_pot = _integrate_profile_1d(
        a1, dV_dphi, V, phi_FV, params, rho_max, n_steps
    )

    return BounceSolution1D(
        rho=rho,
        phi=phi,
        dphi=pi,
        action={
            "kinetic_surface": action_kin,
            "potential_offset": action_pot,
            "total": action_kin + action_pot,
        },
        a_shoot=a1,
        phi_FV=phi_FV,
        phi_TV=phi_TV,
    )

