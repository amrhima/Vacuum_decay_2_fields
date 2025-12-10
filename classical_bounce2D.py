import numpy as np
from dataclasses import dataclass
from typing import Callable, Sequence, Tuple, List, Optional

from scipy.optimize import root

# Import vacuum-finding utilities and full gradient from potential.py
from potential2D import gradV_numeric, find_vacua_from_potential

@dataclass
class BounceSolution2D:
    rho: np.ndarray
    phi: np.ndarray
    dphi: np.ndarray
    action: float
    a_shoot: np.ndarray
    fv_point: Tuple[float, float]
    tv_point: Tuple[float, float]


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


def _bounce_ode_factory_2d(
    params: Sequence[float],
    eps: float = 1e-8,
) -> Callable[[float, np.ndarray], np.ndarray]:
    def ode(r: float, y: np.ndarray) -> np.ndarray:
        phi = y[:2]
        pi = y[2:]

        if abs(r) < eps:
            friction = 0.0
        else:
            friction = 3.0 / r

        # Clip field values to avoid overflow in the polynomial potential
        phi_clipped = np.clip(phi, -10.0, 10.0)

        grad = gradV_numeric(phi_clipped, params)  # shape (2,)
        dpi = grad - friction * pi

        # Replace any NaNs/Infs by large but finite numbers to keep the integrator running
        dpi = np.nan_to_num(dpi, nan=0.0, posinf=1e6, neginf=-1e6)

        return np.concatenate([pi, dpi]).astype(float)

    return ode


def _integrate_profile_2d(
    a: Sequence[float],
    params: Sequence[float],
    rho_max: float,
    n_steps: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    ode = _bounce_ode_factory_2d(params)

    rho = np.linspace(0.0, rho_max, n_steps + 1)
    h = rho[1] - rho[0]

    a_vec = np.asarray(a, dtype=float)
    if a_vec.shape != (2,):
        raise ValueError("Initial condition a must be a 2-component vector for 2D bounce.")

    # State y = (phi1, phi2, pi1, pi2)
    y = np.concatenate([a_vec, np.zeros(2, dtype=float)])

    phi_vals = np.empty((rho.size, 2), dtype=float)
    pi_vals = np.empty((rho.size, 2), dtype=float)

    action = 0.0

    for i, r in enumerate(rho):
        phi_vals[i] = y[:2]
        pi_vals[i] = y[2:]

        if i > 0:
            pi_mid = 0.5 * (pi_vals[i] + pi_vals[i - 1])  # shape (2,)
            r_mid = 0.5 * (rho[i] + rho[i - 1])
            pi_sq = np.dot(pi_mid, pi_mid)
            action += 0.5 * pi_sq * r_mid**3 * h

        if i < n_steps:
            y = rk4_step(ode, r, y, h)

    return rho, phi_vals, pi_vals, action

def _end_value_2d(
    a: Sequence[float],
    fv_point: Sequence[float],
    params: Sequence[float],
    rho_max: float,
    n_steps: int,
) -> np.ndarray:
    """Boundary mismatch F(a) = phi(rho_max; a) - phi_FV for the 2D bounce."""
    _, phi, _, _ = _integrate_profile_2d(a, params, rho_max, n_steps)
    return phi[-1] - np.asarray(fv_point, dtype=float)


def shoot_bounce_2d(
    params: Sequence[float],
    rho_max: float,
    n_steps: int = 2000,
    a0: Optional[Sequence[float]] = None,
    tol: float = 1e-8,
    max_iter: int = 50,
) -> BounceSolution2D:
    # Locate false and true vacua in the full 2D potential
    fv_point, fv_V, tv_point, tv_V = find_vacua_from_potential(params)

    fv = np.asarray(fv_point, dtype=float)
    tv = np.asarray(tv_point, dtype=float)

    if fv.shape != (2,) or tv.shape != (2,):
        raise ValueError(
            "shoot_bounce_2d is currently implemented for a 2-field potential only."
        )

    # Initial guess for a: start near the true vacuum, slightly nudged toward FV
    if a0 is None:
        a0 = tv + 0.1 * (fv - tv)

    a0 = np.asarray(a0, dtype=float)

    def F(a_vec: np.ndarray) -> np.ndarray:
        return _end_value_2d(a_vec, fv, params, rho_max, n_steps)

    sol = root(F, a0, tol=tol)

    if not sol.success:
        raise RuntimeError(
            f"2D bounce shooting failed to converge: {sol.message}"
        )

    a_sol = sol.x
    rho, phi, pi, action = _integrate_profile_2d(a_sol, params, rho_max, n_steps)

    return BounceSolution2D(
        rho=rho,
        phi=phi,
        dphi=pi,
        action=action,
        a_shoot=a_sol,
        fv_point=tuple(fv_point),
        tv_point=tuple(tv_point),
    )


