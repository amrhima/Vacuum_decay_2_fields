import numpy as np
from scipy.integrate import simpson
from G_Y_WKB.WKB_terms import build_W_profile, calculate_A1, calculate_A3
from G_Y_WKB.wkb_subtracted_one_loop_action import wkb_subtracted_one_loop_action

def dm_renormalized_one_loop_action(
    R_bounce,
    X_bounce,
    Y_bounce,
    phi_fv,
    pot,
    r_min,
    Rmax,
    n_max,
    mu_ren,
    wkb_start_n=2,
):

    # -----------------------------------------------------
    # Background-dependent fluctuation potential
    # -----------------------------------------------------

    W_profile, H_fv = build_W_profile(
        R_bounce=R_bounce,
        X_bounce=X_bounce,
        Y_bounce=Y_bounce,
        phi_fv=phi_fv,
        pot=pot,
    )

    # W_profile[k] is assumed to be a 2x2 matrix at R_bounce[k]

    tr_W = np.trace(W_profile, axis1=1, axis2=2)

    # tr[W^2 + 2 H_fv W]
    tr_W2_plus_2M2W = np.array([
        np.trace(W @ W + 2.0 * H_fv @ W)
        for W in W_profile
    ])

    r = np.asarray(R_bounce)

    # -----------------------------------------------------
    # Partial-wave sum
    # -----------------------------------------------------

    partial_wave_sum = wkb_subtracted_one_loop_action(
        R_bounce=R_bounce,
        X_bounce=X_bounce,
        Y_bounce=Y_bounce,
        phi_fv=phi_fv,
        pot=pot,
        r_min=r_min,
        Rmax=Rmax,
        n_max=n_max,
        wkb_start_n=wkb_start_n)

    # -----------------------------------------------------
    # Finite local remainder
    # -----------------------------------------------------

    # r^3 log(r) -> 0 at r=0.
    # Avoid evaluating log(0) numerically.
    log_factor = np.zeros_like(r)

    mask = r > 0.0
    log_factor[mask] = (
        np.log(mu_ren * r[mask] / 2.0)
        + np.euler_gamma
        + 1.0
    )

    finite_local_term = (
        -1.0 / 16.0
        * simpson(
            r**3
            * tr_W2_plus_2M2W
            * log_factor,
            x=r,
        )
    )

    delta_S_reg = (
        partial_wave_sum
        + finite_local_term
    )

    return delta_S_reg


def zero_mode_jacobian(S_bounce: float, dimension: int = 4) -> float:
    """Return the translational zero-mode Jacobian in ``dimension`` dimensions."""
    if S_bounce <= 0.0:
        raise ValueError("S_bounce must be positive.")
    if dimension <= 0:
        raise ValueError("dimension must be positive.")
    return float((S_bounce / (2.0 * np.pi)) ** (dimension / 2.0))


def one_loop_action_with_zero_modes(
    S_bounce: float,
    delta_S_reg: float,
    dimension: int = 4,
) -> float:
    """Convert ``delta_S_reg`` to BubbleDet's one-loop-action convention.

    BubbleDet's reported one-loop action includes the translational
    zero-mode normalization, so ``S1 = delta_S_reg - log(J_zero)``.
    The negative-mode contour factor is deliberately not included here; it is
    a separate factor of 1/2 in the decay rate.
    """
    J_zero = zero_mode_jacobian(S_bounce, dimension)
    return float(delta_S_reg - np.log(J_zero))


def decay_rate(
    S_bounce: float,
    delta_S_reg: float,
    dimension: int = 4,
    negative_mode_factor: float = 0.5,
) -> float:
    """Return the semiclassical decay-rate factor including collective modes."""
    if negative_mode_factor <= 0.0:
        raise ValueError("negative_mode_factor must be positive.")
    J_zero = zero_mode_jacobian(S_bounce, dimension)
    return float(
        negative_mode_factor
        * J_zero
        * np.exp(-delta_S_reg)
        * np.exp(-S_bounce)
    )
