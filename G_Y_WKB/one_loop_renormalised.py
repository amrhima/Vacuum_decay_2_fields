import numpy as np
from G_Y_WKB.WKB_terms import build_W_profile, calculate_A1, calculate_A3
from G_Y_WKB.wkb_subtracted_one_loop_action import wkb_subtracted_one_loop_action
from G_Y_WKB.renormalize import msbar_local_term_4d

def dm_renormalized_one_loop_action(
    R_bounce,
    X_bounce,
    Y_bounce,
    phi_fv,
    pot,
    r_min,
    Rmax,
    l_max,
    mu_ren,
    wkb_start_l=2,
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
        l_max=l_max,
        wkb_start_l=wkb_start_l)

    # Explicit d=4 MS-bar local finite remainder.
    finite_local_term = msbar_local_term_4d(
        R=r,
        W_profile=W_profile,
        H_fv=H_fv,
        mu_ren=mu_ren,
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
