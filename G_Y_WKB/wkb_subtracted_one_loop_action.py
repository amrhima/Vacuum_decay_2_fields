from G_Y_WKB.WKB_terms import (
    build_W_profile,
    estimate_wkb_tail,
)
import numpy as np

from base.fluctuation_operator import (
    make_V_matrix,
    make_V_matrix_fv
)

from G_Y_WKB.determinant_evaluation import (
    determinant_ratio,
    pick_mu_value,
    partial_wave_degeneracy
)

def wkb_subtracted_one_loop_action(
    R_bounce,
    X_bounce,
    Y_bounce,
    phi_fv,
    pot,
    r_min,
    Rmax,
    l_max,
    wkb_start_l=2,
):
    if r_min <= 0.0:
        raise ValueError(
            "r_min must be > 0; the Liouville-transformed radial "
            "equation is singular at r=0."
        )

    if Rmax <= r_min:
        raise ValueError("Rmax must be larger than r_min.")

    if l_max < 0:
        raise ValueError("l_max must be non-negative.")

    if wkb_start_l < 2:
        raise ValueError("wkb_start_l must be at least 2.")

    if wkb_start_l > l_max + 1:
        raise ValueError(
            "wkb_start_l must be no larger than l_max + 1; "
            f"got wkb_start_l={wkb_start_l}, l_max={l_max}."
        )

    # -----------------------------------------------------
    # WKB coefficients
    # -----------------------------------------------------

    W_profile, H_fv = build_W_profile(
        R_bounce=R_bounce,
        X_bounce=X_bounce,
        Y_bounce=Y_bounce,
        phi_fv=phi_fv,
        pot=pot,
    )

    wkb_by_l, wkb_tail = estimate_wkb_tail(
        R_bounce=R_bounce,
        W_profile=W_profile,
        H_fv=H_fv,
        l_max=l_max + 1,
        start_l=wkb_start_l,
    )

    delta_S_reg = 0.0

    for l in range(l_max + 1):

        # -------------------------------------------------
        # Construct radial operators
        # -------------------------------------------------

        V_bounce = make_V_matrix(
            l,
            R_bounce,
            X_bounce,
            Y_bounce,
            pot,
        )

        V_fv = make_V_matrix_fv(
            l,
            phi_fv,
            pot,
        )

        # -------------------------------------------------
        # Zero modes
        # -------------------------------------------------

        if l == 1:
            # Remove the one radial zero eigenvalue.
            # The angular degeneracy d_1 = 4 accounts for
            # the four translational zero modes.

            mu_best, D_l = pick_mu_value(
                l=l,
                r_min=r_min,
                Rmax=Rmax,
                V_matrix=V_bounce,
                V_matrix_fv=V_fv,
                isolated_eigenvalues=(0.0,),
            )

            if D_l is None:
                raise RuntimeError(
                    "No stable mu plateau found for the l=1 zero mode."
                )

        else:
            # The l=0 negative mode is retained in the
            # determinant. Its sign is removed below when
            # forming log(abs(D_l)).

            D_l = determinant_ratio(
                mu=0.0,
                l=l,
                r_min=r_min,
                Rmax=Rmax,
                V_matrix=V_bounce,
                V_matrix_fv=V_fv,
                isolated_eigenvalues=(),
            )

        if not np.isfinite(D_l):
            raise RuntimeError(
                f"Non-finite determinant ratio obtained for l={l}: {D_l}"
            )

        if D_l == 0.0:
            raise RuntimeError(
                f"Zero determinant ratio obtained for l={l}."
            )

        # -------------------------------------------------
        # WKB subtraction and angular degeneracy
        # -------------------------------------------------

        j = l + 1
        d_l = partial_wave_degeneracy(l)

        log_D_l = np.log(abs(D_l))

        # The low partial waves are treated exactly.  BubbleDet subtracts its
        # higher-order WKB approximation only from the high-wave remainder.
        if l >= wkb_start_l:
            log_D_l_subtracted = log_D_l - wkb_by_l[l]
        else:
            log_D_l_subtracted = log_D_l

        delta_S_reg += 0.5 * d_l * log_D_l_subtracted

        print(
            f"l={l:3d}, "
            f"j={j:3d}, "
            f"d_l={d_l:4d}, "
            f"D_l={D_l:.12e}, "
            f"log_D_l={log_D_l:+.12e}, "
            f"subtracted={log_D_l_subtracted:+.12e}, "
        )

    # Add back the WKB approximation summed from wkb_start_l to infinity.
    # This is the finite high-wave tail that was missing from the previous
    # finite-l implementation.
    return delta_S_reg + wkb_tail
# -------------------------------------------------

def decay_rate(S_bounce, delta_S_reg):
    zero_mode_jacobian = (S_bounce / (2.0 * np.pi))**2

    determinant_prefactor = np.exp(-delta_S_reg)

    return (
        0.5
        *
        zero_mode_jacobian
        * determinant_prefactor
        * np.exp(-S_bounce)
    )
