from G_Y_WKB.WKB_terms import (
    build_W_profile,
    estimate_bubbledet_wkb_tail,
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
    n_max,
    wkb_start_n=2,
):
    if r_min <= 0.0:
        raise ValueError(
            "r_min must be > 0; the Liouville-transformed radial "
            "equation is singular at r=0."
        )

    if Rmax <= r_min:
        raise ValueError("Rmax must be larger than r_min.")

    if n_max < 0:
        raise ValueError("n_max must be non-negative.")

    if wkb_start_n < 2:
        raise ValueError("wkb_start_n must be at least 2.")

    if wkb_start_n > n_max + 1:
        raise ValueError(
            "wkb_start_n must be no larger than n_max + 1; "
            f"got wkb_start_n={wkb_start_n}, n_max={n_max}."
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

    wkb_by_n, wkb_tail = estimate_bubbledet_wkb_tail(
        R_bounce=R_bounce,
        W_profile=W_profile,
        H_fv=H_fv,
        l_max=n_max + 1,
        start_l=wkb_start_n,
    )

    delta_S_reg = 0.0

    for n in range(n_max + 1):

        # -------------------------------------------------
        # Construct radial operators
        # -------------------------------------------------

        V_bounce = make_V_matrix(
            n,
            R_bounce,
            X_bounce,
            Y_bounce,
            pot,
        )

        V_fv = make_V_matrix_fv(
            n,
            phi_fv,
            pot,
        )

        # -------------------------------------------------
        # Zero modes
        # -------------------------------------------------

        if n == 1:
            # Remove the one radial zero eigenvalue.
            # The angular degeneracy d_1 = 4 accounts for
            # the four translational zero modes.

            mu_best, D_n = pick_mu_value(
                n=n,
                r_min=r_min,
                Rmax=Rmax,
                V_matrix=V_bounce,
                V_matrix_fv=V_fv,
                isolated_eigenvalues=(0.0,),
            )

            if D_n is None:
                raise RuntimeError(
                    "No stable mu plateau found for the n=1 zero mode."
                )

        else:
            # The n=0 negative mode is retained in the
            # determinant. Its sign is removed below when
            # forming log(abs(D_n)).

            D_n = determinant_ratio(
                mu=0.0,
                n=n,
                r_min=r_min,
                Rmax=Rmax,
                V_matrix=V_bounce,
                V_matrix_fv=V_fv,
                isolated_eigenvalues=(),
            )

        if not np.isfinite(D_n):
            raise RuntimeError(
                f"Non-finite determinant ratio obtained for n={n}: {D_n}"
            )

        if D_n == 0.0:
            raise RuntimeError(
                f"Zero determinant ratio obtained for n={n}."
            )

        # -------------------------------------------------
        # WKB subtraction and angular degeneracy
        # -------------------------------------------------

        j = n + 1
        d_n = partial_wave_degeneracy(n)

        log_D_n = np.log(abs(D_n))

        # The low partial waves are treated exactly.  BubbleDet subtracts its
        # higher-order WKB approximation only from the high-wave remainder.
        if n >= wkb_start_n:
            log_D_n_subtracted = log_D_n - wkb_by_n[n]
        else:
            log_D_n_subtracted = log_D_n

        delta_S_reg += 0.5 * d_n * log_D_n_subtracted

        print(
            f"n={n:3d}, "
            f"j={j:3d}, "
            f"d_n={d_n:4d}, "
            f"D_n={D_n:.12e}, "
            f"log_D_n={log_D_n:+.12e}, "
            f"subtracted={log_D_n_subtracted:+.12e}, "
        )

    # Add back the WKB approximation summed from wkb_start_n to infinity.
    # This is the finite high-wave tail that was missing from the previous
    # finite-n implementation.
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
