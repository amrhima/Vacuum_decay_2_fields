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


def full_log_determinant_ratio(
    R_bounce,
    X_bounce,
    Y_bounce,
    phi_fv,
    pot,
    r_min,
    Rmax,
    n_max
):
    if r_min <= 0.0:
        raise ValueError(
            "r_min must be > 0; the Liouville-transformed radial equation "
            "is singular at r=0."
        )
    if Rmax <= r_min:
        raise ValueError("Rmax must be larger than r_min.")

    log_det_ratio = 0.0

    for n in range(n_max + 1):

        # -------------------------------------------------
        # Construct bounce and false-vacuum radial operators
        # -------------------------------------------------

        V_bounce = make_V_matrix(
            n,
            R_bounce,
            X_bounce,
            Y_bounce,
            pot
        )

        V_fv = make_V_matrix_fv(
            n,
            phi_fv,
            pot
        )

        # -------------------------------------------------
        # Radial GY determinant ratio
        # -------------------------------------------------

        if n == 1:
            # Translational zero mode:
            #
            # D_1' = lim_{mu -> 0} D_1(mu)/mu

            mu_best, D_n = pick_mu_value(
                n=n,
                r_min=r_min,
                Rmax=Rmax,
                V_matrix=V_bounce,
                V_matrix_fv=V_fv,
                isolated_eigenvalues=(0.0,)
            )

            if D_n is None:
                raise RuntimeError(
                    "No stable mu plateau found for n=1 zero mode."
                )

        else:
            # No zero mode.
            #
            # In particular n=0 contains the negative mode.
            # We leave it inside the determinant.

            D_n = determinant_ratio(
                mu=0.0,
                n=n,
                r_min=r_min,
                Rmax=Rmax,
                V_matrix=V_bounce,
                V_matrix_fv=V_fv,
                isolated_eigenvalues=()
            )

        # -------------------------------------------------
        # Angular degeneracy
        # -------------------------------------------------

        d_n = partial_wave_degeneracy(n)

        log_det_ratio += d_n * np.log(abs(D_n))

        print(
            f"n={n:3d}, "
            f"d_n={d_n:4d}, "
            f"D_n={D_n:.12e}"
        )

    return log_det_ratio

def decay_rate(S_bounce, log_det_ratio):
    prefactor_zero_modes = (S_bounce / (2.0 * np.pi))**2

    determinant_prefactor = np.exp(
        -0.5 * log_det_ratio
    )

    return (
        prefactor_zero_modes
        * determinant_prefactor
        * np.exp(-S_bounce)
    )
