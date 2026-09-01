# A file for testing the fluctuation_operator.py and determinant_evaluation.py modules.

import numpy as np

from base.fluctuation_operator import (
    make_V_matrix,
    make_V_matrix_fv
)

from G_Y_WKB.determinant_evaluation import determinant_ratio


# ------------------------------------------------------------
# Dummy potential
# ------------------------------------------------------------

class ToyPotential:
    def H(self, phi):
        x, y = phi

        return np.array([
            [2.0 + 0.1*x*x, 0.05*x*y],
            [0.05*x*y,      3.0 + 0.1*y*y]
        ])

class DummyPotential:
    def H(self, phi):
        # Constant positive-definite 2x2 Hessian
        return np.array([
            [2.0, 0.2],
            [0.2, 3.0]
        ])


pot = ToyPotential()
# pot = DummyPotential()


# ------------------------------------------------------------
# Fake "bounce" profile
#
# Since H is constant, the actual values do not matter.
# Bounce and FV operators will be identical.
# ------------------------------------------------------------

r_min = 1e-4
Rmax = 5.0


R_bounce = np.linspace(r_min, Rmax, 200)

# X_bounce = np.zeros_like(R_bounce)
# Y_bounce = np.zeros_like(R_bounce)

# phi_fv = np.array([0.0, 0.0])

X_bounce = 0.5 * np.exp(-R_bounce**2)
Y_bounce = 0.3 * np.exp(-0.5 * R_bounce**2)

phi_fv = np.array([0.0, 0.0])



# ------------------------------------------------------------
# Test several partial waves
# ------------------------------------------------------------

for n in [0, 2, 5]:

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

    ratio = determinant_ratio(
        mu=0.0,
        n=n,
        r_min=r_min,
        Rmax=Rmax,
        V_matrix=V_bounce,
        V_matrix_fv=V_fv,
        isolated_eigenvalues=()
    )

    print(
        f"n = {n}, "
        f"determinant ratio = {ratio:.12e}"
    )
