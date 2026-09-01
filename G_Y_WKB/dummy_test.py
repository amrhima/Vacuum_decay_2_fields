# A file for testing the fluctuation_operator.py and determinant_evaluation.py modules.

import numpy as np

from base.fluctuation_operator import (
    make_V_matrix,
    make_V_matrix_fv
)

from base.bounce import (
    compute_bounce_for_all_pairs
)

from G_Y_WKB.decay_rate import full_log_determinant_ratio

bounce_pairs = compute_bounce_for_all_pairs();
bounce_pair = bounce_pairs[0]
r_min = 0.0001
r_max = bounce_pair["R_bounce"][-1]

R = full_log_determinant_ratio(
    R_bounce= bounce_pair["R_bounce"],
        X_bounce= bounce_pair["X_prime"],
        Y_bounce= bounce_pair["Y_prime"],
        # CTShiftedLiftedPotential uses primed coordinates; the false vacuum
        # is therefore at the origin in this coordinate system.
        phi_fv = np.zeros(2),
        pot = bounce_pair["pot_prime"],
        r_min =r_min,
        Rmax =r_max,
        n_max = 5
)
    
print("Full log determinant ratio:", R)
