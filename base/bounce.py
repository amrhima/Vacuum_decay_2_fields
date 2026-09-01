"""Stage 1 of the two-field false-vacuum-decay fluctuation-determinant pipeline.

Computes the Euclidean O(4) bounce profile for each downward false -> true
vacuum pair via CosmoTransitions' ``pathDeformation.fullTunneling``. The bounce
is solved in the shifted/lifted/rotated "primed" coordinates (false vacuum at
the origin) and the resulting profile is mapped back to original field-space
coordinates. For every successful pair the script writes
``bounce_data_F{iF}_T{iT}.npz`` -- the input that every downstream pipeline
script consumes -- and produces diagnostic plots (per-pair primed profile and a
combined field-space path plot). Run as a script to also display the plots
interactively; when imported, only ``compute_bounce_for_pair`` is used.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from cosmoTransitions import pathDeformation as pd
from G_F_.v2.potential import find_all_minima
from .potential import PARAMS_DEFAULT, CTShiftedLiftedPotential, find_vacua_from_potential, V_numeric, gradV_numeric


# ---------------------------------------------------------------------------
# Bounce for one false → true pair (using shifted+lifted+rotated potential)
# ---------------------------------------------------------------------------
def compute_bounce_for_pair(pot_prime,
                            false_vac_orig,
                            true_vac_orig):
    false_vac_orig = np.asarray(false_vac_orig, dtype=float)
    true_vac_orig  = np.asarray(true_vac_orig,  dtype=float)

    # φ' = L^T (φ - φ_F)
    false_prime = pot_prime.to_prime(false_vac_orig)   # should be ~0
    true_prime  = pot_prime.to_prime(true_vac_orig)

    # sanity: V'(false') should be ~0
    V_false_prime = float(pot_prime.V(false_prime))
    print("  Check: V'(false') =", V_false_prime, " (should be ~0)")

    path_guess_prime = np.vstack([true_prime, false_prime])

    Y = pd.fullTunneling(
        path_guess_prime,
        pot_prime.V,
        pot_prime.dV,
        maxiter=60,
        verbose=True,
        tunneling_init_params={"alpha": 3},  # O(4)
    )

    R = Y.profile1D.R
    profile1D = Y.profile1D
    Phi_prime = Y.Phi   # (Nr,2)
    X_prime = Phi_prime[:, 0]
    Y_prime = Phi_prime[:, 1]

    # back to ORIGINAL coords, φ = φ_F + L φ'
    Phi_orig = np.array([pot_prime.to_original(phi_p) for phi_p in Phi_prime])
    X_orig   = Phi_orig[:, 0]
    Y_orig   = Phi_orig[:, 1]

    return (profile1D,
            R, 
            X_prime, 
            Y_prime, 
            X_orig, 
            Y_orig,
            Y.action)

# ----------------------------------------------------------------------
# CosmoTransitions tunneling: find all downward pairs of vacua and compute bounce for each pair
# ----------------------------------------------------------------------
def compute_bounce_for_all_pairs(params=PARAMS_DEFAULT, force_recompute=False, tag=""):                    
    # 1) Identify true and false vacuum
    minima = find_all_minima(params)
    n_min = len(minima)
    if n_min < 2:
        raise RuntimeError("Need at least two minima for any bounce.")

    vac_points = [m["point"] for m in minima]
    vac_values = [m["V"] for m in minima]
    
    # Build all downward pairs F->T with V_F > V_T
    pairs = []
    results = []
    for falseVacuumIndex in range(n_min):
        for trueVacuumIndex in range(n_min):
            if falseVacuumIndex == trueVacuumIndex:
                continue
            if vac_values[falseVacuumIndex] > vac_values[trueVacuumIndex]:
                pairs.append((falseVacuumIndex, trueVacuumIndex))
                
    for (falseVacuumIndex, trueVacuumIndex) in pairs:
        filename = f"bounce_data_F{falseVacuumIndex}_T{trueVacuumIndex}_{tag}.npz"
        if (not force_recompute) and os.path.exists(filename):
            loaded = loadBounceData(falseVacuumIndex, trueVacuumIndex, tag)
            loaded_params = np.asarray(loaded["params"], dtype=float)
            if loaded_params.shape == params.shape and np.allclose(loaded_params, params, rtol=0.0, atol=1e-12):
                result = {
                    "falseVacuumIndex": int(loaded["false_index"]),
                    "trueVacuumIndex": int(loaded["true_index"]),
                    "params": loaded_params,
                    "true_vacuum": loaded["true_vac"],
                    "false_vacuum": loaded["false_vac"],
                    "profile1D": None,
                    "R_bounce": loaded["R"],
                    "X_prime": loaded["X_bounce_prime"],
                    "Y_prime": loaded["Y_bounce_prime"],
                    "X_orig": loaded["X_bounce_orig"],
                    "Y_orig": loaded["Y_bounce_orig"],
                    "Action": float(loaded["S_CT"]),
                }
                print(f"\n[LOAD] Reusing saved bounce data: {filename}")
                results.append(result)
                continue
            print(f"\n[RECOMPUTE] Cached params differ from requested params: {filename}")

        false_vac = vac_points[falseVacuumIndex]
        true_vac  = vac_points[trueVacuumIndex]

        pot_prime = CTShiftedLiftedPotential(params, false_vac)
        
        (
            profile1D, 
            R_bounce, 
            X_prime, 
            Y_prime, 
            X_orig, 
            Y_orig,
            S_CT
        ) = compute_bounce_for_pair(
            pot_prime,
            false_vac,
            true_vac,
        )
        result = {
            "falseVacuumIndex": falseVacuumIndex,
            "trueVacuumIndex": trueVacuumIndex,
            "params": params,
            "true_vacuum": true_vac,
            "false_vacuum": false_vac,
            "profile1D": profile1D,
            "R_bounce": R_bounce,
            "X_prime": X_prime,
            "Y_prime": Y_prime,
            "X_orig": X_orig,
            "Y_orig": Y_orig,
            "Action": S_CT,
        }
        saveBounceData(result["falseVacuumIndex"], result["trueVacuumIndex"], result, tag)
        results.append(result)
    
    return results



# ---------------------------------------------------------------------------
# plotting function for bounce profile
# ---------------------------------------------------------------------------
def plot_bounce_profile(R_bounce, X_prime, Y_prime, S_CT):
    # individual primed profile
    plt.figure()
    plt.plot(R_bounce, X_prime, label=r"$x'(\rho)$")
    plt.plot(R_bounce, Y_prime, label=r"$y'(\rho)$")
    plt.xlabel(r"$\rho$")
    plt.ylabel("primed field value")
    plt.title(f"Bounce profile in primed coords,  S = {S_CT:.4f}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------------------
# save bounce data to file
# ---------------------------------------------------------------------------
def saveBounceData(falseVacuumIndex, trueVacuumIndex, result, tag=""):
    filename = f"bounce_data_F{falseVacuumIndex}_T{trueVacuumIndex}_{tag}.npz"
    if os.path.exists(filename):
        print(f"\n[SKIP] Bounce data already exists: {filename}")
        return
    np.savez(filename, 
            params=result["params"],
            false_vac=result["false_vacuum"],
            true_vac=result["true_vacuum"],
            R=result["R_bounce"],
            X_bounce_prime=result["X_prime"],
            Y_bounce_prime=result["Y_prime"],
            X_bounce_orig=result["X_orig"],
            Y_bounce_orig=result["Y_orig"],
            S_CT=result["Action"],
            false_index=falseVacuumIndex,
            true_index=trueVacuumIndex,
            tag=tag,
             )
    print(f"Saved bounce data to {filename}")
    
# ---------------------------------------------------------------------------
# load bounce data from file
# ---------------------------------------------------------------------------
def loadBounceData(falseVacuumIndex, trueVacuumIndex, tag=""):
    filename = f"bounce_data_F{falseVacuumIndex}_T{trueVacuumIndex}_{tag}.npz"
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Bounce data not found: {filename}")
    return np.load(filename)


if __name__ == "__main__":
    results = compute_bounce_for_all_pairs()
    for res in results:
        print("\n\nBounce result:")
        print("  params =", res["params"])
        print("  true_vacuum =", res["true_vacuum"])
        print("  false_vacuum =", res["false_vacuum"])
        print("  Action =", res["Action"])
        plot_bounce_profile(res["R_bounce"], res["X_prime"], res["Y_prime"], res["Action"])
