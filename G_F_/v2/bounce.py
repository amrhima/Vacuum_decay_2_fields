import os
import numpy as np
import matplotlib.pyplot as plt
from cosmoTransitions import pathDeformation as pd
from cosmoTransitions.tunneling1D import PotentialError
from config import DATA_DIR
from potential import (
    PARAMS_DEFAULT,
    find_all_minima,
    CTShiftedLiftedPotential,   
)

# ---------------------------------------------------------------------------
# Bounce for one false → true pair (using shifted+lifted+rotated potential)
# ---------------------------------------------------------------------------

def compute_bounce_for_pair(pot_prime,
                            false_vac_orig,
                            true_vac_orig,
                            tag=""):

    false_vac_orig = np.asarray(false_vac_orig, dtype=float)
    true_vac_orig  = np.asarray(true_vac_orig,  dtype=float)

    # φ' = L^T (φ - φ_F)
    false_prime = pot_prime.to_prime(false_vac_orig)   # should be ~0
    true_prime  = pot_prime.to_prime(true_vac_orig)

    print("\n========================================================")
    print("Computing bounce for pair", tag)
    print("  false_vac (orig) =", false_vac_orig)
    print("  true_vac  (orig) =", true_vac_orig)
    print("  false'          =", false_prime)
    print("  true'           =", true_prime)
    print("========================================================")

    # sanity: false' should be (0,0) numerically
    print("  |false'| = ", np.linalg.norm(false_prime))

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

    print("CosmoTransitions action S =", Y.action)
    print("Final fRatio =", Y.fRatio)

    R = Y.profile1D.R
    Phi_prime = Y.Phi   # (Nr,2)
    X_prime = Phi_prime[:, 0]
    Y_prime = Phi_prime[:, 1]

    # back to ORIGINAL coords, φ = φ_F + L φ'
    Phi_orig = np.array([pot_prime.to_original(phi_p) for phi_p in Phi_prime])
    X_orig   = Phi_orig[:, 0]
    Y_orig   = Phi_orig[:, 1]

    return (true_vac_orig, false_vac_orig,
            R, X_prime, Y_prime, X_orig, Y_orig,
            Y.action)


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    os.chdir(DATA_DIR)
    print(f"[INFO] Writing outputs to {DATA_DIR}")

    params = PARAMS_DEFAULT.copy()
    print("Using parameters:", params)

    minima = find_all_minima(params)

    print("\nLocal minima of ORIGINAL potential (sorted by V):")
    for i, m in enumerate(minima):
        pt = m["point"]
        Vv = m["V"]
        print(f"  M{i}: x={pt[0]: .6f}, y={pt[1]: .6f}, V={Vv: .6f}")

    n_min = len(minima)
    if n_min < 2:
        raise RuntimeError("Need at least two minima for any bounce.")

    vac_points = [m["point"] for m in minima]
    vac_values = [m["V"] for m in minima]

    # Build all downward pairs F->T with V_F > V_T
    pairs = []
    for iF in range(n_min):
        for iT in range(n_min):
            if iF == iT:
                continue
            if vac_values[iF] > vac_values[iT]:
                pairs.append((iF, iT))

    if not pairs:
        raise RuntimeError("No (false,true) pairs with V_false > V_true found.")

    print("\nBounce pairs (indices in minima list, V_F > V_T):")
    for (iF, iT) in pairs:
        print(f"  F=M{iF}, T=M{iT}, V_F={vac_values[iF]: .6f}, V_T={vac_values[iT]: .6f}")

    all_paths = []
    successful = 0

    for (iF, iT) in pairs:
        false_vac = vac_points[iF]
        true_vac  = vac_points[iT]
        V_false   = vac_values[iF]
        V_true    = vac_values[iT]

        tag = f"F(M{iF}, V={V_false:.4f}) -> T(M{iT}, V={V_true:.4f})"
        fname = f"bounce_data_F{iF}_T{iT}.npz"
        if os.path.exists(fname):
            print(f"\n[SKIP] Bounce data already exists: {fname}")
            continue

        pot_prime = CTShiftedLiftedPotential(params, false_vac)

        try:
            (true_vac_arr, false_vac_arr,
             R_bounce,
             X_prime, Y_prime,
             X_orig,  Y_orig,
             S_CT) = compute_bounce_for_pair(
                pot_prime,
                false_vac,
                true_vac,
                tag,
            )
        except PotentialError as e:
            print("CosmoTransitions rejected pair", tag, ":", e)
            continue
        except Exception as e:
            print("Unexpected error for pair", tag, ":", e)
            continue

        all_paths.append({
            "R": R_bounce,
            "X_prime": X_prime,
            "Y_prime": Y_prime,
            "X_orig": X_orig,
            "Y_orig": Y_orig,
            "false_orig": false_vac_arr,
            "true_orig":  true_vac_arr,
            "iF": iF,
            "iT": iT,
            "tag":   tag,
            "S":     S_CT,
        })

        # individual primed profile
        plt.figure()
        plt.plot(R_bounce, X_prime, label=r"$x'(\rho)$")
        plt.plot(R_bounce, Y_prime, label=r"$y'(\rho)$")
        plt.xlabel(r"$\rho$")
        plt.ylabel("primed field value")
        plt.title(f"Bounce profile in primed coords\n{tag},  S = {S_CT:.4f}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # save bounce
        np.savez(
            fname,
            params=params,
            false_vac=false_vac_arr,
            true_vac=true_vac_arr,
            R=R_bounce,
            X_bounce_prime=X_prime,
            Y_bounce_prime=Y_prime,
            X_bounce_orig=X_orig,
            Y_bounce_orig=Y_orig,
            S_CT=S_CT,
            false_index=iF,
            true_index=iT,
            tag=tag,
        )
        print(f"\nSaved bounce data for {tag} to {fname}")

        successful += 1

    if successful == 0:
        print("\nNo successful bounces were computed.")
    else:
        print(f"\nComputed {successful} bounce(s).")

    # Combined field-space plot in original coord.
    if successful > 0:
        plt.figure()
        for p in all_paths:
            plt.plot(p["X_orig"], p["Y_orig"], label=p["tag"])
            plt.scatter(p["false_orig"][0], p["false_orig"][1],
                        c="red",  marker="o")
            plt.scatter(p["true_orig"][0],  p["true_orig"][1],
                        c="blue", marker="x")

        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Bounce paths in field space (all F→T bounces, ORIGINAL coords)")
        plt.axis("equal")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.show()

        print("\n[INFO] Plotted all bounce paths in one field-space plot.")
