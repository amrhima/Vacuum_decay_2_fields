import glob
import numpy as np
import matplotlib.pyplot as plt
from potential import CTShiftedLiftedPotential  

# Green-equation test for the RK Green function G_rk(r,r'):
# L_r G(r,r') = δ(r - r') / r'^3 * I_2
# Test with a simple test function φ(r)=1:
# ∫_0^∞ dr r^3 L_r G(r,r')  ≈  I_2  (for each r')
# On the grid:
# S(r'_ℓ) = Σ_k [ r_k^3 Δr L_r G(r_k, r'_ℓ) ]  ≈  I_2
# With L_r, the full O(4) fluctuation operator in the PRIMED basis:
# ------------------------------------------------------------
# Helper: build H'(r) along the bounce in PRIMED basis
# ------------------------------------------------------------
def build_H_full_prime_on_r_grid(r_grid, R_bounce, X_prime, Y_prime,
                                 params, false_vac):
    r_grid = np.asarray(r_grid, dtype=float)
    R_bounce = np.asarray(R_bounce, dtype=float)
    X_prime = np.asarray(X_prime, dtype=float)
    Y_prime = np.asarray(Y_prime, dtype=float)
    false_vac = np.asarray(false_vac, dtype=float)

    pot_lin = CTShiftedLiftedPotential(params, false_vac)

    def x_prime_of_r(r):
        return np.interp(r, R_bounce, X_prime)

    def y_prime_of_r(r):
        return np.interp(r, R_bounce, Y_prime)

    Nr = len(r_grid)
    H_full = np.zeros((Nr, 2, 2))

    for k, r in enumerate(r_grid):
        xp = x_prime_of_r(r)
        yp = y_prime_of_r(r)
        phi_prime = np.array([xp, yp], dtype=float)
        H_full[k] = pot_lin.H(phi_prime)   # H'(φ'(r)) in primed basis

    return H_full


# ------------------------------------------------------------
# Helper: finite-difference derivatives of G wrt r
# ------------------------------------------------------------
def fd_derivatives_G(r_grid, G_rk):
    r_grid = np.asarray(r_grid, dtype=float)
    Nr = len(r_grid)
    dr = r_grid[1] - r_grid[0]

    dG_dr = np.zeros_like(G_rk)
    d2G_dr2 = np.zeros_like(G_rk)

    for k in range(1, Nr - 1):
        dG_dr[k] = (G_rk[k + 1] - G_rk[k - 1]) / (2.0 * dr)
        d2G_dr2[k] = (G_rk[k + 1] - 2.0 * G_rk[k] + G_rk[k - 1]) / (dr * dr)

    return dG_dr, d2G_dr2


# ------------------------------------------------------------
# Main smeared Green-equation test
# ------------------------------------------------------------
def main():
    rk_files = sorted(glob.glob("rk_green_data_F*_T*.npz"))
    if not rk_files and glob.glob("rk_green_data.npz"):
        rk_files = ["rk_green_data.npz"]

    if not rk_files:
        print("[ERROR] No rk_green_data_*.npz or rk_green_data.npz found.")
        return

    print("[INFO] Will run smeared Green-equation test for RK files:")
    for f in rk_files:
        print("   ", f)

    for rk_file in rk_files:
        print("\n========================================")
        print("[GREEN SMEARED TEST] File:", rk_file)
        print("========================================")

        data = np.load(rk_file, allow_pickle=True)

        r_grid = data["r_grid"]          # (Nr,)
        G_rk   = data["G_rk"]            # (Nr, Nr, 2, 2)
        params = data["params"]
        false_vac = data["false_vac"]

        R_bounce = data["R_bounce"]
        X_prime  = data["X_bounce_prime"]
        Y_prime  = data["Y_bounce_prime"]

        s2    = float(data["s2"])
        n_mode = int(data["n_mode"])

        Nr = len(r_grid)
        dr = r_grid[1] - r_grid[0]
        print(f"  Nr = {Nr},  r in [{r_grid[0]}, {r_grid[-1]}],  dr = {dr}")
        print(f"  n_mode = {n_mode},  s2 = {s2}")

        # ---- build H'(r) along r_grid in primed basis ----
        print("  [INFO] Rebuilding H'(r) along the bounce (primed basis)...")
        H_full = build_H_full_prime_on_r_grid(
            r_grid, R_bounce, X_prime, Y_prime, params, false_vac
        )  # (Nr, 2, 2)

        # operator pieces n(n+2)/r^2 and s2 I
        ell_term = np.zeros((Nr, 2, 2))
        s2_term = np.zeros((Nr, 2, 2))
        for k, r in enumerate(r_grid):
            inv_r2 = 0.0 if r == 0.0 else 1.0 / (r * r)
            n_fac = n_mode * (n_mode + 2) * inv_r2
            ell_term[k] = n_fac * np.eye(2)
            s2_term[k] = s2 * np.eye(2)

        # ---- derivatives of G wrt r ----
        print("  [INFO] Computing finite-difference derivatives of G(r,r') wrt r...")
        dG_dr, d2G_dr2 = fd_derivatives_G(r_grid, G_rk)

        # ---- build L_r G(r,r') on the grid ----
        print("  [INFO] Constructing L_r G on the r-grid...")
        L_G = np.zeros_like(G_rk)  # (Nr, Nr, 2, 2)

        for k, r in enumerate(r_grid):
            if k == 0 or k == Nr - 1:
                continue
            inv_r = 0.0 if r == 0.0 else 1.0 / r

            V_k = ell_term[k] + H_full[k] + s2_term[k]  # (2,2)

            for l in range(Nr):
                # Full matrix equation in field space:
                # L_r G = -G'' - (3/r) G' + V_k @ G
                L_G[k, l] = (
                    -d2G_dr2[k, l]
                    - 3.0 * inv_r * dG_dr[k, l]
                    + V_k @ G_rk[k, l]
                )

        # ---- smeared test with φ(r) = 1 ----
        print("  [INFO] Performing smeared test with φ(r)=1 ...")
        r3 = r_grid**3
        weights = (r3 * dr).reshape(Nr, 1, 1, 1)   # (Nr,1,1,1)

        L_G_weighted = weights * L_G
        S = np.sum(L_G_weighted[1:Nr-1], axis=0)  # (Nr, 2, 2)

        # average S(r') over interior r' range
        l_min = int(0.1 * Nr)
        l_max = int(0.9 * Nr)
        S_slice = S[l_min:l_max+1]  # (N_slice, 2, 2)
        S_avg = np.mean(S_slice, axis=0)

        print(f"  r' range used for averaging S: {r_grid[l_min]} → {r_grid[l_max]}")
        print("  S_avg (full 2x2 matrix) =\n", S_avg)

        # per-component comparison to identity
        target = np.eye(2)
        diff = S_avg - target
        print("  S_avg - I =\n", diff)
        print("  ||S_avg - I||_F =", np.linalg.norm(diff))

        for i in range(2):
            for j in range(2):
                tij = target[i, j]
                print(f"    Component ({i},{j}): "
                      f"S_avg = {S_avg[i,j]: .6e}, "
                      f"target = {tij: .6e}, "
                      f"error = {S_avg[i,j]-tij: .6e}")

        # ---- Plots for each component S_ij(r') ----
        r_prime = r_grid

        # diagonals
        plt.figure(figsize=(7, 5))
        plt.plot(r_prime, S[:, 0, 0], label=r"$S_{11}(r')$")
        plt.plot(r_prime, S[:, 1, 1], label=r"$S_{22}(r')$")
        plt.axhline(1.0, color="k", linestyle="--", label="target 1")
        plt.xlabel(r"$r'$")
        plt.ylabel(r"$S_{ii}(r')$")
        plt.title(f"Smeared Green-equation test (diagonals)\n{rk_file}")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # off-diagonals
        plt.figure(figsize=(7, 5))
        plt.plot(r_prime, S[:, 0, 1], label=r"$S_{12}(r')$")
        plt.plot(r_prime, S[:, 1, 0], label=r"$S_{21}(r')$")
        plt.axhline(0.0, color="k", linestyle="--", label="target 0")
        plt.xlabel(r"$r'$")
        plt.ylabel(r"$S_{ij}(r')$")
        plt.title(f"Smeared Green-equation test (off-diagonals)\n{rk_file}")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()