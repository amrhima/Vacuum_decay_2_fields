import glob
import numpy as np
import matplotlib.pyplot as plt

# jump condition: - r^3 [ G'(r+0,r) - G'(r-0,r) ] = I_2

def compute_jump_matrix(r_grid, G_rk):
    r_grid = np.asarray(r_grid, dtype=float)
    Nr = len(r_grid)
    J = np.full((Nr, 2, 2), np.nan)

    for k in range(1, Nr - 1):
        r_k = r_grid[k]
        dr_plus = r_grid[k + 1] - r_grid[k]
        dr_minus = r_grid[k] - r_grid[k - 1]

        # one-sided derivatives of G(r, r_k) wrt r
        G_plus = (G_rk[k + 1, k] - G_rk[k, k]) / dr_plus
        G_minus = (G_rk[k, k] - G_rk[k - 1, k]) / dr_minus

        J[k] = - (r_k ** 3) * (G_plus - G_minus)

    return J


def main():
    rk_files = sorted(glob.glob("rk_green_data_F*_T*.npz"))
    if not rk_files and glob.glob("rk_green_data.npz"):
        rk_files = ["rk_green_data.npz"]

    if not rk_files:
        print("[ERROR] No rk_green_data_*.npz or rk_green_data.npz found.")
        return

    print("[INFO] Will test jump condition for RK files:")
    for f in rk_files:
        print("  ", f)

    for rk_file in rk_files:
        print("\n===================================")
        print("[JUMP TEST] File:", rk_file)
        print("===================================")

        data = np.load(rk_file, allow_pickle=True)
        r_grid = data["r_grid"]
        G_rk   = data["G_rk"]

        J = compute_jump_matrix(r_grid, G_rk)
        Nr = len(r_grid)

        i_min = int(0.05 * Nr)
        i_max = int(0.95 * Nr)

        J_slice = J[i_min:i_max]
        J_avg = np.nanmean(J_slice, axis=0)

        print("  r range used for averaging J:", r_grid[i_min], "→", r_grid[i_max])
        print("  J_avg =\n", J_avg)
        print("  J_avg - I =\n", J_avg - np.eye(2))
        print("  ||J_avg - I||_F =", np.linalg.norm(J_avg - np.eye(2)))

        # Plot diagonals
        plt.figure(figsize=(7, 5))
        plt.plot(r_grid, J[:, 0, 0], label=r"$J_{11}(r)$")
        plt.plot(r_grid, J[:, 1, 1], label=r"$J_{22}(r)$")
        plt.axhline(1.0, color="k", linestyle="--", label="target 1")
        plt.xlabel(r"$r$")
        plt.ylabel(r"$J_{ii}(r)$")
        plt.title(f"Jump matrix diagonals vs r\n{rk_file}")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Plot off-diagonals
        plt.figure(figsize=(7, 5))
        plt.plot(r_grid, J[:, 0, 1], label=r"$J_{12}(r)$")
        plt.plot(r_grid, J[:, 1, 0], label=r"$J_{21}(r)$")
        plt.axhline(0.0, color="k", linestyle="--", label="target 0")
        plt.xlabel(r"$r$")
        plt.ylabel(r"$J_{ij}(r)$")
        plt.title(f"Jump matrix off-diagonals vs r\n{rk_file}")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()