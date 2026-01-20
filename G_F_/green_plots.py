import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  
from potential import H_numeric

# potential.py defines the original potential and it's automatic Hessian is H_numeric

# ------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------
def load_npz_or_none(filename):
    try:
        data = np.load(filename, allow_pickle=True)
    except FileNotFoundError:
        print(f"[WARN] File {filename} not found.")
        return None
    except Exception as e:
        print(f"[WARN] Could not load {filename}: {e}")
        return None
    return data


def plot_surface_single_matrix(r_grid, G_ij, title, zlabel,
                               clip_percentile=99.0):
    
    #r_grid : (Nr)
    #G_ij   : (Nr, Nr)  single component

    Nr = len(r_grid)
    Rm, Rp = np.meshgrid(r_grid, r_grid, indexing="ij")

    flat = G_ij.ravel()
    if flat.size == 0:
        print("[WARN] Empty matrix, skipping plot:", title)
        return

    vmax = np.percentile(np.abs(flat), clip_percentile)
    if vmax == 0.0:
        vmax = np.max(np.abs(flat)) or 1.0

    Z_vis = np.clip(G_ij, -vmax, vmax)

    # make figure larger so title/labels are readable
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        Rm, Rp, Z_vis,
        rstride=max(1, Nr // 50),
        cstride=max(1, Nr // 50),
        cmap=cm.coolwarm,
        linewidth=0,
        antialiased=True,
    )

    # bigger fonts + extra space for title
    ax.set_xlabel("r", fontsize=12, labelpad=10)
    ax.set_ylabel("r'", fontsize=12, labelpad=10)
    ax.set_zlabel(zlabel, fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=11, pad=20)

    ax.tick_params(labelsize=10)
    ax.set_zlim(-vmax, vmax)

    cb = fig.colorbar(surf, shrink=0.6, pad=0.1)
    cb.set_label(zlabel, fontsize=11)

    # leave room for the (long) title at the top
    fig.subplots_adjust(top=0.88)

    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# Plots for h, f, B (from RK data) 
# ------------------------------------------------------------
def plot_h_basis_all(r_grid, h_plus, h_minus, tag, n_mode, s2):

    # h_plus, h_minus: shape (Nr, 2, 2), indices [k, i, alpha]

    Nr = len(r_grid)
    if h_plus.shape != (Nr, 2, 2) or h_minus.shape != (Nr, 2, 2):
        print("[WARN] h_plus/h_minus have unexpected shape, skipping h-basis plots.")
        return

    for alpha in range(2):
        plt.figure(figsize=(7, 5))
        plt.plot(r_grid, h_plus[:, 0, alpha],
                 label=rf"$h_1^{{+,\alpha={alpha}}}(r)$")
        plt.plot(r_grid, h_plus[:, 1, alpha],
                 label=rf"$h_2^{{+,\alpha={alpha}}}(r)$")
        plt.plot(r_grid, h_minus[:, 0, alpha], "--",
                 label=rf"$h_1^{{-,\alpha={alpha}}}(r)$")
        plt.plot(r_grid, h_minus[:, 1, alpha], "--",
                 label=rf"$h_2^{{-,\alpha={alpha}}}(r)$")

        plt.xlabel(r"$r$")
        plt.ylabel(r"$h_i^{\pm,\alpha}(r)$")
        plt.title(
            fr"$h$-basis functions (primed fields), {tag}, "
            fr"$n={n_mode}$, $\nu^2={s2}$, $\alpha={alpha}$"
        )
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()


def plot_full_modes_all(r_grid, f_plus, f_minus, tag, n_mode, s2):

    # f_plus, f_minus: shape (Nr, 2, 2), indices [k, i, alpha]

    Nr = len(r_grid)
    if f_plus.shape != (Nr, 2, 2) or f_minus.shape != (Nr, 2, 2):
        print("[WARN] f_plus/f_minus have unexpected shape, skipping full-mode plots.")
        return

    for alpha in range(2):
        plt.figure(figsize=(7, 5))
        f_p0 = np.abs(f_plus[:, 0, alpha])
        f_p1 = np.abs(f_plus[:, 1, alpha])
        f_m0 = np.abs(f_minus[:, 0, alpha])
        f_m1 = np.abs(f_minus[:, 1, alpha])

        plt.plot(r_grid, f_p0, label=rf"$|f_1^{{+,\alpha={alpha}}}(r)|$")
        plt.plot(r_grid, f_p1, label=rf"$|f_2^{{+,\alpha={alpha}}}(r)|$")
        plt.plot(r_grid, f_m0, "--", label=rf"$|f_1^{{-,\alpha={alpha}}}(r)|$")
        plt.plot(r_grid, f_m1, "--", label=rf"$|f_2^{{-,\alpha={alpha}}}(r)|$")

        plt.yscale("log")
        plt.xlabel(r"$r$")
        plt.ylabel(r"$|f_i^{\pm,\alpha}(r)|$")
        plt.title(
            fr"Full fluctuation modes (primed fields), "
            fr"{tag}, $n={n_mode}$, $\nu^2={s2}$, $\alpha={alpha}$"
        )
        plt.grid(True, which="both", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()


def plot_free_modes_all(r_grid, B_plus, B_minus, tag, n_mode, s2):
    
    # B_plus, B_minus: shape (Nr, 2), indices [k, i]
    
    Nr = len(r_grid)
    if B_plus.shape != (Nr, 2) or B_minus.shape != (Nr, 2):
        print("[WARN] B_plus/B_minus have unexpected shape, skipping free-mode plots.")
        return

    plt.figure(figsize=(7, 5))
    Bp0 = np.abs(B_plus[:, 0])
    Bp1 = np.abs(B_plus[:, 1])
    Bm0 = np.abs(B_minus[:, 0])
    Bm1 = np.abs(B_minus[:, 1])

    plt.plot(r_grid, Bp0, label=r"$|B_1^{+}(r)|$")
    plt.plot(r_grid, Bp1, label=r"$|B_2^{+}(r)|$")
    plt.plot(r_grid, Bm0, "--", label=r"$|B_1^{-}(r)|$")
    plt.plot(r_grid, Bm1, "--", label=r"$|B_2^{-}(r)|$")

    plt.yscale("log")
    plt.xlabel(r"$r$")
    plt.ylabel(r"$|B_i^{\pm}(r)|$")
    plt.title(
        fr"Free Bessel modes (primed fields), {tag}, "
        fr"$n={n_mode}$, $\nu^2={s2}$"
    )
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# NEW: plot r^3–scaled Wronskian that enters G_rk
# ------------------------------------------------------------
def plot_wronskian_scaled(r_grid, W_scaled, tag):
    
    # W_scaled : (Nr, 2, 2) array with components
    # W_scaled[k, α, β] = r_k^3 * W_{αβ}(r_k)


    r_grid = np.asarray(r_grid, dtype=float)
    Nr = len(r_grid)

    if W_scaled.shape != (Nr, 2, 2):
        print("[WARN] W_scaled has unexpected shape, skipping Wronskian plot.")
        return

    # same tail definition as in rk_builder.py
    r_min_tail = 0.05
    r_max_tail = 0.9 * r_grid[-1]
    i_min = np.searchsorted(r_grid, r_min_tail)
    i_max = np.searchsorted(r_grid, r_max_tail)

    if i_min >= i_max:
        print("[WARN] Not enough points in Wronskian tail region; "
              "skipping plateau lines.")
        Omega = None
    else:
        Omega = np.mean(W_scaled[i_min:i_max+1, :, :], axis=0)

    plt.figure(figsize=(7, 5))
    comp_labels = {(0, 0): "11", (0, 1): "12", (1, 0): "21", (1, 1): "22"}

    for (a, b), lbl in comp_labels.items():
        plt.plot(r_grid, W_scaled[:, a, b],
                 label=rf"$r^3 W_{{{lbl}}}(r)$")
        if Omega is not None:
            plt.axhline(Omega[a, b], linestyle="--", alpha=0.4)

    plt.xlabel(r"$r$")
    plt.ylabel(r"$r^3 W_{\alpha\beta}(r)$")
    plt.title(f"r^3–scaled Wronskian components (primed basis, {tag})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# Hessian eigenvalues along the bounce vs M_free
# ------------------------------------------------------------
def plot_hessian_eigs_along_bounce_prime(
    R_bounce,
    X_prime,
    Y_prime,
    params,
    M_free,
    false_vac,
    L,
    tag,
):
    # φ'(r) = (x'(r), y'(r)) is the bounce in primed coordinates.
    # φ(r) = φ_F + L φ'(r) is the original bounce.
    # H'(r) = L^T H_orig(φ(r)) L is the Hessian in primed coordinates.
    # Next: Plot the eigenvalues of H'(r) and compare to the eigenvalues
    # of M_free, which should equal H'(0) in the same basis.
    R_bounce = np.asarray(R_bounce, dtype=float)
    X_prime  = np.asarray(X_prime, dtype=float)
    Y_prime  = np.asarray(Y_prime, dtype=float)
    false_vac = np.asarray(false_vac, dtype=float)
    L = np.asarray(L, dtype=float)

    Nr = len(R_bounce)
    lam1 = np.zeros(Nr)
    lam2 = np.zeros(Nr)

    for k in range(Nr):
        phi_prime = np.array([X_prime[k], Y_prime[k]])
        phi_orig  = false_vac + L @ phi_prime

        H_orig = H_numeric(phi_orig, params)
        H_prime = L.T @ H_orig @ L

        eigs = np.linalg.eigvals(H_prime)
        eigs = np.sort(eigs.real)
        lam1[k], lam2[k] = eigs[0], eigs[1]

    eigs_free = np.linalg.eigvals(M_free)
    eigs_free = np.sort(eigs_free.real)

    plt.figure(figsize=(7, 5))
    plt.plot(R_bounce, lam1, label=r"$\lambda_1[H'(r)]$")
    plt.plot(R_bounce, lam2, label=r"$\lambda_2[H'(r)]$")

    plt.axhline(eigs_free[0], linestyle="--", color="k",
                label=r"$\lambda_1(M_\mathrm{free})$")
    plt.axhline(eigs_free[1], linestyle=":", color="k",
                label=r"$\lambda_2(M_\mathrm{free})$")

    plt.xlabel(r"$r$")
    plt.ylabel("eigenvalues in primed basis")
    plt.title(f"Hessian eigenvalues along bounce (primed basis, {tag})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_hessian_eigs_along_bounce_orig(
    R_bounce,
    X_bounce,
    Y_bounce,
    params,
    M_free,
    tag,
):
    # Fallback: original-basis Hessian eigenvalues along the bounce.
    # Used only if primed data / L / false_vac are not available.

    R_bounce = np.asarray(R_bounce, dtype=float)
    X_bounce = np.asarray(X_bounce, dtype=float)
    Y_bounce = np.asarray(Y_bounce, dtype=float)
    Nr = len(R_bounce)

    lam1 = np.zeros(Nr)
    lam2 = np.zeros(Nr)

    for k in range(Nr):
        xk = X_bounce[k]
        yk = Y_bounce[k]
        Hk = H_numeric((xk, yk), params)
        eigs = np.linalg.eigvals(Hk)
        eigs = np.sort(eigs.real)
        lam1[k], lam2[k] = eigs[0], eigs[1]

    eigs_free = np.linalg.eigvals(M_free)
    eigs_free = np.sort(eigs_free.real)

    plt.figure(figsize=(7, 5))
    plt.plot(R_bounce, lam1, label=r"$\lambda_1[H(r)]$")
    plt.plot(R_bounce, lam2, label=r"$\lambda_2[H(r)]$")

    plt.axhline(eigs_free[0], linestyle="--", color="k",
                label=r"$\lambda_1(M_\mathrm{free})$")
    plt.axhline(eigs_free[1], linestyle=":", color="k",
                label=r"$\lambda_2(M_\mathrm{free})$")

    plt.xlabel(r"$r$")
    plt.ylabel("eigenvalues (original basis)")
    plt.title(f"Hessian eigenvalues along bounce (original basis, {tag})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# Per-bounce plotting
# ------------------------------------------------------------
def process_one_bounce(rk_filename):

    base = os.path.basename(rk_filename)

    # two cases:
    #  1) rk_green_<TAG>.npz → TAG = <TAG>
    #  2) rk_green_data.npz → TAG = "data"
    if base == "rk_green_data.npz":
        tag_suffix = "data"
    else:
        tag_suffix = base[len("rk_green_") : -len(".npz")]

    print("\n===============================================")
    print(f"[BOUNCE] Processing RK file: {rk_filename}")
    print(f"         tag suffix = '{tag_suffix}'")
    print("===============================================")

    # ---- Load RK Green ----
    rk_data = load_npz_or_none(rk_filename)
    if rk_data is None:
        print("[ERROR] Could not load RK data, skipping this bounce.")
        return

    r_rk = rk_data["r_grid"]
    G_rk = rk_data["G_rk"]
    tag_rk    = str(rk_data.get("tag", f"RK_{tag_suffix}"))
    s2_rk    = float(rk_data["s2"])
    n_mode_rk = int(rk_data.get("n_mode", -1))
    print(f"[INFO] Loaded RK Green: shape G_rk = {G_rk.shape}")

    # optional extras
    h_plus  = rk_data["h_plus"]  if "h_plus"  in rk_data.files else None
    h_minus = rk_data["h_minus"] if "h_minus" in rk_data.files else None
    f_plus  = rk_data["f_plus"]  if "f_plus"  in rk_data.files else None
    f_minus = rk_data["f_minus"] if "f_minus" in rk_data.files else None
    B_plus  = rk_data["B_plus"]  if "B_plus"  in rk_data.files else None
    B_minus = rk_data["B_minus"] if "B_minus" in rk_data.files else None
    W_scaled = rk_data["W_scaled"] if "W_scaled" in rk_data.files else None

    # bounce / params
    R_bounce = rk_data["R_bounce"] if "R_bounce" in rk_data.files else None

    # prefer primed bounce if available
    if "X_bounce_prime" in rk_data.files and "Y_bounce_prime" in rk_data.files:
        X_prime = rk_data["X_bounce_prime"]
        Y_prime = rk_data["Y_bounce_prime"]
    else:
        X_prime = None
        Y_prime = None

    X_bounce_orig = rk_data["X_bounce"] if "X_bounce" in rk_data.files else None
    Y_bounce_orig = rk_data["Y_bounce"] if "Y_bounce" in rk_data.files else None

    params = rk_data["params"] if "params" in rk_data.files else None

    # M_free preferred; fallback to H_free for very old files
    if "M_free" in rk_data.files:
        M_free = rk_data["M_free"]
    elif "H_free" in rk_data.files:
        M_free = rk_data["H_free"]
    else:
        M_free = None

    # linear field redefinition info (for primed Hessian)
    L = rk_data["L"] if "L" in rk_data.files else None
    false_vac = rk_data["false_vac"] if "false_vac" in rk_data.files else None

    # ---- Load matching FD and spectral Greens, if present ----
    fd_filename   = f"fd_green_{tag_suffix}.npz"
    spec_filename = f"spec_green_{tag_suffix}.npz"

    fd_data   = load_npz_or_none(fd_filename)
    spec_data = load_npz_or_none(spec_filename)

    if fd_data is not None:
        r_fd   = fd_data["r_grid"]
        G_fd   = fd_data["G_fd"]
        tag_fd = str(fd_data.get("tag", f"FD_{tag_suffix}"))
        s2_fd = float(fd_data["s2"])
        n_mode_fd = int(fd_data.get("n_mode", -1))
        print(f"[INFO] Loaded FD Green: {fd_filename}, shape G_fd = {G_fd.shape}")
    else:
        r_fd = G_fd = None
        tag_fd = f"FD_{tag_suffix}"
        s2_fd = -1.0
        n_mode_fd = -1

    if spec_data is not None:
        r_sp   = spec_data["r_grid"]
        G_sp   = spec_data["G_spec"]
        tag_sp = str(spec_data.get("tag", f"spec_{tag_suffix}"))
        s2_sp = float(spec_data["s2"])
        n_mode_sp = int(spec_data.get("n_mode", -1))
        print(f"[INFO] Loaded spectral Green: {spec_filename}, shape G_spec = {G_sp.shape}")
    else:
        r_sp = G_sp = None
        tag_sp = f"spec_{tag_suffix}"
        s2_sp = -1.0
        n_mode_sp = -1

    # --------------------------------------------------------
    # h, full modes, free modes from RK data
    # --------------------------------------------------------
    if (h_plus is not None) and (h_minus is not None):
        print("\n[INFO] Plotting h-basis functions from RK data...")
        plot_h_basis_all(r_rk, h_plus, h_minus, tag_rk, n_mode_rk, s2_rk)
    else:
        print("\n[INFO] h-basis arrays not found in RK file; skipping h-plots.")

    if (f_plus is not None) and (f_minus is not None):
        print("[INFO] Plotting full mode functions f^± from RK data...")
        plot_full_modes_all(r_rk, f_plus, f_minus, tag_rk, n_mode_rk, s2_rk)
    else:
        print("[INFO] f^± arrays not found in RK file; skipping full-mode plots.")

    if (B_plus is not None) and (B_minus is not None):
        print("[INFO] Plotting free modes B^± from RK data...")
        plot_free_modes_all(r_rk, B_plus, B_minus, tag_rk, n_mode_rk, s2_rk)
    else:
        print("[INFO] B^± arrays not found in RK file; skipping free-mode plots.")

    # --------------------------------------------------------
    # NEW: r^3–scaled Wronskian that enters G_rk
    # --------------------------------------------------------
    if W_scaled is not None:
        print("[INFO] Plotting r^3–scaled Wronskian components...")
        plot_wronskian_scaled(r_rk, W_scaled, tag_rk)
    else:
        print("[INFO] No W_scaled array found; skipping Wronskian plot.")

    # --------------------------------------------------------
    # Hessian eigenvalues along the bounce vs M_free
    # --------------------------------------------------------
    if (R_bounce is not None and params is not None and M_free is not None):
        if (X_prime is not None and Y_prime is not None
                and L is not None and false_vac is not None):
            print("\n[INFO] Plotting primed-basis Hessian eigenvalues along bounce...")
            plot_hessian_eigs_along_bounce_prime(
                R_bounce, X_prime, Y_prime,
                params, M_free, false_vac, L, tag_rk
            )
        elif (X_bounce_orig is not None and Y_bounce_orig is not None):
            print("\n[INFO] Primed data not complete; "
                  "plotting original-basis Hessian eigenvalues instead...")
            plot_hessian_eigs_along_bounce_orig(
                R_bounce, X_bounce_orig, Y_bounce_orig,
                params, M_free, tag_rk
            )
        else:
            print("\n[INFO] No bounce coordinates found for Hessian plot.")
    else:
        print("\n[INFO] Bounce/Hessian/parameter data not fully present; "
              "skipping Hessian-eigenvalue plot.")

    # --------------------------------------------------------
    # 3D Green components: 11, 12, 21, 22
    # --------------------------------------------------------
    components = [
        (0, 0, "11"),
        (0, 1, "12"),
        (1, 0, "21"),
        (1, 1, "22"),
    ]

    for (i_idx, j_idx, label_ij) in components:
        print(f"\n=== [Bounce {tag_suffix}] Green component {label_ij} "
              f"(i={i_idx}, j={j_idx}) ===")

        # ----- RK -----
        if G_rk is not None:
            G_ij = G_rk[:, :, i_idx, j_idx]
            title = (rf"$G_{{{label_ij}}}(r,r')$ (RK, primed fields, {tag_rk}) "
                     rf"$n={n_mode_rk}$, $\nu^2={s2_rk}$")
            zlabel = rf"$G_{{{label_ij}}}^\mathrm{{RK}}$ (primed, clipped)"
            plot_surface_single_matrix(r_rk, G_ij, title, zlabel,
                                       clip_percentile=99.0)

        # ----- FD -----
        if G_fd is not None:
            G_ij = G_fd[:, :, i_idx, j_idx]
            title = (rf"$G_{{{label_ij}}}(r,r')$ (FD, {tag_fd}) "
                     rf"$n={n_mode_fd}$, $\nu^2={s2_fd}$")
            zlabel = rf"$G_{{{label_ij}}}^\mathrm{{FD}}$ (clipped)"
            plot_surface_single_matrix(r_fd, G_ij, title, zlabel,
                                       clip_percentile=99.0)

        # ----- spectral -----
        if G_sp is not None:
            G_ij = G_sp[:, :, i_idx, j_idx]
            title = (rf"$G_{{{label_ij}}}(r,r')$ (spectral, {tag_sp}) "
                     rf"$n={n_mode_sp}$, $\nu^2={s2_sp}$")
            zlabel = rf"$G_{{{label_ij}}}^\mathrm{{spec}}$ (clipped)"
            plot_surface_single_matrix(r_sp, G_ij, title, zlabel,
                                       clip_percentile=99.0)

    print(f"\n[INFO] Finished plotting for bounce tag '{tag_suffix}'.\n")


# ------------------------------------------------------------
# Main: loop over all bounces
# ------------------------------------------------------------
if __name__ == "__main__":
    # find all RK files
    rk_files = sorted(glob.glob("rk_green_*.npz"))

    # also support older single-file name
    if os.path.exists("rk_green_data.npz") and "rk_green_data.npz" not in rk_files:
        rk_files.append("rk_green_data.npz")

    if not rk_files:
        print("[ERROR] No rk_green_*.npz or rk_green_data.npz found.")
        print("        Run the RK builder for your bounces first.")
        raise SystemExit

    print("[INFO] Found RK files:")
    for f in rk_files:
        print("   ", f)

    # process each bounce separately
    for rk_file in rk_files:
        process_one_bounce(rk_file)

    print("\n[INFO] Done plotting all bounces.")