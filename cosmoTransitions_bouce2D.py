import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

from cosmoTransitions import pathDeformation as pd
from potential2D import V_numeric, find_critical_points, find_vacua_from_potential, V_func, gradV_func


# ----------------------------------------------------------------------
# CosmoTransitions tunneling: fullTunneling
# ----------------------------------------------------------------------

def run_cosmotransitions_bounce():
    params = (1.0, 1.2, 0.5, 0.6, 0.2, 0.3)


    # 1) Identify true and false vacuum
    fv, fv_V, tv,tv_V  = find_vacua_from_potential(params=params)
    points = [{'x': tv}, {'x': fv}]
    def V(x):
        x = np.asarray(x)
        phi1 = x[..., 0]
        phi2 = x[..., 1]
        return V_func(phi1, phi2, *params)  # broadcasts over array

    def grad_V(x):
        x = np.asarray(x)
        phi1 = x[..., 0]
        phi2 = x[..., 1]
        dphi1, dphi2 = gradV_func(phi1, phi2, *params)
        return np.stack([dphi1, dphi2], axis=-1)  # shape like x
        
    print("\n=== Selected vacua ===")
    print(f"  True vacuum : phi = {tv[0]},  V = {tv_V}")
    print(f"  False vacuum: phi = {fv[0]},  V = {fv_V}")

    # 2) Initial path: straight line from TV (first) to FV (last)
    path_pts = np.vstack([tv, fv])

    print(path_pts)
    # 3) Run CosmoTransitions tunneling
    print("\nRunning CosmoTransitions pathDeformation.fullTunneling ...")
    Y = pd.fullTunneling(path_pts, V, grad_V, verbose=True)

    profile1D = Y.profile1D
    Phi_path = Y.Phi          # shape (n_points, 2)
    action = Y.action
    fRatio = Y.fRatio

    print("\n=== Bounce result ===")
    print(f"  Action S4 ≈ {action:.6f}")
    print(f"  fRatio (transverse force / |grad V|) ≈ {fRatio:.3e}")
    print(f"  Number of points along path: {Phi_path.shape[0]}")

    # ------------------------------------------------------------------
    # 4. Plots: field-space contour + path, and radial profiles
    # ------------------------------------------------------------------

    # (a) Field-space contour with tunneling path
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    ax = axes[0]
    # Choose a box that comfortably contains both vacua
    all_x = np.array([p["x"] for p in points])
    x_min = np.min(all_x[:, 0]) - 0.5
    x_max = np.max(all_x[:, 0]) + 0.5
    y_min = np.min(all_x[:, 1]) - 0.5
    y_max = np.max(all_x[:, 1]) + 0.5

    nx = ny = 200
    X = np.linspace(x_min, x_max, nx)
    Ygrid = np.linspace(y_min, y_max, ny)
    XX, YY = np.meshgrid(X, Ygrid)
    XY = np.stack([XX, YY], axis=-1)
    ZZ = V(XY)

    cs = ax.contour(
        XX,
        YY,
        ZZ,
        levels=50,
        linewidths=0.5,
    )
    ax.clabel(cs, inline=True, fontsize=6)
    ax.plot(Phi_path[:, 0], Phi_path[:, 1], "k-", lw=2, label="bounce path")
    ax.plot(tv[0], tv[1], "ro", label="TV")
    ax.plot(fv[0], fv[1], "bo", label="FV")
    ax.set_xlabel(r"$\phi_1$")
    ax.set_ylabel(r"$\phi_2$")
    ax.set_title("Field-space potential and tunneling path")
    ax.legend(loc="best", fontsize=8)

    # (b) Radial profiles phi1(r), phi2(r)
    ax2 = axes[1]

    # 1. Radial grid
    R = profile1D.R          # shape (N_rho,)

    # 2. Path parameter along Phi_path
    N_path = Phi_path.shape[0]
    s_path = np.linspace(0.0, 1.0, N_path)   # parameter along the field-space path

    # 3. Coordinate along the path from the 1D instanton
    lam = profile1D.Phi      # shape (N_rho,) – field along the path (up to scaling)

    # Rescale lam to [0,1] so it matches the s_path range
    lam_min, lam_max = lam.min(), lam.max()
    if lam_max == lam_min:
        # trivial path; avoid division by zero
        s_r = np.zeros_like(lam)
    else:
        s_r = (lam - lam_min) / (lam_max - lam_min)

    # 4. Interpolate the 2D fields along the path
    phi1_r = np.interp(s_r, s_path, Phi_path[:, 0])
    phi2_r = np.interp(s_r, s_path, Phi_path[:, 1])

    # 5. Plot
    ax2.plot(R, phi1_r, label=r"$\phi_1(\rho)$")
    ax2.plot(R, phi2_r, label=r"$\phi_2(\rho)$")
    ax2.set_xlabel(r"$\rho$")
    ax2.set_ylabel("field values")
    ax2.set_title("Bounce profiles")
    ax2.legend(loc="best")


    plt.tight_layout()
    plt.show()

    return {
        "params": params,
        "points": points,
        "true_vacuum": tv,
        "false_vacuum": fv,
        "profile1D": profile1D,
        "Phi_path": Phi_path,
        "action": action,
        "fRatio": fRatio,
    }


if __name__ == "__main__":
    run_cosmotransitions_bounce()
