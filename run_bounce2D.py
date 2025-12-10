import numpy as np
import matplotlib.pyplot as plt

from classical_bounce2D import shoot_bounce_2d
from potential2D import find_critical_points, V_numeric, find_vacua_from_potential


def main():
    # 1. Set up model parameters
    params = (1.0, 1.2, 0.5, 0.6, 0.2, 0.3)

    # 2. Locate vacua for diagnostics
    fv_point, fv_V, tv_point, tv_V = find_vacua_from_potential(params)

    print("False vacuum (highest minimum):", fv_point, "V =", fv_V)
    print("True  vacuum (lowest  minimum):", tv_point, "V =", tv_V)

    # 3. Compute the 2D bounce using the classical shooting solver
    rho_max = 20.0
    n_steps = 4000
    tv_arr = np.array(tv_point, dtype=float)
    fv_arr = np.array(fv_point, dtype=float)

    ct_guided_a0 = tv_arr + 0.3 * (fv_arr - tv_arr)
    print("Heuristic CT-guided initial guess a0:", ct_guided_a0)

    bounce = shoot_bounce_2d(
        params=params,
        rho_max=rho_max,
        n_steps=n_steps,
        a0=ct_guided_a0,
    )

    print("Bounce shooting parameter a = (phi1(0), phi2(0)):", bounce.a_shoot)
    print(f"Bounce action (kinetic only) S_E ≈ {bounce.action:.6g}")

    # 4. Prepare data for plotting
    rho = bounce.rho
    phi1 = bounce.phi[:, 0]
    phi2 = bounce.phi[:, 1]

    # ------------------------------------------------------------------
    # 4a. Field-space contour with bounce path and vacua (like CosmoTransitions)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    ax0 = axes[0]

    # Choose a box that matches the CosmoTransitions script:
    # use all stationary points (critical points) to set the range.
    crit_points = find_critical_points(params, guesses)
    all_x = np.array(crit_points, dtype=float)
    x_min = np.min(all_x[:, 0]) - 0.5
    x_max = np.max(all_x[:, 0]) + 0.5
    y_min = np.min(all_x[:, 1]) - 0.5
    y_max = np.max(all_x[:, 1]) + 0.5

    nx = ny = 200
    X = np.linspace(x_min, x_max, nx)
    Ygrid = np.linspace(y_min, y_max, ny)
    XX, YY = np.meshgrid(X, Ygrid)

    # Evaluate potential on the grid
    ZZ = np.empty_like(XX)
    for i in range(nx):
        for j in range(ny):
            ZZ[j, i] = V_numeric((XX[j, i], YY[j, i]), params)

    cs = ax0.contour(
        XX,
        YY,
        ZZ,
        levels=50,
        linewidths=0.5,
    )
    ax0.clabel(cs, inline=True, fontsize=6)
    ax0.plot(phi1, phi2, "k-", lw=2, label="bounce path")
    ax0.plot(tv_point[0], tv_point[1], "ro", label="TV")
    ax0.plot(fv_point[0], fv_point[1], "bo", label="FV")
    ax0.set_xlabel(r"$\phi_1$")
    ax0.set_ylabel(r"$\phi_2$")
    ax0.set_title("Field-space potential and tunneling path")
    ax0.legend(loc="best", fontsize=8)

    # ------------------------------------------------------------------
    # 4b. Radial profiles phi1(rho), phi2(rho)
    # ------------------------------------------------------------------
    ax1 = axes[1]
    ax1.plot(rho, phi1, label=r"$\phi_1(\rho)$")
    ax1.plot(rho, phi2, label=r"$\phi_2(\rho)$")
    ax1.set_xlabel(r"$\rho$")
    ax1.set_ylabel(r"field values")
    ax1.set_title("Bounce profiles")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")

    # Annotate the action on the radial-profile panel
    textstr = rf"$S_E \approx {bounce.action:.3g}$"
    ax1.text(
        0.97,
        0.95,
        textstr,
        transform=ax1.transAxes,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig("bounce_profile.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()


