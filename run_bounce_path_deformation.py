import numpy as np
import matplotlib.pyplot as plt
from potential2D import V_numeric
from path_deformation import path_deformation

def plot_tunneling_path(
    Phi_path,
    tv_point,
    fv_point,
    V,
    params,
    n_grid=200,
    filename=None,
):
    Phi_path = np.asarray(Phi_path, dtype=float)
    tv_point = np.asarray(tv_point, dtype=float)
    fv_point = np.asarray(fv_point, dtype=float)

    all_pts = [Phi_path, tv_point[None, :], fv_point[None, :]]

    all_pts = np.vstack(all_pts)

    # Define a field-space box that contains everything with a small margin
    margin = 0.5
    x_min = np.min(all_pts[:, 0]) - margin
    x_max = np.max(all_pts[:, 0]) + margin
    y_min = np.min(all_pts[:, 1]) - margin
    y_max = np.max(all_pts[:, 1]) + margin

    # Build grid and evaluate the potential
    x = np.linspace(x_min, x_max, n_grid)
    y = np.linspace(y_min, y_max, n_grid)
    XX, YY = np.meshgrid(x, y)
    ZZ = np.zeros_like(XX)

    # conservative, works even if V is not vectorized
    for i in range(n_grid):
        for j in range(n_grid):
            ZZ[i, j] = V((XX[i, j], YY[i, j]), params)

    # Make the plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    ax = axes[0]

    # Contours of the potential
    cs = ax.contour(XX, YY, ZZ, levels=50, linewidths=0.5)
    ax.clabel(cs, inline=True, fontsize=6)

    # Your path
    ax.plot(
        Phi_path[:, 0],
        Phi_path[:, 1],
        "k-",
        lw=2,
        label="my path (deformation)",
    )

    # Vacua
    ax.plot(tv_point[0], tv_point[1], "ro", label="TV")
    ax.plot(fv_point[0], fv_point[1], "bo", label="FV")

    ax.set_xlabel(r"$\phi_1$")
    ax.set_ylabel(r"$\phi_2$")
    ax.set_title("Field-space potential and tunneling path")
    ax.legend(loc="best", fontsize=8)
    
    ax2 = axes[1]
    phi1_r = Phi_path[:, 0]
    phi2_r = Phi_path[:, 1]
    
    R = np.linspace(0, 35, phi1_r.size)

    ax2.plot(R, phi1_r, label=r"$\phi_1(\rho)$")
    ax2.plot(R, phi2_r, label=r"$\phi_2(\rho)$")
    ax2.set_xlabel(r"$\rho$")
    ax2.set_ylabel(r"field values")
    ax2.set_title("Bounce profiles")
    ax2.legend(loc="best")

    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, dpi=150)

    plt.show()
    
params = (1.0, 1.2, 0.5, 0.6, 0.2, 0.3)
Phi_path,fv_point, phi_FV, tv_point, phi_TV = path_deformation(params)

plot_tunneling_path(
    Phi_path=Phi_path,
    tv_point=tv_point,
    fv_point=fv_point,
    V=V_numeric,
    params=params,
    filename="vacuum_decay_path.png",
)