import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

from cosmoTransitions import pathDeformation as pd


# ----------------------------------------------------------------------
# 1. Parameters and potential
# ----------------------------------------------------------------------

def build_parameters():
    params = (1.0, 1.2, 0.5, 0.6, 0.2, 0.3)
    return params


def build_guesses(params):
    """
    Construct a set of initial guesses for the vacuum finder based on
    the single-field vevs of the quartic pieces.
    """
    mu1, mu2, lam1, lam2, lam12, kappa = params

    v1 = mu1 / np.sqrt(lam1)
    v2 = mu2 / np.sqrt(lam2)

    guesses = [
        (0.0, 0.0),
        (v1, 0.0),
        (-v1, 0.0),
        (0.0, v2),
        (0.0, -v2),
        (v1, v2),
        (-v1, v2),
        (v1, -v2),
        (-v1, -v2),
    ]
    return guesses


class TwoFieldPotential:
    def __init__(self, params=None):
        if params is None:
            params = build_parameters()
        self.params = tuple(params)
        self.mu1, self.mu2, self.lam1, self.lam2, self.lam12, self.kappa = self.params

    # Potential value
    def V(self, X):
        X = np.asanyarray(X)
        phi1, phi2 = X[..., 0], X[..., 1]
        mu1, mu2, lam1, lam2, lam12, kappa = self.params
        V = (
            -0.5 * mu1**2 * phi1**2
            + 0.25 * lam1 * phi1**4
            - 0.5 * mu2**2 * phi2**2
            + 0.25 * lam2 * phi2**4
            + lam12 * phi1**2 * phi2**2
            + kappa * phi1**3
        )
        return V

    # Gradient (same shape as X)
    def dV(self, X):
        X = np.asanyarray(X)
        phi1, phi2 = X[..., 0], X[..., 1]
        mu1, mu2, lam1, lam2, lam12, kappa = self.params

        dphi1 = (
            -mu1**2 * phi1
            + lam1 * phi1**3
            + 2.0 * lam12 * phi1 * phi2**2
            + 3.0 * kappa * phi1**2
        )
        dphi2 = (
            -mu2**2 * phi2
            + lam2 * phi2**3
            + 2.0 * lam12 * phi1**2 * phi2
        )

        grad = np.empty_like(X)
        grad[..., 0] = dphi1
        grad[..., 1] = dphi2
        return grad


# ----------------------------------------------------------------------
# 2. Zero-temperature vacua finder (very simple)
# ----------------------------------------------------------------------

def numerical_hessian(pot, x, eps=1e-4):
    x = np.asarray(x, dtype=float)
    H = np.zeros((2, 2), dtype=float)
    f0 = float(pot.V(x))

    for i in range(2):
        dx_i = np.zeros_like(x)
        dx_i[i] = eps
        f_ip = float(pot.V(x + dx_i))
        f_im = float(pot.V(x - dx_i))
        H[i, i] = (f_ip - 2.0 * f0 + f_im) / eps**2

    # off-diagonal
    dx0 = np.array([eps, 0.0])
    dx1 = np.array([0.0, eps])
    f_pp = float(pot.V(x + dx0 + dx1))
    f_pm = float(pot.V(x + dx0 - dx1))
    f_mp = float(pot.V(x - dx0 + dx1))
    f_mm = float(pot.V(x - dx0 - dx1))
    H[0, 1] = H[1, 0] = (f_pp - f_pm - f_mp + f_mm) / (4.0 * eps**2)

    return H


def classify_point(pot, x, tol=1e-6):
    H = numerical_hessian(pot, x)
    eigs = np.linalg.eigvals(H)
    Vx = float(pot.V(x))

    if np.all(eigs > tol):
        tp = "minimum"
    elif np.all(eigs < -tol):
        tp = "maximum"
    else:
        tp = "saddle"
    return tp, eigs, Vx


def find_stationary_points(pot, guesses, tol=1e-6):
    pts = []

    for g in guesses:
        x0 = np.array(g, dtype=float)

        # Minimization in R^2
        res = optimize.minimize(
            fun=lambda x: float(pot.V(x)),
            x0=x0,
            jac=lambda x: np.asarray(pot.dV(x)),
            method="BFGS",
            tol=1e-10,
        )

        if not res.success:
            continue

        x = res.x

        # Deduplicate: skip if very close to an existing point
        if any(np.linalg.norm(x - p["x"]) < 1e-3 for p in pts):
            continue

        tp, eigs, Vx = classify_point(pot, x)
        pts.append({"x": x, "V": Vx, "type": tp, "eigs": eigs})

    return pts


def pick_true_and_false_vacua(points):
    mins = [p for p in points if p["type"] == "minimum"]
    if len(mins) < 2:
        raise ValueError("Need at least two distinct minima to define FV and TV.")

    mins_sorted = sorted(mins, key=lambda p: p["V"])
    tv = mins_sorted[0]
    fv = mins_sorted[1]
    return tv, fv


# ----------------------------------------------------------------------
# 3. CosmoTransitions tunneling: fullTunneling
# ----------------------------------------------------------------------

def run_cosmotransitions_bounce():
    params = build_parameters()
    pot = TwoFieldPotential(params=params)

    # 1) Find stationary points from your guesses
    guesses = build_guesses(params)
    points = find_stationary_points(pot, guesses)

    print("=== Stationary points found ===")
    for p in points:
        x1, x2 = p["x"]
        print(
            f"  type={p['type']:8s}  "
            f"phi1={x1: .4f}, phi2={x2: .4f},  V={p['V']: .6f},  eigs={p['eigs']}"
        )

    # 2) Identify true and false vacuum
    tv, fv = pick_true_and_false_vacua(points)
    print("\n=== Selected vacua ===")
    print(f"  True vacuum : phi = {tv['x']},  V = {tv['V']}")
    print(f"  False vacuum: phi = {fv['x']},  V = {fv['V']}")

    # 3) Initial path: straight line from TV (first) to FV (last)
    path_pts = np.vstack([tv["x"], fv["x"]])

    # 4) Run CosmoTransitions tunneling
    print("\nRunning CosmoTransitions pathDeformation.fullTunneling ...")
    Y = pd.fullTunneling(path_pts, pot.V, pot.dV, verbose=True)

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
    ZZ = pot.V(XY)

    cs = ax.contour(
        XX,
        YY,
        ZZ,
        levels=50,
        linewidths=0.5,
    )
    ax.clabel(cs, inline=True, fontsize=6)
    ax.plot(Phi_path[:, 0], Phi_path[:, 1], "k-", lw=2, label="bounce path")
    ax.plot(tv["x"][0], tv["x"][1], "ro", label="TV")
    ax.plot(fv["x"][0], fv["x"][1], "bo", label="FV")
    ax.set_xlabel(r"$\phi_1$")
    ax.set_ylabel(r"$\phi_2$")
    ax.set_title("Field-space potential and tunneling path")
    ax.legend(loc="best", fontsize=8)

    # (b) Radial profiles phi1(r), phi2(r)
    ax2 = axes[1]
    R = profile1D.R
    phi1_r = Phi_path[:, 0]
    phi2_r = Phi_path[:, 1]

    ax2.plot(R, phi1_r, label=r"$\phi_1(\rho)$")
    ax2.plot(R, phi2_r, label=r"$\phi_2(\rho)$")
    ax2.set_xlabel(r"$\rho$")
    ax2.set_ylabel(r"field values")
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
