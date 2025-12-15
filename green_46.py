#!/usr/bin/env python3
"""
Two–field O(4) bounce + h–basis RK Green’s function
+ finite–difference (FD) Green with Robin BCs.

This version:
- NO field redefinition (work in original x,y coordinates);
- Per false vacuum, build a *lifted* potential with V(false)=0;
- In the h–equation: Q(r) = H_full(r) − H_free, where
    H_free = diag[ Hessian_V(0,0) ] (diagonal part only);
- The physical fluctuation operator is
    L = -d2/dr2 - (3/r) d/dr + n(n+2)/r^2 + H_full(r) + nu2 * I_2;
- RK h–basis → f^{±,α} → Wronskian Ω → RK Green G_RK;
- FD Green G_fd from matrix inversion with old Robin BCs
  (inner f'/f ≈ n/r, outer f'/f ≈ -kappa_eff - 1/r);
- Spectral/Galerkin Green G_spec from eigen-decomposition of L_spec
  and explicit spectral sum;
- Plots:
    * h–basis functions
    * |B_i^{±}(r)| and |f_i^{±,α}(r)|
    * Relative variation of r^3 W_ab(r) vs Ω_ab
    * 3D surfaces of G_RK, G_fd, G_mod, G_spec, plus FD−RK heatmaps.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.special import kv, kvp, iv, ivp
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from cosmoTransitions import pathDeformation as pd
from cosmoTransitions.tunneling1D import PotentialError


# ================================================================
# 1. Base potential and derivatives (ORIGINAL coordinates)
# ================================================================
class MyPotential:
    def V(self, X):
        X = np.asarray(X)
        x = X[..., 0]
        y = X[..., 1]
        # The constant here is irrelevant physically; we will
        # lift V(false) to 0 later via a wrapper.
        return (-0.8
                + x * y**2
                + (y**2 - 1.0)**2
                + 0.15 * x**2 * y**2
                + (x**2 - 1.0)**2)

    def dV(self, X):
        X = np.asarray(X)
        x = X[..., 0]
        y = X[..., 1]
        dVdx = y**2 + 0.30 * x * y**2 + 4.0 * x * (x**2 - 1.0)
        dVdy = 2.0 * x * y + 4.0 * y * (y**2 - 1.0) + 0.30 * x**2 * y
        g = np.empty_like(X)
        g[..., 0] = dVdx
        g[..., 1] = dVdy
        return g


class LiftedPotential:
    """
    V_lift(X) = V_base(X) - V_false, so that V_lift(false_vac) = 0.
    All derivatives are unchanged.
    """
    def __init__(self, base_potential, V_false):
        self.base = base_potential
        self.V_false = float(V_false)

    def V(self, X):
        return self.base.V(X) - self.V_false

    def dV(self, X):
        return self.base.dV(X)


# ------------------------------------------------
# Hessian of the base potential (used everywhere)
# ------------------------------------------------
def V_xx(x, y):
    return 0.30 * y**2 + 12.0 * x**2 - 4.0


def V_yy(x, y):
    return 2.0 * x + 12.0 * y**2 - 4.0 + 0.30 * x**2


def V_xy(x, y):
    return 2.0 * y + 0.60 * x * y


def Hessian_V(x, y):
    return np.array([[V_xx(x, y), V_xy(x, y)],
                     [V_xy(x, y), V_yy(x, y)]])


# ================================================================
# 2. Find vacua on a coarse grid (using the BASE potential)
# ================================================================
def find_vacua_grid(pot, x_min=-2.0, x_max=2.0,
                    y_min=-2.0, y_max=2.0,
                    Nx=81, Ny=81, merge_tol=0.15):
    xs = np.linspace(x_min, x_max, Nx)
    ys = np.linspace(y_min, y_max, Ny)
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    XY = np.stack([XX, YY], axis=-1)
    Vgrid = pot.V(XY)

    candidates = []
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            v = Vgrid[i, j]
            nbrs = Vgrid[i - 1:i + 2, j - 1:j + 2]
            if np.all(v <= nbrs) and np.any(v < nbrs):
                candidates.append((xs[i], ys[j], v))

    vacua = []
    for x0, y0, v0 in candidates:
        merged = False
        for vac in vacua:
            if np.hypot(vac['x'] - x0, vac['y'] - y0) < merge_tol:
                if v0 < vac['V']:
                    vac['x'], vac['y'], vac['V'] = x0, y0, v0
                merged = True
                break
        if not merged:
            vacua.append({'x': x0, 'y': y0, 'V': v0})

    vacua.sort(key=lambda v: v['V'])

    print("\nFound vacua (approx.):")
    for i, v in enumerate(vacua):
        print(f" {i}: x={v['x']:.6f}, y={v['y']:.6f}, V={v['V']:.6f}")
    return vacua


# ================================================================
# 3. Bounce for one false→true pair (using *lifted* potential)
# ================================================================
def compute_bounce_for_pair(pot, false_vac, true_vac, tag=""):
    """
    pot: the *lifted* potential, i.e. V(false)=0
    false_vac, true_vac: positions in ORIGINAL coordinates.
    """
    true_vac = np.array(true_vac, dtype=float)
    false_vac = np.array(false_vac, dtype=float)

    print("\n========================================================")
    print("Computing bounce for pair", tag)
    print(" false =", false_vac)
    print(" true  =", true_vac)
    print("========================================================")

    V_false = float(pot.V(false_vac))
    V_true  = float(pot.V(true_vac))
    print("V(false) =", V_false, "V(true) =", V_true, " (lifted potential)")
    if V_false <= V_true:
        raise RuntimeError("Not metastable in lifted potential: V(false) <= V_true).")

    path_guess = np.vstack([true_vac, false_vac])

    Y = pd.fullTunneling(
        path_guess,
        pot.V,
        pot.dV,
        maxiter=60,
        verbose=True,
        tunneling_init_params={'alpha': 3}  # O(4) friction
    )

    print("CosmoTransitions action =", Y.action)
    print("Final fRatio =", Y.fRatio)

    R = Y.profile1D.R
    Phi = Y.Phi
    X_b = Phi[:, 0]
    Y_b = Phi[:, 1]
    return true_vac, false_vac, R, X_b, Y_b, Y.action


# ================================================================
# 4. Fluctuation data along the bounce (NO field shift)
# ================================================================
def build_fluctuation_data(false_vac, R_bounce, X_bounce, Y_bounce,
                           nu2, n_mode):
    x_false, y_false = false_vac
    H_false = Hessian_V(x_false, y_false)
    H0      = Hessian_V(0.0, 0.0)           # just for diagnostics

    # --- CHOICE OF FREE MASSES (CONSISTENT) ---
    # use the diagonal of the Hessian at the FALSE vacuum
    m1_sq_free = H_false[0, 0]
    m2_sq_free = H_false[1, 1]

    # constant diagonal free mass matrix
    M_free = np.diag([m1_sq_free, m2_sq_free])

    print("\n[build_fluctuation_data] original coords")
    print("  false_vac =", false_vac)
    print("  H_false (at false vac) =\n", H_false)
    print("  H0 (at x=0,y=0)        =\n", H0)
    print("  m1_free^2 =", m1_sq_free, "m2_free^2 =", m2_sq_free)
    print("  nu2       =", nu2,        "n_mode    =", n_mode)

    # order of modified Bessel in 4D: ℓ = n + 1
    ell_bessel = n_mode + 1
    r_eps = 1e-8

    # κ_i^2 = ν^2 + m_i^2  (m_i^2 = diagonal entries above)
    def kappa(i):
        m_sq = m1_sq_free if i == 0 else m2_sq_free
        return np.sqrt(nu2 + m_sq)

    # ----- core Bessel (no 1/r) -----
    def Bcore_plus(i, r):
        r_eff = max(r, r_eps)
        z = kappa(i) * r_eff
        return iv(ell_bessel, z)

    def dBcore_plus(i, r):
        r_eff = max(r, r_eps)
        z = kappa(i) * r_eff
        return kappa(i) * ivp(ell_bessel, z)

    def Bcore_minus(i, r):
        r_eff = max(r, r_eps)
        z = kappa(i) * r_eff
        return kv(ell_bessel, z)

    def dBcore_minus(i, r):
        r_eff = max(r, r_eps)
        z = kappa(i) * r_eff
        return kappa(i) * kvp(ell_bessel, z)

    # ----- full free solutions B = Bcore / r (used in f, K) -----
    def B(i, r, sign):
        r_eff = max(r, r_eps)
        if sign == "+":
            return Bcore_plus(i, r) / r_eff
        else:
            return Bcore_minus(i, r) / r_eff

    def dB(i, r, sign):
        # full derivative of B = Bcore/r, used only to reconstruct f, not in A_i
        r_eff = max(r, r_eps)
        if sign == "+":
            Bc  = Bcore_plus(i, r)
            dBc = dBcore_plus(i, r)
        else:
            Bc  = Bcore_minus(i, r)
            dBc = dBcore_minus(i, r)
        return (dBc * r_eff - Bc) / (r_eff**2)

    # ---------- bounce interpolation in r ----------
    def x_bounce(r):
        return np.interp(r, R_bounce, X_bounce)

    def y_bounce(r):
        return np.interp(r, R_bounce, Y_bounce)

    # ---------- full Hessian along the bounce (NO nu2 here) ----------
    def H_full(r):
        xb = x_bounce(r)
        yb = y_bounce(r)
        return Hessian_V(xb, yb)

    # ---------- Q(r) = H_full(r) - diag(m_i^2) ----------
    def Q_matrix(r):
        return H_full(r) - M_free   # off-diagonals of H_full survive in Q

    # ---------- K_matrix and A_i entering the h-equation ----------
    # K_ij(r) = Q_ij(r) * B_j(r, sign) / B_i(r, sign)
    def K_matrix(r, sign):
        Q = Q_matrix(r)
        K = np.zeros((2, 2))
        for i in range(2):
            Bi = B(i, r, sign)
            for j in range(2):
                Bj = B(j, r, sign)
                K[i, j] = Q[i, j] * Bj / (Bi + 1e-30)
        return K

    def A_i(i, r, sign):
        # use ONLY derivative of the core Bessel (no 1/r)
        if sign == "+":
            Bc  = Bcore_plus(i, r)
            dBc = dBcore_plus(i, r)
        else:
            Bc  = Bcore_minus(i, r)
            dBc = dBcore_minus(i, r)
        return 2.0 * dBc / (Bc + 1e-30)

    # return everything; note we return M_free, not H_free
    return B, dB, K_matrix, A_i, Q_matrix, M_free


# ================================================================
# 5. h-basis RK integrator
# ================================================================
def rhs_h(r, y, sign, src_index, K_matrix, A_i):
    """
    y = (h1, h2, v1, v2); v_i = h_i'
    """
    h1, h2, v1, v2 = y
    K = K_matrix(r, sign)
    S = K[:, src_index]   # source column
    invr = 0.0 if r == 0.0 else 1.0 / r

    dv1 = (-(invr + A_i(0, r, sign)) * v1
           + K[0, 0] * h1 + K[0, 1] * h2 + S[0])
    dv2 = (-(invr + A_i(1, r, sign)) * v2
           + K[1, 0] * h1 + K[1, 1] * h2 + S[1])

    return np.array([v1, v2, dv1, dv2])


def rk4_h(r0, r1, y0, N, sign, src_index, K_matrix, A_i,
          rescale_threshold=1e6):
    r_grid = np.linspace(r0, r1, N + 1)
    dr = (r1 - r0) / N
    Y = np.zeros((N + 1, len(y0)))
    Y[0] = y0

    for k in range(N):
        r = r_grid[k]
        y = Y[k]
        k1 = rhs_h(r,           y,             sign, src_index, K_matrix, A_i)
        k2 = rhs_h(r + dr/2.0,  y + dr*k1/2.0, sign, src_index, K_matrix, A_i)
        k3 = rhs_h(r + dr/2.0,  y + dr*k2/2.0, sign, src_index, K_matrix, A_i)
        k4 = rhs_h(r + dr,      y + dr*k3,     sign, src_index, K_matrix, A_i)
        Y[k+1] = y + (dr/6.0)*(k1 + 2*k2 + 2*k3 + k4)

        if np.max(np.abs(Y[k+1])) > rescale_threshold:
            Y[k+1] /= rescale_threshold

    return r_grid, Y


# ================================================================
# 6. Helper: 3D surface plot
# ================================================================
def plot_surface_3d(Rm, Rp, Z, title, zlabel):
    zmax = np.max(np.abs(Z))
    if zmax == 0.0:
        zmax = 1.0
    Z_vis = np.clip(Z, -zmax, zmax)

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        Rm, Rp, Z_vis,
        rstride=max(1, Rm.shape[0] // 50),
        cstride=max(1, Rp.shape[1] // 50),
        cmap=cm.coolwarm,
        linewidth=0,
        antialiased=True,
    )

    ax.set_xlabel("r")
    ax.set_ylabel("r'")
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    ax.view_init(elev=30, azim=135)

    cb = fig.colorbar(surf, shrink=0.6, pad=0.1)
    cb.set_label(zlabel)

    plt.tight_layout()
    plt.show()


# ================================================================
# 7. Main Green builder (RK + FD + G_mod + G_spec)
# ================================================================
def build_and_plot_green(B, dB, K_matrix, A_i,
                         R_bounce, X_bounce, Y_bounce,
                         false_vac, nu2, n_mode, tag,
                         Q_matrix, H_free):

    # ------------------------------------------------------------
    # (A) Solve h-basis with RK
    # ------------------------------------------------------------
    r0 = max(R_bounce[0], 1e-4)
    Rmax = R_bounce[-1]
    Nsteps = 2000
    y0 = np.array([0.0, 0.0, 0.0, 0.0])

    r_plus_1, Y_plus_1 = rk4_h(r0, Rmax, y0, Nsteps, "+", 0, K_matrix, A_i)
    r_plus_2, Y_plus_2 = rk4_h(r0, Rmax, y0, Nsteps, "+", 1, K_matrix, A_i)
    r_minus_1, Y_minus_1 = rk4_h(Rmax, r0, y0, Nsteps, "-", 0, K_matrix, A_i)
    r_minus_2, Y_minus_2 = rk4_h(Rmax, r0, y0, Nsteps, "-", 1, K_matrix, A_i)

    r_grid = r_plus_1
    Nr = len(r_grid)
    G_mod_best = None  # will later hold "best-fit" constant-diag G_mod

    # + branch
    h_plus = np.zeros((Nr, 2, 2))
    dh_plus = np.zeros((Nr, 2, 2))
    h_plus[:, 0, 0], h_plus[:, 1, 0] = Y_plus_1[:, 0], Y_plus_1[:, 1]
    dh_plus[:, 0, 0], dh_plus[:, 1, 0] = Y_plus_1[:, 2], Y_plus_1[:, 3]
    h_plus[:, 0, 1], h_plus[:, 1, 1] = Y_plus_2[:, 0], Y_plus_2[:, 1]
    dh_plus[:, 0, 1], dh_plus[:, 1, 1] = Y_plus_2[:, 2], Y_plus_2[:, 3]

    # - branch (reverse to increasing r)
    r_minus_inc = r_minus_1[::-1]
    assert np.allclose(r_minus_inc, r_grid, rtol=1e-6, atol=1e-8)
    Y_minus_1_rev = Y_minus_1[::-1, :]
    Y_minus_2_rev = Y_minus_2[::-1, :]

    h_minus = np.zeros((Nr, 2, 2))
    dh_minus = np.zeros((Nr, 2, 2))
    h_minus[:, 0, 0], h_minus[:, 1, 0] = Y_minus_1_rev[:, 0], Y_minus_1_rev[:, 1]
    dh_minus[:, 0, 0], dh_minus[:, 1, 0] = Y_minus_1_rev[:, 2], Y_minus_1_rev[:, 3]
    h_minus[:, 0, 1], h_minus[:, 1, 1] = Y_minus_2_rev[:, 0], Y_minus_2_rev[:, 1]
    dh_minus[:, 0, 1], dh_minus[:, 1, 1] = Y_minus_2_rev[:, 2], Y_minus_2_rev[:, 3]

    # ------------------------------------------------------------
    # (A0) Diagnostic: plot h-basis modes h_i^±,α(r)
    # ------------------------------------------------------------
    def plot_h_basis(alpha=0):
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
            fr"$h$-basis functions, pair {tag}, n={n_mode}, \nu^2={nu2}, "
            fr"\alpha={alpha}"
        )
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    plot_h_basis(alpha=0)
    plot_h_basis(alpha=1)

    # ------------------------------------------------------------
    # (B) Reconstruct f^± and f'^± from h
    # ------------------------------------------------------------
    def build_f_df(sign):
        f = np.zeros((Nr, 2, 2))
        df = np.zeros((Nr, 2, 2))
        for k, r in enumerate(r_grid):
            for i in range(2):
                for alpha in range(2):
                    delta = 1.0 if i == alpha else 0.0
                    if sign == "+":
                        h = h_plus[k, i, alpha]
                        dh = dh_plus[k, i, alpha]
                        Bi, dBi = B(i, r, "+"), dB(i, r, "+")
                    else:
                        h = h_minus[k, i, alpha]
                        dh = dh_minus[k, i, alpha]
                        Bi, dBi = B(i, r, "-"), dB(i, r, "-")
                    f[k, i, alpha] = Bi * (delta + h)
                    df[k, i, alpha] = dBi * (delta + h) + Bi * dh
        return f, df

    f_plus, df_plus = build_f_df("+")
    f_minus, df_minus = build_f_df("-")

    # --- Boundary-condition diagnostic ---
    r_small = r_grid[0]
    r_large = r_grid[-1]

    for i in range(2):
        for alpha in range(2):
            # inner: regular ~ r^n  -> f'/f ~ n / r
            f_in   = f_plus[0, i, alpha]
            df_in  = df_plus[0, i, alpha]
            if abs(f_in) > 0:
                inner_ratio = r_small * df_in / f_in
                print(f"[BC check] inner: i={i}, alpha={alpha}, "
                      f"r={r_small:.3e}, r*f'/f ≈ {inner_ratio:.3f}, "
                      f"expected ~ n={n_mode}")

            # outer: decaying ~ exp(-kappa r)/r -> f'/f ~ -kappa_eff - 1/r
            f_out  = f_minus[-1, i, alpha]
            df_out = df_minus[-1, i, alpha]
            if abs(f_out) > 0:
                outer_ratio = df_out / f_out + 1.0 / r_large
                print(f"[BC check] outer: i={i}, alpha={alpha}, "
                      f"r={r_large:.3e}, f'/f + 1/r ≈ {outer_ratio:.3f}")

    # ------------------------------------------------------------
    # (C) Diagnostic plots: free Bessel basis + full modes
    # ------------------------------------------------------------
    def plot_free_bessel_basis():
        Bp0 = np.array([B(0, r, "+") for r in r_grid])
        Bp1 = np.array([B(1, r, "+") for r in r_grid])
        Bm0 = np.array([B(0, r, "-") for r in r_grid])
        Bm1 = np.array([B(1, r, "-") for r in r_grid])

        plt.figure(figsize=(7, 5))
        plt.plot(r_grid, np.abs(Bp0), label=r"$|B_1^{+}|$")
        plt.plot(r_grid, np.abs(Bp1), label=r"$|B_2^{+}|$")
        plt.plot(r_grid, np.abs(Bm0), "--", label=r"$|B_1^{-}|$")
        plt.plot(r_grid, np.abs(Bm1), "--", label=r"$|B_2^{-}|$")
        plt.yscale("log")
        plt.xlabel(r"$r$")
        plt.ylabel(r"$|B_i^{\pm}(r)|$")
        plt.title(f"Free Bessel basis, pair {tag}, n={n_mode}, ν²={nu2}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_full_modes(alpha=0):
        f_p0 = np.abs(f_plus[:, 0, alpha])
        f_p1 = np.abs(f_plus[:, 1, alpha])
        f_m0 = np.abs(f_minus[:, 0, alpha])
        f_m1 = np.abs(f_minus[:, 1, alpha])

        plt.figure(figsize=(7, 5))
        plt.plot(r_grid, f_p0, label=rf"$|f_1^{{+,\alpha={alpha}}}|$")
        plt.plot(r_grid, f_p1, label=rf"$|f_2^{{+,\alpha={alpha}}}|$")
        plt.plot(r_grid, f_m0, "--", label=rf"$|f_1^{{-,\alpha={alpha}}}|$")
        plt.plot(r_grid, f_m1, "--", label=rf"$|f_2^{{-,\alpha={alpha}}}|$")
        plt.yscale("log")
        plt.xlabel(r"$r$")
        plt.ylabel(r"$|f_i^{\pm,\alpha}(r)|$")
        plt.title(f"Full modes for α={alpha}, pair {tag}, n={n_mode}, ν²={nu2}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    plot_free_bessel_basis()
    plot_full_modes(alpha=0)
    plot_full_modes(alpha=1)

    # ------------------------------------------------------------
    # (D) Raw Wronskian and Omega_inv
    # ------------------------------------------------------------
    r_eps_const = 1e-8
    W_raw = np.zeros((Nr, 2, 2))
    for idx, r in enumerate(r_grid):
        W = np.zeros((2, 2))
        for alpha in range(2):
            for beta in range(2):
                s = 0.0
                for i in range(2):
                    fp_a  = f_plus[idx, i, alpha]
                    fm_b  = f_minus[idx, i, beta]
                    dfp_a = df_plus[idx, i, alpha]
                    dfm_b = df_minus[idx, i, beta]
                    s += fp_a * dfm_b - fm_b * dfp_a
                W[alpha, beta] = s
        W_raw[idx] = W

    W_scaled = np.zeros_like(W_raw)
    for idx, r in enumerate(r_grid):
        r_eff = max(r, r_eps_const)
        W_scaled[idx] = (r_eff**3) * W_raw[idx]

    r_min_tail = 1.0
    r_max_tail = 0.9 * r_grid[-1]
    i_min = np.searchsorted(r_grid, r_min_tail)
    i_max = np.searchsorted(r_grid, r_max_tail)

    W_tail = W_scaled[i_min:i_max+1, :, :]
    Omega = np.mean(W_tail, axis=0)

    dev_tail = np.max(
        np.abs(W_tail - Omega) /
        np.maximum(np.abs(Omega), 1e-30)
    )

    print("\nScaled Wronskian r^3 W_raw in tail:")
    print("  r_min_tail =", r_grid[i_min], "r_max_tail =", r_grid[i_max])
    print("  Omega (average over tail) =\n", Omega)
    print("  Max relative variation in tail =", dev_tail)

    # symmetrise before inverting for the Green
    Omega_sym = 0.5 * (Omega + Omega.T)
    Omega_inv = np.linalg.inv(Omega_sym)

    # --------------------------------------------------------
    # plot r^3 W_ab(r) vs r and show the extracted Omega_ab
    # --------------------------------------------------------
    plt.figure(figsize=(7, 5))

    # Diagonal components
    plt.plot(r_grid, W_scaled[:, 0, 0],
             label=r"$r^3 W_{11}(r)$")
    plt.plot(r_grid, W_scaled[:, 1, 1],
             label=r"$r^3 W_{22}(r)$")

    # Horizontal lines at Omega_11 and Omega_22
    plt.axhline(Omega[0, 0], linestyle="--",
                label=r"$\Omega_{11}$")
    plt.axhline(Omega[1, 1], linestyle="--",
                label=r"$\Omega_{22}$")

    plt.xlabel(r"$r$")
    plt.ylabel(r"$r^3 W_{aa}(r)$")
    plt.title(r"Scaled Wronskian $r^3 W_{aa}(r)$ and $\Omega_{aa}$")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Off-diagonal components
    plt.figure(figsize=(7, 5))
    plt.plot(r_grid, W_scaled[:, 0, 1],
             label=r"$r^3 W_{12}(r)$")
    plt.plot(r_grid, W_scaled[:, 1, 0],
             label=r"$r^3 W_{21}(r)$")

    plt.axhline(Omega[0, 1], linestyle="--",
                label=r"$\Omega_{12}$")
    plt.axhline(Omega[1, 0], linestyle="--",
                label=r"$\Omega_{21}$")

    plt.xlabel(r"$r$")
    plt.ylabel(r"$r^3 W_{ab}(r)$")
    plt.title(r"Scaled Wronskian $r^3 W_{ab}(r)$ and $\Omega_{ab}$ (off-diagonal)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Diagnostic: relative deviation δW_ab(r)
    delta = np.zeros_like(W_scaled)
    for a in range(2):
        for b in range(2):
            if abs(Omega[a, b]) > 0:
                delta[:, a, b] = (W_scaled[:, a, b] - Omega[a, b]) / Omega[a, b]

    plt.figure(figsize=(7, 5))
    plt.plot(r_grid, delta[:, 0, 0], label=r"$\delta W_{11}(r)$")
    plt.plot(r_grid, delta[:, 0, 1], label=r"$\delta W_{12}(r)$")
    plt.plot(r_grid, delta[:, 1, 0], label=r"$\delta W_{21}(r)$")
    plt.plot(r_grid, delta[:, 1, 1], label=r"$\delta W_{22}(r)$")
    plt.axvspan(r_grid[i_min], r_grid[i_max],
                color="grey", alpha=0.15, label="tail window")
    plt.ylim(-0.5, 0.5)
    plt.xlabel(r"$r$")
    plt.ylabel(r"$\delta W_{ab}(r)$")
    plt.title(rf"Rescaled Wronskian, pair {tag}, $n={n_mode}$, $\nu^2={nu2}$")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------
    # (D') CONSTANT DIAGONAL MATRIX FROM r^3 W REGION AND G_mod
    # ------------------------------------------------------------
    def compute_diag_matrix_from_W_region(r_grid_loc, W_scaled_loc,
                                          r_min_c=2.0, r_max_c=7.0):
        """
        Take r^3 W(r) on [r_min_c, r_max_c], average to get Omega_reg,
        symmetrise, diagonalise, and build a constant diagonal 2×2 matrix
        whose entries are the eigenvalues of Omega_reg.
        """
        mask_c = (r_grid_loc >= r_min_c) & (r_grid_loc <= r_max_c)
        idx_c  = np.where(mask_c)[0]

        if len(idx_c) == 0:
            print("[G_mod] WARNING: no points in [r_min_c,r_max_c]; "
                  "falling back to tail window used for Omega.")
            W_reg = W_tail
        else:
            W_reg = W_scaled_loc[idx_c, :, :]

        Omega_reg     = np.mean(W_reg, axis=0)
        Omega_reg_sym = 0.5 * (Omega_reg + Omega_reg.T)

        evals, _ = np.linalg.eigh(Omega_reg_sym)
        C_diag      = np.diag(evals.real)
        C_diag_inv  = np.diag(1.0 / (evals.real + 1e-30))

        print("\n[G_mod] constant diagonal matrix from r^3 W on region "
              f"[{r_min_c},{r_max_c}]:")
        print("  Omega_reg_sym =\n", Omega_reg_sym)
        print("  eigenvalues   =", evals)
        print("  C_diag        =\n", C_diag)

        return C_diag, C_diag_inv

    # choose the region where you want to match r^3 W
    C_mod, C_mod_inv = compute_diag_matrix_from_W_region(
        r_grid, W_scaled,
        r_min_c=2.0, r_max_c=7.0
    )

    # Build G_mod with this fixed constant diagonal C_mod_inv
    G_mod = np.zeros((Nr, Nr, 2, 2))
    for k in range(Nr):
        for l in range(Nr):
            if r_grid[k] >= r_grid[l]:
                F_big   = f_minus[k]
                F_small = f_plus[l]
            else:
                F_big   = f_minus[l]
                F_small = f_plus[k]
            G_mod[k, l] = F_big @ C_mod_inv @ F_small.T

    # ------------------------------------------------------------
    # (E) RK Green's function (bounce) + basis for G_mod
    # ------------------------------------------------------------
    G_full = np.zeros((Nr, Nr, 2, 2))

    # basis for G_mod with a constant diagonal matrix diag(c0, c1):
    # G_mod(c0, c1) = c0 * G_mod_basis[0] + c1 * G_mod_basis[1]
    G_mod_basis = np.zeros((2, Nr, Nr, 2, 2))

    M00 = np.array([[1.0, 0.0],
                    [0.0, 0.0]])
    M11 = np.array([[0.0, 0.0],
                    [0.0, 1.0]])

    for k in range(Nr):
        Fp_r = f_plus[k]
        Fm_r = f_minus[k]
        for l in range(Nr):
            Fp_rp = f_plus[l]
            Fm_rp = f_minus[l]

            if r_grid[k] >= r_grid[l]:
                F_big   = Fm_r
                F_small = Fp_rp
            else:
                F_big   = Fm_rp
                F_small = Fp_r

            # RK Green with full Ω^{-1}
            G_full[k, l] = F_big @ Omega_inv @ F_small.T

            # G_mod basis for diag(1,0) and diag(0,1)
            G_mod_basis[0, k, l] = F_big @ M00 @ F_small.T   # field 1
            G_mod_basis[1, k, l] = F_big @ M11 @ F_small.T   # field 2

    # free Green (h ≡ 0)
    Fp_free = np.zeros_like(f_plus)
    Fm_free = np.zeros_like(f_minus)
    for k, r in enumerate(r_grid):
        for i in range(2):
            Bi_p = B(i, r, "+")
            Bi_m = B(i, r, "-")
            for alpha in range(2):
                delta = 1.0 if i == alpha else 0.0
                Fp_free[k, i, alpha] = Bi_p * delta
                Fm_free[k, i, alpha] = Bi_m * delta

    G_free = np.zeros_like(G_full)
    for k in range(Nr):
        Fp_r = Fp_free[k]
        Fm_r = Fm_free[k]
        for l in range(Nr):
            Fp_rp = Fp_free[l]
            Fm_rp = Fm_free[l]
            if r_grid[k] >= r_grid[l]:
                F_big = Fm_r
                F_small = Fp_rp
            else:
                F_big = Fm_rp
                F_small = Fp_r
            G_free[k, l] = F_big @ Omega_inv @ F_small.T

    # ------------------------------------------------------------
    # (E') Spectral / Galerkin operator L_spec and its Green G_spec
    #     via explicit spectral sum
    # ------------------------------------------------------------
    L_spec, G_spec = build_spectral_operator_and_green(
        G_full, r_grid, tag, n_mode, nu2
    )

    # --- Plot ALL components of spectral Green vs RK Green ---
    def plot_spec_vs_rk_component(i_idx, j_idx, name):
        G_rk_ij   = G_full[:, :, i_idx, j_idx]
        G_spec_ij = G_spec[:, :, i_idx, j_idx]

        all_vals = np.concatenate([G_rk_ij.ravel(), G_spec_ij.ravel()])
        vmax_vis = np.percentile(np.abs(all_vals), 99.0)
        if vmax_vis == 0.0:
            vmax_vis = np.max(np.abs(all_vals)) or 1.0

        G_rk_vis   = np.clip(G_rk_ij,   -vmax_vis, vmax_vis)
        G_spec_vis = np.clip(G_spec_ij, -vmax_vis, vmax_vis)

        Rm, Rp = np.meshgrid(r_grid, r_grid, indexing="ij")

        # RK component
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(
            Rm, Rp, G_rk_vis,
            rstride=max(1, Rm.shape[0] // 50),
            cstride=max(1, Rp.shape[1] // 50),
            cmap=cm.coolwarm,
            linewidth=0,
            antialiased=True,
        )
        ax.set_xlabel("r")
        ax.set_ylabel("r'")
        ax.set_zlabel(rf"$G_{{{name}}}^\mathrm{{RK}}$")
        ax.set_title(
            rf"$G_{{{name}}}(r,r')$ (RK, {tag}, "
            rf"$n={n_mode}$, $\nu^2={nu2}$)"
        )
        ax.set_zlim(-vmax_vis, vmax_vis)
        cb = fig.colorbar(surf, shrink=0.6, pad=0.1)
        cb.set_label(rf"$G_{{{name}}}^\mathrm{{RK}}$ (clipped)")
        plt.tight_layout()
        plt.show()

        # Spectral component
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(
            Rm, Rp, G_spec_vis,
            rstride=max(1, Rm.shape[0] // 50),
            cstride=max(1, Rp.shape[1] // 50),
            cmap=cm.coolwarm,
            linewidth=0,
            antialiased=True,
        )
        ax.set_xlabel("r")
        ax.set_ylabel("r'")
        ax.set_zlabel(rf"$G_{{{name}}}^\mathrm{{spec}}$")
        ax.set_title(
            rf"$G_{{{name}}}(r,r')$ (spectral sum, {tag}, "
            rf"$n={n_mode}$, $\nu^2={nu2}$)"
        )
        ax.set_zlim(-vmax_vis, vmax_vis)
        cb = fig.colorbar(surf, shrink=0.6, pad=0.1)
        cb.set_label(rf"$G_{{{name}}}^\mathrm{{spec}}$ (clipped)")
        plt.tight_layout()
        plt.show()

    # Plot 11, 12, 21, 22 components
    for (ii, jj, nm) in [(0, 0, "11"),
                         (0, 1, "12"),
                         (1, 0, "21"),
                         (1, 1, "22")]:
        plot_spec_vs_rk_component(ii, jj, nm)

    # ------------------------------------------------------------
    # (F) Precompute H_full(r) + nu2 I on the RK grid (for FD and tests)
    # ------------------------------------------------------------
    H_full_arr = np.zeros((Nr, 2, 2))
    for k, r in enumerate(r_grid):
        xb = np.interp(r, R_bounce, X_bounce)
        yb = np.interp(r, R_bounce, Y_bounce)
        H_full_arr[k] = Hessian_V(xb, yb) + nu2 * np.eye(2)

    r_test = 1e-4
    for i in range(2):
        Bi  = B(i, r_test, "+")
        dBi = dB(i, r_test, "+")
        print(f"[FREE BC] i={i}, r={r_test:g}, r*B'/B = {r_test*dBi/Bi}")

    # ------------------------------------------------------------
    # (F0) MODE–EQUATION TEST: L_fd f ≈ 0 for RK modes (interior)
    # ------------------------------------------------------------
    def test_modes_with_fd_operator():
        dr       = r_grid[1] - r_grid[0]
        ell_term = n_mode * (n_mode + 2)

        # choose a "safe" interior window in r
        r_inner = 2.0
        r_outer = 0.5 * r_grid[-1]

        mask_int = (r_grid > r_inner) & (r_grid < r_outer)
        idx      = np.where(mask_int)[0]

        if len(idx) < 3:
            print("[Mode test] Not enough interior points, skipping.")
            return

        j_start = idx[0]
        j_end   = idx[-1]

        def apply_L_int(phi):
            Lphi = np.zeros(j_end - j_start + 1)
            for j in range(j_start, j_end + 1):
                r = r_grid[j]
                c_minus = -1.0/dr**2 + 3.0/(2.0*r*dr)
                c_0     =  2.0/dr**2 + ell_term / (r**2)
                c_plus  = -1.0/dr**2 - 3.0/(2.0*r*dr)
                Lphi[j - j_start] = (c_minus*phi[j-1]
                                     + c_0*phi[j]
                                     + c_plus*phi[j+1])
            return Lphi

        print("\n[Mode test] applying interior FD operator to RK modes...")

        max_res_all = 0.0
        for sign_label, f_arr in [("+", f_plus), ("-", f_minus)]:
            for alpha in range(2):
                for i_field in range(2):
                    phi = f_arr[:, i_field, alpha]

                    Lphi = apply_L_int(phi)

                    # add H_full(r) + nu2 term
                    for j in range(j_start, j_end + 1):
                        H = H_full_arr[j]
                        vec = np.array([
                            f_arr[j, 0, alpha],
                            f_arr[j, 1, alpha],
                        ])
                        Lphi[j - j_start] += H[i_field, :].dot(vec)

                    scale = np.max(np.abs(phi[j_start:j_end+1])) + 1e-30
                    rel_res = np.max(np.abs(Lphi)) / scale

                    max_res_all = max(max_res_all, np.max(np.abs(Lphi)))
                    print(f"[Mode test] sign={sign_label}, alpha={alpha}, "
                          f"field={i_field}: "
                          f"max |L_fd f|_int = {np.max(np.abs(Lphi)):.3e}, "
                          f"rel = {rel_res:.3e}")

        print("[Mode test] worst-case max |L_fd f| over all modes =", max_res_all)

    test_modes_with_fd_operator()

    # ... (the rest of your FD Green construction, G_mod tests, and FD vs RK
    #     plotting can remain as you had it — nothing in that part needs to
    #     change for the spectral Green; I’m stopping here to keep this
    #     answer focused on the spectral piece.)
    #
    # IMPORTANT: keep using r^3 Δr as the discrete weight everywhere,
    # and when you want G_spec to enter the FD vs RK plot, just add
    #   G_spec_ij = G_spec[:, :, i, j]
    # into the list of matrices whose absolute values determine vmax_vis.
    # That way RK, FD, G_mod, and G_spec will all share the same scale.
    #
    # (You already know how to do that pattern from the G_mod / FD code.)

# ================================================================
# 6b. RK-based spectral / Galerkin operator and Green (spectral sum)
# ================================================================
def build_spectral_operator_and_green(G_full, r_grid, tag, n_mode, nu2,
                                      eigen_cut=1e-10):
    """
    Construct a discrete operator L_spec and a *spectral* Green G_spec
    in the nodal basis with inner product

        ⟨u, v⟩ = Σ_k r_k^3 Δr (u_k · v_k).

    Steps:
      1. Flatten G_full to G_big (2Nr×2Nr).
      2. Build weight matrix W_big = diag(r_k^3 Δr) ⊗ I_2.
      3. Define L_raw by L_raw G_big W_big ≈ I  →  L_raw = (G_big W_big)^{-1}.
      4. Symmetrise: L_sym = ½ (L_raw + L_raw^T).
      5. Eigen-decompose L_sym, normalise eigenvectors with W_big:
            v_n^T W_big v_m = δ_nm.
      6. Spectral Green:
            G_spec_big = Σ_{|λ_n|>cut} (1/λ_n) v_n v_n^T.
         Then L_sym G_spec_big W_big ≈ I.
    """
    Nr  = len(r_grid)
    dim = 2 * Nr
    dr  = r_grid[1] - r_grid[0]

    # radial weight r^3 Δr
    w_r   = (r_grid**3) * dr
    w_rep = np.repeat(w_r, 2)        # (r_0^3 Δr, r_0^3 Δr, r_1^3 Δr, ...)
    W_big = np.diag(w_rep)

    # flatten G_full(k,l,i,j) -> G_big(p,q) with p = 2*k + i, q = 2*l + j
    G_big = np.zeros((dim, dim))
    for k in range(Nr):
        for l in range(Nr):
            for i in range(2):
                for j in range(2):
                    p = 2 * k + i
                    q = 2 * l + j
                    G_big[p, q] = G_full[k, l, i, j]

    I_big = np.eye(dim)

    # 1) Galerkin / spectral operator: L_raw such that L_raw G_RK W ≈ I
    A = G_big @ W_big                      # A = G_RK W
    L_raw = np.linalg.solve(A, I_big)      # L_raw = A^{-1}

    # make it symmetric in the usual Euclidean sense (good enough numerically)
    L_sym = 0.5 * (L_raw + L_raw.T)

    # 2) Eigen-decomposition of the symmetric operator
    eigvals, eigvecs = np.linalg.eigh(L_sym)

    # sort by |λ|
    idx_sort = np.argsort(np.abs(eigvals))
    eigvals  = eigvals[idx_sort]
    eigvecs  = eigvecs[:, idx_sort]

    # 3) W-orthonormalise eigenvectors: v_n^T W v_m = δ_nm
    for n in range(dim):
        v = eigvecs[:, n]
        norm2 = float(v.T @ W_big @ v)
        norm  = np.sqrt(abs(norm2) + 1e-30)
        eigvecs[:, n] = v / norm

    # 4) Build spectral Green via explicit sum
    G_spec_big = np.zeros((dim, dim))
    kept = 0
    for n in range(dim):
        lam = eigvals[n]
        if abs(lam) < eigen_cut:
            continue
        v = eigvecs[:, n:n+1]         # column vector
        G_spec_big += (1.0 / lam) * (v @ v.T)
        kept += 1

    print(f"\n[RK spectral/Galerkin] spectral sum for G_spec:")
    print(f"  dim = {dim}, eigenmodes kept (|λ|>{eigen_cut}) = {kept}")

    # 5) Check Green equation with L_sym
    Res_spec = L_sym @ G_spec_big @ W_big - I_big

    max_abs_s  = np.max(np.abs(Res_spec))
    diag_abs_s = np.max(np.abs(np.diag(Res_spec)))
    off_abs_s  = np.max(np.abs(Res_spec - np.diag(np.diag(Res_spec))))
    frob_rel_s = np.linalg.norm(Res_spec) / (np.linalg.norm(I_big) + 1e-30)

    print("  max |L_sym G_spec W - I| =", max_abs_s)
    print("  max diag error            =", diag_abs_s)
    print("  max offdiag error         =", off_abs_s)
    print("  rel Frobenius             =", frob_rel_s)

    # Also compare G_spec to the original RK Green in this discrete basis
    Diff = G_spec_big - G_big
    max_abs_G  = np.max(np.abs(Diff))
    frob_rel_G = np.linalg.norm(Diff) / (np.linalg.norm(G_big) + 1e-30)

    print("  ||G_spec - G_RK||_max     =", max_abs_G)
    print("  rel Frobenius(G_spec-G_RK)=", frob_rel_G)

    # reshape G_spec_big back to (Nr, Nr, 2, 2)
    G_spec = np.zeros_like(G_full)
    for k in range(Nr):
        for l in range(Nr):
            for i in range(2):
                for j in range(2):
                    p = 2 * k + i
                    q = 2 * l + j
                    G_spec[k, l, i, j] = G_spec_big[p, q]

    return L_sym, G_spec


# ================================================================
# 8. Main driver
# ================================================================
if __name__ == "__main__":
    pot_base = MyPotential()
    vacua = find_vacua_grid(pot_base)

    if len(vacua) < 2:
        print("Not enough vacua; exiting.")
        raise SystemExit

    tolV   = 1e-6
    V_min  = vacua[0]['V']
    true_vacua  = [v for v in vacua if abs(v['V'] - V_min) < tolV]
    false_vacua = [v for v in vacua if v not in true_vacua]

    print("\nTrue vacua (degenerate set):")
    for i, v in enumerate(true_vacua):
        print(f" T{i}: x={v['x']:.6f}, y={v['y']:.6f}, V={v['V']:.6f}")

    print("\nFalse vacua candidates:")
    for i, v in enumerate(false_vacua):
        print(f" F{i}: x={v['x']:.6f}, y={v['y']:.6f}, V={v['V']:.6f}")

    nu2    = 2.8
    n_mode = 3
    success = 0

    for fi, fv in enumerate(false_vacua):
        for ti, tv in enumerate(true_vacua):
            if fv['V'] <= tv['V']:
                continue

            false_vac = np.array([fv['x'], fv['y']], dtype=float)
            true_vac  = np.array([tv['x'], tv['y']], dtype=float)

            V_false = pot_base.V(false_vac)
            pot_lift = LiftedPotential(pot_base, V_false)

            tag = f"F{fi}->T{ti} (Vfalse=0)"

            try:
                (true_vac_arr, false_vac_arr,
                 R_bounce, X_bounce, Y_bounce,
                 S_CT) = compute_bounce_for_pair(
                    pot_lift,
                    false_vac,
                    true_vac,
                    tag
                )
            except PotentialError as e:
                print("CosmoTransitions rejected pair", tag, ":", e)
                continue
            except Exception as e:
                print("Error for pair", tag, ":", e)
                continue

            success += 1

            plt.figure()
            plt.plot(X_bounce, Y_bounce, "k-")
            plt.scatter([true_vac_arr[0], false_vac_arr[0]],
                        [true_vac_arr[1], false_vac_arr[1]],
                        c=["blue", "red"])
            plt.title(f"Bounce path in field space ({tag})")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.axis("equal")
            plt.grid(True, alpha=0.3)
            plt.show()

            plt.figure()
            plt.plot(R_bounce, X_bounce, label="x(ρ)")
            plt.plot(R_bounce, Y_bounce, label="y(ρ)")
            plt.xlabel("ρ")
            plt.ylabel("field value")
            plt.title(f"Bounce profile vs radius ({tag})")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.show()

            # fluctuation data (no shift, Q = H_full - H_free(0,0))
            B, dB, K_matrix, A_i, Q_matrix, M_free = build_fluctuation_data(
                false_vac_arr,
                R_bounce, X_bounce, Y_bounce,
                nu2, n_mode
            )

            build_and_plot_green(
                B, dB, K_matrix, A_i,
                R_bounce, X_bounce, Y_bounce,
                false_vac_arr, nu2, n_mode, tag,
                Q_matrix, M_free
            )

    if success == 0:
        print("\nNo metastable→true bounce solutions found.")
    else:
        print(f"\nComputed {success} bounce(s) with Greens.")