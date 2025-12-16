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
- FD Green G_fd from matrix inversion with RK–based Robin BCs;
- G_mod_const: RK Green with Ω^{-1} replaced by a constant diagonal
  matrix from the r^3 W region;
- G_mod_best: same structure but with a best–fit diagonal matrix found
  by solving the Green equation in a thinned window;
- Plots:
    * h–basis functions
    * |B_i^{±}(r)| and |f_i^{±,α}(r)|
    * Relative variation of r^3 W_ab(r) vs Ω_ab
    * 3D surfaces of G_RK, G_fd, G_mod_const, G_mod_best
      plus heatmaps of ΔG(FD−RK).
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
    H0      = Hessian_V(0.0, 0.0)   # this is your H_free

# >>> use H0 (x=0,y=0) for the free masses <<<
    m1_sq_free = -H0[0, 0]
    m2_sq_free = -H0[1, 1]
    M_free     = np.diag([m1_sq_free, m2_sq_free])

    print("\n[build_fluctuation_data] original coords")
    print("  false_vac =", false_vac)
    print("  H_false (at false vac) =\n", H_false)
    print("  H0 (at x=0,y=0)        =\n", H0)
    print("  m1_free^2 =", m1_sq_free, "m2_free^2 =", m2_sq_free)
    print("  nu2       =", nu2,        "n_mode    =", n_mode)

    ell_bessel = n_mode + 1
    r_eps = 1e-8

    def kappa(i):
        m_sq = m1_sq_free if i == 0 else m2_sq_free
        return np.sqrt(nu2 + m_sq)

    # ---------- core Bessel (no 1/r) ----------
    def Bcore_plus(i, r):
        r_eff = max(r, r_eps)
        z = kappa(i) * r_eff
        return iv(ell_bessel, z)

    def dBcore_plus(i, r):
        r_eff = max(r, r_eps)
        z = kappa(i) * r_eff
        return ivp(ell_bessel, z)     # d/dz

    def Bcore_minus(i, r):
        r_eff = max(r, r_eps)
        z = kappa(i) * r_eff
        return kv(ell_bessel, z)

    def dBcore_minus(i, r):
        r_eff = max(r, r_eps)
        z = kappa(i) * r_eff
        return kvp(ell_bessel, z)     # d/dz

    # ---------- full free solutions B = Bcore/r ----------
    def B(i, r, sign):
        r_eff = max(r, r_eps)
        if sign == "+":
            return Bcore_plus(i, r) / r_eff
        else:
            return Bcore_minus(i, r) / r_eff

    def dB(i, r, sign):
        r_eff = max(r, r_eps)
        k = kappa(i)

        if sign == "+":
            Bc     = Bcore_plus(i, r)
            dBc_dz = dBcore_plus(i, r)
        else:
            Bc     = Bcore_minus(i, r)
            dBc_dz = dBcore_minus(i, r)

        dBc_dr = k * dBc_dz
        return (dBc_dr * r_eff - Bc) / (r_eff**2)

    # ---------- bounce interpolation ----------
    def x_bounce(r):
        return np.interp(r, R_bounce, X_bounce)

    def y_bounce(r):
        return np.interp(r, R_bounce, Y_bounce)

    # ---------- full Hessian along bounce (no nu2 here) ----------
    def H_full(r):
        xb = x_bounce(r)
        yb = y_bounce(r)
        return Hessian_V(xb, yb)

    # ---------- Q(r) = H_full - M_free ----------
    def Q_matrix(r):
        return H_full(r) - M_free

    # ---------- K_matrix and A_i entering h-equation ----------
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
        if sign == "+":
            Bc     = Bcore_plus(i, r)
            dBc_dz = dBcore_plus(i, r)
        else:
            Bc     = Bcore_minus(i, r)
            dBc_dz = dBcore_minus(i, r)
        # THIS is the correct formula you wanted:
        return 2.0 * kappa(i) * dBc_dz / (Bc + 1e-30)

    return B, dB, K_matrix, A_i, Q_matrix, M_free


# ================================================================
# 5. h-basis RK integrator
# ================================================================
def rhs_h(r, y, sign, src_index, K_matrix, A_i, Q_matrix):
    """
    y = (h1, h2, v1, v2); v_i = h_i'

    S_i(r) = K_{i, src_index}(r) is the correct source.
    """
    h1, h2, v1, v2 = y
    K = K_matrix(r, sign)
    invr = 0.0 if r == 0.0 else 1.0 / r

    # correct source: S_i = K_{i,α} with α = src_index
    S0 = K[0, src_index]
    S1 = K[1, src_index]

    dv1 = (-(invr + A_i(0, r, sign)) * v1
           + K[0, 0] * h1 + K[0, 1] * h2 + S0)
    dv2 = (-(invr + A_i(1, r, sign)) * v2
           + K[1, 0] * h1 + K[1, 1] * h2 + S1)

    return np.array([v1, v2, dv1, dv2])


def rk4_h(r0, r1, y0, N, sign, src_index,
          K_matrix, A_i, Q_matrix,
          rescale_threshold=1e6):

    r_grid = np.linspace(r0, r1, N + 1)
    dr = (r1 - r0) / N
    Y = np.zeros((N + 1, len(y0)))
    Y[0] = y0

    for k in range(N):
        r = r_grid[k]
        y = Y[k]
        k1 = rhs_h(r,           y,             sign, src_index, K_matrix, A_i, Q_matrix)
        k2 = rhs_h(r + dr/2.0,  y + dr*k1/2.0, sign, src_index, K_matrix, A_i, Q_matrix)
        k3 = rhs_h(r + dr/2.0,  y + dr*k2/2.0, sign, src_index, K_matrix, A_i, Q_matrix)
        k4 = rhs_h(r + dr,      y + dr*k3,     sign, src_index, K_matrix, A_i, Q_matrix)
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
# 7. Main Green builder (RK + FD with RK–based Robin BCs) + G_mod
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

    r_plus_1, Y_plus_1 = rk4_h(
        r0, Rmax, y0, Nsteps, "+", 0, K_matrix, A_i, Q_matrix
    )
    r_plus_2, Y_plus_2 = rk4_h(
        r0, Rmax, y0, Nsteps, "+", 1, K_matrix, A_i, Q_matrix
    )
    r_minus_1, Y_minus_1 = rk4_h(
        Rmax, r0, y0, Nsteps, "-", 0, K_matrix, A_i, Q_matrix
    )
    r_minus_2, Y_minus_2 = rk4_h(
        Rmax, r0, y0, Nsteps, "-", 1, K_matrix, A_i, Q_matrix
    )

    r_grid = r_plus_1
    Nr = len(r_grid)

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
        W_raw[idx] = -W

    W_scaled = np.zeros_like(W_raw)
    for idx, r in enumerate(r_grid):
        r_eff = max(r, r_eps_const)
        W_scaled[idx] = (r_eff**3) * W_raw[idx]

    r_min_tail = 2.0
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

        # === NEW FOR G_mod_rot: diagonalize Omega_sym ===
    eig_Omega, R_Omega = np.linalg.eigh(Omega_sym)
    # Projectors onto the eigen-directions of Omega_sym
    N0 = R_Omega @ np.diag([1.0, 0.0]) @ R_Omega.T
    N1 = R_Omega @ np.diag([0.0, 1.0]) @ R_Omega.T

    print("\n[Omega_sym eigendecomposition]")
    print("  eig(Omega_sym) =", eig_Omega)
    print("  R_Omega =\n", R_Omega)

    # Plots of r^3 W and Omega
    plt.figure(figsize=(7, 5))
    plt.plot(r_grid, W_scaled[:, 0, 0], label=r"$r^3 W_{11}(r)$")
    plt.plot(r_grid, W_scaled[:, 1, 1], label=r"$r^3 W_{22}(r)$")
    plt.axhline(Omega[0, 0], linestyle="--", label=r"$\Omega_{11}$")
    plt.axhline(Omega[1, 1], linestyle="--", label=r"$\Omega_{22}$")
    plt.xlabel(r"$r$")
    plt.ylabel(r"$r^3 W_{aa}(r)$")
    plt.title(r"Scaled Wronskian $r^3 W_{aa}(r)$ and $\Omega_{aa}$")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.plot(r_grid, W_scaled[:, 0, 1], label=r"$r^3 W_{12}(r)$")
    plt.plot(r_grid, W_scaled[:, 1, 0], label=r"$r^3 W_{21}(r)$")
    plt.axhline(Omega[0, 1], linestyle="--", label=r"$\Omega_{12}$")
    plt.axhline(Omega[1, 0], linestyle="--", label=r"$\Omega_{21}$")
    plt.xlabel(r"$r$")
    plt.ylabel(r"$r^3 W_{ab}(r)$")
    plt.title(r"Scaled Wronskian $r^3 W_{ab}(r)$ and $\Omega_{ab}$ (off-diagonal)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

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
    # (D') CONSTANT DIAGONAL MATRIX FROM r^3 W REGION
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
        C_diag     = np.diag(evals.real)
        C_diag_inv = np.diag(1.0 / (evals.real + 1e-30))

        print("\n[G_mod] constant diagonal matrix from r^3 W on region "
              f"[{r_min_c},{r_max_c}]:")
        print("  Omega_reg_sym =\n", Omega_reg_sym)
        print("  eigenvalues   =", evals)
        print("  C_diag        =\n", C_diag)

        return C_diag, C_diag_inv

    C_mod, C_mod_inv = compute_diag_matrix_from_W_region(
        r_grid, W_scaled, r_min_c=2.0011, r_max_c=6.6948
    )

    # ------------------------------------------------------------
    # (E) RK Green's function G_full, G_mod basis, and G_mod_const
    # ------------------------------------------------------------
    G_full = np.zeros((Nr, Nr, 2, 2))
    G_mod_basis = np.zeros((2, Nr, Nr, 2, 2))
    G_mod_const = np.zeros((Nr, Nr, 2, 2))
    # === NEW FOR G_mod_rot: basis in rotated (Omega) eigenbasis ===
    G_rot_basis = np.zeros((2, Nr, Nr, 2, 2))

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

            # Basis for constant diagonal matrix diag(c0,c1) in ORIGINAL basis
            G_mod_basis[0, k, l] = F_big @ M00 @ F_small.T
            G_mod_basis[1, k, l] = F_big @ M11 @ F_small.T

            # === NEW FOR G_mod_rot: basis in the ROTATED (Omega) eigenbasis ===
            # any R_Omega diag(c0,c1) R_Omega^T = c0*N0 + c1*N1
            G_rot_basis[0, k, l] = F_big @ N0 @ F_small.T
            G_rot_basis[1, k, l] = F_big @ N1 @ F_small.T

            # Explicit G_mod_const from C_mod_inv (diagonal in original basis)
            G_mod_const[k, l] = F_big @ C_mod_inv @ F_small.T
            

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
    # (F) Precompute H_full(r) + nu2 I on the RK grid (for FD)
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

    def test_continuum_green_equation(r_grid, G, H_full_arr, n_mode, tag):
        Nr = len(r_grid)
        dr = r_grid[1] - r_grid[0]
        ell_term = n_mode * (n_mode + 2)

    # choose a few representative r' columns to test
        cols_to_test = [Nr // 4, Nr // 2, 3 * Nr // 4]
        cols_to_test = sorted(set(c for c in cols_to_test if 0 < c < Nr-1))

        print(f"\n[Continuum Green test] {tag}:  L_cont G ≈ δ(r-r')/r'^3")
        print("  testing columns (r') at indices:", cols_to_test)

    # radial weights for the δ-normalization check
        w_r = (r_grid**3) * dr

        for ell in cols_to_test:
            rprime = r_grid[ell]
            print(f"\n  --- column ell={ell}, r'={rprime:.6e} ---")

        # LG[k, i_field, j_field] = (L_cont G)_{ij}(r_k, r'_ell)
            LG = np.zeros((Nr, 2, 2))

        # interior points: use the same 3-point stencil as in your FD L
            for k in range(1, Nr-1):
                r = r_grid[k]
                c_minus = -1.0/dr**2 + 3.0/(2.0*r*dr)
                c_0     =  2.0/dr**2 + ell_term / (r**2)
                c_plus  = -1.0/dr**2 - 3.0/(2.0*r*dr)

                H = H_full_arr[k]  # this is Hessian_V(x_b,y_b) + nu2 I_2

                for j_field in range(2):  # column field index
                # build φ_i(r) = G_{ij}(r, r')
                    G_im1 = G[k-1, ell, :, j_field]  # shape (2,)
                    G_i   = G[k,   ell, :, j_field]
                    G_ip1 = G[k+1, ell, :, j_field]

                # free radial operator part on each field i
                    L_free = (
                        c_minus * G_im1 +
                        c_0     * G_i   +
                        c_plus  * G_ip1
                    )  # shape (2,)

                # mixing via H_full_arr: (H · G)_i = H_{i m} G_{mj}
                    mix = H @ G_i  # shape (2,)

                    LG[k, :, j_field] = L_free + mix

        # we ignore k=0, k=Nr-1 in the analysis (stencil incomplete there)

        # (a) off-diagonal residual: r_k != r'_ell
            mask_off = np.ones(Nr, dtype=bool)
            mask_off[ell] = False
            mask_off[0] = False
            mask_off[-1] = False

            max_off = np.max(np.abs(LG[mask_off, :, :]))
            print(f"    max |L_cont G| away from diagonal (all fields) = {max_off:.3e}")

        # (b) δ-normalization via discrete integral
            for j_field in range(2):
                for i_field in range(2):
                # discrete integral over r: sum_k r_k^3 Δr * (L_cont G)_{ij}(r_k, r')
                    integral_ij = np.sum(w_r * LG[:, i_field, j_field])
                    print(f"    ∑_k r_k^3 Δr [L G]_(i={i_field},j={j_field}) ≈ {integral_ij:+.6e} "
                      f"(expect δ_ij)")
                    
    # ------------------------------------------------------------
    # NEW: direct continuum-style Green equation test on G_full
    #      L_cont G_RK(r,r') ≈ δ(r-r') / r'^3
    # ------------------------------------------------------------


    # ------------------------------------------------------------
    # (F') RK & G_mod_const Green-equation tests on thinned window
    #      + best-fit diagonal matrix Cinv_best
    # ------------------------------------------------------------
    print("\n[RK/G_mod Green test] building FD operator on a thinned window...")

    Cinv_best = None       # best-fit diag(c0,c1) in ORIGINAL basis
    Cinv_rot_best = None   # === NEW FOR G_mod_rot: best-fit R diag(c0,c1) R^T ===

    # Choose an interior window away from r=0 and far tail
    r_core_cut_G = 3.0
    r_outer_G    = 0.9 * r_grid[-1]

    mask_rk     = (r_grid > r_core_cut_G) & (r_grid < r_outer_G)
    idx_rk_full = np.where(mask_rk)[0]
    Nw_full     = len(idx_rk_full)

    if Nw_full < 5:
        print("[RK/G_mod Green test] Not enough points, skipping test.")
    else:
        # Thin the window to keep matrix sizes manageable
        target_N = 200
        step     = max(1, Nw_full // target_N)
        idx_rk   = idx_rk_full[::step]
        Nw_rk    = len(idx_rk)

        r_rk     = r_grid[idx_rk]
        dim_rk   = 2 * Nw_rk

        dr_full  = r_grid[1] - r_grid[0]      # spacing on original grid
        dr_th    = dr_full * step             # spacing on thinned grid

        ell_term = n_mode * (n_mode + 2)

        L_rk = np.zeros((dim_rk, dim_rk))

        # --- exact Robin slopes from full RK modes on the thinned window ---
        k_inner_rk = idx_rk[0]
        k_outer_rk = idx_rk[-1]

        sigma_inner_rk = np.zeros(2)  # s_in(i) = (f'/f)_+ at inner edge
        sigma_outer_rk = np.zeros(2)  # s_out(i) = (f'/f)_- at outer edge

        for i in range(2):   # field index
            num_in = 0.0
            den_in = 0.0
            num_out = 0.0
            den_out = 0.0

            for alpha in range(2):
                # inner edge: take regular '+' modes
                f_in   = f_plus[k_inner_rk,  i, alpha]
                df_in  = df_plus[k_inner_rk, i, alpha]
                # outer edge: take decaying '−' modes
                f_out  = f_minus[k_outer_rk,  i, alpha]
                df_out = df_minus[k_outer_rk, i, alpha]

                if abs(f_in) > 0:
                    num_in += df_in / f_in
                    den_in += 1.0
                if abs(f_out) > 0:
                    num_out += df_out / f_out
                    den_out += 1.0

            sigma_inner_rk[i] = num_in  / (den_in  + 1e-30)
            sigma_outer_rk[i] = num_out / (den_out + 1e-30)

        print("\n[RK BC] exact RK slopes on thinned window:")
        for i in range(2):
            print(f" field {i}: s_in = {sigma_inner_rk[i]:.6e}, "
                  f"s_out = {sigma_outer_rk[i]:.6e}")

        # build FD operator with those Robin slopes
        for jloc, k_idx in enumerate(idx_rk):
            r = r_rk[jloc]

            if jloc == 0:
                # inner Robin BC: (f1 - f0)/dr = s_in f0
                for i in range(2):
                    row = 2*jloc + i
                    L_rk[row, :] = 0.0
                    s = sigma_inner_rk[i]
                    L_rk[row, 2*jloc     + i] = -1.0/dr_th - s   # f0
                    L_rk[row, 2*(jloc+1) + i] =  1.0/dr_th       # f1
                continue

            if jloc == Nw_rk - 1:
                # outer Robin BC: (fN - f_{N-1})/dr = s_out fN
                for i in range(2):
                    row = 2*jloc + i
                    L_rk[row, :] = 0.0
                    s = sigma_outer_rk[i]
                    L_rk[row, 2*(jloc-1) + i] = -1.0/dr_th       # f_{N-1}
                    L_rk[row, 2*jloc     + i] =  1.0/dr_th - s   # fN
                continue

            # interior stencil for L = -d2/dr2 - (3/r)d/dr + ell/r^2 + H_full + nu^2
            c_minus = -1.0/dr_th**2 + 3.0/(2.0*r*dr_th)
            c_0     =  2.0/dr_th**2 + ell_term / (r**2)
            c_plus  = -1.0/dr_th**2 - 3.0/(2.0*r*dr_th)

            for i in range(2):
                row       = 2*jloc + i
                col_minus = 2*(jloc-1) + i
                col_0     = 2*jloc      + i
                col_plus  = 2*(jloc+1)  + i

                L_rk[row, col_minus] += c_minus
                L_rk[row, col_0]     += c_0
                L_rk[row, col_plus]  += c_plus

            # add H_full(r) + nu^2 I_2 at this radial point
            H     = H_full_arr[k_idx]
            block = slice(2*jloc, 2*jloc+2)
            L_rk[block, block] += H

        # build weight matrix on thinned window: ∫dr r^3 → sum r_k^3 Δr
        w_rk      = (r_rk**3) * dr_th
        w_rep_rk  = np.repeat(w_rk, 2)
        W_rk      = np.diag(w_rep_rk)

        # extract G_RK and G_mod_const on this window as big (2N × 2N) matrices
        G_rk_win  = G_full[np.ix_(idx_rk, idx_rk)]
        G_modc_win = G_mod_const[np.ix_(idx_rk, idx_rk)]

        G_rk_big   = np.zeros((dim_rk, dim_rk))
        G_modc_big = np.zeros((dim_rk, dim_rk))
        for a_loc in range(Nw_rk):
            for b_loc in range(Nw_rk):
                for i in range(2):
                    for j in range(2):
                        p = 2*a_loc + i
                        q = 2*b_loc + j
                        G_rk_big[p, q]   = G_rk_win[a_loc, b_loc, i, j]
                        G_modc_big[p, q] = G_modc_win[a_loc, b_loc, i, j]

        I_rk = np.eye(dim_rk)

        # ---------- RK Green equation test ----------
        A_rk = L_rk @ G_rk_big @ W_rk
        Rk_res_raw = A_rk - I_rk

        max_rk_raw   = np.max(np.abs(Rk_res_raw))
        diag_rk_raw  = np.max(np.abs(np.diag(Rk_res_raw)))
        off_rk_raw   = np.max(np.abs(Rk_res_raw - np.diag(np.diag(Rk_res_raw))))
        frob_rk_raw  = np.linalg.norm(Rk_res_raw) / (np.linalg.norm(I_rk) + 1e-30)

        # scalar renormalization for RK
        diagA = np.diag(A_rk)
        num   = np.sum(diagA)
        den   = np.sum(diagA**2) + 1e-30
        c_opt_rk = num / den

        A_rk_scaled   = c_opt_rk * A_rk
        Rk_res_scaled = A_rk_scaled - I_rk

        max_rk_sc   = np.max(np.abs(Rk_res_scaled))
        diag_rk_sc  = np.max(np.abs(np.diag(Rk_res_scaled)))
        off_rk_sc   = np.max(np.abs(Rk_res_scaled - np.diag(np.diag(Rk_res_scaled))))
        frob_rk_sc  = np.linalg.norm(Rk_res_scaled) / (np.linalg.norm(I_rk) + 1e-30)

        print("[RK Green test] (thinned window)")
        print("  Nw_rk           =", Nw_rk, "  dim_rk =", dim_rk)
        print("  --- without rescaling ---")
        print("    max |L G_RK W - I|   =", max_rk_raw)
        print("    max diag error       =", diag_rk_raw)
        print("    max offdiag error    =", off_rk_raw)
        print("    rel Frobenius        =", frob_rk_raw)
        print("  --- with scalar rescaling, G -> c_opt_rk * G ---")
        print("    c_opt_rk              =", c_opt_rk)
        print("    max |L (cG_RK) W - I| =", max_rk_sc)
        print("    max diag error        =", diag_rk_sc)
        print("    max offdiag error     =", off_rk_sc)
        print("    rel Frobenius         =", frob_rk_sc)

        # radial dependence of RK residual
        row_abs = np.max(np.abs(Rk_res_raw), axis=1)      # shape (dim_rk,)
        row_abs = row_abs.reshape(Nw_rk, 2)
        row_abs_r = np.max(row_abs, axis=1)

        plt.figure(figsize=(7, 5))
        plt.semilogy(r_rk, row_abs_r, marker="o", linestyle="-", ms=3)
        plt.xlabel(r"$r$")
        plt.ylabel(r"$\max_{i,j} |(L G_{\rm RK} W - I)_{(\text{row } r)}|$")
        plt.title(
            rf"RK Green residual vs $r$ (window, {tag}, "
            rf"$n={n_mode}$, $\nu^2={nu2}$)"
        )
        plt.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        plt.show()

        # ---------- G_mod_const Green equation test ----------
        A_modc = L_rk @ G_modc_big @ W_rk
        Rmodc_res_raw = A_modc - I_rk

        max_modc_raw   = np.max(np.abs(Rmodc_res_raw))
        diag_modc_raw  = np.max(np.abs(np.diag(Rmodc_res_raw)))
        off_modc_raw   = np.max(np.abs(Rmodc_res_raw - np.diag(np.diag(Rmodc_res_raw))))
        frob_modc_raw  = np.linalg.norm(Rmodc_res_raw) / (np.linalg.norm(I_rk) + 1e-30)

        print("[G_mod_const Green test] (thinned window, explicit C_mod_inv)")
        print("    max |L G_mod_const W - I|   =", max_modc_raw)
        print("    max diag error              =", diag_modc_raw)
        print("    max offdiag error           =", off_modc_raw)
        print("    rel Frobenius               =", frob_modc_raw)

        # --------------------------------------------------------
        # (Best-fit G_mod) Solve for diag(c0,c1) in
        #   G_mod(c) = c0 * G_mod_basis[0] + c1 * G_mod_basis[1]
        # by minimizing ||L_rk G_mod(c) W_rk - I||_F^2.
        # --------------------------------------------------------
        Gm0_win = G_mod_basis[0][np.ix_(idx_rk, idx_rk)]
        Gm1_win = G_mod_basis[1][np.ix_(idx_rk, idx_rk)]

        Gm0_big = np.zeros((dim_rk, dim_rk))
        Gm1_big = np.zeros((dim_rk, dim_rk))
        for a_loc in range(Nw_rk):
            for b_loc in range(Nw_rk):
                for i in range(2):
                    for j in range(2):
                        p = 2*a_loc + i
                        q = 2*b_loc + j
                        Gm0_big[p, q] = Gm0_win[a_loc, b_loc, i, j]
                        Gm1_big[p, q] = Gm1_win[a_loc, b_loc, i, j]

        A0 = L_rk @ Gm0_big @ W_rk
        A1 = L_rk @ Gm1_big @ W_rk

        def frob_inner(X, Y):
            return np.sum(X * Y)

        Gmat = np.array([
            [frob_inner(A0, A0), frob_inner(A0, A1)],
            [frob_inner(A1, A0), frob_inner(A1, A1)]
        ])
        bvec = np.array([
            frob_inner(A0, I_rk),
            frob_inner(A1, I_rk)
        ])

        cvec = np.linalg.solve(Gmat, bvec)
        c0, c1 = cvec
        Cinv_best = np.diag([c0, c1])

        print("\n[G_mod best-fit Green test] constant diagonal matrix C_inv_best")
        print("  C_inv_best = diag(c0, c1)")
        print("  c0 =", c0, "  (field 1)")
        print("  c1 =", c1, "  (field 2)")

        A_best = c0 * A0 + c1 * A1
        R_best = A_best - I_rk

        max_best   = np.max(np.abs(R_best))
        diag_best  = np.max(np.abs(np.diag(R_best)))
        off_best   = np.max(np.abs(R_best - np.diag(np.diag(R_best))))
        frob_best  = np.linalg.norm(R_best) / (np.linalg.norm(I_rk) + 1e-30)

        print("  --- residual for G_mod with fitted diag(c0,c1) ---")
        print("    max |L G_mod_best W - I|   =", max_best)
        print("    max diag error            =", diag_best)
        print("    max offdiag error         =", off_best)
        print("    rel Frobenius             =", frob_best)

        # ------------------------------------------------------------
        # Homogeneous test: L_rk G ≈ 0 away from the diagonal
        # ------------------------------------------------------------
        def homogeneous_test_away_from_diagonal(L_rk_loc, G_big_loc, label=""):
            dim_loc = L_rk_loc.shape[0]
            max_abs_loc = 0.0
            max_rel_loc = 0.0
            band_points = 2

            for q in range(dim_loc):
                Lphi = L_rk_loc @ G_big_loc[:, q]
                k_col = q // 2

                mask = np.ones(dim_loc, dtype=bool)
                for p in range(dim_loc):
                    k_row = p // 2
                    if abs(k_row - k_col) <= band_points:
                        mask[p] = False

                if not np.any(mask):
                    continue

                Lphi_off = Lphi[mask]
                Gcol_off = G_big_loc[mask, q]

                abs_max = np.max(np.abs(Lphi_off))
                scale   = np.max(np.abs(Gcol_off)) + 1e-30
                rel     = abs_max / scale

                max_abs_loc = max(max_abs_loc, abs_max)
                max_rel_loc = max(max_rel_loc, rel)

            print(f"[{label} homogeneous test] band = {band_points} radial pts")
            print(f"  max |L G|_offdiag = {max_abs_loc:.3e}")
            print(f"  max rel |L G|     = {max_rel_loc:.3e}")

        homogeneous_test_away_from_diagonal(L_rk, G_rk_big,   label="RK")
        homogeneous_test_away_from_diagonal(L_rk, G_modc_big, label="G_mod_const")

            # --------------------------------------------------------
        # === NEW FOR G_mod_rot ===
        # Best-fit constants in the ROTATED eigenbasis of Omega_sym:
        #   C_rot = R_Omega diag(c0_rot, c1_rot) R_Omega^T
        # via G_rot(c) = c0_rot * G_rot_basis[0] + c1_rot * G_rot_basis[1]
        # --------------------------------------------------------
        Grot0_win = G_rot_basis[0][np.ix_(idx_rk, idx_rk)]
        Grot1_win = G_rot_basis[1][np.ix_(idx_rk, idx_rk)]

        Grot0_big = np.zeros((dim_rk, dim_rk))
        Grot1_big = np.zeros((dim_rk, dim_rk))
        for a_loc in range(Nw_rk):
            for b_loc in range(Nw_rk):
                for i in range(2):
                    for j in range(2):
                        p = 2 * a_loc + i
                        q = 2 * b_loc + j
                        Grot0_big[p, q] = Grot0_win[a_loc, b_loc, i, j]
                        Grot1_big[p, q] = Grot1_win[a_loc, b_loc, i, j]

        Ar0 = L_rk @ Grot0_big @ W_rk
        Ar1 = L_rk @ Grot1_big @ W_rk

        Gmat_rot = np.array([
            [frob_inner(Ar0, Ar0), frob_inner(Ar0, Ar1)],
            [frob_inner(Ar1, Ar0), frob_inner(Ar1, Ar1)]
        ])
        bvec_rot = np.array([
            frob_inner(Ar0, I_rk),
            frob_inner(Ar1, I_rk)
        ])

        cvec_rot = np.linalg.solve(Gmat_rot, bvec_rot)
        c0_rot, c1_rot = cvec_rot

        # full 2×2 constant matrix C_inv_rot = R_Omega diag(c0_rot,c1_rot) R_Omega^T
        C_diag_rot = np.diag([c0_rot, c1_rot])
        Cinv_rot_best = R_Omega @ C_diag_rot @ R_Omega.T

        print("\n[G_mod_rot best-fit Green test] rotated constant matrix C_inv_rot_best")
        print("  C_inv_rot_best = R_Omega diag(c0_rot, c1_rot) R_Omega^T")
        print("  c0_rot =", c0_rot, "  (along eigvec 1 of Omega)")
        print("  c1_rot =", c1_rot, "  (along eigvec 2 of Omega)")
        print("  C_inv_rot_best =\n", Cinv_rot_best)

        A_rot = c0_rot * Ar0 + c1_rot * Ar1
        R_rot = A_rot - I_rk

        max_rot   = np.max(np.abs(R_rot))
        diag_rot  = np.max(np.abs(np.diag(R_rot)))
        off_rot   = np.max(np.abs(R_rot - np.diag(np.diag(R_rot))))
        frob_rot  = np.linalg.norm(R_rot) / (np.linalg.norm(I_rk) + 1e-30)

        print("  --- residual for G_mod_rot with fitted diag(c0_rot,c1_rot) ---")
        print("    max |L G_mod_rot W - I|   =", max_rot)
        print("    max diag error            =", diag_rot)
        print("    max offdiag error         =", off_rot)
        print("    rel Frobenius             =", frob_rot)

        # homogeneous test for G_mod_rot in the same window
        G_rot_best_big = c0_rot * Grot0_big + c1_rot * Grot1_big
        homogeneous_test_away_from_diagonal(L_rk, G_rot_best_big,
                                            label="G_mod_rot")

    # ------------------------------------------------------------
    # (G) FD DISCRETE GREEN with RK-based Robin BCs, on FULL RK GRID
    #     L_win G_fd W_win ≈ I, with r_fd == r_grid
    # ------------------------------------------------------------
    print("\n----------  FD DISCRETE GREEN (RK-based Robin BCs, full grid)  ----------")

    dr  = r_grid[1] - r_grid[0]
    Nr  = len(r_grid)

    idx_fd = np.arange(Nr, dtype=int)
    r_fd   = r_grid.copy()
    Nw     = Nr

    dim      = 2 * Nw
    ell_term = n_mode * (n_mode + 2)

    L_win = np.zeros((dim, dim))

    # --- exact Robin slopes from FULL RK modes on THIS (full) grid ---
    j_inner = idx_fd[0]
    j_outer = idx_fd[-1]

    r_inner_fd = r_fd[0]
    r_outer_fd = r_fd[-1]

    s_inner_fd = np.zeros(2)
    s_outer_fd = np.zeros(2)
    eps = 1e-14

    for i in range(2):
        # inner slope from + branch
        num_in = 0.0
        cnt_in = 0
        for alpha in range(2):
            f_val  = f_plus[j_inner, i, alpha]
            if abs(f_val) < eps:
                continue
            df_val = df_plus[j_inner, i, alpha]
            num_in += df_val / f_val
            cnt_in += 1
        s_inner_fd[i] = num_in / (cnt_in + 1e-30)

        # outer slope from - branch
        num_out = 0.0
        cnt_out = 0
        for alpha in range(2):
            f_val  = f_minus[j_outer, i, alpha]
            if abs(f_val) < eps:
                continue
            df_val = df_minus[j_outer, i, alpha]
            num_out += df_val / f_val
            cnt_out += 1
        s_outer_fd[i] = num_out / (cnt_out + 1e-30)

        print(f"[FD BC (RK full)] field {i}: s_in = {s_inner_fd[i]:+.6e}, "
              f"s_out = {s_outer_fd[i]:+.6e},  "
              f"r_in = {r_inner_fd:.3e}, r_out = {r_outer_fd:.3e}")

    for jloc, k_idx in enumerate(idx_fd):
        r = r_fd[jloc]

        if jloc == 0:
            for i in range(2):
                row = 2*jloc + i
                L_win[row, :] = 0.0
                s = s_inner_fd[i]
                L_win[row, 2*jloc     + i] = -1.0/dr - s
                L_win[row, 2*(jloc+1) + i] =  1.0/dr
            continue

        if jloc == Nw - 1:
            for i in range(2):
                row = 2*jloc + i
                L_win[row, :] = 0.0
                s = s_outer_fd[i]
                L_win[row, 2*(jloc-1) + i] = -1.0/dr
                L_win[row, 2*jloc     + i] =  1.0/dr - s
            continue

        c_minus = -1.0/dr**2 + 3.0/(2.0*r*dr)
        c_0     =  2.0/dr**2 + ell_term / (r**2)
        c_plus  = -1.0/dr**2 - 3.0/(2.0*r*dr)

        for i in range(2):
            row       = 2 * jloc + i
            col_minus = 2*(jloc-1) + i
            col_0     = 2*jloc      + i
            col_plus  = 2*(jloc+1)  + i

            L_win[row, col_minus] += c_minus
            L_win[row, col_0]     += c_0
            L_win[row, col_plus]  += c_plus

        H = H_full_arr[k_idx]
        block = slice(2*jloc, 2*jloc+2)
        L_win[block, block] += H

    w_fd   = (r_fd**3) * dr
    w_rep  = np.repeat(w_fd, 2)
    W_win  = np.diag(w_rep)
    Winv   = np.diag(1.0 / (w_rep + 1e-30))

    plt.figure(figsize=(7, 5))
    plt.plot(r_fd, w_fd, label=r"$r'^3 \,\Delta r$")
    plt.axhline(abs(Omega[0, 0]), linestyle="--", label=r"$|\Omega_{11}|$")
    plt.axhline(abs(Omega[1, 1]), linestyle=":", label=r"$|\Omega_{22}|$")
    plt.yscale("log")
    plt.xlabel(r"$r'$")
    plt.ylabel(r"weight / $|\Omega|$")
    plt.title(r"Discrete weights $r'^3\Delta r$ vs $|\Omega_{aa}|$")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    I_big     = np.eye(dim)
    G_inv_big = np.linalg.solve(L_win, Winv)

    M   = L_win @ G_inv_big @ W_win
    Err = M - I_big

    max_abs     = np.max(np.abs(Err))
    diag_abs    = np.max(np.abs(np.diag(Err)))
    offdiag_abs = np.max(np.abs(Err - np.diag(np.diag(Err))))
    frob_rel    = np.linalg.norm(Err) / (np.linalg.norm(I_big) + 1e-30)

    print("[FD inverse full] max |L G_inv W − I| =", max_abs)
    print("[FD inverse full] max diag error      =", diag_abs)
    print("[FD inverse full] max offdiag error   =", offdiag_abs)
    print("[FD inverse full] rel Frobenius       =", frob_rel)

    # reshape big G_fd matrix back to (Nr, Nr, 2, 2)
    G_fd = np.zeros((Nr, Nr, 2, 2))
    for a_loc in range(Nr):
        for b_loc in range(Nr):
            for i in range(2):
                for j in range(2):
                    p = 2 * a_loc + i
                    q = 2 * b_loc + j
                    G_fd[a_loc, b_loc, i, j] = G_inv_big[p, q]

        # Build full-grid G_mod_best if we found Cinv_best
    G_mod_best = None
    if Cinv_best is not None:
        G_mod_best = np.zeros_like(G_full)
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
                G_mod_best[k, l] = F_big @ Cinv_best @ F_small.T

        print("\n[G_mod_best] built full-grid G_mod_best from best-fit diag(c0,c1).")
    else:
        print("\n[G_mod_best] WARNING: no best-fit diagonal matrix found; "
              "G_mod_best not built.")

    # === NEW FOR G_mod_rot: full-grid Green with rotated constant matrix ===
    G_mod_rot = None
    if Cinv_rot_best is not None:
        G_mod_rot = np.zeros_like(G_full)
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
                G_mod_rot[k, l] = F_big @ Cinv_rot_best @ F_small.T

        print("\n[G_mod_rot] built full-grid G_mod_rot from best-fit rotated matrix.")
    else:
        print("\n[G_mod_rot] WARNING: no best-fit rotated matrix found; "
              "G_mod_rot not built.")

    # Continuum Green-equation tests for the modified Greens
    if G_mod_best is not None:
        test_continuum_green_equation(
            r_grid,
            G_mod_best,
            H_full_arr,
            n_mode,
            tag + " [G_mod_best]"
        )
    if G_mod_rot is not None:
        test_continuum_green_equation(
            r_grid,
            G_mod_rot,
            H_full_arr,
            n_mode,
            tag + " [G_mod_rot]"
        )

    if G_full is not None:
        test_continuum_green_equation(
            r_grid,
            G_full,
            H_full_arr,
            n_mode,
            tag + " [G_RK]"
        )
        
    Rm_fd, Rp_fd = np.meshgrid(r_fd, r_fd, indexing="ij")

    def plot_fd_vs_rk_component(i, j, name):
        G_fd_ij       = G_fd[:, :, i, j]
        G_rk_ij       = G_full[:, :, i, j]
        G_modc_ij     = G_mod_const[:, :, i, j]
        matrices      = [G_fd_ij, G_rk_ij, G_modc_ij]

        if G_mod_best is not None:
            G_mod_best_ij = G_mod_best[:, :, i, j]
            matrices.append(G_mod_best_ij)
        else:
            G_mod_best_ij = None

        # === NEW FOR G_mod_rot: include rotated Green in the scaling ===
        if 'G_mod_rot' in locals() and G_mod_rot is not None:
            G_mod_rot_ij = G_mod_rot[:, :, i, j]
            matrices.append(G_mod_rot_ij)
        else:
            G_mod_rot_ij = None

        all_vals = np.concatenate([M.ravel() for M in matrices])
        vmax_vis = np.percentile(np.abs(all_vals), 99.0)
        if vmax_vis == 0.0:
            vmax_vis = np.max(np.abs(all_vals)) or 1.0

        G_fd_vis       = np.clip(G_fd_ij,       -vmax_vis, vmax_vis)
        G_rk_vis       = np.clip(G_rk_ij,       -vmax_vis, vmax_vis)
        G_modc_vis     = np.clip(G_modc_ij,     -vmax_vis, vmax_vis)
        if G_mod_best_ij is not None:
            G_mod_best_vis = np.clip(G_mod_best_ij, -vmax_vis, vmax_vis)

        def plot_surface_with_vmax(Z_vis, title, zlabel):
            fig = plt.figure(figsize=(7, 5))
            ax = fig.add_subplot(111, projection="3d")
            surf = ax.plot_surface(
                Rm_fd, Rp_fd, Z_vis,
                rstride=max(1, Rm_fd.shape[0] // 50),
                cstride=max(1, Rp_fd.shape[1] // 50),
                cmap=cm.coolwarm,
                linewidth=0,
                antialiased=True,
            )
            ax.set_xlabel("r")
            ax.set_ylabel("r'")
            ax.set_zlabel(zlabel)
            ax.set_title(title)
            ax.set_zlim(-vmax_vis, vmax_vis)
            cb = fig.colorbar(surf, shrink=0.6, pad=0.1)
            cb.set_label(zlabel)
            plt.tight_layout()
            plt.show()

        # RK, FD, G_mod_const, G_mod_best
        plot_surface_with_vmax(
            G_rk_vis,
            title=rf"$G_{{{name}}}(r,r')$ (RK, full grid) for {tag}, "
                  rf"$n={n_mode}$, $\nu^2={nu2}$",
            zlabel=rf"$G_{{{name}}}^\mathrm{{RK}}$ (clipped)",
        )

        plot_surface_with_vmax(
            G_fd_vis,
            title=rf"$G_{{{name}}}(r,r')$ (FD inverse, full grid) for {tag}$",
            zlabel=rf"$G_{{{name}}}^\mathrm{{FD}}$ (clipped)",
        )

        plot_surface_with_vmax(
            G_modc_vis,
            title=rf"$G_{{{name}}}(r,r')$ (G_mod_const, explicit diag)$",
            zlabel=rf"$G_{{{name}}}^\mathrm{{mod}}$ (const, clipped)",
        )
        if G_mod_rot_ij is not None:
            G_mod_rot_vis = np.clip(G_mod_rot_ij, -vmax_vis, vmax_vis)

        if G_mod_best_ij is not None:
            plot_surface_with_vmax(
                G_mod_best_vis,
                title=rf"$G_{{{name}}}(r,r')$ (G_mod_best, best-fit diag)$",
                zlabel=rf"$G_{{{name}}}^\mathrm{{mod,\,best}}$ (clipped)",
            )

        # === NEW FOR G_mod_rot: plot rotated-best Green with same vmax ===
        if G_mod_rot_ij is not None:
            plot_surface_with_vmax(
                G_mod_rot_vis,
                title=rf"$G_{{{name}}}(r,r')$ (G_mod_rot, best-fit rotated)$",
                zlabel=rf"$G_{{{name}}}^\mathrm{{mod,\,rot}}$ (clipped)",
            )

        # FD − RK heatmap
        DG     = G_fd_ij - G_rk_ij
        DG_abs = np.abs(DG)
        dgmax  = np.percentile(DG_abs, 99.0)
        if dgmax == 0.0:
            dgmax = DG_abs.max() if DG_abs.max() > 0 else 1.0
        DG_vis = np.clip(DG, -dgmax, dgmax)

        plt.figure(figsize=(6, 5))
        plt.imshow(
            DG_vis,
            origin="lower",
            extent=[r_fd[0], r_fd[-1], r_fd[0], r_fd[-1]],
            aspect="auto",
            cmap="seismic",
            vmin=-dgmax, vmax=dgmax,
        )
        plt.colorbar(
            label=rf"$\Delta G_{{{name}}} = "
                  r"G_{{{name}}}^\mathrm{FD} - "
                  r"G_{{{name}}}^\mathrm{RK}$"
        )
        plt.xlabel(r"$r'$")
        plt.ylabel(r"$r$")
        plt.title(
            rf"$\Delta G_{{{name}}}(r,r')$ (FD − RK, full grid, {tag})"
        )
        plt.tight_layout()
        plt.show()

    # plot a few representative components on the full grid
    plot_fd_vs_rk_component(0, 0, "11")
    plot_fd_vs_rk_component(0, 1, "12")
    plot_fd_vs_rk_component(1, 1, "22")


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