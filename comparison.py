"""Compare BubbleDet with the G_Y_WKB one-loop calculation.

This comparison intentionally uses the ``bdet`` model from
``G_F_/computation/potential_coupled_toy_model.py``.  It is the decoupled
reference model in this repository: only one scalar field tunnels, so it is
the two-field setup that can be passed to BubbleDet's scalar interface.

The script solves one CosmoTransitions 1-D bounce and passes that same profile
to both calculations.  It writes ``comparison_results.json`` and PNG plots to
the selected output directory.

Run from the project root, after installing BubbleDet, for example::

    python comparison.py --out-dir comparison_output --n-max 12 --l-max 12

BubbleDet's API is documented at https://bubbledet.readthedocs.io/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import simpson


PROJECT_ROOT = Path(__file__).resolve().parent
COMPUTATION_DIR = PROJECT_ROOT / "G_F_" / "computation"
if str(COMPUTATION_DIR) not in sys.path:
    sys.path.insert(0, str(COMPUTATION_DIR))

from potential_coupled_toy_model import (  # noqa: E402
    CTShiftedLiftedPotential,
    H_numeric,
    get_model,
)
from bounce_coupled_toy_model import single_field_V_dV  # noqa: E402
from G_Y_WKB.one_loop_renormalised import (  # noqa: E402
    dm_renormalized_one_loop_action,
    one_loop_action_with_zero_modes,
    zero_mode_jacobian,
)
from G_Y_WKB.WKB_terms import build_W_profile  # noqa: E402
from G_Y_WKB.WKB_terms import estimate_bubbledet_wkb_tail  # noqa: E402
from G_Y_WKB.determinant_evaluation import (  # noqa: E402
    determinant_ratio,
    partial_wave_degeneracy,
    pick_mu_value,
)
from base.fluctuation_operator import (  # noqa: E402
    make_V_matrix,
    make_V_matrix_fv,
)


class VectorisedPotential:
    """Expose the existing two-field potential with scalar and batch H APIs."""

    def __init__(self, potential):
        self.potential = potential

    def H(self, points):
        points = np.asarray(points, dtype=float)
        if points.ndim == 1:
            return np.asarray(self.potential.H(points), dtype=float)
        return np.asarray(self.potential.H_batch(points), dtype=float)


def _jsonable(value):
    """Convert BubbleDet/scientific-Python return values to JSON values."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (float, int, str, bool)) or value is None:
        return value
    return repr(value)


def solve_reference_bounce(model_name="bdet"):
    """Solve the one-field reference bounce used by both calculations."""
    model = get_model(model_name)
    false_index, true_index = model["pair"]
    false_vac = model["minima"][false_index]
    true_vac = model["minima"][true_index]

    # The bdet pair flips field x only.
    moving_field = int(np.flatnonzero(~np.isclose(false_vac, true_vac))[0])
    V1, dV1 = single_field_V_dV(model["params"], false_vac, moving_field)

    from cosmoTransitions.tunneling1D import SingleFieldInstanton

    instanton = SingleFieldInstanton(
        phi_absMin=float(true_vac[moving_field]),
        phi_metaMin=float(false_vac[moving_field]),
        V=V1,
        dV=dV1,
        alpha=3,
    )
    profile = instanton.findProfile(xtol=1e-8, phitol=1e-8)
    R = np.asarray(profile.R, dtype=float)
    phi_moving = np.asarray(profile.Phi, dtype=float)
    S0 = float(instanton.findAction(profile))

    phi_orig = np.tile(false_vac, (len(R), 1))
    phi_orig[:, moving_field] = phi_moving

    pot_prime = CTShiftedLiftedPotential(model["params"], false_vac)
    phi_prime = np.asarray(pot_prime.to_prime(phi_orig), dtype=float)

    # BubbleDet defaults to the false-vacuum Higgs mass when no scale is
    # supplied. Record that value and pass it explicitly to both calculations
    # so the comparison uses one renormalisation convention.
    mu_bubbledet = float(np.sqrt(H_numeric(false_vac, model["params"])
                                [moving_field, moving_field]))

    return {
        "model": model,
        "false_vac": np.asarray(false_vac, dtype=float),
        "true_vac": np.asarray(true_vac, dtype=float),
        "moving_field": moving_field,
        "R": R,
        "phi_orig": phi_orig,
        "phi_prime": phi_prime,
        "S0": S0,
        "mu_bubbledet": mu_bubbledet,
        "pot_prime": VectorisedPotential(pot_prime),
        "V1": V1,
        "dV1": dV1,
    }


def run_bubbledet(data, l_max=None, mu_ren=None):
    """Run BubbleDet and collect its public determinant diagnostics."""
    try:
        from BubbleDet import BubbleConfig, BubbleDeterminant, ParticleConfig
    except ImportError as exc:
        raise RuntimeError(
            "BubbleDet is not installed. Install it in the active Python "
            "environment before running comparison.py."
        ) from exc

    idx = data["moving_field"]
    false_vac = data["false_vac"]
    R = data["R"]
    phi = data["phi_orig"][:, idx]
    V1 = data["V1"]
    dV1 = data["dV1"]

    # Analytic second derivative of the 1-D potential, obtained from the
    # symbolic two-field Hessian while keeping the spectator at its vacuum.
    model = data["model"]
    params = model["params"]
    from potential_coupled_toy_model import H_numeric, H_numeric_batch

    def d2V1(field_value):
        values = np.asarray(field_value, dtype=float)
        if values.ndim == 0:
            point = np.array(false_vac, dtype=float, copy=True)
            point[idx] = float(values)
            return float(H_numeric(point, params)[idx, idx])

        points = np.broadcast_to(false_vac, values.shape + (2,)).copy()
        points[..., idx] = values
        return H_numeric_batch(points, params)[..., idx, idx]

    bubble = BubbleConfig(
        V=V1,
        dV=dV1,
        d2V=d2V1,
        phi_metaMin=float(false_vac[idx]),
        R=R,
        Phi=phi,
        dim=4,
    )
    particle = ParticleConfig(
        W_Phi=d2V1,
        spin=0,
        dof_internal=1,
        zero_modes="Higgs",
    )
    # BubbleDet's implicit default is the false-vacuum Higgs mass. Supplying
    # it explicitly makes the shared scale visible and reproducible.
    if mu_ren is None:
        mu_ren = data["mu_bubbledet"]
    determinant = BubbleDeterminant(
        bubble, particle, renormalisation_scale=float(mu_ren)
    )

    kwargs = {}
    if l_max is not None:
        kwargs["l_max"] = int(l_max)

    def optional(label, function):
        try:
            return function()
        except Exception as exc:  # diagnostics should not hide the main result
            return {"error": f"{type(exc).__name__}: {exc}"}

    result = {}
    result["determinant"] = determinant.findDeterminant(**kwargs)
    result["determinant_full"] = determinant.findDeterminant(full=True, **kwargs)
    result["negative_eigenvalue"] = optional(
        "negative_eigenvalue", determinant.findNegativeEigenvalue
    )
    result["log_phi_infinity"] = optional(
        "log_phi_infinity", determinant.findLogPhiInfinity
    )
    result["single_determinant"] = determinant.findSingleDeterminant(
        particle, **kwargs
    )
    result["single_determinant_full"] = determinant.findSingleDeterminant(
        particle, full=True, **kwargs
    )
    result["renormalisation_term"] = optional(
        "renormalisation_term",
        lambda: determinant.findRenormalisationTerm(particle),
    )
    result["derivative_expansion_LO"] = optional(
        "derivative_expansion_LO",
        lambda: determinant.findDerivativeExpansion(particle, NLO=False),
    )
    result["derivative_expansion_NLO"] = optional(
        "derivative_expansion_NLO",
        lambda: determinant.findDerivativeExpansion(particle, NLO=True),
    )
    result["WKB"] = optional(
        "WKB",
        lambda: determinant.findWKB(particle, max(2, int(l_max or 2))),
    )
    result["gelfand_yaglom"] = []
    if l_max is not None:
        for ell in range(int(l_max) + 1):
            result["gelfand_yaglom"].append(
                {
                    "l": ell,
                    "F": optional(
                        f"F_l={ell}",
                        lambda ell=ell: determinant.findGelfandYaglomFl(
                            particle, ell
                        ),
                    ),
                    "T": optional(
                        f"T_l={ell}",
                        lambda ell=ell: determinant.findGelfandYaglomTl(
                            particle, ell
                        ),
                    ),
                }
            )

    config = determinant.configs[0]
    result["radial_profile"] = {
        "R": config.R,
        "W": config.W,
        "W_inf": config.W_inf,
        "Delta_W": config.Delta_W,
        "m_W_meta": config.m_W_meta,
        "renormalisation_scale": config.renormalisation_scale,
    }

    return _jsonable(result), determinant, particle


def run_gy(data, n_max, mu_ren, wkb_start_n=2):
    """Run the repository calculation and retain its radial diagnostics."""
    R = data["R"]
    phi_prime = data["phi_prime"]
    X, Y = phi_prime[:, 0], phi_prime[:, 1]
    pot = data["pot_prime"]
    r_min = max(float(R[0]), 1e-4)

    delta_s = float(dm_renormalized_one_loop_action(
        R_bounce=R,
        X_bounce=X,
        Y_bounce=Y,
        phi_fv=np.zeros(2),
        pot=pot,
        r_min=r_min,
        Rmax=float(R[-1]),
        n_max=int(n_max),
        mu_ren=float(mu_ren),
        wkb_start_n=int(wkb_start_n),
    ))
    s1_comparable = one_loop_action_with_zero_modes(data["S0"], delta_s)
    J_zero = zero_mode_jacobian(data["S0"])
    log_rate = (
        np.log(0.5)
        + np.log(J_zero)
        - delta_s
        - data["S0"]
    )

    W_profile, H_fv = build_W_profile(R, X, Y, np.zeros(2), pot)
    wkb_by_n, wkb_tail = estimate_bubbledet_wkb_tail(
        R, W_profile, H_fv, l_max=int(n_max) + 1, start_l=int(wkb_start_n)
    )
    log_factor = np.zeros_like(R)
    mask = R > 0.0
    log_factor[mask] = np.log(float(mu_ren) * R[mask] / 2.0) + np.euler_gamma + 1.0
    local_term = float(-1.0 / 16.0 * simpson(
        R**3 * np.array([
            np.trace(W @ W + 2.0 * H_fv @ W) for W in W_profile
        ]) * log_factor,
        x=R,
    ))
    partial = []
    for n in range(int(n_max) + 1):
        vb = make_V_matrix(n, R, X, Y, pot)
        vf = make_V_matrix_fv(n, np.zeros(2), pot)
        if n == 1:
            _, D = pick_mu_value(
                n=n, r_min=r_min, Rmax=float(R[-1]),
                V_matrix=vb, V_matrix_fv=vf,
                isolated_eigenvalues=(0.0,),
            )
        else:
            D = determinant_ratio(
                mu=0.0, n=n, r_min=r_min, Rmax=float(R[-1]),
                V_matrix=vb, V_matrix_fv=vf,
                isolated_eigenvalues=(),
            )
        partial.append({
            "n": n,
            "degeneracy": partial_wave_degeneracy(n),
            "determinant_ratio": float(D),
            "log_determinant_ratio": float(np.log(abs(D))),
            "wkb_approximation": float(wkb_by_n[n]),
            "log_subtracted": float(
                np.log(abs(D)) - wkb_by_n[n]
                if n >= wkb_start_n else np.log(abs(D))
            ),
            "exact_contribution": float(
                0.5 * partial_wave_degeneracy(n) * np.log(abs(D))
            ),
            "subtracted_contribution": float(
                0.5 * partial_wave_degeneracy(n) * (
                    np.log(abs(D)) - wkb_by_n[n]
                    if n >= wkb_start_n else np.log(abs(D))
                )
            ),
        })

    # Store the same aligned decomposition used in the plots.  In
    # BubbleDet's ``full`` output the WKB tail and counterterm are attached
    # to l=0, while the translational zero-mode normalization is attached to
    # l=1.
    for item in partial:
        aligned = item["subtracted_contribution"]
        if item["n"] == 0:
            aligned += wkb_tail + local_term
        elif item["n"] == 1:
            aligned -= np.log(J_zero)
        item["bubbledet_convention_contribution"] = float(aligned)

    return {
        "delta_S_reg": delta_s,
        "zero_mode_jacobian": J_zero,
        "S1_comparable_to_BubbleDet": s1_comparable,
        "prefactor": float(np.exp(-delta_s)),
        "negative_mode_factor": 0.5,
        "log_decay_rate_factor": float(log_rate),
        "decay_rate_factor": float(np.exp(log_rate)),
        "H_false": H_fv,
        "W_trace": np.trace(W_profile, axis1=1, axis2=2),
        "W_profile": W_profile,
        "local_renormalisation_term": local_term,
        "wkb_tail": float(wkb_tail),
        "wkb_by_n": wkb_by_n,
        "partial_waves": partial,
    }


def make_plots(data, bubbledet, gy, out_dir):
    import matplotlib.pyplot as plt

    def scalar_result(value):
        if isinstance(value, list) and value:
            return float(value[0])
        if isinstance(value, (float, int)):
            return float(value)
        return np.nan

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(out_dir / name, dpi=160)
        plt.close(fig)

    R = data["R"]
    idx = data["moving_field"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(R, data["phi_orig"][:, idx], label="shared bounce / BubbleDet")
    ax.plot(R, data["phi_orig"][:, idx], "--", label="G_Y_WKB input")
    ax.set(xlabel=r"$r$", ylabel=rf"$\phi_{idx}(r)$", title="Bounce profile")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "bounce_profile_comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(R, gy["W_trace"], label=r"$\mathrm{Tr}[H_b-H_F]$")
    bd_profile = bubbledet.get("radial_profile", {})
    if "Delta_W" in bd_profile:
        ax.plot(R, bd_profile["Delta_W"], "--", label=r"BubbleDet $\Delta W$")
    ax.set(xlabel=r"$r$", ylabel="Hessian insertion", title="Fluctuation background")
    ax.grid(alpha=0.25)
    ax.legend()
    save(fig, "fluctuation_background.png")

    ns = np.array([item["n"] for item in gy["partial_waves"]])
    ratios = np.array([item["determinant_ratio"] for item in gy["partial_waves"]])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ns, np.log(np.abs(ratios)), "o-", label="G_Y_WKB log determinant ratio")
    bd_gy = bubbledet.get("gelfand_yaglom", [])
    bd_l = []
    bd_f = []
    bd_t = []
    for item in bd_gy:
        f = scalar_result(item.get("F"))
        t = scalar_result(item.get("T"))
        if np.isfinite(f):
            bd_l.append(item["l"])
            bd_f.append(f)
            bd_t.append(t)
    if bd_l:
        ax.plot(bd_l, bd_f, "x--", label="BubbleDet F_l")
    ax.set(xlabel="partial-wave n", ylabel=r"$\log|D_n|$",
           title="G_Y_WKB partial-wave determinants")
    ax.grid(alpha=0.25)
    ax.legend()
    save(fig, "partial_wave_determinants.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if bd_l:
        ax.plot(bd_l, bd_t, "x--", label="BubbleDet T_l")
    ax.plot(ns, ratios, "o-", label="G_Y_WKB D_n")
    ax.set(xlabel="partial wave", ylabel="determinant ratio",
           title="Gelfand–Yaglom determinant ratios")
    ax.set_yscale("symlog", linthresh=1e-5)
    ax.grid(alpha=0.25)
    ax.legend()
    save(fig, "gelfand_yaglom_ratios.png")

    bd_full = np.asarray(bubbledet["determinant_full"][0][0], dtype=float)
    bd_err = np.asarray(bubbledet["determinant_full"][1][0], dtype=float)
    gy_raw_terms = np.array([
        item["exact_contribution"] for item in gy["partial_waves"]
    ])
    gy_terms = np.array([
        item["subtracted_contribution"] for item in gy["partial_waves"]
    ])
    # BubbleDet's ``full`` array is not a raw partial-wave decomposition:
    # it puts the renormalisation term and the analytically summed WKB tail
    # into l=0, and the translational zero-mode normalization into l=1.
    # Make the equivalent decomposition for G_Y_WKB before comparing the
    # plotted entries.
    gy_bubbledet_convention = gy_terms.copy()
    gy_bubbledet_convention[0] += (
        gy["wkb_tail"] + gy["local_renormalisation_term"]
    )
    if len(gy_bubbledet_convention) > 1:
        gy_bubbledet_convention[1] -= np.log(gy["zero_mode_jacobian"])
    m = min(len(bd_full), len(gy_terms))
    ell = np.arange(m)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(ell, bd_full[:m], yerr=bd_err[:m], fmt="o-",
                capsize=2, label="BubbleDet per-wave contribution")
    ax.plot(
        ell, gy_bubbledet_convention[:m], "x--",
        label="G_Y_WKB in BubbleDet convention",
    )
    ax.set(xlabel="partial wave", ylabel="contribution to S1",
           title="Partial-wave contributions (aligned conventions)")
    ax.grid(alpha=0.25)
    ax.legend()
    save(fig, "partial_wave_contributions.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ell, gy_raw_terms[:m], "o-", label="G_Y_WKB raw exact contribution")
    ax.plot(ell, gy_terms[:m], "x--", label="G_Y_WKB WKB-subtracted contribution")
    ax.set(xlabel="partial wave", ylabel="contribution before global factors",
           title="G_Y_WKB partial-wave bookkeeping")
    ax.grid(alpha=0.25)
    ax.legend()
    save(fig, "partial_wave_contributions_gy_bookkeeping.png")

    # Isolate the two special low waves.  This makes it clear that n=0 is
    # retained, while n=1 is the translational zero-mode sector.
    low_m = min(2, m)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xlow = np.arange(low_m)
    width = 0.24
    ax.bar(xlow - width, bd_full[:low_m], width,
           yerr=bd_err[:low_m], capsize=2, label="BubbleDet full")
    ax.bar(xlow, gy_raw_terms[:low_m], width, label="G_Y_WKB raw")
    ax.bar(xlow + width, gy_bubbledet_convention[:low_m], width,
           label="G_Y_WKB aligned")
    ax.set_xticks(xlow, ["n=0 / l=0", "n=1 / l=1"][:low_m])
    ax.set_ylabel("contribution to S1")
    ax.set_title("Low partial waves and zero-mode bookkeeping")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save(fig, "low_partial_waves_zero_mode.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ell, np.cumsum(bd_full[:m]), "o-", label="BubbleDet cumulative")
    ax.plot(
        ell, np.cumsum(gy_bubbledet_convention[:m]), "x--",
        label="G_Y_WKB cumulative (aligned)",
    )
    ax.axhline(float(bubbledet["determinant"][0]), color="tab:orange",
               linestyle=":", label="BubbleDet extrapolated S1")
    ax.axhline(gy["S1_comparable_to_BubbleDet"], color="tab:blue",
               linestyle=":", label="G_Y_WKB comparable S1")
    ax.set(xlabel="partial wave", ylabel="cumulative one-loop action",
           title="Cumulative partial-wave sums")
    ax.grid(alpha=0.25)
    ax.legend()
    save(fig, "cumulative_partial_wave_sums.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    n_wkb = np.arange(len(gy["wkb_by_n"]))
    ax.plot(n_wkb, gy["wkb_by_n"], "o-", label="G_Y_WKB WKB approximation")
    bd_wkb = bubbledet.get("WKB", [])
    if isinstance(bd_wkb, list) and len(bd_wkb) >= 1:
        ax.plot(np.arange(len(bd_wkb[0])), bd_wkb[0], "x--",
                label="BubbleDet WKB approximation")
    ax.set(xlabel="partial wave", ylabel=r"$F_l^{\rm WKB}$",
           title="WKB approximations by partial wave")
    ax.set_yscale("symlog", linthresh=1e-8)
    ax.grid(alpha=0.25)
    ax.legend()
    save(fig, "wkb_by_partial_wave.png")

    bd_ren = scalar_result(bubbledet.get("renormalisation_term"))
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(["BubbleDet", "G_Y_WKB"],
           [bd_ren, gy["local_renormalisation_term"]],
           color=["tab:orange", "tab:blue"])
    ax.set_ylabel("renormalisation contribution")
    ax.set_title("Local renormalisation term")
    ax.grid(axis="y", alpha=0.25)
    save(fig, "renormalisation_terms.png")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    neg = scalar_result(bubbledet.get("negative_eigenvalue"))
    ax.bar(["BubbleDet negative mode"], [neg], color="tab:red")
    ax.set_ylabel(r"$\lambda_-$")
    ax.set_title("Negative eigenvalue diagnostic")
    ax.grid(axis="y", alpha=0.25)
    save(fig, "negative_mode.png")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    log_phi = scalar_result(bubbledet.get("log_phi_infinity"))
    ax.bar(["BubbleDet"], [log_phi], color="tab:orange")
    ax.set_ylabel(r"$\log\phi_\infty$")
    ax.set_title("Asymptotic tail coefficient")
    ax.grid(axis="y", alpha=0.25)
    save(fig, "asymptotic_tail_coefficient.png")

    labels = ["BubbleDet S1", "G_Y_WKB ΔS"]
    values = [
        float(bubbledet["determinant"][0]),
        gy["S1_comparable_to_BubbleDet"],
    ]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(labels, values, color=["tab:orange", "tab:blue"])
    ax.set_ylabel("one-loop action (same zero-mode convention)")
    ax.set_title("Renormalised one-loop result")
    ax.grid(axis="y", alpha=0.25)
    save(fig, "one_loop_result_comparison.png")


def make_convergence_plot(data, gy, convergence, out_dir):
    """Plot the finite-cutoff one-loop action for both implementations."""
    import matplotlib.pyplot as plt

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(out_dir / name, dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    gy_curve = convergence["G_Y_WKB"]
    bd_curve = convergence["BubbleDet"]
    ax.plot(
        [item["n_max"] for item in gy_curve],
        [item["S1"] for item in gy_curve],
        "o-", label="G_Y_WKB finite n_max",
    )
    ax.errorbar(
        [item["l_max"] for item in bd_curve],
        [item["S1"] for item in bd_curve],
        yerr=[item["error"] for item in bd_curve],
        fmt="s--", capsize=2, label="BubbleDet finite l_max/extrapolation",
    )
    ax.axhline(
        convergence["BubbleDet_reference_S1"], color="k", linestyle=":",
        label="BubbleDet largest-l_max reference",
    )
    ax.set(
        xlabel="partial-wave cutoff",
        ylabel="one-loop action S1",
        title="One-loop convergence with partial-wave cutoff",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    save(fig, "S1_convergence.png")


def convergence_sweep(data, max_cutoff, mu_ren, wkb_start_n=2, gy_existing=None):
    """Evaluate both codes at a range of partial-wave cutoffs.

    The GY curve is reconstructed from one calculation at ``max_cutoff``;
    this keeps every point on the curve based on identical bounce and
    determinant data. BubbleDet is evaluated independently at each cutoff,
    because its finite-cutoff extrapolation is part of the output being
    tested.
    """
    max_cutoff = int(max_cutoff)
    gy_max = (
        gy_existing
        if gy_existing is not None
        else run_gy(data, max_cutoff, mu_ren, wkb_start_n)
    )
    log_jacobian = np.log(gy_max["zero_mode_jacobian"])
    running = gy_max["local_renormalisation_term"] + gy_max["wkb_tail"]
    gy_curve = []
    for item in gy_max["partial_waves"]:
        running += item["subtracted_contribution"]
        if item["n"] == 1:
            running -= log_jacobian
        if item["n"] >= 2:
            gy_curve.append({"n_max": item["n"], "S1": float(running)})

    cutoffs = sorted(set([
        2, 4, 6, 8, 10, 12, 16, 20, 30, 40, max_cutoff,
    ]))
    cutoffs = [cutoff for cutoff in cutoffs if cutoff <= max_cutoff]
    bubbledet_curve = []
    for cutoff in cutoffs:
        bd, _, _ = run_bubbledet(data, cutoff, mu_ren)
        bubbledet_curve.append({
            "l_max": cutoff,
            "S1": float(bd["determinant"][0]),
            "error": float(bd["determinant"][1]),
        })

    return {
        "G_Y_WKB": gy_curve,
        "BubbleDet": bubbledet_curve,
        "BubbleDet_reference_S1": bubbledet_curve[-1]["S1"],
        "settings": {
            "max_cutoff": max_cutoff,
            "wkb_start_n": int(wkb_start_n),
            "mu_ren": float(mu_ren),
        },
    }, gy_max


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("comparison_output"))
    parser.add_argument("--n-max", type=int, default=8,
                        help="maximum G_Y_WKB partial wave")
    parser.add_argument("--l-max", type=int, default=8,
                        help="maximum BubbleDet orbital and diagnostic wave (>=2)")
    parser.add_argument("--mu-ren", type=float, default=None,
                        help="shared renormalisation scale; default: BubbleDet's false-vacuum mass")
    parser.add_argument("--wkb-start-n", type=int, default=2,
                        help="first partial wave receiving WKB subtraction (default: 2)")
    parser.add_argument("--convergence-sweep", action="store_true",
                        help="also scan and plot S1 versus both partial-wave cutoffs")
    parser.add_argument("--sweep-max", type=int, default=None,
                        help="largest cutoff in the convergence sweep; enables the sweep")
    args = parser.parse_args()

    if args.n_max < 1 or args.l_max < 2:
        parser.error("--n-max must be >= 1 and --l-max must be >= 2")
    if args.wkb_start_n < 2 or args.wkb_start_n > args.n_max + 1:
        parser.error("--wkb-start-n must be >= 2 and <= n_max + 1")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = solve_reference_bounce()
    mu_ren = data["mu_bubbledet"] if args.mu_ren is None else args.mu_ren
    bubbledet, _, _ = run_bubbledet(data, args.l_max, mu_ren)
    gy = run_gy(data, args.n_max, mu_ren, args.wkb_start_n)

    bubbledet_s1 = float(bubbledet["determinant"][0])
    bubbledet_error = float(bubbledet["determinant"][1])
    bubbledet_log_rate = (
        np.log(0.5) - data["S0"] - bubbledet_s1
    )

    results = {
        "model": data["model"]["name"],
        "false_vac": data["false_vac"],
        "true_vac": data["true_vac"],
        "moving_field": data["moving_field"],
        "mu_ren": mu_ren,
        "S0": data["S0"],
        "BubbleDet": bubbledet,
        "G_Y_WKB": gy,
        "rate_comparison": {
            "BubbleDet_S1": bubbledet_s1,
            "BubbleDet_S1_error": bubbledet_error,
            "BubbleDet_negative_mode_factor": 0.5,
            "BubbleDet_log_decay_rate_factor": float(bubbledet_log_rate),
            "G_Y_WKB_log_decay_rate_factor": gy["log_decay_rate_factor"],
        },
        "settings": {
            "n_max": args.n_max,
            "l_max": args.l_max,
            "mu_ren": mu_ren,
            "wkb_start_n": args.wkb_start_n,
        },
    }
    results = _jsonable(results)
    with (args.out_dir / "comparison_results.json").open("w") as stream:
        json.dump(results, stream, indent=2)

    make_plots(data, bubbledet, gy, args.out_dir)
    if args.convergence_sweep or args.sweep_max is not None:
        sweep_max = args.n_max if args.sweep_max is None else args.sweep_max
        if sweep_max < 2:
            parser.error("--sweep-max must be >= 2")
        convergence, gy_sweep = convergence_sweep(
            data, sweep_max, mu_ren, args.wkb_start_n,
            gy_existing=gy if sweep_max == args.n_max else None,
        )
        results["convergence"] = _jsonable(convergence)
        with (args.out_dir / "comparison_results.json").open("w") as stream:
            json.dump(results, stream, indent=2)
        make_convergence_plot(data, gy_sweep, convergence, args.out_dir)
    print(json.dumps({
        "S0": data["S0"],
        "BubbleDet_S1": bubbledet_s1,
        "BubbleDet_S1_error": bubbledet_error,
        "G_Y_WKB_delta_S_reg": gy["delta_S_reg"],
        "G_Y_WKB_S1_comparable": gy["S1_comparable_to_BubbleDet"],
        "output": str(args.out_dir.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
