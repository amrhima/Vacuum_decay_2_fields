"""Plot BubbleDet versus GY_WKB for a decoupled two-field model.

The model used here is

    V(phi, chi) = 1/2 m2 phi^2 - g phi^3/3 + lam phi^4/4
                  + 1/2 spectator_m2 chi^2,

with the false vacuum at (phi, chi) = (0, 0).  Only ``phi`` tunnels, so the
Hessian is diagonal everywhere and the same one-dimensional bounce can be
passed to BubbleDet and to the two-field GY_WKB calculation.

Examples
--------
    python plot-GY.py --scan cubic --out cubic.png
    python plot-GY.py --scan quartic --out quartic.png
    python plot-GY.py --scan mass --out mass.png

BubbleDet must be installed in the active Python environment.  The scan
functions are also importable through ``runpy`` or from a copy of this file;
the hyphenated filename is intentionally kept as requested.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from G_Y_WKB.one_loop_renormalised import (  # noqa: E402
    dm_renormalized_one_loop_action,
    one_loop_action_with_zero_modes,
)


@dataclass(frozen=True)
class DecoupledQuartic:
    """Parameters and Hessian for the decoupled two-field potential."""

    mass_sq: float = 1.0
    cubic: float = 3.0
    quartic: float = 1.0
    spectator_mass_sq: float = 1.0

    def __post_init__(self):
        if self.mass_sq <= 0.0 or self.quartic <= 0.0:
            raise ValueError("mass_sq and quartic must be positive.")
        if self.spectator_mass_sq <= 0.0:
            raise ValueError("spectator_mass_sq must be positive.")
        if self.cubic <= 0.0:
            raise ValueError("cubic must be positive.")
        if self.cubic**2 <= 4.0 * self.quartic * self.mass_sq:
            raise ValueError(
                "Parameters do not produce a barrier: require cubic^2 > "
                "4*quartic*mass_sq."
            )
        if self.V(self.true_vacuum()) >= 0.0:
            raise ValueError(
                "The nonzero stationary point is not a lower true vacuum; "
                "increase cubic or decrease mass_sq/quartic."
            )

    def V(self, phi):
        phi = np.asarray(phi, dtype=float)
        return (
            0.5 * self.mass_sq * phi**2
            - self.cubic * phi**3 / 3.0
            + self.quartic * phi**4 / 4.0
        )

    def dV(self, phi):
        phi = np.asarray(phi, dtype=float)
        return (
            self.mass_sq * phi
            - self.cubic * phi**2
            + self.quartic * phi**3
        )

    def d2V(self, phi):
        phi = np.asarray(phi, dtype=float)
        return (
            self.mass_sq
            - 2.0 * self.cubic * phi
            + 3.0 * self.quartic * phi**2
        )

    def true_vacuum(self) -> float:
        discriminant = self.cubic**2 - 4.0 * self.quartic * self.mass_sq
        return float(
            (self.cubic + np.sqrt(discriminant)) / (2.0 * self.quartic)
        )

    def H(self, point):
        """Return the 2x2 Hessian used by GY_WKB."""
        point = np.asarray(point, dtype=float)
        phi = point[..., 0]
        hessian = np.zeros(phi.shape + (2, 2), dtype=float)
        hessian[..., 0, 0] = self.d2V(phi)
        hessian[..., 1, 1] = self.spectator_mass_sq
        return hessian


def _solve_bounce(model: DecoupledQuartic):
    """Solve the common one-dimensional O(4) bounce."""
    from cosmoTransitions.tunneling1D import SingleFieldInstanton

    instanton = SingleFieldInstanton(
        phi_absMin=model.true_vacuum(),
        phi_metaMin=0.0,
        V=model.V,
        dV=model.dV,
        alpha=3,
    )
    profile = instanton.findProfile(xtol=1e-8, phitol=1e-8)
    return {
        "R": np.asarray(profile.R, dtype=float),
        "phi": np.asarray(profile.Phi, dtype=float),
        "S0": float(instanton.findAction(profile)),
    }


def _run_bubbledet(model, bounce, l_max):
    """Evaluate BubbleDet in the same renormalisation convention."""
    try:
        from BubbleDet import BubbleConfig, BubbleDeterminant, ParticleConfig
    except ImportError as exc:
        raise RuntimeError(
            "BubbleDet is required for this plot. Install it in the active "
            "Python environment."
        ) from exc

    R = bounce["R"]
    phi = bounce["phi"]
    bubble = BubbleConfig(
        V=model.V,
        dV=model.dV,
        d2V=model.d2V,
        phi_metaMin=0.0,
        R=R,
        Phi=phi,
        dim=4,
    )
    particle = ParticleConfig(
        W_Phi=model.d2V,
        spin=0,
        dof_internal=1,
        zero_modes="Higgs",
    )
    determinant = BubbleDeterminant(
        bubble,
        particle,
        renormalisation_scale=np.sqrt(model.mass_sq),
    )
    result = determinant.findDeterminant(l_max=int(l_max))
    return float(np.asarray(result).reshape(-1)[0])


def _run_gy_wkb(model, bounce, l_max, wkb_start_l=2):
    """Evaluate GY_WKB using the same bounce and renormalisation scale."""
    R = bounce["R"]
    r_min = max(float(R[0]), 1e-4)
    X = bounce["phi"]
    Y = np.zeros_like(X)
    delta_s = dm_renormalized_one_loop_action(
        R_bounce=R,
        X_bounce=X,
        Y_bounce=Y,
        phi_fv=np.zeros(2),
        pot=model,
        r_min=r_min,
        Rmax=float(R[-1]),
        l_max=int(l_max),
        mu_ren=np.sqrt(model.mass_sq),
        wkb_start_l=int(wkb_start_l),
    )
    return float(one_loop_action_with_zero_modes(bounce["S0"], delta_s))


def evaluate_point(model, l_max=12, wkb_start_l=2):
    """Return one BubbleDet/GY_WKB comparison point."""
    bounce = _solve_bounce(model)
    return {
        "bubble_det": _run_bubbledet(model, bounce, l_max),
        "gy_wkb": _run_gy_wkb(model, bounce, l_max, wkb_start_l),
        "bounce_action": bounce["S0"],
    }


def _scan(values, make_model, l_max, wkb_start_l):
    values = np.asarray(values, dtype=float)
    bubble_det = np.full(values.shape, np.nan)
    gy_wkb = np.full(values.shape, np.nan)
    bounce_action = np.full(values.shape, np.nan)
    for i, value in enumerate(values):
        result = evaluate_point(make_model(float(value)), l_max, wkb_start_l)
        bubble_det[i] = result["bubble_det"]
        gy_wkb[i] = result["gy_wkb"]
        bounce_action[i] = result["bounce_action"]
        print(
            f"{i + 1}/{len(values)}: parameter={value:g}, "
            f"BubbleDet={bubble_det[i]:+.8e}, GY_WKB={gy_wkb[i]:+.8e}"
        )
    return values, bubble_det, gy_wkb, bounce_action


def plot_varying_three_point(
    cubic_values=np.linspace(2.2, 4.0, 7),
    mass_sq=1.0,
    quartic=1.0,
    spectator_mass_sq=1.0,
    l_max=12,
    wkb_start_l=2,
    out=None,
):
    """Plot the one-loop action versus the cubic/three-point coupling ``g``."""
    values, bubble, gy, _ = _scan(
        cubic_values,
        lambda g: DecoupledQuartic(
            mass_sq, g, quartic, spectator_mass_sq
        ),
        l_max,
        wkb_start_l,
    )
    return _plot_scan(values, bubble, gy, r"three-point coupling $g$", out)


def plot_varying_four_point(
    quartic_values=np.linspace(0.5, 1.75, 7),
    mass_sq=1.0,
    cubic=3.0,
    spectator_mass_sq=1.0,
    l_max=12,
    wkb_start_l=2,
    out=None,
):
    """Plot the one-loop action versus the quartic/four-point coupling ``λ``."""
    values, bubble, gy, _ = _scan(
        quartic_values,
        lambda quartic_value: DecoupledQuartic(
            mass_sq, cubic, quartic_value, spectator_mass_sq
        ),
        l_max,
        wkb_start_l,
    )
    return _plot_scan(values, bubble, gy, r"four-point coupling $\lambda$", out)


def plot_varying_mass(
    mass_sq_values=np.linspace(0.25, 1.75, 7),
    cubic=3.0,
    quartic=1.0,
    spectator_mass_sq=1.0,
    l_max=12,
    wkb_start_l=2,
    out=None,
):
    """Plot the one-loop action versus the tunnelling mass-squared ``m²``."""
    values, bubble, gy, _ = _scan(
        mass_sq_values,
        lambda mass_sq: DecoupledQuartic(
            mass_sq, cubic, quartic, spectator_mass_sq
        ),
        l_max,
        wkb_start_l,
    )
    return _plot_scan(values, bubble, gy, r"tunnelling mass squared $m^2$", out)


def _plot_scan(values, bubble, gy, xlabel, out):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(values, bubble, "o-", label="BubbleDet")
    ax.plot(values, gy, "s--", label="GY_WKB")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("one-loop correction to effective action")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    if out is not None:
        fig.savefig(out, dpi=180)
    return fig, ax


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", choices=("cubic", "quartic", "mass"),
                        default="cubic")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--l-max", type=int, default=12)
    parser.add_argument("--wkb-start-l", type=int, default=2)
    args = parser.parse_args()
    if args.l_max < 1:
        parser.error("--l-max must be positive")
    if args.wkb_start_l < 2 or args.wkb_start_l > args.l_max + 1:
        parser.error("--wkb-start-l must satisfy 2 <= start <= l-max+1")

    if args.scan == "cubic":
        plot_varying_three_point(
            l_max=args.l_max, wkb_start_l=args.wkb_start_l, out=args.out
        )
    elif args.scan == "quartic":
        plot_varying_four_point(
            l_max=args.l_max, wkb_start_l=args.wkb_start_l, out=args.out
        )
    else:
        plot_varying_mass(
            l_max=args.l_max, wkb_start_l=args.wkb_start_l, out=args.out
        )


if __name__ == "__main__":
    main()
