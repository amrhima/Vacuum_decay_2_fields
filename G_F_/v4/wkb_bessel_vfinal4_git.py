"""
wkb_bessel_vfinal4.py -- scipy-only Bessel evaluator (v3 pipeline).

This is the v3 pipeline's drop-in replacement for the v2
pipeline_fd_rk_wkb_v2/wkb_bessel_vfinal4.py.  v2 dispatches between
scipy / Hankel / Olver based on (nu, z).  v3 uses scipy directly
for all Bessel evaluations.

Rationale (based on our scipy-vs-Olver accuracy test bench):
  - scipy.special.iv/kv/ivp/kvp delegate to AMOS -- the Fortran
    library (originally distributed via Slatec) that scipy.special
    wraps for Bessel functions -- which automatically picks between
    Miller's recurrence, continued fractions, and its own asymptotic
    series based on (nu, z).
  - AMOS achieves ~1e-15 relative accuracy uniformly across our
    production regime (nu <= 31, z <= 222 at s^2 = 1000).
  - The v2 regime dispatch (Hankel for z > max(30, 5nu), Olver for
    nu >= 11) is auditable and slightly faster, but not a
    numerical requirement: scipy handles all (nu, z) correctly.

API: same as v2's wkb_bessel_vfinal4.py.  iv, kv, ivp, kvp return the
true unscaled Bessel function values.  pick_regime is retained as
a no-op stub for downstream code that introspects it.

All threshold constants are set to infinity to make the intent
crystal clear: this module never dispatches to Hankel or Olver.
"""

from scipy.special import iv as _scipy_iv, kv as _scipy_kv
from scipy.special import ivp as _scipy_ivp, kvp as _scipy_kvp


# ============================================================================ #
#   Threshold constants (kept for API compatibility; set to infinity)          #
# ============================================================================ #

NU_OLVER_THRESHOLD = float('inf')
Z_HANKEL_MIN       = float('inf')
FIVE_NU            = float('inf')
HANKEL_TERMS       = 0     # no Hankel series used
OLVER_TERMS        = 0     # no Olver series used


# ============================================================================ #
#   Public API -- all routed through scipy.special                             #
# ============================================================================ #

def iv(nu, z):
    """I_nu(z) via scipy.special.iv.  Returns float."""
    return float(_scipy_iv(nu, z))


def kv(nu, z):
    """K_nu(z) via scipy.special.kv.  Returns float."""
    return float(_scipy_kv(nu, z))


def ivp(nu, z):
    """I'_nu(z) via scipy.special.ivp.  Returns float."""
    return float(_scipy_ivp(nu, z))


def kvp(nu, z):
    """K'_nu(z) via scipy.special.kvp.  Returns float."""
    return float(_scipy_kvp(nu, z))


def pick_regime(nu, z):
    """No-op: always returns 'scipy' (the only branch we use).

    Kept for API compatibility with the v2 wkb_bessel module.
    Downstream code that checks pick_regime(...) will see 'scipy'.
    """
    return 'scipy'


# ============================================================================ #
#   Diagnostic helpers (preserved API)                                         #
# ============================================================================ #

def _print_thresholds():
    """For tests/diagnostics: print the (trivial) threshold state."""
    print(f"  thresholds: NU_OLVER={NU_OLVER_THRESHOLD}, "
          f"Z_HANKEL_MIN={Z_HANKEL_MIN}, FIVE_NU={FIVE_NU}")
    print(f"  Hankel/Olver disabled; using scipy.special everywhere.")


if __name__ == "__main__":
    print(__doc__)
    _print_thresholds()
    print("\nQuick sanity check at a few (nu, z) cells:")
    for (nu, z) in [(1, 0.5), (5, 30), (10, 50), (18, 100), (31, 200)]:
        print(f"  I_{nu}({z}) = {iv(nu, z):.6e},  K_{nu}({z}) = {kv(nu, z):.6e}")
