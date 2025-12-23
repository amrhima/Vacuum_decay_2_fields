# bounce_profile_from_path.py

import numpy as np
from collections import namedtuple

from classical_bounce1D import shoot_bounce_1d


class _LinearPath1D:
    def __init__(self, pts: np.ndarray, V, dV):
        self.pts_array = np.asarray(pts, dtype=float)
        if self.pts_array.ndim != 2:
            raise ValueError("path_pts must have shape (n_points, n_fields)")
        if self.pts_array.shape[0] < 2:
            raise ValueError("path_pts must have at least two points")

        self.V_field = V
        self.dV_field = dV

        diffs = np.diff(self.pts_array, axis=0)
        seg_len = np.linalg.norm(diffs, axis=1)
        if np.any(seg_len <= 0):
            raise ValueError("path_pts contains duplicate or zero-length segments")

        self.diffs = diffs
        self.seg_len = seg_len
        self.s = np.concatenate([[0.0], np.cumsum(seg_len)])
        self.L = float(self.s[-1])

    def _locate(self, x: np.ndarray):
        x_flat = np.clip(np.asarray(x, dtype=float).ravel(), 0.0, self.L)
        idx = np.searchsorted(self.s, x_flat, side="right") - 1
        idx = np.clip(idx, 0, len(self.seg_len) - 1)
        t = (x_flat - self.s[idx]) / self.seg_len[idx]
        return x_flat, idx, t

    def pts(self, x: np.ndarray) -> np.ndarray:
        """Map x (scalar or array) to field-space coordinates along the path."""
        x_flat, idx, t = self._locate(x)
        phi = self.pts_array[idx] + t[:, None] * self.diffs[idx]
        return phi.reshape(np.asarray(x).shape + (self.pts_array.shape[1],))

    def _unit_vecs(self, idx: np.ndarray) -> np.ndarray:
        return self.diffs[idx] / self.seg_len[idx][:, None]

    def V(self, x: np.ndarray) -> np.ndarray:
        phi = self.pts(x)
        phi_flat = phi.reshape(-1, phi.shape[-1])
        vals = np.array([self.V_field(p) for p in phi_flat])
        return vals.reshape(np.asarray(x).shape)

    def dV(self, x: np.ndarray) -> np.ndarray:
        x_flat, idx, _ = self._locate(x)
        u = self._unit_vecs(idx)
        phi = self.pts(x_flat)
        phi_flat = phi.reshape(-1, phi.shape[-1])
        grad = np.array([self.dV_field(p) for p in phi_flat])
        dVdx = np.einsum("ij,ij->i", grad, u)
        return dVdx.reshape(np.asarray(x).shape)

    def d2V(self, x: np.ndarray, h: float = None) -> np.ndarray:
        if h is None:
            h = 1e-3 * self.L if self.L > 0 else 1e-3
        # Central difference on dV(x)
        return (self.dV(np.asarray(x) + h) - self.dV(np.asarray(x) - h)) / (2.0 * h)


def bounce_profile_from_path(
    path_pts: np.ndarray,
    V,          # callable: V(X) with X shape (..., nfields) -> (...,)
    dV,         # callable: dV(X) with X shape (..., nfields) -> (..., nfields)
    *,
    alpha: float = 3.0,               # 4D O(4) bounce => alpha = 3
    # pass-through to classical_bounce1D.shoot_bounce_1d:
    findProfile_kwargs=None,
):
    findProfile_kwargs = {} if findProfile_kwargs is None else dict(findProfile_kwargs)

    path = _LinearPath1D(path_pts, V, dV)

    # x is the distance along the path; by construction:
    # x=0 at the first point, x=L at the last point.
    x_1 = 0.0
    x_2 = float(path.L)
    V1 = lambda x, params: float(path.V(x))
    dV1 = lambda x, params: float(path.dV(x))

    # Allow users to override integrator settings via findProfile_kwargs.
    rho_max = findProfile_kwargs.pop("rho_max", 20.0)
    n_steps = findProfile_kwargs.pop("n_steps", 2000)
    a0 = findProfile_kwargs.pop("a0", None)
    a1 = findProfile_kwargs.pop("a1", None)
    tol = findProfile_kwargs.pop("tol", 1e-8)
    max_iter = findProfile_kwargs.pop("max_iter", 50)

    if alpha != 3.0:
        # Warn gently via print; classical_bounce1D is hard-coded to alpha=3 (O(4))
        print("Warning: classical_bounce1D uses alpha=3; provided alpha=", alpha)

    sol = shoot_bounce_1d(
        params=(),
        dV_dphi=dV1,
        V=V1,
        phi_FV=x_1,
        phi_TV=x_2,
        rho_max=rho_max,
        n_steps=n_steps,
        a0=a0,
        a1=a1,
        tol=tol,
        max_iter=max_iter,
    )

    Profile1D = namedtuple("Profile1D", ["R", "Phi", "dPhi", "Rerr"])
    profile1D = Profile1D(R=sol.rho, Phi=sol.phi, dPhi=sol.dphi, Rerr=np.zeros_like(sol.rho))

    # Map x(r) back to the full field profile phi_i(r)
    x_of_r = np.asarray(profile1D.Phi, dtype=float)   # this is x(r)
    Phi_fields = path.pts(x_of_r)                     # shape (npoints, nfields)

    return path, sol, profile1D, Phi_fields
