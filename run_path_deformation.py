# run_path_deformation.py

from ct_profile_from_path import ct_find_profile_on_path
import numpy as np
from path_deformation import path_deformation
from plots import plot_tunneling_path
from potential2D import V_func, V_numeric, gradV_func

# mu1, mu2, lam1, lam2, lam12, kappa
params = (1.0, 1.2, 0.4, 0.5, 0.35, 0.1)

Phi_path,fv_point, _, tv_point, _ = path_deformation(params, N=6000, alpha=0.001,max_iter=1200, tol=10e-5)

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

spath, inst, prof1D, Phi_fields = ct_find_profile_on_path(
    path_pts=Phi_path,   # (n_points, 2)
    V=V,              # your V(X)
    dV=grad_V,        # your gradV(X)
    alpha=3.0,
    findProfile_kwargs=dict(npoints=4000, phitol=1e-6, xtol=1e-6, rmax=1e4),
)
r = np.asarray(prof1D.R)

plot_tunneling_path(
    Phi_path=Phi_path,
    tv_point=tv_point,
    fv_point=fv_point,
    V=V_numeric,
    params=params,
    R = None,
    n_grid=6000,
    filename="vacuum_decay_path_2.png",
)