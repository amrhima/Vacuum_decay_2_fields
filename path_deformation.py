from tkinter import constants
import numpy as np
from potential2D import find_vacua_from_potential, gradV_numeric

def path_deformation(
    params,
    N=100,
    alpha=0.01,
    max_iter=200,
    tol=1e-3,
):
    fv_point, phi_FV, tv_point, phi_TV = find_vacua_from_potential(params)
    # initial straight-line path
    # initial straight-line path
    s = np.linspace(0.0, 1.0, N)
    Phi = np.zeros((N, 2))
    Phi[0]  = tv_point
    Phi[-1] = fv_point
    for i in range(1, N-1):
        Phi[i] = (1 - s[i]) * phi_TV + s[i] * phi_FV

    gradV = gradV_numeric
    for it in range(max_iter):
        Phi, grad, grad_perp = deform_path(Phi, params, alpha, gradV)
        Phi = reparametrize(Phi)

        # measure convergence: max |F_perp| / |grad|
        grad_norm = np.linalg.norm(grad, axis=1)
        perp_norm = np.linalg.norm(grad_perp, axis=1)
        # avoid div by zero
        mask = grad_norm > 0
        ratio = np.zeros_like(grad_norm)
        ratio[mask] = perp_norm[mask] / grad_norm[mask]
        max_ratio = np.max(ratio)

        print(f"iter {it}: max fRatio = {max_ratio:.3e}")

        if max_ratio < tol:
            print("Converged.")
            break

    return Phi, fv_point, phi_FV, tv_point, phi_TV

def compute_tangent(Phi):
    # Phi shape (N, 2)
    N = Phi.shape[0]
    t = np.zeros_like(Phi)
    # interior points: centered differences
    t[1:-1] = Phi[2:] - Phi[:-2]
    # endpoints: one-sided
    t[0]  = Phi[1] - Phi[0]
    t[-1] = Phi[-1] - Phi[-2]
    # normalize
    norms = np.linalg.norm(t, axis=1, keepdims=True)
    # avoid division by zero
    norms[norms == 0.0] = 1.0
    t /= norms
    return t

def compute_forces(Phi, params, gradV):
    N = Phi.shape[0]
    grad = np.zeros_like(Phi)
    for i in range(N):
        grad[i] = gradV(Phi[i], params)  # shape (2,)
    t = compute_tangent(Phi)
    # project grad onto tangent: (grad·t) t
    dot = np.sum(grad * t, axis=1, keepdims=True)
    grad_parallel = dot * t
    grad_perp = grad - grad_parallel
    return grad, grad_perp

def deform_path(Phi, params, alpha, gradV):
    Phi_new = Phi.copy()
    grad, grad_perp = compute_forces(Phi, params, gradV)
    # endpoints fixed
    Phi_new[1:-1] = Phi[1:-1] - alpha * grad_perp[1:-1]
    return Phi_new, grad, grad_perp

def reparametrize(Phi):
    # compute cumulative arc-length
    dPhi = Phi[1:] - Phi[:-1]
    seglen = np.linalg.norm(dPhi, axis=1)
    s = np.zeros(Phi.shape[0])
    s[1:] = np.cumsum(seglen)
    # normalize to [0,1]
    total = s[-1]
    if total == 0.0:
        return Phi  # path collapsed; nothing to do
    s /= total

    # new uniform parameter
    s_uniform = np.linspace(0.0, 1.0, Phi.shape[0])

    # interpolate each component
    Phi_new = np.zeros_like(Phi)
    for dim in range(Phi.shape[1]):
        Phi_new[:, dim] = np.interp(s_uniform, s, Phi[:, dim])

    return Phi_new
