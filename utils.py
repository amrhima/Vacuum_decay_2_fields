from sympy import Symbol, sympify, diff, simplify
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# 1- DE methods for finding green functions
# ---------------------------------------------------------------------------

Array = np.ndarray

@dataclass
class FundamentalMatrix:
    r: Array                 # shape (N,)
    Y: Array                 # shape (N, 2, 2)
    Yp: Array                # shape (N, 2, 2)  (dY/dr)
    l: int

def _rk4_step(f: Callable[[float, Array], Array], r: float, y: Array, h: float) -> Array:
    k1 = f(r, y)
    k2 = f(r + 0.5*h, y + 0.5*h*k1)
    k3 = f(r + 0.5*h, y + 0.5*h*k2)
    k4 = f(r + h, y + h*k3)
    return y + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def _pack(Y: Array, Yp: Array) -> Array:
    # pack 2x2 matrices into a 8-vector: [Y(:), Yp(:)]
    return np.concatenate([Y.reshape(-1), Yp.reshape(-1)])

def _unpack(z: Array) -> Tuple[Array, Array]:
    Y = z[:4].reshape(2, 2)
    Yp = z[4:].reshape(2, 2)
    return Y, Yp
