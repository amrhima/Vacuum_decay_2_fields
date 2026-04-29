# G_F_/v2 — Bounce-side Green's function and fluctuation determinant

This directory contains two independent numerical methods for computing
the same physical quantity — the fluctuation determinant ratio D and
its building block G_bar(n, s^2) — in the bounce background:

- **`rk_method/`** — Runge-Kutta solution of the radial ODE
- **`fd_method/`** — Finite-difference discretisation of the operator

Shared utilities (potential, bounce profile, RK builders, FV background)
live at the root of `v2/` and are imported by both methods.

## Setup

The scripts read and write data from a directory specified by the
`G_PROJECT_DATA` environment variable. Set it once in your shell:

```sh
export G_PROJECT_DATA=/path/to/your/G_project_data
```

If the variable is unset, scripts fall back to a `data/` subfolder next
to `config.py`.

## Running

All scripts must be launched from `G_F_/v2/` so the shared modules
(`potential.py`, `config.py`, etc.) at the root resolve correctly.

```sh
cd G_F_/v2

# RK method
python rk_method/compute_gbar_n0.py
python rk_method/compute_gbar_n1.py
python rk_method/compute_D_integral.py
python rk_method/plot_D_integral.py

# FD method (v2)
python fd_method/compute_gbar_n0_fd_v2.py
python fd_method/compute_gbar_n1_fd_v2.py
python fd_method/compute_D_integral_fd_v2.py
python fd_method/plot_D_integral_fd_v2.py
```

## Layout

```
v2/
├── config.py                       data-directory resolver (env var)
├── potential.py                    shared: 2-field potential
├── bounce.py                       shared: Euclidean bounce solver
├── rk_builder_adapt_v2.py          shared: RK Green's function builder
├── rk_builder_fv.py                shared: RK builder, false-vacuum background
├── compute_gbar_fv.py              shared: G_bar in the FV background
├── compute_gbar_npos.py            shared: G_bar for n >= 2 (both methods reuse)
│
├── rk_method/
│   ├── compute_D_integral.py       D-integral assembly (RK inputs)
│   ├── compute_gbar_n0.py          G_bar(n=0) via RK + pole fit
│   ├── compute_gbar_n1.py          G_bar(n=1) via RK + zero-mode fit
│   ├── plot_D_integral.py          plots
│   ├── plot_gbar_n0.py
│   └── plot_gbar_n1.py
│
└── fd_method/
    ├── fd_builder_n0_v2.py         FD M-tilde builder for n=0
    ├── fd_builder_n1_v2.py         FD M-tilde builder for n=1
    ├── compute_gbar_n0_fd_v2.py    G_bar(n=0) via FD + projector
    ├── compute_gbar_n1_fd_v2.py    G_bar(n=1) via FD + discrete zero mode
    ├── compute_D_integral_fd_v2.py D-integral assembly (FD inputs)
    ├── plot_D_integral_fd_v2.py    plots
    ├── plot_gbar_n0_fd_v2.py
    └── plot_gbar_n1_fd_v2.py
```
