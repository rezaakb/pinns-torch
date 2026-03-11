# Wave Equation

## Problem

Solves the 1D wave equation using a physics-informed neural network:

$$u_{tt} = c^2 u_{xx}, \quad x \in [0, 1], \quad t \in [0, 1]$$

with Dirichlet boundary conditions $u(0, t) = u(1, t) = 0$ and initial conditions:

- $u(x, 0) = \sin(\pi x)$
- $u_t(x, 0) = 0$

The exact solution is $u(x, t) = \sin(\pi x)\cos(\pi c t)$ with wave speed $c = 1$.

## Features

- Demonstrates PINN training on a **hyperbolic** PDE (wave propagation)
- Reference data generated analytically (no `.mat` file needed)
- Enforces both initial displacement and zero initial velocity via `output_fn`
- Complements the existing parabolic (heat/diffusion) and dispersive (Schrödinger, KdV) examples

## Usage

```bash
python train.py
```
