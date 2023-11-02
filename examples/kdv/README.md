## Discrete Inverse Korteweg–de Vries Equation

Given the non-linear KdV equation:

$$ u_t + \lambda_1 uu_x + \lambda_2 u_{xxx} = 0,$$

we use Runge–Kutta methods with q stages to identify parameters $\lambda = (\lambda_1, \lambda_2)$. The network outputs:

$$ [u^n_1(x),\dots, u^n_q(x), u^n_{q+1}(x)] $$

with $u^n$ as data at time $t^n$. Data is sampled at $t^n = 0.2$ and $t^{n+1} = 0.8$.

### Problem Setup

| Discrete Inverse KdV Equation | |
|------------------------------|---|
| PDE equations | $f^{n+c_j} = -\lambda_1 u^{n+c_j}u_x^{n+c_j} - \lambda_2  u^{n+c_j}_{xxx}$ |
| Dirichlet boundary conditions | $u(t,-1) = u(t, 1) = 0$ |
| The output of net | $[u^n_1(x),\dots, u^n_q(x), u^n_{q+1}(x)]$ |
| Layers of net | $[1] + 3 \times [50] +[50]$ |
| The number of stages (q) | $50$ |
| Sample count from collection points at $t_0$ | $250^*$ |
| Sample count from solutions at $t_0$ | $250^*$ |
| $t_0 \rightarrow t_1$ | $0.1 \rightarrow 0.9$ |
| Loss function | $\text{SSE}^{0}_s  + \text{SSE}^{0}_c + \text{SSE}^{1}_b$ |
\* Same points used for collocation and solutions.
