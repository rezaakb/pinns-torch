## Discrete Forward Allen-Cahn Equation

Given the non-linear AC equation:

$$u_t - 0.0001u_{xx} + 5u^3 - 5u = 0,$$
$$u(0, x) = x^2 \cos(\pi x),$$
$$u(t,-1) = u(t, 1),$$
$$u_x(t,-1) = u_x(t, 1),$$

with $x \in [-1, 1]$ and $t \in [0, 1]$, we adopt Runge–Kutta methods with q stages. The neural network output is:
$$[u^n_1(x),\dots, u^n_q(x), u^n_{q+1}(x)]$$
where $u^n$ is data at time $t^n$. We extract data from the exact solution at $t_0 = 0.1$ aiming to predict the solution at $t_1 = 0.9$ using a single time-step of $\Delta t = 0.8$.

### Problem Setup

| Discrete Forward AC Equation | |
|------------------------------|---|
| PDE equations | $f^{n+c_j} = 5.0 u^{n+c_j} - 5.0 (u^{n+c_j})^3 + 0.0001 u^{n+c_j}_{xx}$ |
| Periodic boundary conditions | $u(t,-1) = u(t, 1), u_{x}(t,-1) = u_{x}(t, 1)$ |
| The output of net | $[u^n_1(x),\dots, u^n_q(x), u^n_{q+1}(x)]$ |
| Layers of net | $[1] + 4 * [200] + [101]$ |
| The number of stages (q) | $100$ |
| Sample count from collection points at $t_0$ | $200^*$ |
| Sample count from solutions at $t_0$ | $200^*$ |
| $t_0 \rightarrow t_1$ | $0.1 \rightarrow 0.9$ |
| Loss function | $\text{SSE}^{0}_s  + \text{SSE}^{0}_c + \text{SSE}^{1}_b$ |
\* Same points used for collocation and solutions.
