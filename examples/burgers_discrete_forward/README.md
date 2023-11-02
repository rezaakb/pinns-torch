## Discrete Forward Burgurs' Equation

For this problem, we use data from $t_1 = 0.1$ to predict solutions at $t_2 = 0.9$ utilizing Runge-Kutta methods with q stages. The equation is:
$$f^{n+c_j} = u_t + u^{n+c_j}u_x^{n+c_j} - (0.01/\pi)u_{xx}^{n+c_j}$$
Here, $u^n$ indicates information at time $t^n$.

### Problem Setup

| Discrete Forward Burgurs' Equation | |
|------------------------------|---|
| PDE equations | $f = u_t + uu_x - (0.01 /\pi) u_{xx}$ |
| Dirichlet boundary conditions | $u(t,-1) = u(t, 1) = 0$ |
| The output of net | $[u^n_1(x),\dots, u^n_q(x), u^n_{q+1}(x)]$ |
| Layers of net | $[1] + 3 \times [50] +[501]$ |
| The number of stages (q) | $500$ |
| Sample count from collection points at $t_0$ | $250^*$ |
| Sample count from solutions at $t_0$ | $250^*$ |
| $t_0 \rightarrow t_1$ | $0.1 \rightarrow 0.9$ |
| Loss function | $\text{SSE}^{0}_s  + \text{SSE}^{0}_c + \text{SSE}^{1}_b$ |
\* Same points used for collocation and solutions.
