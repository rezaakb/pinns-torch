## Discrete Inverse Burgurs' Equation

Similar to its forward counterpart, we utilize Runge-Kutta methods with q stages. The equation here is given by:

$$f^{n+c_j} = u_t + \lambda_1 u^{n+c_j}u_x^{n+c_j} - \lambda_2 u_{xx}^{n+c_j}$$

The goal is to determine $\lambda_1$ and $\lambda_2$. Data points are taken from $t=0.1$ to $t=0.9$.

### Problem Setup

| Discrete Inverse Burgurs' Equation | |
|------------------------------|---|
| PDE equations | $f = u_t + \lambda_1 uu_x - \lambda_2 u_{xx}$ |
| The output of net | $[u^n_1(x),\dots, u^n_q(x), u^n_{q+1}(x)]$ |
| Layers of net | $[1] + 4 \times [50] +[81]$ |
| The number of stages (q) | $81$ |
| Sample count from collection points at $t_0$ | $199^*$ |
| Sample count from solutions at $t_0$ | $199^*$ |
| Sample count from collection points at $t_1$ | $201^*$ |
| Sample count from solutions at $t_1$ | $201^*$ |
| $t_0 \rightarrow t_1$ | $0.1 \rightarrow 0.9$ |
| Loss function | $\text{SSE}^{0}_s  + \text{SSE}^{0}_c + \text{SSE}^{1}_s + \text{SSE}^{1}_c$ |
\* Same points used for collocation and solutions.

