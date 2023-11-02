## Continuous Inverse Burgers' Equation
Considering the equation:
$$u_t + \lambda_1uu_x - \lambda_2 u_{xx} = 0,$$
we aim to both predict the solution $u(t, x)$ and determine the unknown parameters $\lambda = (\lambda_1, \lambda_2)$.

### Problem Setup 

| Continuous Inverse Burgers' Equation | |
|------------------------------|---|
| PDE equations | $f = u_t + \lambda_1 uu_x - \lambda_2 u_{xx}$ |
| The output of net | $[u(t,x)]$ |
| Layers of net | $[2] + 8 \times [20] + [1]$ |
| Sample count from collection points | $2000^*$ |
| Sample count from solution | $2000^*$ |
| Loss function | $\text{MSE}_s  + \text{MSE}_c$ |
\* Same points used for collocation and solutions.
