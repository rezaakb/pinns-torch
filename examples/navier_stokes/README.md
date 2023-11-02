## Continuous Inverse Navier-Stokes Equation
Given the 2D nonlinear Navier-Stokes equation:

$$u_t + \lambda_{1}(uu_x + vu_y) = -p_x + \lambda_{2}(u_{xx} + u_{yy}), v_t + \lambda_{1}(uv_x + vv_y) = -p_y + \lambda_{2}(v_{xx} + v_{yy}),$$

where $u(t, x, y)$ and $v(t, x, y)$ are the x and y components of the velocity field, and $p(t, x, y)$ is the pressure, we seek the unknowns $\lambda = (\lambda_1, \lambda_2)$. When required, we integrate the constraints:

$$ 0 = u_x + v_y, u = \psi_y, v = -\psi_x,$$

We use a dual-output neural network to approximate $[\psi(t, x, y), p(t, x, y)]$, leading to a physics-informed neural network $[f(t, x, y), g(t, x, y)]$. 

### Problem Setup 

| Continuous Inverse Navier-Stokes Equation | |
|------------------------------|---|
| PDE equations | $f =  u_t + \lambda_1 (u u_x + v u_y) + p_x - \lambda_2  (u_{xx} + u_{yy}), g = v_t + \lambda_1 (u v_x + v  v_y) + p_y - \lambda_2  (v_{xx} + v_{yy})$ |
| Assumptions | $u = \psi_y, v = -\psi_x$ |
| The output of net | $[\psi(t, x, y), p(t, x, y)]$ |
| Layers of net | $[3] + 8 \times [20] +[2]$ |
| Sample count from collection points | $5000^*$ |
| Sample count from solution | $5000^*$ |
| Loss function | $\text{SSE}_s  + \text{SSE}_c$ |
\* Same points used for collocation and solutions.
