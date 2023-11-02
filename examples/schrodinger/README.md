## Continuous Forward Schrodinger Equation
For the nonlinear Schrodinger equation given by:
$$ih_t + 0.5h_{xx} + {|h|}^2h = 0,$$
$$h(0,x) = 2 \text{sech}(x),$$
$$h(t,-5) = h(t, 5),$$
$$h_x(t,-5) = h_x(t, 5),$$
with $x\in [-5,5]$, $t\in [0,\pi/2]$, and $h(t, x)$ as the complex solution, we partition $h(t, x)$ into its real part $u$ and imaginary part $v$. Thus, our complex-valued neural network representation is $[u(t, x), v(t, x)]$.

### Problem Setup 

| Continuous Forward Schrodinger Equation | |
|------------------------------|---|
| PDE equations | $f_u = u_t + 0.5v_{xx} + v(u^2 +v^2),f_v = v_t + 0.5u_{xx} + u(u^2 +v^2)$|
| Initial conditions | $u(0, x) = 2 \text{sech}(x), v(0, x) = 0$|
| Periodic boundary conditions | $u(t,-5) = u(t, 5), v(t,-5) = v(t, 5), u_x(t,-5) = u_x(t, 5), v_x(t,-5) = v_x(t, 5) $|
| The output of net | $[u(t,x),v(t,x)]$|
| Layers of net | $[2] + 4 \times [100] +[2]$ |
| Sample count from collection points | $20000$ |
| Sample count from the initial condition | $50$ |
| Sample count from boundary conditions | $50$ |
| Loss function | $\text{MSE}_0  + \text{MSE}_b + \text{MSE}_c$ |
