## Continuous Forward Burgers' Equation
Given the Burgers' equation:

$$ u_t + uu_x - (\dfrac{0.01}{\pi})u_{xx} = 0, $$

with domain $x \in [-1, 1]$ and $t \in [0, 1]$, and the initial and boundary conditions:
$$u(0, x) = -\sin(\pi x), $$
$$u(t,-1) = 0,$$
$$u(t, 1) = 0.$$

### Problem Setup 

| Continuous Forward Burgers' Equation | |
|------------------------------|---|
| PDE equations | $f = u_t + uu_x - (0.01 /π) u_{xx}$ |
| Initial conditions | $u(0, x) = -\text{sin}(π x)$ |
| Dirichlet boundary conditions | $u(t, -1) = u(t, 1) = 0$|
| The output of net | $[u(t, x)]$ |
| Layers of net | $[2] + 8 \times [20] + [1]$ |
| Sample count from collection points | $10000$ |
| Sample count from the initial condition | $50$ |
| Sample count from boundary conditions | $50$ |
| Loss function | $\text{MSE}_0  + \text{MSE}_b + \text{MSE}_c$ |
