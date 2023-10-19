## Non-Linear AC Equation

Given the non-linear AC equation:

![equation](https://latex.codecogs.com/svg.latex?u_t%20-%200.0001u_{xx}%20&plus;%205u^3%20-%205u%20=%200)

with the initial and boundary conditions:

![equation](https://latex.codecogs.com/svg.latex?u(0,%20x)%20=%20x^2%20\cos(\pi%20x))

![equation](https://latex.codecogs.com/svg.latex?u(t,-1)%20=%20u(t,%201))

![equation](https://latex.codecogs.com/svg.latex?u_x(t,-1)%20=%20u_x(t,%201))

where \(x \in [-1, 1]\) and \(t \in [0, 1]\). We adopt Runge–Kutta methods with \(q\) stages as described in [1,2]. The neural network output is:

![equation](https://latex.codecogs.com/svg.latex?[u^n_1(x),\dots,%20u^n_q(x),%20u^n_{q+1}(x)])

where \(u^n\) is data at time \(t^n\). We extract data from the exact solution at \(t_0 = 0.1\) aiming to predict the solution at \(t_1 = 0.9\) using a single time-step of \(\Delta t = 0.8\). The \(l_2\)-norm errors for \(u(x)\) at \(t_1\) are shown in the table below.

### Problem Setup for Discrete Forward Allen-Cahn Equation

| Discrete Forward AC Equation | |
|------------------------------|---|
| PDE equations | ![equation](https://latex.codecogs.com/svg.latex?f^{n+c_j}%20=%205.0%20u^{n+c_j}%20-%205.0%20(u^{n+c_j})^3%20&plus;%200.0001%20u^{n+c_j}_{xx}) |
| Periodic boundary conditions | ![equation](https://latex.codecogs.com/svg.latex?u(t,-1)%20=%20u(t,%201),%20u_x(t,-1)%20=%20u_x(t,%201)) |
| The output of net | ![equation](https://latex.codecogs.com/svg.latex?[u^n_1(x),\dots,%20u^n_q(x),%20u^n_{q&plus;1}(x)]) |
| Layers of net | \([1] + 4 \times [200] +[101]\) |
| The number of stages (q) | 100 |
| Sample count from collection points at \(t_0\) | \(200^*\) |
| Sample count from solutions at \(t_0\) | \(200^*\) |
| \(t_0 \rightarrow t_1\) | \(0.1 \rightarrow 0.9\) |
| Loss function | \(\text{SSE}^{0}_s  + \text{SSE}^{0}_c + \text{SSE}^{1}_b\) |

\* Same points used for collocation and solutions.

### References

1. Raissi, Maziar, Paris Perdikaris, and George Em Karniadakis. "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations." Journal of Computational Physics 378 (2019): 686-707.

2. Iserles, A., and S. P. Nørsett. "Efficient implementation of Runge-Kutta methods with many stages." IMA Journal of Numerical Analysis 30.4 (2010): 1003-1018.