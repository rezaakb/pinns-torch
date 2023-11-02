## Continuous Forward 3D Navier-Stokes Equation
In this example, the fluid's dynamics are represented by the non-dimensional Navier-Stokes and continuity equations:
\begin{align*}
c_t + u c_x + v c_y + w c_z &= \text{Pec}^{-1} (c_{xx} + c_{yy} + c_{zz}),\\
u_t + u u_x + v u_y + w u_z &= - p_x + \text{Re}^{-1} (u_{xx} + u_{yy} + u_{zz}),\\
v_t + u v_x + v v_y + w v_z&= - p_y + \text{Re}^{-1} (v_{xx} + v_{yy} + v_{zz}),\\
w_t + u w_x + v w_y + w w_z &= - p_z + \text{Re}^{-1} (w_{xx} + w_{yy} + w_{zz}),\\
u_x + v_y + w_z &= 0.
\end{align*}

Velocity components are given by $u=(u,v,w)$, and $p$ is the pressure.

### Problem Setup 

| Continuous Forward 3D Navier-Stokes Equation | |
|------------------------------|---|
| PDE equations | \begin{aligned}
        e_1 =& c_t + (u c_x + v c_y + w c_z) \\&- (1.0 / \text{Pec})  (c_xx + c_yy + c_zz) \\
e_2 =& u_t + (u u_x + v u_y + w u_z) + p_x \\&- (1.0 / \text{Re}) (u_xx + u_yy + u_zz) \\
e_3 =& v_t + (u v_x + v v_y + w v_z) + p_y \\&- (1.0 / \text{Re}) (v_xx + v_yy + v_zz) \\
e_4 =& w_t + (u w_x + v w_y + w w_z) + p_z \\&- (1.0 / \text{Re}) (w_xx + w_yy + w_zz) \\
e_5 =& u_x + v_y + w_z
    \end{aligned}  |
| The output of net | $[c(t, x, y, z),u(t, x, y, z),v(t, x, y, z), \\ w(t, x, y, z),p(t, x, y, z)]$ |
| Layers of net | $[4] + 10 \times [250] +[5]$ |
| Batch size of collection points | $10000$ |
| Batch size of solutions in $c(t, x, y, z)$ | $10000$ |
| Loss function | $\text{MSE}_s  + \text{MSE}_c$ |
