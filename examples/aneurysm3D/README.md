## Continuous Forward 3D Navier-Stokes Equation
In this example, the fluid's dynamics are represented by the non-dimensional Navier-Stokes and continuity equations:

$$ c_t + u c_x + v c_y + w c_z = \text{Pec}^{-1} (c_{xx} + c_{yy} + c_{zz}), $$

$$ u_t + u u_x + v u_y + w u_z = - p_x + \text{Re}^{-1} (u_{xx} + u_{yy} + u_{zz}), $$

$$ v_t + u v_x + v v_y + w v_z = - p_y + \text{Re}^{-1} (v_{xx} + v_{yy} + v_{zz}), $$ 

$$ w_t + u w_x + v w_y + w w_z = - p_z + \text{Re}^{-1} (w_{xx} + w_{yy} + w_{zz}), $$

$$ u_x + v_y + w_z = 0 $$

Velocity components are given by $u=(u,v,w)$, and $p$ is the pressure. For the detailed problem setup, please refer to our paper.
