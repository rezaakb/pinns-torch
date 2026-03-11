from typing import Dict, Optional

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

import pinnstorch


C = 1.0  # wave speed


def read_data_fn(root_path):
    """Generate analytical reference data for 1D wave equation.

    Exact solution: u(x, t) = sin(pi*x) * cos(pi*c*t)
    satisfying u_tt = c^2 * u_xx with u(0,t)=u(1,t)=0.
    """
    nx, nt = 256, 201
    x = np.linspace(0, 1, nx)
    t = np.linspace(0, 1, nt)
    X, T = np.meshgrid(x, t, indexing="ij")
    exact_u = np.sin(np.pi * X) * np.cos(np.pi * C * T)
    return {"u": exact_u}


def pde_fn(
    outputs: Dict[str, torch.Tensor],
    x: torch.Tensor,
    t: torch.Tensor,
):
    """Define the wave equation PDE residual: u_tt - c^2 * u_xx = 0."""
    u_x, u_t = pinnstorch.utils.gradient(outputs["u"], [x, t])
    u_xx = pinnstorch.utils.gradient(u_x, x)[0]
    u_tt = pinnstorch.utils.gradient(u_t, t)[0]
    outputs["f"] = u_tt - C**2 * u_xx
    return outputs


def output_fn(
    outputs: Dict[str, torch.Tensor],
    x: torch.Tensor,
    t: torch.Tensor,
):
    """Compute velocity field for initial velocity enforcement."""
    outputs["v"] = pinnstorch.utils.gradient(outputs["u"], t)[0]
    return outputs


@hydra.main(version_base="1.3", config_path="configs", config_name="config.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training."""
    pinnstorch.utils.extras(cfg)
    metric_dict, _ = pinnstorch.train(
        cfg, read_data_fn=read_data_fn, pde_fn=pde_fn, output_fn=output_fn
    )
    metric_value = pinnstorch.utils.get_metric_value(
        metric_dict=metric_dict, metric_names=cfg.get("optimized_metric")
    )
    return metric_value


if __name__ == "__main__":
    main()
