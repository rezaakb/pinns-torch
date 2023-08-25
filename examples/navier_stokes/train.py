from typing import Any, Dict, List, Optional, Tuple

import hydra
import numpy as np
import rootutils
import torch
from omegaconf import DictConfig

import pinnstorch


def read_data_fn(root_path):
    """Read and preprocess data from the specified root path.

    :param root_path: The root directory containing the data.
    :return: Processed data in the form of a PointCloudData object.
    """
    
    data = pinnstorch.utils.load_data(root_path, "cylinder_nektar_wake.mat")
    x = data["X_star"][:, 0:1]  # N x 1
    y = data["X_star"][:, 1:2]  # N x 1
    t = data["t"]  # T x 1
    U_star = data["U_star"]  # N x 2 x T
    exact_u = U_star[:, 0, :]  # N x T
    exact_v = U_star[:, 1, :]  # N x T
    exact_p = data["p_star"]  # N x T
    return pinnstorch.data.PointCloudData(
        spatial=[x, y], time=[t], solution={"u": exact_u, "v": exact_v, "p": exact_p}
    )


def output_fn(outputs, x, y, t):
    """Define `output_fn` funtion that will be applied to outputs of net.
    """
    outputs["u"] = pinnstorch.utils.gradient(outputs["psi"], y)
    outputs["v"] = -pinnstorch.utils.gradient(outputs["psi"], x)

    return outputs


def pde_fn(outputs, x, y, t, extra_variables):
    """Define the partial differential equations (PDEs).
    """
    
    u_x, u_y, u_t = pinnstorch.utils.gradient(outputs["u"], [x, y, t])
    u_xx = pinnstorch.utils.gradient(u_x, x)
    u_yy = pinnstorch.utils.gradient(u_y, y)

    v_x, v_y, v_t = pinnstorch.utils.gradient(outputs["v"], [x, y, t])
    v_xx = pinnstorch.utils.gradient(v_x, x)
    v_yy = pinnstorch.utils.gradient(v_y, y)

    p_x, p_y = pinnstorch.utils.gradient(outputs["p"], [x, y])

    outputs["f_u"] = (
        u_t
        + extra_variables["l1"] * (outputs["u"] * u_x + outputs["v"] * u_y)
        + p_x
        - extra_variables["l2"] * (u_xx + u_yy)
    )

    outputs["f_v"] = (
        v_t
        + extra_variables["l1"] * (outputs["u"] * v_x + outputs["v"] * v_y)
        + p_y
        - extra_variables["l2"] * (v_xx + v_yy)
    )

    return outputs


@hydra.main(version_base="1.3", config_path="configs", config_name="config.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """

    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    pinnstorch.utils.extras(cfg)

    # train the model
    metric_dict, _ = pinnstorch.train(
        cfg, read_data_fn=read_data_fn, pde_fn=pde_fn, output_fn=output_fn
    )

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = pinnstorch.utils.get_metric_value(
        metric_dict=metric_dict, metric_names=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
