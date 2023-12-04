from typing import Any, Dict, List, Optional, Tuple

import hydra
import numpy as np
import torch

import pinnstorch

from omegaconf import DictConfig


def read_data_fn(root_path):
    """Read and preprocess data from the specified root path.

    :param root_path: The root directory containing the data.
    :return: Processed data in the form of a PointCloudData object.
    """

    data = pinnstorch.utils.load_data(root_path, "Aneurysm3D.mat")

    t_star = data["t_star"]  # T x 1
    x_star = data["x_star"]  # N x 1
    y_star = data["y_star"]  # N x 1
    z_star = data["z_star"]  # N x 1

    U_star = data["U_star"]  # N x T
    V_star = data["V_star"]  # N x T
    W_star = data["W_star"]  # N x T
    P_star = data["P_star"]  # N x T
    C_star = data["C_star"]  # N x T

    return pinnstorch.data.PointCloudData(
        spatial=[x_star, y_star, z_star],
        time=[t_star],
        solution={"u": U_star, "v": V_star, "w": W_star, "p": P_star, "c": C_star},
    )

def pde_fn(outputs: Dict[str, torch.Tensor],
           x: torch.Tensor,
           y: torch.Tensor,
           z: torch.Tensor,
           t: torch.Tensor):   
    """Define the partial differential equations (PDEs).

    :param outputs: Dictionary containing the network outputs for different variables.
    :param x: Spatial coordinate x.
    :param y: Spatial coordinate y.
    :param z: Spatial coordinate z.
    :param t: Temporal coordinate t.
    :param extra_variables: Additional variables if available (optional).
    :return: Dictionary of computed PDE terms for each variable.
    """

    Pec = 1.0 / 0.0101822
    Rey = 1.0 / 0.0101822

    Y = torch.cat([outputs["c"], outputs["u"], outputs["v"], outputs["w"], outputs["p"]], 1)

    Y_t, Y_x, Y_y, Y_z = pinnstorch.utils.fwd_gradient(Y, [t, x, y, z])

    Y_xx = pinnstorch.utils.fwd_gradient(Y_x, x)[0]
    Y_yy = pinnstorch.utils.fwd_gradient(Y_y, y)[0]
    Y_zz = pinnstorch.utils.fwd_gradient(Y_z, z)[0]

    c, u, v, w, p = torch.split(Y, (1), dim=1)

    c_t, u_t, v_t, w_t, _ = torch.split(Y_t, (1), dim=1)
    c_x, u_x, v_x, w_x, p_x = torch.split(Y_x, (1), dim=1)
    c_y, u_y, v_y, w_y, p_y = torch.split(Y_y, (1), dim=1)
    c_z, u_z, v_z, w_z, p_z = torch.split(Y_z, (1), dim=1)

    c_xx, u_xx, v_xx, w_xx, _ = torch.split(Y_xx, (1), dim=1)
    c_yy, u_yy, v_yy, w_yy, _ = torch.split(Y_yy, (1), dim=1)
    c_zz, u_zz, v_zz, w_zz, _ = torch.split(Y_zz, (1), dim=1)

    outputs["e1"] = c_t + (u * c_x + v * c_y + w * c_z) - (1.0 / Pec) * (c_xx + c_yy + c_zz)
    outputs["e2"] = u_t + (u * u_x + v * u_y + w * u_z) + p_x - (1.0 / Rey) * (u_xx + u_yy + u_zz)
    outputs["e3"] = v_t + (u * v_x + v * v_y + w * v_z) + p_y - (1.0 / Rey) * (v_xx + v_yy + v_zz)
    outputs["e4"] = w_t + (u * w_x + v * w_y + w * w_z) + p_z - (1.0 / Rey) * (w_xx + w_yy + w_zz)
    outputs["e5"] = u_x + v_y + w_z

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
        cfg, read_data_fn=read_data_fn, pde_fn=pde_fn, output_fn=None
    )

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = pinnstorch.utils.get_metric_value(
        metric_dict=metric_dict, metric_names=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
