from typing import Any, Dict, List, Optional, Tuple

import hydra
import numpy as np
import rootutils
import torch
from omegaconf import DictConfig

import pinnstorch


def read_data_fn(root_path):
    data = pinnstorch.utils.load_data(root_path, "burgers_shock.mat")
    exact_u = np.real(data["usol"])
    return {"u": exact_u}


def pde_fn(outputs, x, extra_variables):
    U = outputs["u"]
    U_x = pinnstorch.utils.fwd_gradient(U, x)
    U_xx = pinnstorch.utils.fwd_gradient(U_x, x)
    outputs["f"] = -extra_variables["l1"] * U * U_x + torch.exp(extra_variables["l2"]) * U_xx
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
