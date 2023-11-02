import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import numpy as np
import pytest
import rootutils
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

import pinnstorch
from pinnstorch.train import train
from tests.helpers.run_if import RunIf


def read_data_fn(root_path):
    """Read and preprocess data from the specified root path.

    :param root_path: The root directory containing the data.
    :return: Processed data will be used in Mesh class.
    """

    data = pinnstorch.utils.load_data(root_path, "KdV.mat")
    exact_u = np.real(data["uu"]).astype("float32")
    return {"u": exact_u}


def pde_fn(outputs, x, extra_variables):
    """Define the partial differential equations (PDEs)."""

    U = outputs["u"]
    U_x = pinnstorch.utils.fwd_gradient(U, x)[0]
    U_xx = pinnstorch.utils.fwd_gradient(U_x, x)[0]
    U_xxx = pinnstorch.utils.fwd_gradient(U_xx, x)[0]
    outputs["f"] = -extra_variables["l1"] * U * U_x - torch.exp(extra_variables["l2"]) * U_xxx
    return outputs


def test_train_config(cfg_train: DictConfig) -> None:
    """Tests the training configuration provided by the `cfg_train` pytest fixture.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    assert cfg_train
    assert cfg_train.data
    assert cfg_train.model
    assert cfg_train.net
    assert cfg_train.trainer

    HydraConfig().set_config(cfg_train)

    net: torch.nn.Module = hydra.utils.instantiate(cfg_train.net)(lb=[0.0, 0.0], ub=[1.0, 1.0])
    hydra.utils.instantiate(cfg_train.model)(net=net, pde_fn=pde_fn, output_fn=None)
    hydra.utils.instantiate(cfg_train.trainer)


def test_eval_config(cfg_eval: DictConfig) -> None:
    """Tests the evaluation configuration provided by the `cfg_eval` pytest fixture.

    :param cfg_train: A DictConfig containing a valid evaluation configuration.
    """
    assert cfg_eval
    assert cfg_eval.data
    assert cfg_eval.model
    assert cfg_eval.trainer

    HydraConfig().set_config(cfg_eval)

    net: torch.nn.Module = hydra.utils.instantiate(cfg_eval.net)(lb=[0.0, 0.0], ub=[1.0, 1.0])
    hydra.utils.instantiate(cfg_eval.model)(net=net, pde_fn=pde_fn, output_fn=None)
    hydra.utils.instantiate(cfg_eval.trainer)
