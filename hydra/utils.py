from __future__ import annotations

import importlib
from omegaconf import DictConfig


def instantiate(cfg: DictConfig):
    """Simplified instantiate that supports _target_ and _partial_."""
    if cfg is None:
        return None
    if isinstance(cfg, DictConfig):
        cfg_dict = cfg
    else:
        cfg_dict = DictConfig(cfg)
    if "_target_" not in cfg_dict:
        raise ValueError("Config missing '_target_' field")
    target_str = cfg_dict["_target_"]
    module_path, attr = target_str.rsplit(".", 1)
    module = importlib.import_module(module_path)
    target_cls = getattr(module, attr)
    params = {k: v for k, v in cfg_dict.items() if not k.startswith("_")}
    if cfg_dict.get("_partial_", False):
        def partial_func(*args, **kwargs):
            merged = {**params, **kwargs}
            return target_cls(*args, **merged)
        return partial_func
    return target_cls(**params)
