from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

from omegaconf import OmegaConf

_global_config_path: Optional[Path] = None

class _HydraContext:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.prev = None

    def __enter__(self):
        global _global_config_path
        self.prev = _global_config_path
        _global_config_path = self.path

    def __exit__(self, exc_type, exc, tb):
        global _global_config_path
        _global_config_path = self.prev


def initialize(*, version_base: str | None = None, config_path: str | Path) -> _HydraContext:
    """Minimal initialize context manager used in tests."""
    return _HydraContext(config_path)


def compose(*, config_name: str, overrides: Sequence[str] | None = None, return_hydra_config: bool = False):
    """Load a config file and optionally apply overrides."""
    if _global_config_path is None:
        raise RuntimeError("Hydra not initialized")
    cfg_path = (_global_config_path / config_name).resolve()
    cfg = OmegaConf.load(cfg_path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return cfg
