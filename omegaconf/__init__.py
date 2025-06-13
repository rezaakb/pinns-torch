from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List


class DictConfig(dict):
    pass


@contextmanager
def open_dict(cfg: DictConfig):
    yield cfg


class OmegaConf:
    _resolvers: Dict[str, Any] = {}

    @staticmethod
    def register_new_resolver(name: str, resolver):
        OmegaConf._resolvers[name] = resolver

    @staticmethod
    def load(path: str | Path) -> DictConfig:
        text = Path(path).read_text()
        try:
            data = json.loads(text)
        except Exception:
            # extremely naive YAML -> JSON conversion
            import yaml as _yaml  # type: ignore
            data = _yaml.safe_load(text)
        return DictConfig(data)

    @staticmethod
    def merge(*configs: DictConfig) -> DictConfig:
        result = DictConfig()
        for cfg in configs:
            result.update(cfg)
        return result

    @staticmethod
    def from_dotlist(items: Iterable[str]) -> DictConfig:
        cfg = DictConfig()
        for item in items:
            if '=' in item:
                key, value = item.split('=', 1)
            else:
                key, value = item, 'true'
            sub = cfg
            parts = key.split('.')
            for p in parts[:-1]:
                sub = sub.setdefault(p, DictConfig())
            sub[parts[-1]] = OmegaConf._parse_value(value)
        return cfg

    @staticmethod
    def _parse_value(value: str) -> Any:
        if value.lower() in {"true", "false"}:
            return value.lower() == "true"
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
        return value
