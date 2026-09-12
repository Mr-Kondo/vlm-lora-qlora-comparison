"""YAML configuration loading with single-level inheritance and CLI overrides.

Both training methods and all three evaluation variants read their settings
through this module, so "LoRA and QLoRA used the same value for X" is a
property of the config files rather than of duplicated Python code.
"""

from __future__ import annotations

import copy
import json
import os
from typing import Any, Iterable, Mapping

import yaml

_MISSING = object()


class ConfigError(ValueError):
    """Raised when a configuration file is malformed or a key is missing."""


class Config(Mapping):
    """Read-only nested view over a config dict with attribute access.

    ``cfg.training.learning_rate`` and ``cfg["training"]["learning_rate"]`` are
    equivalent. Missing keys raise :class:`ConfigError` instead of returning
    ``None``, so a typo in a config file fails loudly at start-up rather than
    silently changing an experimental condition.
    """

    def __init__(self, data: Mapping[str, Any], path: str = ""):
        self._data = dict(data)
        self._path = path

    # -- Mapping protocol -------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        try:
            value = self._data[key]
        except KeyError:
            raise ConfigError(f"missing config key: {self._qualify(key)}") from None
        if isinstance(value, dict):
            return Config(value, self._qualify(key))
        return value

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __getattr__(self, key: str) -> Any:
        if key.startswith("_"):
            raise AttributeError(key)
        return self[key]

    def __repr__(self) -> str:
        return f"Config({json.dumps(self._data, default=str, sort_keys=True)})"

    # -- helpers ----------------------------------------------------------
    def _qualify(self, key: str) -> str:
        return f"{self._path}.{key}" if self._path else key

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except ConfigError:
            return default

    def to_dict(self) -> dict:
        """Return a deep copy as plain Python containers (JSON serializable)."""
        return copy.deepcopy(self._data)


def _deep_merge(base: dict, override: Mapping[str, Any]) -> dict:
    """Recursively merge ``override`` into ``base``, returning a new dict."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top-level YAML must be a mapping")
    return data


def load_config(path: str, overrides: Iterable[str] = ()) -> Config:
    """Load ``path``, resolving a single ``extends:`` parent and CLI overrides.

    ``extends`` is resolved relative to the directory of ``path``. Overrides are
    ``dotted.key=value`` strings; values are parsed as YAML scalars so
    ``training.learning_rate=1e-4`` and ``data.max_train_samples=null`` behave
    as expected.
    """
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise ConfigError(f"config file not found: {path}")

    child = _load_yaml(path)
    parent_ref = child.pop("extends", None)
    if parent_ref:
        parent_path = os.path.join(os.path.dirname(path), str(parent_ref))
        parent = _load_yaml(parent_path)
        if "extends" in parent:
            raise ConfigError(
                f"{parent_path}: nested 'extends' is not supported (keep inheritance one level deep)"
            )
        data = _deep_merge(parent, child)
        data["_parent_config"] = os.path.relpath(parent_path, os.path.dirname(path))
    else:
        data = child

    for override in overrides:
        data = _apply_override(data, override)

    data["_config_path"] = path
    return Config(data)


def _apply_override(data: dict, override: str) -> dict:
    if "=" not in override:
        raise ConfigError(f"override must look like key.path=value, got: {override!r}")
    dotted, raw = override.split("=", 1)
    keys = [k for k in dotted.strip().split(".") if k]
    if not keys:
        raise ConfigError(f"override has an empty key path: {override!r}")
    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ConfigError(f"could not parse override value in {override!r}: {exc}") from exc

    node = data
    for key in keys[:-1]:
        nxt = node.get(key, _MISSING)
        if nxt is _MISSING or not isinstance(nxt, dict):
            node[key] = {}
        node = node[key]
    node[keys[-1]] = value
    return data


def config_fingerprint(cfg: Config, exclude: Iterable[str] = ()) -> str:
    """Stable hash of the configuration, used to detect accidental drift.

    Keys listed in ``exclude`` (dotted paths) are removed first, so the LoRA and
    QLoRA runs can be compared on everything *except* the keys that are
    intentionally different.
    """
    import hashlib

    data = cfg.to_dict()
    data.pop("_config_path", None)
    data.pop("_parent_config", None)
    for dotted in exclude:
        keys = dotted.split(".")
        node = data
        for key in keys[:-1]:
            node = node.get(key) if isinstance(node, dict) else None
            if node is None:
                break
        if isinstance(node, dict):
            node.pop(keys[-1], None)
    payload = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]
