"""Central configuration loader. All modules read settings from here
instead of hardcoding constants."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"


@lru_cache(maxsize=None)
def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    """Load and cache the project configuration.

    Raises FileNotFoundError with a clear message if the config is missing,
    rather than failing downstream with an opaque KeyError.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found at {path}. "
            "Expected config/settings.yaml at the project root."
        )
    with path.open("r") as f:
        config = yaml.safe_load(f)
    if not config:
        raise ValueError(f"Configuration file at {path} is empty or invalid.")
    return config
