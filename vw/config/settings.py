"""配置管理：从 defaults.yaml 加载，支持工作区级别覆盖."""

from pathlib import Path
import yaml


_DEFAULTS_PATH = Path(__file__).parent / "defaults.yaml"


def load_defaults() -> dict:
    with open(_DEFAULTS_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_default(key: str, default=None):
    """点号分隔的 key，如 'inference.confidence'."""
    defaults = load_defaults()
    parts = key.split(".")
    for p in parts:
        if isinstance(defaults, dict):
            defaults = defaults.get(p, {})
        else:
            return default
    return defaults if defaults != {} else default
