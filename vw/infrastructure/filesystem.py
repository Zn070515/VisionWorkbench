"""工作区目录管理."""

from pathlib import Path


def ensure_workspace(base_dir: str | Path) -> Path:
    """确保工作区目录存在并返回路径."""
    base = Path(base_dir).expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)
    return base


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_default_workspace() -> Path:
    """获取默认工作区路径."""
    return Path.home() / "visionworkbench"
