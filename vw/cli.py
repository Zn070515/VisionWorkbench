"""VisionWorkbench CLI 入口."""

import subprocess
import sys
from pathlib import Path

import vw


def cmd_start():
    """启动 VisionWorkbench Web UI."""
    main_py = Path(__file__).parent / "main.py"
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(main_py)],
        check=False,
    )


def cmd_version():
    """显示版本."""
    print(f"VisionWorkbench v{vw.__version__}")


HELP_TEXT = """VisionWorkbench — 轻量级本地 AI 视觉研发工作台

命令:
  start     启动 Web UI
  version   显示版本信息
  help      显示此帮助
"""


def main():
    if len(sys.argv) < 2:
        print(HELP_TEXT)
        return

    cmd = sys.argv[1]
    if cmd == "start":
        cmd_start()
    elif cmd == "version":
        cmd_version()
    elif cmd in ("help", "-h", "--help"):
        print(HELP_TEXT)
    else:
        print(f"未知命令: {cmd}")
        print(HELP_TEXT)


if __name__ == "__main__":
    main()
