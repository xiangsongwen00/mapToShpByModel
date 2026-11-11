from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.gui import PipelineWindow
from trainModel.src.pipeline import run


def main() -> int:
    parser = argparse.ArgumentParser(description="道路/建筑二值分割模型训练")
    parser.add_argument("--config", default=str(ROOT / "trainModel" / "config" / "building.yaml"))
    parser.add_argument("--no-gui", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        from common.config import load_yaml
        print(load_yaml(args.config))
        return 0
    if args.no_gui:
        run(args.config)
        return 0
    window = PipelineWindow(
        "阶段二：模型训练", Path(args.config), run,
        "分别训练道路或建筑模型。正式训练需要 CUDA 版 PyTorch；窗口在后台线程运行并持续显示训练日志。",
    )
    window.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

