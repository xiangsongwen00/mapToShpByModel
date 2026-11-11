from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.gui import PipelineWindow
from getBuildAndRoad.src.pipeline import run


def main() -> int:
    parser = argparse.ArgumentParser(description="道路和建筑生产推理")
    parser.add_argument("--config", default=str(ROOT / "getBuildAndRoad" / "config" / "infer.yaml"))
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
        "阶段三：道路与建筑提取", Path(args.config), run,
        "加载训练阶段导出的运行目录或 best.pt，对 TIFF 加权滑窗推理，并导出概率、掩膜和 GeoPackage。",
    )
    window.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

