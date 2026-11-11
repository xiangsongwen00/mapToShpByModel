from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.gui import PipelineWindow
from makeLabel.src.pipeline import run
from makeLabel.src.osm_downloader import convert_legacy_json, run as run_osm_download


def main() -> int:
    parser = argparse.ArgumentParser(description="OSM 预标注、栅格化和数据集切片")
    parser.add_argument("--config", default=None)
    parser.add_argument("--no-gui", action="store_true")
    osm_actions = parser.add_mutually_exclusive_group()
    osm_actions.add_argument("--download-osm", action="store_true", help="仅下载 OSM 建筑/道路候选图斑")
    osm_actions.add_argument("--convert-osm", action="store_true", help="把旧 JSON OSM 缓存转换为 GeoPackage")
    parser.add_argument("--check", action="store_true", help="仅检查配置与导入")
    args = parser.parse_args()
    config_path = args.config or str(
        ROOT / "makeLabel" / "config" / ("osm_download.yaml" if args.download_osm or args.convert_osm else "default.yaml")
    )
    if args.check:
        from common.config import load_yaml
        print(load_yaml(config_path))
        return 0
    if args.no_gui:
        runner = convert_legacy_json if args.convert_osm else run_osm_download if args.download_osm else run
        runner(config_path)
        return 0
    if args.convert_osm:
        window = PipelineWindow(
            "OSM 数据标准化：JSON 转 GeoPackage", Path(config_path), convert_legacy_json,
            "把旧 Overpass JSON 分块转换成 GeoPackage，重新汇总标准图层；全部校验成功后移除旧 JSON。",
        )
        window.mainloop()
        return 0
    if args.download_osm:
        window = PipelineWindow(
            "标注准备：下载 OSM 建筑与道路图斑",
            Path(config_path),
            run_osm_download,
            "按每幅 TIFF 范围分块下载 OSM；支持缓存、断点续传、要素去重和备用节点。输出建筑面、道路中心线、道路面三层 GeoPackage，不覆盖人工标注。",
        )
        window.mainloop()
        return 0
    window = PipelineWindow(
        "阶段一：标注与数据集制作",
        Path(config_path),
        run,
        "生成 OSM 候选标注、全图掩膜和训练切片。OSM 结果只是候选，正式训练前请在 GIS 工具中复核 data/labels/vectors。",
    )
    window.add_pipeline_action(
        "下载 OSM 图斑",
        "标注准备：下载 OSM 建筑与道路图斑",
        ROOT / "makeLabel" / "config" / "osm_download.yaml",
        run_osm_download,
        "按 TIFF 范围分块下载并缓存 OSM，输出 building、road_centerline、road_surface 三个候选图层。",
    )
    window.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
