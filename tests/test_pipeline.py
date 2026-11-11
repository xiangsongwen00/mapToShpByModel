from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio
import torch
import yaml
from PIL import Image
from rasterio.transform import from_origin
from shapely.geometry import Polygon

from common.geo import sliding_windows
from common.model import TinyUNet
from getBuildAndRoad.src.pipeline import run as run_inference
from makeLabel.src.pipeline import run as run_labeling
from makeLabel.src.building_geometry import roof_to_base
from makeLabel.src.osm_downloader import make_road_surfaces, parse_payloads, split_bounds
from trainModel.src.pipeline import _loss, run as run_training


class CoreTests(unittest.TestCase):
    def test_partial_label_unknown_pixel_has_no_gradient(self):
        logits = torch.zeros((1, 1, 1, 2), requires_grad=True)
        mask = torch.tensor([[[1, 255]]])
        loss = _loss(logits, mask, torch.tensor([1.0]), torch.tensor([0.0]))
        loss.backward()
        self.assertNotEqual(float(logits.grad[0, 0, 0, 0]), 0.0)
        self.assertEqual(float(logits.grad[0, 0, 0, 1]), 0.0)

    def test_roof_to_base_pixel_shift(self):
        class RasterStub:
            crs = "EPSG:3857"
            transform = from_origin(0, 100, 1, 1)

        roofs = gpd.GeoDataFrame(
            {"geometry": [Polygon([(10, 80), (20, 80), (20, 90), (10, 90)])]}, crs="EPSG:3857"
        )
        bases = roof_to_base(
            roofs, RasterStub(),
            {"roof_to_base": {"mode": "fixed_pixels", "dx_pixels": 2, "dy_pixels": 3}}, "area",
        )
        self.assertEqual(tuple(round(value, 3) for value in bases.total_bounds), (12.0, 77.0, 22.0, 87.0))

    def test_roof_cannot_be_silently_used_as_base(self):
        class RasterStub:
            crs = "EPSG:3857"
            transform = from_origin(0, 100, 1, 1)

        roofs = gpd.GeoDataFrame(
            {"geometry": [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]}, crs="EPSG:3857"
        )
        with self.assertRaises(ValueError):
            roof_to_base(roofs, RasterStub(), {"roof_to_base": {"mode": "none"}}, "area")

    def test_roof_to_base_angle_height_shift(self):
        class RasterStub:
            crs = "EPSG:32648"
            transform = from_origin(200000, 3300000, 0.5, 0.5)

        roofs = gpd.GeoDataFrame(
            {"height_m": [10.0], "geometry": [Polygon([
                (200100, 3299900), (200110, 3299900), (200110, 3299910), (200100, 3299910),
            ])]},
            crs="EPSG:32648",
        )
        base = roof_to_base(roofs, RasterStub(), {"roof_to_base": {
            "mode": "angle_height", "off_nadir_deg": 45,
            "correction_azimuth_deg": 90, "height_field": "height_m",
        }}, "area")
        # Reprojection via the locally estimated UTM zone can introduce centimetre-scale differences.
        self.assertAlmostEqual(base.geometry.iloc[0].centroid.x - roofs.geometry.iloc[0].centroid.x, 10.0, delta=0.05)
        self.assertAlmostEqual(base.geometry.iloc[0].centroid.y - roofs.geometry.iloc[0].centroid.y, 0.0, delta=0.05)

    def test_osm_grid_parse_and_road_surface(self):
        cells = split_bounds((106.0, 29.0, 106.031, 29.021), 0.015)
        self.assertEqual(len(cells), 6)
        building_payload = {"elements": [{
            "type": "way", "id": 1, "tags": {"building": "yes"},
            "geometry": [
                {"lon": 106.0, "lat": 29.0}, {"lon": 106.001, "lat": 29.0},
                {"lon": 106.001, "lat": 29.001}, {"lon": 106.0, "lat": 29.001},
                {"lon": 106.0, "lat": 29.0},
            ],
        }]}
        road_payload = {"elements": [{
            "type": "way", "id": 2, "tags": {"highway": "residential"},
            "geometry": [{"lon": 106.0, "lat": 29.0}, {"lon": 106.01, "lat": 29.01}],
        }]}
        buildings, roads = parse_payloads([building_payload, building_payload], [road_payload, road_payload])
        self.assertEqual(len(buildings), 1)
        self.assertEqual(len(roads), 1)
        surfaces = make_road_surfaces(roads, {"residential": 6}, 5)
        self.assertEqual(len(surfaces), 1)
        self.assertEqual(float(surfaces.iloc[0].width_m), 6.0)

    def test_sliding_windows_cover_edges(self):
        windows = list(sliding_windows(100, 70, 32, 16))
        self.assertEqual(int(windows[-1].col_off + windows[-1].width), 100)
        self.assertEqual(int(windows[-1].row_off + windows[-1].height), 70)

    def test_model_preserves_shape(self):
        model = TinyUNet(base_channels=4)
        output = model(torch.zeros(2, 3, 65, 67))
        self.assertEqual(tuple(output.shape), (2, 1, 65, 67))


class EndToEndSmokeTest(unittest.TestCase):
    def test_label_pipeline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_dir = root / "data" / "tiff"
            vector_dir = root / "data" / "labels" / "vectors"
            image_dir.mkdir(parents=True)
            vector_dir.mkdir(parents=True)
            tif = image_dir / "area.tif"
            with rasterio.open(
                tif, "w", driver="GTiff", width=48, height=48, count=3, dtype="uint8",
                crs="EPSG:3857", transform=from_origin(0, 48, 1, 1),
            ) as dst:
                dst.write(np.full((3, 48, 48), 100, dtype="uint8"))
            buildings = gpd.GeoDataFrame(
                {"kind": ["yes"], "geometry": [Polygon([(8, 8), (24, 8), (24, 24), (8, 24)])]},
                crs="EPSG:3857",
            )
            buildings.to_file(vector_dir / "area.gpkg", layer="building", driver="GPKG")
            label_config = root / "label.yaml"
            label_config.write_text(yaml.safe_dump({
                "project_root": str(root), "input_dir": "data/tiff", "labels_dir": "data/labels",
                "dataset_dir": "data/datasets", "dataset_version": "smoke", "download_osm": False,
                "tile_size": 32, "stride": 16, "split": {"train": 0.7, "val": 0.15, "test": 0.15},
            }), encoding="utf-8")
            result = run_labeling(label_config, lambda _: None)
            self.assertEqual(result["tiles"], 4)
            self.assertGreater(result["positive_building"], 0)
            self.assertTrue(Path(result["manifest"]).exists())

    def test_train_then_infer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "data" / "datasets" / "smoke"
            (dataset / "images").mkdir(parents=True)
            (dataset / "masks_building").mkdir()
            rows = []
            for index, split in enumerate(("train", "train", "val")):
                image = np.zeros((32, 32, 3), dtype="uint8")
                image[8:24, 8:24] = 180 + index
                mask = np.zeros((32, 32), dtype="uint8")
                mask[8:24, 8:24] = 1
                image_path = dataset / "images" / f"{index}.png"
                mask_path = dataset / "masks_building" / f"{index}.png"
                Image.fromarray(image).save(image_path)
                Image.fromarray(mask).save(mask_path)
                rows.append({
                    "sample_id": str(index), "image_path": str(image_path.relative_to(root)),
                    "building_mask_path": str(mask_path.relative_to(root)), "split": split, "scene": "urban",
                })
            manifest = dataset / "manifest.csv"
            with manifest.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            train_config = root / "train.yaml"
            train_config.write_text(yaml.safe_dump({
                "project_root": str(root), "manifest": str(manifest), "task": "building",
                "runs_dir": "runs", "run_id": "smoke", "allow_cpu": True, "device": "cpu",
                "model": {"name": "tiny_unet", "base_channels": 4}, "tile_size": 32,
                "inference_stride": 16, "batch_size": 1, "epochs": 1, "num_workers": 0,
                "gradient_accumulation": 1, "early_stopping_patience": 2,
            }), encoding="utf-8")
            result = run_training(train_config, lambda _: None)
            self.assertTrue(Path(result["best_checkpoint"]).exists())

            input_dir = root / "input"
            input_dir.mkdir()
            tif = input_dir / "sample.tif"
            data = np.zeros((3, 48, 48), dtype="uint8")
            data[:, 12:36, 12:36] = 190
            with rasterio.open(
                tif, "w", driver="GTiff", width=48, height=48, count=3, dtype="uint8",
                crs="EPSG:3857", transform=from_origin(0, 48, 1, 1),
            ) as dst:
                dst.write(data)
            infer_config = root / "infer.yaml"
            infer_config.write_text(yaml.safe_dump({
                "project_root": str(root), "input_dir": str(input_dir), "output_dir": "outputs",
                "job_id": "smoke", "device": "cpu", "batch_size": 1,
                "models": {"building": result["run_dir"], "road": None},
                "postprocess": {"building": {"threshold": 0.5, "min_area_m2": 1.0, "close_gap_m": 0}},
            }), encoding="utf-8")
            inferred = run_inference(infer_config, lambda _: None)
            self.assertEqual(inferred["images"], 1)
            self.assertTrue((Path(inferred["job_dir"]) / "sample" / "building_base_probability.tif").exists())
            self.assertTrue((Path(inferred["job_dir"]) / "sample" / "building_base_mask.tif").exists())


if __name__ == "__main__":
    unittest.main()
