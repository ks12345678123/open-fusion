import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import open3d as o3d


Rgb = Tuple[int, int, int]


@dataclass
class ObjectCandidate:
    object_id: int
    scene: str
    semantic_label: str
    semantic_index: Optional[int]
    semantic_rgb: List[int]
    point_count: int
    centroid: List[float]
    bbox_min: List[float]
    bbox_max: List[float]
    bbox_size: List[float]
    confidence: float
    visual_rgb: List[int]


def _rgb_tuple(color: Iterable[float]) -> Rgb:
    return tuple(int(v) for v in np.round(np.asarray(color) * 255).clip(0, 255))


def load_label_map(path: Path) -> Dict[Rgb, Tuple[str, Optional[int]]]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    mapping = {}
    for item in data.get("labels", []):
        rgb = tuple(int(v) for v in item["rgb"])
        mapping[rgb] = (item.get("name", f"label_{item.get('index', 'unknown')}"), item.get("index"))
    return mapping


def stable_object_color(object_id: int) -> np.ndarray:
    rng = np.random.default_rng(20260623 + object_id)
    color = rng.uniform(0.15, 1.0, size=3)
    return color.astype(np.float64)


def build_object_map(
    scene_dir: Path,
    output_dir: Path,
    scene_name: Optional[str] = None,
    semantic_ply: str = "semantic_pc.ply",
    label_map_name: str = "semantic_label_map.json",
    voxel_size: float = 0.05,
    dbscan_eps: float = 0.22,
    dbscan_min_points: int = 25,
    min_label_points: int = 200,
    min_cluster_points: int = 80,
) -> Path:
    scene_dir = scene_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scene = scene_name or scene_dir.name
    semantic_path = scene_dir / semantic_ply
    if not semantic_path.exists():
        raise FileNotFoundError(f"Missing semantic point cloud: {semantic_path}")

    pcd = o3d.io.read_point_cloud(str(semantic_path))
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    if len(points) == 0:
        raise ValueError(f"Empty semantic point cloud: {semantic_path}")
    if len(colors) != len(points):
        raise ValueError("Semantic point cloud must contain per-point colors.")

    rgb = np.round(colors * 255).clip(0, 255).astype(np.uint8)
    unique_rgb, inverse, counts = np.unique(rgb, axis=0, return_inverse=True, return_counts=True)
    label_map = load_label_map(scene_dir / label_map_name)

    objects: List[ObjectCandidate] = []
    vis_points: List[np.ndarray] = []
    vis_colors: List[np.ndarray] = []
    label_summaries = []
    object_id = 0

    order = np.argsort(-counts)
    for fallback_index, color_pos in enumerate(order):
        count = int(counts[color_pos])
        color_tuple: Rgb = tuple(int(v) for v in unique_rgb[color_pos])
        label_name, semantic_index = label_map.get(
            color_tuple,
            (f"semantic_color_{fallback_index:03d}", None),
        )
        label_summaries.append(
            {
                "label": label_name,
                "semantic_index": semantic_index,
                "rgb": list(color_tuple),
                "point_count": count,
                "used_for_clustering": count >= min_label_points,
            }
        )
        if count < min_label_points:
            continue

        label_points = points[inverse == color_pos]
        label_pcd = o3d.geometry.PointCloud()
        label_pcd.points = o3d.utility.Vector3dVector(label_points)
        if voxel_size > 0:
            label_pcd = label_pcd.voxel_down_sample(voxel_size)
        down_points = np.asarray(label_pcd.points)
        if len(down_points) < min_cluster_points:
            continue

        cluster_ids = np.asarray(
            label_pcd.cluster_dbscan(
                eps=dbscan_eps,
                min_points=dbscan_min_points,
                print_progress=False,
            )
        )
        for cluster_id in sorted(int(v) for v in np.unique(cluster_ids) if v >= 0):
            cluster_points = down_points[cluster_ids == cluster_id]
            if len(cluster_points) < min_cluster_points:
                continue

            bbox_min = cluster_points.min(axis=0)
            bbox_max = cluster_points.max(axis=0)
            bbox_size = bbox_max - bbox_min
            centroid = cluster_points.mean(axis=0)
            confidence = min(1.0, len(cluster_points) / max(1, count))
            obj_color = stable_object_color(object_id)
            obj = ObjectCandidate(
                object_id=object_id,
                scene=scene,
                semantic_label=label_name,
                semantic_index=semantic_index,
                semantic_rgb=list(color_tuple),
                point_count=int(len(cluster_points)),
                centroid=[round(float(v), 5) for v in centroid],
                bbox_min=[round(float(v), 5) for v in bbox_min],
                bbox_max=[round(float(v), 5) for v in bbox_max],
                bbox_size=[round(float(v), 5) for v in bbox_size],
                confidence=round(float(confidence), 5),
                visual_rgb=[int(round(channel * 255)) for channel in obj_color],
            )
            objects.append(obj)

            vis_points.append(cluster_points)
            vis_colors.append(np.repeat(obj_color[None, :], len(cluster_points), axis=0))
            object_id += 1

    output_json = output_dir / f"{scene}_object_map.json"
    output_ply = output_dir / f"{scene}_objects.ply"

    payload = {
        "scene": scene,
        "source_scene_dir": str(scene_dir),
        "source_semantic_ply": str(semantic_path),
        "label_map": str(scene_dir / label_map_name) if (scene_dir / label_map_name).exists() else None,
        "parameters": {
            "voxel_size": voxel_size,
            "dbscan_eps": dbscan_eps,
            "dbscan_min_points": dbscan_min_points,
            "min_label_points": min_label_points,
            "min_cluster_points": min_cluster_points,
        },
        "semantic_color_summary": label_summaries,
        "object_count": len(objects),
        "objects": [asdict(obj) for obj in objects],
    }
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    if vis_points:
        out_pcd = o3d.geometry.PointCloud()
        out_pcd.points = o3d.utility.Vector3dVector(np.concatenate(vis_points, axis=0))
        out_pcd.colors = o3d.utility.Vector3dVector(np.concatenate(vis_colors, axis=0))
        o3d.io.write_point_cloud(str(output_ply), out_pcd)

    return output_json


def parse_args():
    parser = argparse.ArgumentParser(description="Build an object-level map from Open-Fusion semantic point clouds.")
    parser.add_argument("--scene-dir", required=True, help="Directory containing semantic_pc.ply.")
    parser.add_argument("--output-dir", default="outputs/object_maps", help="Directory for object map outputs.")
    parser.add_argument("--scene-name", default=None, help="Override scene name used in output filenames.")
    parser.add_argument("--voxel-size", type=float, default=0.05)
    parser.add_argument("--dbscan-eps", type=float, default=0.22)
    parser.add_argument("--dbscan-min-points", type=int, default=25)
    parser.add_argument("--min-label-points", type=int, default=200)
    parser.add_argument("--min-cluster-points", type=int, default=80)
    return parser.parse_args()


def main():
    args = parse_args()
    output_json = build_object_map(
        scene_dir=Path(args.scene_dir),
        output_dir=Path(args.output_dir),
        scene_name=args.scene_name,
        voxel_size=args.voxel_size,
        dbscan_eps=args.dbscan_eps,
        dbscan_min_points=args.dbscan_min_points,
        min_label_points=args.min_label_points,
        min_cluster_points=args.min_cluster_points,
    )
    with output_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    print(f"scene={payload['scene']}")
    print(f"object_count={payload['object_count']}")
    print(f"object_map={output_json}")


if __name__ == "__main__":
    main()
