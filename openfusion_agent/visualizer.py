import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import cv2
import numpy as np
import open3d as o3d

from openfusion_agent.object_map import stable_object_color


TARGET_COLORS = [
    np.array([1.0, 0.12, 0.08]),
    np.array([1.0, 0.58, 0.05]),
    np.array([1.0, 0.95, 0.10]),
]
REFERENCE_COLOR = np.array([0.0, 0.78, 1.0])
CONTEXT_COLOR = np.array([0.24, 0.24, 0.24])


def default_objects_ply(object_map_path: Path) -> Path:
    name = object_map_path.name.replace("_object_map.json", "_objects.ply")
    return object_map_path.with_name(name)


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def object_lookup(object_map: Dict) -> Dict[int, Dict]:
    return {int(obj["object_id"]): obj for obj in object_map.get("objects", [])}


def object_color_rgb(obj: Dict) -> Tuple[int, int, int]:
    if "visual_rgb" in obj:
        return tuple(int(v) for v in obj["visual_rgb"])
    color = stable_object_color(int(obj["object_id"]))
    return tuple(int(round(v * 255)) for v in color)


def collect_reference_ids(query_result: Dict) -> Set[int]:
    ref_ids: Set[int] = set()
    for result in query_result.get("results", []):
        for relation in result.get("relations", []):
            ref_id = relation.get("reference_object_id")
            if ref_id is not None:
                ref_ids.add(int(ref_id))
    return ref_ids


def color_mask(colors: np.ndarray, rgb: Tuple[int, int, int], tolerance: int = 1) -> np.ndarray:
    colors_rgb = np.round(colors * 255).astype(np.int16)
    target_rgb = np.asarray(rgb, dtype=np.int16)
    return np.all(np.abs(colors_rgb - target_rgb) <= tolerance, axis=1)


def bbox_mask(points: np.ndarray, obj: Dict, pad: float = 0.03) -> np.ndarray:
    bbox_min = np.asarray(obj["bbox_min"], dtype=np.float64) - pad
    bbox_max = np.asarray(obj["bbox_max"], dtype=np.float64) + pad
    return np.all((points >= bbox_min) & (points <= bbox_max), axis=1)


def make_highlight_cloud(
    object_map: Dict,
    query_result: Dict,
    objects_ply: Path,
) -> Tuple[o3d.geometry.PointCloud, Dict]:
    pcd = o3d.io.read_point_cloud(str(objects_ply))
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    if len(points) == 0:
        raise ValueError(f"Empty object point cloud: {objects_ply}")

    objects = object_lookup(object_map)
    selected_ids = [int(item["object_id"]) for item in query_result.get("results", [])]
    reference_ids = sorted(collect_reference_ids(query_result) - set(selected_ids))

    output_colors = np.repeat(CONTEXT_COLOR[None, :], len(points), axis=0)
    selected_point_counts = {}

    for obj_id in reference_ids:
        obj = objects.get(obj_id)
        if not obj:
            continue
        mask = color_mask(colors, object_color_rgb(obj))
        if not np.any(mask):
            mask = bbox_mask(points, obj)
        output_colors[mask] = REFERENCE_COLOR
        selected_point_counts[f"reference_{obj_id}"] = int(mask.sum())

    for rank, obj_id in enumerate(selected_ids):
        obj = objects.get(obj_id)
        if not obj:
            continue
        mask = color_mask(colors, object_color_rgb(obj))
        if not np.any(mask):
            mask = bbox_mask(points, obj)
        color = TARGET_COLORS[min(rank, len(TARGET_COLORS) - 1)]
        output_colors[mask] = color
        selected_point_counts[f"target_{obj_id}"] = int(mask.sum())

    out_pcd = o3d.geometry.PointCloud()
    out_pcd.points = o3d.utility.Vector3dVector(points)
    out_pcd.colors = o3d.utility.Vector3dVector(output_colors)
    metadata = {
        "selected_object_ids": selected_ids,
        "reference_object_ids": reference_ids,
        "selected_point_counts": selected_point_counts,
    }
    return out_pcd, metadata


def render_projection(points: np.ndarray, colors: np.ndarray, axes: Tuple[int, int, int], title: str, size: int = 900):
    image = np.full((size, size, 3), 24, dtype=np.uint8)
    xy = points[:, axes[:2]]
    depth = points[:, axes[2]]

    lo = np.percentile(xy, 1, axis=0)
    hi = np.percentile(xy, 99, axis=0)
    span = np.maximum(hi - lo, 1e-5)
    pad = span.max() * 0.06
    lo -= pad
    hi += pad
    span = hi - lo

    px = ((xy[:, 0] - lo[0]) / span[0] * (size - 1)).astype(np.int32)
    py = ((xy[:, 1] - lo[1]) / span[1] * (size - 1)).astype(np.int32)
    py = size - 1 - py
    valid = (px >= 0) & (px < size) & (py >= 0) & (py < size)
    order = np.argsort(depth[valid])
    bgr = (np.clip(colors[valid], 0, 1) * 255).astype(np.uint8)[:, ::-1]
    image[py[valid][order], px[valid][order]] = bgr[order]

    mask = np.any(image != 24, axis=2).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(image, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1).astype(bool)
    image[mask] = dilated[mask]
    cv2.putText(image, title, (24, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (235, 235, 235), 2, cv2.LINE_AA)
    return image


def query_caption(query_result: Dict) -> str:
    plan = query_result.get("query_plan", {})
    target = ", ".join(plan.get("target", [])) or "target"
    relations = plan.get("relations", [])
    if relations:
        rel_text = "; ".join(f"{rel.get('type')} {rel.get('object')}" for rel in relations)
        return f"{target} | {rel_text}"
    return target


def render_sheet(pcd: o3d.geometry.PointCloud, query_result: Dict, output_png: Path) -> None:
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    sheet = np.full((920, 2740, 3), 18, dtype=np.uint8)
    views = [
        render_projection(points, colors, (0, 1, 2), "XY / top"),
        render_projection(points, colors, (0, 2, 1), "XZ / front"),
        render_projection(points, colors, (1, 2, 0), "YZ / side"),
    ]
    x = 20
    for view in views:
        sheet[10:910, x : x + 900] = view
        x += 910

    caption = f"{query_result.get('scene', 'scene')} | {query_caption(query_result)}"
    cv2.putText(sheet, caption[:150], (24, 905), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (235, 235, 235), 2, cv2.LINE_AA)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_png), sheet)


def visualize_query_result(
    query_result_path: Path,
    object_map_path: Optional[Path],
    objects_ply_path: Optional[Path],
    output_dir: Path,
    prefix: Optional[str] = None,
) -> Dict:
    query_result = load_json(query_result_path)
    object_map_path = object_map_path or Path(query_result["object_map"])
    object_map = load_json(object_map_path)
    objects_ply_path = objects_ply_path or default_objects_ply(object_map_path)
    if not objects_ply_path.exists():
        raise FileNotFoundError(f"Missing objects PLY: {objects_ply_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = prefix or query_result_path.stem
    output_ply = output_dir / f"{prefix}_highlight.ply"
    output_png = output_dir / f"{prefix}_highlight.png"
    output_json = output_dir / f"{prefix}_visualization.json"

    pcd, metadata = make_highlight_cloud(object_map, query_result, objects_ply_path)
    o3d.io.write_point_cloud(str(output_ply), pcd)
    render_sheet(pcd, query_result, output_png)

    report = {
        **metadata,
        "query_result": str(query_result_path),
        "object_map": str(object_map_path),
        "objects_ply": str(objects_ply_path),
        "output_ply": str(output_ply),
        "output_png": str(output_png),
    }
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize object query results as highlighted PLY and PNG files.")
    parser.add_argument("--query-result", required=True, help="Path to query result JSON from query_executor.")
    parser.add_argument("--object-map", default=None, help="Optional object map path override.")
    parser.add_argument("--objects-ply", default=None, help="Optional objects PLY path override.")
    parser.add_argument("--output-dir", default="outputs/query_results")
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    report = visualize_query_result(
        query_result_path=Path(args.query_result),
        object_map_path=Path(args.object_map) if args.object_map else None,
        objects_ply_path=Path(args.objects_ply) if args.objects_ply else None,
        output_dir=Path(args.output_dir),
        prefix=args.prefix,
    )
    if args.pretty:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
