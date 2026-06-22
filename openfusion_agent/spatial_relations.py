from typing import Dict, Iterable, Tuple

import numpy as np


def center(obj: Dict) -> np.ndarray:
    return np.asarray(obj["centroid"], dtype=np.float64)


def bbox_min(obj: Dict) -> np.ndarray:
    return np.asarray(obj["bbox_min"], dtype=np.float64)


def bbox_max(obj: Dict) -> np.ndarray:
    return np.asarray(obj["bbox_max"], dtype=np.float64)


def distance(a: Dict, b: Dict) -> float:
    return float(np.linalg.norm(center(a) - center(b)))


def xy_overlap_ratio(a: Dict, b: Dict) -> float:
    a_min, a_max = bbox_min(a), bbox_max(a)
    b_min, b_max = bbox_min(b), bbox_max(b)
    inter_min = np.maximum(a_min[:2], b_min[:2])
    inter_max = np.minimum(a_max[:2], b_max[:2])
    inter = np.maximum(inter_max - inter_min, 0.0)
    inter_area = float(inter[0] * inter[1])
    a_area = float(np.prod(np.maximum(a_max[:2] - a_min[:2], 1e-6)))
    return inter_area / max(a_area, 1e-6)


def relation_score(candidate: Dict, reference: Dict, relation_type: str, near_threshold: float = 1.5) -> Tuple[float, Dict]:
    d = distance(candidate, reference)
    info = {"distance": round(d, 5), "reference_object_id": reference.get("object_id")}

    if relation_type == "closest_to":
        return 1.0 / (1.0 + d), info

    if relation_type == "near":
        score = max(0.0, 1.0 - d / near_threshold)
        return score, info

    cand_min, cand_max = bbox_min(candidate), bbox_max(candidate)
    ref_min, ref_max = bbox_min(reference), bbox_max(reference)
    cand_ctr, ref_ctr = center(candidate), center(reference)
    overlap = xy_overlap_ratio(candidate, reference)
    info["xy_overlap"] = round(overlap, 5)

    if relation_type == "on":
        vertical_gap = abs(cand_min[2] - ref_max[2])
        info["vertical_gap"] = round(float(vertical_gap), 5)
        return (1.0 if overlap > 0.15 and vertical_gap < 0.35 else 0.0), info

    if relation_type == "under":
        vertical_gap = abs(cand_max[2] - ref_min[2])
        info["vertical_gap"] = round(float(vertical_gap), 5)
        return (1.0 if overlap > 0.15 and vertical_gap < 0.35 else 0.0), info

    if relation_type == "inside":
        inside = np.all(cand_min >= ref_min) and np.all(cand_max <= ref_max)
        return (1.0 if inside else 0.0), info

    # Axis-relative relations use the reconstructed scene frame. This is a
    # practical first pass; later navigation work should replace it with a
    # camera- or user-defined frame.
    if relation_type == "left_of":
        return (1.0 if cand_ctr[0] < ref_ctr[0] else 0.0), info
    if relation_type == "right_of":
        return (1.0 if cand_ctr[0] > ref_ctr[0] else 0.0), info
    if relation_type == "in_front_of":
        return (1.0 if cand_ctr[1] < ref_ctr[1] else 0.0), info
    if relation_type == "behind":
        return (1.0 if cand_ctr[1] > ref_ctr[1] else 0.0), info

    if relation_type == "between":
        return 0.0, info

    raise ValueError(f"unsupported relation type: {relation_type}")


def best_relation_score(candidate: Dict, references: Iterable[Dict], relation_type: str, near_threshold: float = 1.5):
    best = (0.0, None, {})
    for reference in references:
        score, info = relation_score(candidate, reference, relation_type, near_threshold=near_threshold)
        if score > best[0]:
            best = (score, reference, info)
    return best

