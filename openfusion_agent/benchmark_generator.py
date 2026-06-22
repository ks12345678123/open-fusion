import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from openfusion_agent.config import DEFAULT_SEMANTIC_LABELS
from openfusion_agent.query_executor import ObjectMapQueryExecutor
from openfusion_agent.schemas import QueryPlan, Relation


ZH_LABELS = {
    "vase": "花瓶",
    "table": "桌子",
    "tv shelf": "电视柜",
    "curtain": "窗帘",
    "wall": "墙",
    "floor": "地板",
    "ceiling": "天花板",
    "door": "门",
    "tv": "电视",
    "room plant": "绿植",
    "light": "灯",
    "sofa": "沙发",
    "cushion": "靠垫",
    "wall paint": "墙面装饰",
    "chair": "椅子",
}

CATEGORY_PRIORITY = [
    "chair",
    "table",
    "sofa",
    "tv",
    "door",
    "light",
    "vase",
    "curtain",
    "room plant",
    "cushion",
    "tv shelf",
    "wall paint",
    "wall",
    "floor",
    "ceiling",
]

FUNCTIONAL_QUERY_SPECS = [
    ("seat", "找可以坐的地方", ["chair", "sofa"], 5),
    ("exit", "找出口", ["door"], 3),
    ("watch", "找可以看视频的东西", ["tv"], 3),
    ("lighting", "找照明设备", ["light"], 5),
    ("plant", "找植物", ["room plant"], 3),
    ("decoration", "找花瓶或墙面装饰", ["vase", "wall paint"], 5),
]

SPATIAL_QUERY_SPECS = [
    ("chair", "table", "near", "找桌子旁边的椅子", 2.0),
    ("sofa", "table", "near", "找桌子旁边的沙发", 2.0),
    ("table", "sofa", "near", "找沙发旁边的桌子", 2.0),
    ("cushion", "sofa", "near", "找沙发旁边的靠垫", 2.0),
    ("tv shelf", "tv", "near", "找电视旁边的电视柜", 2.0),
    ("room plant", "table", "near", "找桌子旁边的绿植", 2.0),
    ("curtain", "wall", "near", "找墙边的窗帘", 2.0),
    ("door", "wall", "near", "找墙边的门", 2.0),
    ("tv", "table", "closest_to", "找离桌子最近的电视", 1.5),
    ("light", "table", "closest_to", "找离桌子最近的灯", 1.5),
    ("chair", "door", "closest_to", "找离门最近的椅子", 1.5),
    ("vase", "table", "closest_to", "找离桌子最近的花瓶", 1.5),
]

EXTRA_RELATION_TARGETS = [
    "chair",
    "table",
    "sofa",
    "tv",
    "light",
    "vase",
    "room plant",
    "cushion",
    "tv shelf",
    "curtain",
]
EXTRA_RELATION_REFS = ["table", "sofa", "tv", "door", "wall"]


def load_object_map(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def group_ids_by_label(objects: Iterable[Dict]) -> Dict[str, List[int]]:
    grouped: Dict[str, List[int]] = defaultdict(list)
    for obj in objects:
        label = str(obj.get("semantic_label", "")).lower()
        if label:
            grouped[label].append(int(obj["object_id"]))
    return {label: sorted(ids) for label, ids in grouped.items()}


def zh_label(label: str) -> str:
    return ZH_LABELS.get(label, label)


def make_plan(targets: Sequence[str], top_k: int, relation: Optional[Tuple[str, str]] = None) -> QueryPlan:
    relations = []
    if relation:
        rel_type, rel_object = relation
        relations = [Relation(type=rel_type, object=rel_object)]
    return QueryPlan(
        intent="find_object",
        target=[target.lower() for target in targets],
        attributes=[],
        relations=relations,
        top_k=top_k,
    )


def result_object_ids(result: Dict) -> List[int]:
    return [int(item["object_id"]) for item in result.get("results", []) if item.get("object_id") is not None]


def result_reference_ids(result: Dict) -> List[int]:
    refs = []
    for item in result.get("results", []):
        for relation in item.get("relations", []):
            ref_id = relation.get("reference_object_id")
            if ref_id is not None:
                refs.append(int(ref_id))
    return sorted(set(refs))


def reference_labels(ref_ids: Iterable[int], object_by_id: Dict[int, Dict]) -> List[str]:
    labels = {
        str(object_by_id[ref_id].get("semantic_label", "")).lower()
        for ref_id in ref_ids
        if ref_id in object_by_id
    }
    return sorted(label for label in labels if label)


def build_case(
    query_id: str,
    scene: str,
    query: str,
    case_type: str,
    object_map_path: Path,
    plan: QueryPlan,
    expected_labels: Sequence[str],
    expected_object_ids: Sequence[int],
    near_threshold: float = 1.5,
    expected_reference_ids: Optional[Sequence[int]] = None,
    expected_reference_labels: Optional[Sequence[str]] = None,
    annotation_source: str = "auto_object_map",
    notes: str = "",
) -> Dict:
    return {
        "query_id": query_id,
        "scene": scene,
        "query": query,
        "type": case_type,
        "object_map": object_map_path.as_posix(),
        "near_threshold": near_threshold,
        "top_k": plan.top_k,
        "plan": plan.to_dict(),
        "expected_labels": sorted({label.lower() for label in expected_labels}),
        "expected_object_ids": sorted({int(obj_id) for obj_id in expected_object_ids}),
        "expected_reference_ids": sorted({int(obj_id) for obj_id in expected_reference_ids or []}),
        "expected_reference_labels": sorted({label.lower() for label in expected_reference_labels or []}),
        "annotation_source": annotation_source,
        "review_status": "auto_unreviewed",
        "notes": notes,
    }


def generate_category_cases(
    scene: str,
    object_map_path: Path,
    ids_by_label: Dict[str, List[int]],
    max_cases: int,
) -> List[Dict]:
    cases = []
    for label in CATEGORY_PRIORITY:
        if label not in ids_by_label:
            continue
        plan = make_plan([label], top_k=min(5, max(3, len(ids_by_label[label]))))
        cases.append(
            build_case(
                query_id="",
                scene=scene,
                query=f"找{zh_label(label)}",
                case_type="category",
                object_map_path=object_map_path,
                plan=plan,
                expected_labels=[label],
                expected_object_ids=ids_by_label[label],
                notes="自动类别查询，期望对象来自 object_map 标签。",
            )
        )
        if len(cases) >= max_cases:
            break
    return cases


def generate_functional_cases(
    scene: str,
    object_map_path: Path,
    ids_by_label: Dict[str, List[int]],
) -> List[Dict]:
    cases = []
    for name, query, targets, top_k in FUNCTIONAL_QUERY_SPECS:
        present_targets = [target for target in targets if target in ids_by_label]
        if not present_targets:
            continue
        expected_ids = []
        for target in present_targets:
            expected_ids.extend(ids_by_label[target])
        plan = make_plan(targets, top_k=min(10, top_k))
        cases.append(
            build_case(
                query_id="",
                scene=scene,
                query=query,
                case_type="functional",
                object_map_path=object_map_path,
                plan=plan,
                expected_labels=present_targets,
                expected_object_ids=expected_ids,
                annotation_source="auto_object_map",
                notes=f"自动功能查询：{name}。",
            )
        )
    return cases


def generate_relation_case(
    scene: str,
    object_map_path: Path,
    objects: List[Dict],
    ids_by_label: Dict[str, List[int]],
    target: str,
    ref: str,
    rel_type: str,
    query: str,
    near_threshold: float,
) -> Optional[Dict]:
    if target not in ids_by_label or ref not in ids_by_label:
        return None

    plan = make_plan([target], top_k=3, relation=(rel_type, ref))
    executor = ObjectMapQueryExecutor(object_map_path, near_threshold=near_threshold)
    result = executor.execute(plan)
    result_ids = result_object_ids(result)
    if not result_ids:
        return None

    object_by_id = {int(obj["object_id"]): obj for obj in objects}
    ref_ids = result_reference_ids(result)
    return build_case(
        query_id="",
        scene=scene,
        query=query,
        case_type="spatial_relation",
        object_map_path=object_map_path,
        plan=plan,
        expected_labels=[target],
        expected_object_ids=result_ids,
        near_threshold=near_threshold,
        expected_reference_ids=ref_ids,
        expected_reference_labels=reference_labels(ref_ids, object_by_id),
        annotation_source="auto_executor",
        notes="自动空间关系查询，期望结果由当前 object_map 和关系执行器生成，后续需人工复核。",
    )


def generate_spatial_cases(
    scene: str,
    object_map_path: Path,
    objects: List[Dict],
    ids_by_label: Dict[str, List[int]],
    max_cases: int,
) -> List[Dict]:
    cases = []
    seen = set()

    def add_case(target: str, ref: str, rel_type: str, query: str, near_threshold: float) -> None:
        key = (target, ref, rel_type, query)
        if key in seen or len(cases) >= max_cases:
            return
        seen.add(key)
        case = generate_relation_case(
            scene,
            object_map_path,
            objects,
            ids_by_label,
            target,
            ref,
            rel_type,
            query,
            near_threshold,
        )
        if case:
            cases.append(case)

    for target, ref, rel_type, query, near_threshold in SPATIAL_QUERY_SPECS:
        add_case(target, ref, rel_type, query, near_threshold)

    for ref in EXTRA_RELATION_REFS:
        for target in EXTRA_RELATION_TARGETS:
            if target == ref:
                continue
            add_case(
                target,
                ref,
                "closest_to",
                f"找离{zh_label(ref)}最近的{zh_label(target)}",
                1.5,
            )
            add_case(
                target,
                ref,
                "near",
                f"找{zh_label(ref)}旁边的{zh_label(target)}",
                2.0,
            )
            if len(cases) >= max_cases:
                return cases
    return cases


def assign_query_ids(scene: str, cases: List[Dict]) -> None:
    scene_suffix = scene.replace("replica_", "")
    for index, case in enumerate(cases, 1):
        case["query_id"] = f"{scene_suffix}_auto_{index:03d}"


def generate_cases_for_map(object_map_path: Path, cases_per_scene: int) -> List[Dict]:
    object_map = load_object_map(object_map_path)
    scene = object_map.get("scene") or object_map_path.stem.replace("_object_map", "")
    objects = object_map.get("objects", [])
    ids_by_label = group_ids_by_label(objects)

    cases: List[Dict] = []
    cases.extend(generate_category_cases(scene, object_map_path, ids_by_label, max_cases=12))
    cases.extend(generate_functional_cases(scene, object_map_path, ids_by_label))

    remaining = max(0, cases_per_scene - len(cases))
    if remaining:
        cases.extend(
            generate_spatial_cases(
                scene,
                object_map_path,
                objects,
                ids_by_label,
                max_cases=remaining,
            )
        )

    unique_cases = []
    seen_queries = set()
    for case in cases:
        key = (case["query"], json.dumps(case["plan"], sort_keys=True, ensure_ascii=False))
        if key in seen_queries:
            continue
        seen_queries.add(key)
        unique_cases.append(case)
        if len(unique_cases) >= cases_per_scene:
            break

    assign_query_ids(scene, unique_cases)
    return unique_cases


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate auto-labeled Replica query benchmark cases.")
    parser.add_argument("--object-map-dir", default="outputs/object_maps")
    parser.add_argument("--output", default="benchmarks/replica_auto_seed_queries.jsonl")
    parser.add_argument("--cases-per-scene", type=int, default=20)
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    object_map_dir = Path(args.object_map_dir)
    object_maps = sorted(object_map_dir.glob("replica_*_object_map.json"))
    if not object_maps:
        raise FileNotFoundError(f"No object maps found in {object_map_dir}")

    all_cases = []
    per_scene_counts = {}
    for object_map_path in object_maps:
        cases = generate_cases_for_map(object_map_path, cases_per_scene=args.cases_per_scene)
        all_cases.extend(cases)
        scene = cases[0]["scene"] if cases else object_map_path.stem.replace("_object_map", "")
        per_scene_counts[scene] = len(cases)

    output_path = Path(args.output)
    write_jsonl(output_path, all_cases)
    summary = {
        "output": output_path.as_posix(),
        "num_cases": len(all_cases),
        "num_scenes": len(object_maps),
        "cases_per_scene": per_scene_counts,
        "labels": DEFAULT_SEMANTIC_LABELS,
        "review_status": "auto_unreviewed",
    }
    if args.pretty:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
