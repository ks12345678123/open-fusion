import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from openfusion_agent.config import DEFAULT_SEMANTIC_LABELS
from openfusion_agent.llm_parser import DeepSeekQueryParser
from openfusion_agent.query_executor import ObjectMapQueryExecutor
from openfusion_agent.schemas import QueryPlan


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc
    return rows


def bool_to_int(value: Optional[bool]) -> Optional[int]:
    if value is None:
        return None
    return 1 if value else 0


def labels_from_results(results: Iterable[Dict]) -> List[str]:
    return [str(item.get("semantic_label", "")).lower() for item in results]


def ids_from_results(results: Iterable[Dict]) -> List[int]:
    return [int(item["object_id"]) for item in results if item.get("object_id") is not None]


def reference_ids_from_results(results: Iterable[Dict]) -> List[int]:
    ids = []
    for item in results:
        for relation in item.get("relations", []):
            ref_id = relation.get("reference_object_id")
            if ref_id is not None:
                ids.append(int(ref_id))
    return ids


def reference_labels_from_results(results: Iterable[Dict], object_by_id: Dict[int, Dict]) -> List[str]:
    labels = []
    for ref_id in reference_ids_from_results(results):
        obj = object_by_id.get(ref_id)
        if obj:
            labels.append(str(obj.get("semantic_label", "")).lower())
    return labels


def any_match(values: Iterable, expected: Iterable) -> Optional[bool]:
    expected_set = set(expected)
    if not expected_set:
        return None
    return bool(set(values) & expected_set)


def top1_match(values: List, expected: Iterable) -> Optional[bool]:
    expected_set = set(expected)
    if not expected_set:
        return None
    if not values:
        return False
    return values[0] in expected_set


def score_case(case: Dict, result: Dict, object_by_id: Dict[int, Dict], latency_ms: float) -> Dict:
    results = result.get("results", [])
    result_labels = labels_from_results(results)
    result_ids = ids_from_results(results)
    expected_labels = [str(v).lower() for v in case.get("expected_labels", [])]
    expected_ids = [int(v) for v in case.get("expected_object_ids", [])]
    expected_ref_ids = [int(v) for v in case.get("expected_reference_ids", [])]
    expected_ref_labels = [str(v).lower() for v in case.get("expected_reference_labels", [])]
    result_ref_ids = reference_ids_from_results(results)
    result_ref_labels = reference_labels_from_results(results, object_by_id)

    return {
        "query_id": case.get("query_id"),
        "scene": case.get("scene"),
        "type": case.get("type"),
        "query": case.get("query"),
        "has_result": bool(results),
        "top1_label_match": top1_match(result_labels, expected_labels),
        "topk_label_match": any_match(result_labels, expected_labels),
        "top1_object_match": top1_match(result_ids, expected_ids),
        "topk_object_match": any_match(result_ids, expected_ids),
        "reference_id_match": any_match(result_ref_ids, expected_ref_ids),
        "reference_label_match": any_match(result_ref_labels, expected_ref_labels),
        "latency_ms": round(latency_ms, 3),
        "result_ids": result_ids,
        "result_labels": result_labels,
        "reference_ids": result_ref_ids,
        "reference_labels": result_ref_labels,
        "plan": result.get("query_plan"),
    }


def summarize(scored: List[Dict]) -> Dict:
    metric_names = [
        "has_result",
        "top1_label_match",
        "topk_label_match",
        "top1_object_match",
        "topk_object_match",
        "reference_id_match",
        "reference_label_match",
    ]
    summary = {"num_cases": len(scored)}
    for metric in metric_names:
        values = [item[metric] for item in scored if item[metric] is not None]
        if values:
            summary[metric] = {
                "count": len(values),
                "score": round(sum(1 for value in values if value) / len(values), 4),
            }
        else:
            summary[metric] = {"count": 0, "score": None}
    if scored:
        summary["avg_latency_ms"] = round(sum(item["latency_ms"] for item in scored) / len(scored), 3)
    return summary


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "query_id",
        "scene",
        "type",
        "query",
        "has_result",
        "top1_label_match",
        "topk_label_match",
        "top1_object_match",
        "topk_object_match",
        "reference_id_match",
        "reference_label_match",
        "latency_ms",
        "result_ids",
        "result_labels",
        "reference_ids",
        "reference_labels",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def parse_labels(value: str) -> List[str]:
    if not value:
        return DEFAULT_SEMANTIC_LABELS
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def run_benchmark(
    benchmark_path: Path,
    output_dir: Path,
    use_llm: bool = False,
    labels: Optional[List[str]] = None,
    model: str = "deepseek-v4-flash",
) -> Dict:
    cases = read_jsonl(benchmark_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    parser = DeepSeekQueryParser(model=model) if use_llm else None
    scored = []

    executor_cache: Dict[str, ObjectMapQueryExecutor] = {}
    object_cache: Dict[str, Dict[int, Dict]] = {}

    for case in cases:
        object_map_path = Path(case["object_map"])
        object_key = str(object_map_path)
        near_threshold = float(case.get("near_threshold", 1.5))
        executor_key = f"{object_key}|near={near_threshold}"
        if executor_key not in executor_cache:
            executor_cache[executor_key] = ObjectMapQueryExecutor(
                object_map_path,
                near_threshold=near_threshold,
            )
        if object_key not in object_cache:
            object_cache[object_key] = {
                int(obj["object_id"]): obj for obj in executor_cache[executor_key].object_map.get("objects", [])
            }
        executor = executor_cache[executor_key]

        start = time.perf_counter()
        if use_llm:
            plan = parser.parse(case["query"], available_labels=labels, top_k=int(case.get("top_k", 3)))
        else:
            plan = QueryPlan.from_dict(case["plan"])
        result = executor.execute(plan)
        latency_ms = (time.perf_counter() - start) * 1000.0
        scored.append(score_case(case, result, object_cache[object_key], latency_ms))

    summary = summarize(scored)
    write_jsonl(output_dir / "results.jsonl", scored)
    write_csv(output_dir / "results.csv", scored)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Run query-agent benchmark cases.")
    parser.add_argument("--benchmark", required=True, help="Path to benchmark JSONL.")
    parser.add_argument("--output-dir", default="outputs/benchmarks")
    parser.add_argument("--use-llm", action="store_true", help="Parse queries with DeepSeek instead of fixed plans.")
    parser.add_argument("--labels", default="", help="Comma-separated labels for LLM parsing.")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    summary = run_benchmark(
        benchmark_path=Path(args.benchmark),
        output_dir=Path(args.output_dir),
        use_llm=args.use_llm,
        labels=parse_labels(args.labels),
        model=args.model,
    )
    if args.pretty:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
