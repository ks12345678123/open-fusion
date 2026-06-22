import argparse
import json
from pathlib import Path
from typing import List, Optional

from openfusion_agent.config import DEFAULT_SEMANTIC_LABELS
from openfusion_agent.llm_parser import DeepSeekQueryParser
from openfusion_agent.query_executor import ObjectMapQueryExecutor
from openfusion_agent.schemas import QueryPlan
from openfusion_agent.visualizer import visualize_query_result


def parse_labels(value: str) -> List[str]:
    if not value:
        return DEFAULT_SEMANTIC_LABELS
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def load_plan(plan_file: Optional[str]) -> Optional[QueryPlan]:
    if not plan_file:
        return None
    with open(plan_file, "r", encoding="utf-8") as f:
        return QueryPlan.from_dict(json.load(f))


def safe_prefix(query: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in query.strip().lower())
    safe = "_".join(part for part in safe.split("_") if part)
    return safe[:48] or "query"


def parse_args():
    parser = argparse.ArgumentParser(description="Run the LLM query agent pipeline on an object map.")
    parser.add_argument("--query", default=None, help="Natural-language query.")
    parser.add_argument("--plan-file", default=None, help="Use an existing QueryPlan JSON instead of calling DeepSeek.")
    parser.add_argument("--object-map", required=True)
    parser.add_argument("--output-dir", default="outputs/query_results")
    parser.add_argument("--labels", default="", help="Comma-separated labels made available to the LLM parser.")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--near-threshold", type=float, default=1.5)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--skip-visualization", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.query and not args.plan_file:
        raise ValueError("either --query or --plan-file is required")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or safe_prefix(args.query or Path(args.plan_file).stem)

    plan = load_plan(args.plan_file)
    if plan is None:
        parser = DeepSeekQueryParser(model=args.model)
        plan = parser.parse(args.query, available_labels=parse_labels(args.labels), top_k=args.top_k)

    plan_path = output_dir / f"{prefix}_plan.json"
    with plan_path.open("w", encoding="utf-8") as f:
        json.dump(plan.to_dict(), f, ensure_ascii=False, indent=2)

    executor = ObjectMapQueryExecutor(Path(args.object_map), near_threshold=args.near_threshold)
    query_result = executor.execute(plan)
    result_path = output_dir / f"{prefix}_result.json"
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(query_result, f, ensure_ascii=False, indent=2)

    report = None
    if not args.skip_visualization:
        report = visualize_query_result(
            query_result_path=result_path,
            object_map_path=Path(args.object_map),
            objects_ply_path=None,
            output_dir=output_dir,
            prefix=prefix,
        )

    output = {
        "query": args.query,
        "plan": plan.to_dict(),
        "result_path": str(result_path),
        "plan_path": str(plan_path),
        "visualization": report,
    }
    if args.pretty:
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
