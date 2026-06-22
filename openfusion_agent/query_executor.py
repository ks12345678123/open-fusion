import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from openfusion_agent.schemas import QueryPlan, extract_json_object
from openfusion_agent.spatial_relations import best_relation_score


SYNONYMS = {
    "lamp": {"light", "lamp"},
    "light": {"light", "lamp"},
    "sofa": {"sofa", "couch"},
    "couch": {"sofa", "couch"},
    "cabinet": {"cabinet", "drawer", "storage"},
    "shelf": {"shelf", "tv shelf", "bookshelf"},
    "chair": {"chair", "stool", "bench"},
    "door": {"door", "exit"},
    "tv": {"tv", "television"},
    "table": {"table", "desk"},
}


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def expanded_terms(term: str) -> List[str]:
    term = normalize(term)
    terms = {term}
    terms.update(SYNONYMS.get(term, set()))
    return sorted(terms)


def semantic_match_score(obj: Dict, terms: Iterable[str]) -> float:
    label = normalize(obj.get("semantic_label", ""))
    if not label:
        return 0.0
    best = 0.0
    for term in terms:
        for expanded in expanded_terms(term):
            if label == expanded:
                best = max(best, 1.0)
            elif expanded in label or label in expanded:
                best = max(best, 0.8)
            else:
                label_tokens = set(label.split())
                term_tokens = set(expanded.split())
                if label_tokens and term_tokens:
                    overlap = len(label_tokens & term_tokens) / len(label_tokens | term_tokens)
                    best = max(best, 0.5 * overlap)
    return best


class ObjectMapQueryExecutor:
    def __init__(self, object_map_path: Path, near_threshold: float = 1.5) -> None:
        with object_map_path.open("r", encoding="utf-8") as f:
            self.object_map = json.load(f)
        self.object_map_path = object_map_path
        self.objects = self.object_map.get("objects", [])
        self.near_threshold = near_threshold

    def execute(self, plan: QueryPlan) -> Dict:
        candidates = []
        target_terms = plan.target + plan.attributes
        for obj in self.objects:
            semantic_score = semantic_match_score(obj, target_terms)
            if target_terms and semantic_score <= 0.0:
                continue

            relation_results = []
            relation_score_total = 0.0
            relation_failed = False
            for relation in plan.relations:
                references = [
                    ref
                    for ref in self.objects
                    if ref.get("object_id") != obj.get("object_id")
                    and semantic_match_score(ref, [relation.object]) > 0.0
                ]
                if not references:
                    relation_failed = True
                    break
                score, reference, info = best_relation_score(
                    obj,
                    references,
                    relation.type,
                    near_threshold=self.near_threshold,
                )
                if score <= 0.0:
                    relation_failed = True
                    break
                relation_score_total += score
                relation_results.append(
                    {
                        "type": relation.type,
                        "object": relation.object,
                        "score": round(float(score), 5),
                        **info,
                    }
                )
            if relation_failed:
                continue

            confidence = float(obj.get("confidence", 0.0))
            total_score = semantic_score + 0.5 * relation_score_total + 0.1 * confidence
            candidates.append(
                {
                    "object_id": obj.get("object_id"),
                    "semantic_label": obj.get("semantic_label"),
                    "score": round(float(total_score), 5),
                    "semantic_score": round(float(semantic_score), 5),
                    "relation_score": round(float(relation_score_total), 5),
                    "confidence": round(confidence, 5),
                    "point_count": obj.get("point_count"),
                    "centroid": obj.get("centroid"),
                    "bbox_min": obj.get("bbox_min"),
                    "bbox_max": obj.get("bbox_max"),
                    "relations": relation_results,
                }
            )

        candidates.sort(key=lambda item: item["score"], reverse=True)
        return {
            "scene": self.object_map.get("scene"),
            "object_map": str(self.object_map_path),
            "query_plan": plan.to_dict(),
            "result_count": len(candidates),
            "results": candidates[: plan.top_k],
        }


def load_plan(plan: Optional[str], plan_file: Optional[str]) -> QueryPlan:
    if plan_file:
        with open(plan_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return QueryPlan.from_dict(data)
    if not plan:
        raise ValueError("either --plan or --plan-file is required")
    return QueryPlan.from_dict(extract_json_object(plan))


def parse_args():
    parser = argparse.ArgumentParser(description="Execute a structured 3D query plan on an object map.")
    parser.add_argument("--object-map", required=True, help="Path to *_object_map.json.")
    parser.add_argument("--plan", default=None, help="QueryPlan JSON string.")
    parser.add_argument("--plan-file", default=None, help="Path to QueryPlan JSON file.")
    parser.add_argument("--near-threshold", type=float, default=1.5)
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    plan = load_plan(args.plan, args.plan_file)
    executor = ObjectMapQueryExecutor(Path(args.object_map), near_threshold=args.near_threshold)
    result = executor.execute(plan)
    if args.pretty:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
