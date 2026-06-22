import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Tuple


VALID_INTENTS = {"find_object"}
VALID_RELATIONS = {
    "near",
    "on",
    "under",
    "inside",
    "left_of",
    "right_of",
    "in_front_of",
    "behind",
    "closest_to",
    "between",
}


@dataclass
class Relation:
    type: str
    object: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Relation":
        if not isinstance(data, dict):
            raise ValueError("relation must be an object")
        rel_type = str(data.get("type", "")).strip().lower()
        rel_object = str(data.get("object", "")).strip().lower()
        if rel_type not in VALID_RELATIONS:
            raise ValueError(f"unsupported relation type: {rel_type}")
        if not rel_object:
            raise ValueError("relation object is required")
        return cls(type=rel_type, object=rel_object)


@dataclass
class QueryPlan:
    intent: str = "find_object"
    target: List[str] = field(default_factory=list)
    attributes: List[str] = field(default_factory=list)
    relations: List[Relation] = field(default_factory=list)
    top_k: int = 3

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueryPlan":
        if not isinstance(data, dict):
            raise ValueError("query plan must be a JSON object")

        intent = str(data.get("intent", "find_object")).strip().lower()
        if intent not in VALID_INTENTS:
            raise ValueError(f"unsupported intent: {intent}")

        target = _string_list(data.get("target", []), "target")
        attributes = _string_list(data.get("attributes", []), "attributes")
        relations = [Relation.from_dict(item) for item in data.get("relations", [])]

        try:
            top_k = int(data.get("top_k", 3))
        except (TypeError, ValueError) as exc:
            raise ValueError("top_k must be an integer") from exc
        top_k = max(1, min(10, top_k))

        if not target and not attributes:
            raise ValueError("at least one target or attribute is required")

        return cls(
            intent=intent,
            target=target,
            attributes=attributes,
            relations=relations,
            top_k=top_k,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _string_list(value: Any, field_name: str) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list of strings")

    out = []
    for item in value:
        text = str(item).strip().lower()
        if text:
            out.append(text)
    return out


def extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("LLM response does not contain a JSON object")
    return json.loads(text[start : end + 1])


def parse_query_plan(text: str) -> QueryPlan:
    return QueryPlan.from_dict(extract_json_object(text))


def validate_query_plan(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    try:
        QueryPlan.from_dict(data)
    except ValueError as exc:
        return False, [str(exc)]
    return True, []

