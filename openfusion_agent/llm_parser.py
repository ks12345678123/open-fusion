import argparse
import json
import os
import urllib.error
import urllib.request
from typing import Iterable, List, Optional

from openfusion_agent.config import DEFAULT_SEMANTIC_LABELS
from openfusion_agent.schemas import QueryPlan, parse_query_plan


DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com"


SYSTEM_PROMPT = """You are a parser for an open-vocabulary 3D semantic map query agent.
Convert the user's natural-language request into exactly one JSON object.
Do not answer the user directly. Do not include markdown.

JSON schema:
{
  "intent": "find_object",
  "target": ["chair"],
  "attributes": [],
  "relations": [
    {"type": "near", "object": "table"}
  ],
  "top_k": 3
}

Rules:
- The only supported intent is "find_object".
- Output target and relation object names in concise English lowercase nouns.
- Prefer labels from the available label list when they fit.
- Functional phrases must be expanded into concrete target classes.
  Examples:
  "place to sit" -> ["chair", "sofa", "bench", "stool"]
  "storage furniture" -> ["cabinet", "shelf", "drawer"]
  "lighting device" -> ["lamp", "light"]
  "exit" -> ["door"]
- Supported relation types are: near, on, under, inside, left_of, right_of, in_front_of, behind, closest_to, between.
- If no relation exists, use an empty relations list.
- top_k must be an integer between 1 and 10.
"""


class DeepSeekQueryParser:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = 60,
        max_retries: int = 1,
    ) -> None:
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY is not set")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

    def parse(self, query: str, available_labels: Optional[Iterable[str]] = None, top_k: int = 3) -> QueryPlan:
        labels = list(available_labels or DEFAULT_SEMANTIC_LABELS)
        last_error = None
        last_content = None
        for _ in range(self.max_retries + 1):
            content = self._call_deepseek(
                query=query,
                available_labels=labels,
                top_k=top_k,
                previous_error=last_error,
                previous_content=last_content,
            )
            try:
                return parse_query_plan(content)
            except (ValueError, json.JSONDecodeError) as exc:
                last_error = str(exc)
                last_content = content
        raise RuntimeError(f"DeepSeek response did not pass schema validation: {last_error}")

    def _call_deepseek(
        self,
        query: str,
        available_labels: List[str],
        top_k: int,
        previous_error: Optional[str] = None,
        previous_content: Optional[str] = None,
    ) -> str:
        user_prompt = {
            "query": query,
            "available_labels": available_labels,
            "default_top_k": top_k,
        }
        if previous_error is not None:
            user_prompt["previous_error"] = previous_error
            user_prompt["previous_content"] = previous_content
            user_prompt["repair_instruction"] = "Return a corrected JSON object only."
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
            ],
            "stream": False,
            "temperature": 0,
            "max_tokens": 512,
            "thinking": {"type": "disabled"},
        }

        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"DeepSeek API HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"DeepSeek API request failed: {exc}") from exc

        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected DeepSeek API response: {body}") from exc


def parse_labels(value: str) -> List[str]:
    if not value:
        return DEFAULT_SEMANTIC_LABELS
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Parse natural language into a 3D map query plan with DeepSeek.")
    parser.add_argument("--query", required=True, help="Natural-language query.")
    parser.add_argument("--labels", default="", help="Comma-separated available semantic labels.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    return parser.parse_args()


def main():
    args = parse_args()
    parser = DeepSeekQueryParser(model=args.model, base_url=args.base_url, max_retries=args.max_retries)
    plan = parser.parse(args.query, available_labels=parse_labels(args.labels), top_k=args.top_k)
    if args.pretty:
        print(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(plan.to_dict(), ensure_ascii=False))


if __name__ == "__main__":
    main()
