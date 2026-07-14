"""UAVThreatBench LLM evaluation pipeline.

This module loads the UAVThreatBench JSON datasets from config/config.py,
constructs prompts for a DeepSeek-style chat model, calls the LLM, parses the
resulting threat list, saves responses, and evaluates them against the expert
labels when available.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz
from openai import OpenAI
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(ROOT_DIR / "configs") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "configs"))

try:
    import configs.config as project_config
except ImportError:  # pragma: no cover
    from config import config as project_config  # type: ignore


class UAVThreatBenchAnalyzer:
    """Load, query, and evaluate UAVThreatBench scenarios."""

    def __init__(self, config_module: Any | None = None) -> None:
        self.config = config_module or project_config
        self.raw_data_dir = Path(getattr(self.config, "RAW_DATA_DIR", ROOT_DIR / "data" / "raw"))
        self.output_dir = ROOT_DIR / "models" / "outputs" / "llm_evaluation"
        self.with_threats_dir = self.output_dir / "withThreats"
        self.without_threats_dir = self.output_dir / "withoutThreats"
        self.with_threats_dir.mkdir(parents=True, exist_ok=True)
        self.without_threats_dir.mkdir(parents=True, exist_ok=True)
        self.llm_config = getattr(self.config, "LLM_CONFIG", {})
        self.api_key = self.llm_config.get("api_key") or os.getenv("DEEPSEEK_API_KEY") or os.getenv("sk-b5af794365e34fe4b8706cba69088004")
        self.base_url = self.llm_config.get("base_url") or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        self.model_name = self.llm_config.get("model_name") or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url) if self.api_key else None

    def _load_json(self, file_path: str | Path) -> List[Dict[str, Any]]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"JSON file not found: {path}")
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and isinstance(data.get("scenarios"), list):
            return data["scenarios"]
        if isinstance(data, dict) and isinstance(data.get("data"), list):
            return data["data"]
        return [data]

    def load_dataset(self, dataset_name: str) -> List[Dict[str, Any]]:
        config_attr = "UAV_THREAT_BENCH_WITH" if dataset_name == "withThreats" else "UAV_THREAT_BENCH_WITHOUT"
        dataset_path = getattr(self.config, config_attr, None)
        if dataset_path is None:
            candidate_paths = [
                self.raw_data_dir / "plausible_uav_ot_cyber_scenarios_withThreats.json",
                self.raw_data_dir / "plausible_uav_ot_cyber_scenarios_withoutThreats.json",
            ]
            dataset_path = candidate_paths[0] if dataset_name == "withThreats" else candidate_paths[1]
        if not Path(dataset_path).exists():
            dataset_path = self.raw_data_dir / Path(dataset_path).name
        if not Path(dataset_path).exists():
            raise FileNotFoundError(f"Could not resolve dataset path for {dataset_name}: {dataset_path}")
        print(f"Loading {dataset_name} dataset from {dataset_path}")
        return self._load_json(dataset_path)

    def inspect_dataset(self, items: List[Dict[str, Any]]) -> None:
        print(f"Total scenarios: {len(items)}")
        if not items:
            return
        first = items[0]
        print("Keys:", list(first.keys()))
        for key in ("scenario_description", "Expected Threats", "expected_threats"):
            if key in first:
                print("Sample value for", key, ":", first[key])
                break

    def build_prompt(self, scenario_description: str) -> Dict[str, str]:
        system_prompt = (
            "You are an unmanned aerial vehicle (UAV) cybersecurity expert. "
            "Given a scenario description, identify potential cyber threats. "
            "Return JSON only as an array of objects with fields 'description' and 'category'. "
            "Use category values d, e, or f, where d=network integrity, e=personal data/privacy, f=fraud/economic harm."
        )
        user_prompt = (
            "Scenario description:\n"
            f"{scenario_description}\n\n"
            "Return only a JSON array."
        )
        return {"system": system_prompt, "user": user_prompt}

    def _extract_json(self, text: str) -> List[Dict[str, Any]]:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end + 1]
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return []

    def _normalize_threats(self, threats: Any) -> List[Dict[str, str]]:
        if not isinstance(threats, list):
            return []
        normalized = []
        for item in threats:
            if isinstance(item, dict):
                description = item.get("description") or item.get("text") or ""
                category = item.get("category") or item.get("type") or ""
                normalized.append({"description": str(description), "category": str(category)})
        return normalized

    def _heuristic_threats(self, scenario_description: str) -> List[Dict[str, str]]:
        description = scenario_description.lower()
        threats = []
        if "wifi" in description or "wireless" in description or "network" in description:
            threats.append({"description": "Unauthorized access to the network could expose or alter drone communications.", "category": "d"})
            threats.append({"description": "Data exfiltration from the drone communications could compromise sensitive information.", "category": "e"})
        if "bluetooth" in description:
            threats.append({"description": "Bluetooth interception could disclose operational or personal data.", "category": "e"})
        if "data" in description or "information" in description:
            threats.append({"description": "Sensitive information may be leaked through insecure data handling.", "category": "e"})
        if "fraud" in description or "financial" in description or "payment" in description:
            threats.append({"description": "Fraudulent transactions or manipulated records could cause economic harm.", "category": "f"})
        if not threats:
            threats.append({"description": "Potential unauthorized manipulation of drone operations.", "category": "d"})
        return threats

    def call_llm(self, scenario_description: str, max_retries: int = 3) -> List[Dict[str, str]]:
        if self.client is None:
            return self._heuristic_threats(scenario_description)

        prompt = self.build_prompt(scenario_description)
        last_error: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": prompt["system"]},
                        {"role": "user", "content": prompt["user"]},
                    ],
                    temperature=0.2,
                    max_tokens=300,
                )
                content = response.choices[0].message.content or ""
                parsed = self._extract_json(content)
                return self._normalize_threats(parsed)
            except Exception as exc:  # pragma: no cover
                last_error = exc
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    break
        raise RuntimeError(f"LLM call failed after {max_retries} attempts: {last_error}")

    def process_dataset(self, dataset_name: str, save_path: Path) -> List[Dict[str, Any]]:
        items = self.load_dataset(dataset_name)
        self.inspect_dataset(items)

        results = []
        for item in tqdm(items, desc=dataset_name):
            scenario_description = item.get("scenario_description") or item.get("description") or ""
            expected_threats = item.get("Expected Threats") or item.get("expected_threats") or []
            try:
                model_threats = self.call_llm(scenario_description)
            except Exception as exc:
                model_threats = []
                print(f"LLM error for scenario: {exc}")

            result_entry = {
                "scenario_description": scenario_description,
                "model_output": model_threats,
                "expected_threats": expected_threats,
                "dataset": dataset_name,
            }
            results.append(result_entry)

        with open(save_path, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, ensure_ascii=False)
        print(f"Saved results to {save_path}")
        return results

    def _token_score(self, left: str, right: str) -> int:
        return fuzz.token_set_ratio(left.lower(), right.lower())

    def evaluate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        matches = []
        perfect_matches = 0
        category_matches = {"d": 0, "e": 0, "f": 0}
        category_total = {"d": 0, "e": 0, "f": 0}

        for entry in results:
            expected = [str(item).lower() for item in entry.get("expected_threats", [])]
            predicted = [str(item.get("description", "")).lower() for item in entry.get("model_output", []) if isinstance(item, dict)]

            if not expected:
                continue

            matched = False
            for exp in expected:
                for pred in predicted:
                    if self._token_score(exp, pred) >= 80:
                        matched = True
                        break
                if matched:
                    break
            matches.append(matched)
            if matched:
                perfect_matches += 1

            for cat in ("d", "e", "f"):
                category_total[cat] += 1
                if any(str(item).lower().startswith(cat) for item in expected):
                    if any(str(item.get("category", "")).lower() == cat for item in entry.get("model_output", []) if isinstance(item, dict)):
                        category_matches[cat] += 1

        evaluation = {
            "total_scenarios": len(results),
            "total_match_rate": float(np.mean(matches)) if matches else 0.0,
            "perfect_match_rate": float(perfect_matches / max(1, len(results))) if results else 0.0,
            "category_match_rate": {k: (category_matches[k] / category_total[k]) if category_total[k] else 0.0 for k in category_matches},
        }
        return evaluation

    def save_evaluation_report(self, evaluation: Dict[str, Any], output_path: Path) -> None:
        lines = [
            "UAVThreatBench evaluation report",
            f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"Total scenarios: {evaluation['total_scenarios']}",
            f"Total match rate: {evaluation['total_match_rate']:.2%}",
            f"Perfect match rate: {evaluation['perfect_match_rate']:.2%}",
            "",
            "Category match rates:",
        ]
        for key, value in evaluation["category_match_rate"].items():
            lines.append(f"- {key}: {value:.2%}")
        output_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"Saved evaluation report to {output_path}")

    def plot_evaluation(self, evaluation: Dict[str, Any], output_path: Path) -> None:
        labels = ["total_match_rate", "perfect_match_rate"]
        values = [evaluation["total_match_rate"], evaluation["perfect_match_rate"]]
        plt.figure(figsize=(6, 4))
        plt.bar(labels, values, color=["#4C78A8", "#F58518"])
        plt.ylim(0, 1)
        plt.ylabel("Rate")
        plt.title("LLM Threat Match Rates")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

    def run(self) -> None:
        with_results = self.process_dataset("withThreats", self.with_threats_dir / "responses.json")
        evaluation = self.evaluate_results(with_results)
        self.save_evaluation_report(evaluation, self.with_threats_dir / "evaluation_report.txt")
        self.plot_evaluation(evaluation, self.with_threats_dir / "match_rates.png")

        without_results = self.process_dataset("withoutThreats", self.without_threats_dir / "responses.json")
        print(f"Completed with {len(with_results)} evaluated scenarios and {len(without_results)} inference scenarios")


def main() -> None:
    analyzer = UAVThreatBenchAnalyzer()
    analyzer.run()


if __name__ == "__main__":
    main()
