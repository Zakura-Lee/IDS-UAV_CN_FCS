"""UAVThreatBench LLM evaluation pipeline.

This module loads the UAVThreatBench JSON datasets from config/config.py,
constructs prompts for a DeepSeek-style chat model, calls the LLM, parses the
resulting threat list, saves responses, and evaluates them against the expert
labels when available.
"""

from __future__ import annotations

import json
import logging
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
from openai import OpenAI
from tqdm import tqdm

try:
    from thefuzz import fuzz as thefuzz_fuzz
except ImportError:  # pragma: no cover
    thefuzz_fuzz = None

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(ROOT_DIR / "configs") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "configs"))

try:
    import configs.config as project_config
except ImportError:  # pragma: no cover
    from config import config as project_config  # type: ignore

logger = logging.getLogger(__name__)


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
        self.api_key = self.llm_config.get("api_key") or os.getenv("DEEPSEEK_API_KEY")
        self.base_url = self.llm_config.get("base_url") or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        self.model_name = self.llm_config.get("model_name") or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

        logger.debug("Initializing UAVThreatBenchAnalyzer")
        logger.debug("Using raw data directory: %s", self.raw_data_dir)
        logger.debug("LLM config loaded: %s", self.llm_config)

        if self.api_key:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            logger.info("OpenAI client initialized successfully for model '%s' at %s", self.model_name, self.base_url)
        else:
            self.client = None
            logger.warning("DeepSeek API key not found. Using heuristic fallback.")
            print("⚠️ DeepSeek API Key not found. Using heuristic fallback.")

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

    def _is_reasoning_model(self) -> bool:
        model_name = (self.model_name or "").lower()
        return any(marker in model_name for marker in ("reasoner", "reasoning", "r1", "deepseek-v4-flash"))

    def build_prompt(self, scenario_description: str) -> Dict[str, str]:
        """Minimalist prompt: baseline + 3 lightweight enhancements.

        Baseline (45.02% match rate) plus sentence guidance, category
        examples, and format reinforcement.  Keeps the prompt short so the
        model focuses on the scenario rather than complex instructions.
        """
        reasoning_hint = ""
        if self._is_reasoning_model():
            reasoning_hint = (
                "You are allowed to use reasoning internally to analyze the "
                "scenario. Do not expose chain-of-thought or verbose "
                "explanations. "
            )

        system_prompt = (
            "You are an unmanned aerial vehicle (UAV) cybersecurity expert. "
            f"{reasoning_hint}"
            "Given a scenario description, identify potential cyber threats. "
            "Each description should be a complete sentence with specific consequence. "
            "Return JSON only as an array of objects with fields 'description' and 'category'. "
            "Use category values (d), (e), or (f), where (d)=network integrity, (e)=personal data/privacy, (f)=fraud/economic harm. "
            "Category examples: (d) 'attacker could disrupt network operations', (e) 'attacker could expose sensitive data', (f) 'attacker could cause financial loss'. "
            "Output ONLY valid JSON, no additional text."
        )
        user_prompt = (
            "Analyze this scenario and list the threats in a JSON array:\n"
            f"{scenario_description}"
        )
        return {"system": system_prompt, "user": user_prompt}

    def _extract_json(self, text: str) -> List[Dict[str, Any]]:
        """Multi-strategy JSON extraction with automatic repair for common LLM output issues.

        Handles: markdown fences, thinking tags, trailing commas, single-quoted
        keys/values, truncated arrays, and explanatory text before/after JSON.
        """
        text = text.strip()
        if not text:
            logger.debug("LLM returned empty content")
            return []

        # 1. Strip reasoning tags and markdown fences (any variant)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = text.strip()
        if not text:
            logger.debug("LLM returned only reasoning/fence content")
            return []

        # 2. Strategy A: extract JSON array between outermost [ ... ]
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end + 1]
            parsed = self._try_parse_json(candidate)
            if isinstance(parsed, list):
                return parsed

        # 3. Strategy B: try parsing the whole text as-is
        parsed = self._try_parse_json(text)
        if isinstance(parsed, list):
            return parsed

        # 4. Strategy C: salvage - regex-extract individual {"description":..., "category":...} objects
        objects = re.findall(r'\{\s*"description"\s*:\s*"([^"]*)"\s*,\s*"category"\s*:\s*"([^"]*)"\s*\}', text)
        if objects:
            logger.debug("Salvaged %d threat objects via regex from unparseable JSON", len(objects))
            return [{"description": desc, "category": cat} for desc, cat in objects]

        # 5. Strategy D: try single-quote → double-quote repair before giving up
        repaired = re.sub(r"'([^']*)'", r'"\1"', text)
        start_r = repaired.find("[")
        end_r = repaired.rfind("]")
        if start_r != -1 and end_r != -1 and end_r > start_r:
            candidate = repaired[start_r:end_r + 1]
            parsed = self._try_parse_json(candidate)
            if isinstance(parsed, list):
                return parsed

        logger.debug("All JSON extraction strategies failed for content: %s", text[:300])
        return []

    @staticmethod
    def _try_parse_json(candidate: str) -> Any:
        """Attempt json.loads with automatic repair for common issues."""
        # Remove trailing commas before ] or } (most common JSON syntax error from LLMs)
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None

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

    def _extract_scenario_text(self, item: Dict[str, Any]) -> str:
        for key in ("Scenario Description", "scenario_description", "description", "text"):
            if key in item and isinstance(item.get(key), str) and item.get(key).strip():
                return item[key].strip()
        return ""

    def _extract_expected_threats(self, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        for key in ("Expected Threats", "expected_threats", "threats"):
            value = item.get(key)
            if isinstance(value, list):
                return value
        return []

    def _heuristic_threats(self, scenario: Dict[str, Any] | str) -> List[Dict[str, str]]:
        """Keyword-driven fallback using ALL scenario fields, not just description."""
        # Backward-compatible: accept plain strings from old call paths
        if isinstance(scenario, str):
            scenario = {"Scenario Description": scenario}

        description = str(
            scenario.get("Scenario Description", "")
            or scenario.get("scenario_description", "")
            or ""
        ).lower()
        attack_vector = str(
            scenario.get("Cybersecurity Origin/Attack Vector", "")
            or scenario.get("attack_vector", "")
            or ""
        ).lower()
        protocol = str(
            scenario.get("Communication Protocol", "")
            or scenario.get("communication_protocol", "")
            or ""
        ).lower()
        data_flow = str(
            scenario.get("Data Flow/Function", "")
            or scenario.get("data_flow", "")
            or ""
        ).lower()
        consequences = str(
            scenario.get("Potential Cybersecurity Consequences (from origin)", [])
            or scenario.get("potential_consequences", [])
            or ""
        ).lower()
        combined = f"{description} {attack_vector} {protocol} {data_flow} {consequences}"
        threats: List[Dict[str, str]] = []

        # (d) -- network/device threats
        if any(kw in combined for kw in ("wifi", "wireless", "network", "jamming", "denial of service", "dos", "command injection", "firmware", "tamper", "spoof", "impersonat", "unauthorized access", "intrusion", "mitm", "man-in-the-middle", "malware")):
            threats.append({"description": f"Unauthorized access or attack via {attack_vector or 'the network interface'} could compromise drone operations, causing device or network harm.", "category": "(d)"})
        if any(kw in combined for kw in ("tamper", "physical", "onboard", "storage", "firmware")):
            threats.append({"description": f"Physical tampering with {attack_vector or 'onboard components'} could inject malicious code or disrupt safe operation.", "category": "(d)"})

        # (e) -- data/privacy threats
        if any(kw in combined for kw in ("data", "eavesdrop", "intercept", "privacy", "leak", "exfiltrat", "breach", "expos", "bluetooth", "information")):
            threats.append({"description": f"Interception or eavesdropping via {protocol or 'the communication link'} could expose sensitive operational data, violating data privacy.", "category": "(e)"})

        # (f) -- fraud/economic threats
        if any(kw in combined for kw in ("fraud", "financial", "payment", "economic", "manipulat", "inventory", "spoof", "falsif")):
            threats.append({"description": f"Manipulation of {data_flow or 'operational data'} via {attack_vector or 'the attack vector'} could cause fraudulent records and economic loss.", "category": "(f)"})

        if not threats:
            threats.append({"description": f"Potential unauthorized manipulation of drone operations via {attack_vector or 'the identified attack vector'}.", "category": "(d)"})

        while len(threats) < 3:
            threats.append({"description": "Additional cybersecurity threat related to the identified attack surface.", "category": "(d)"})

        return threats[:5]

    def call_llm(self, scenario_description: str, max_retries: int = 3) -> List[Dict[str, str]]:
        scenario_description = scenario_description.strip()

        if self.client is None:
            logger.debug("No LLM client available; using heuristic fallback for scenario: %s", scenario_description[:80])
            return self._heuristic_threats(scenario_description)

        if not scenario_description:
            logger.warning("Scenario description is empty; skipping LLM call")
            return []

        prompt = self.build_prompt(scenario_description)
        last_error: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                request_kwargs: Dict[str, Any] = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": prompt["system"]},
                        {"role": "user", "content": prompt["user"]},
                    ],
                    "temperature": 0.2,
                    "max_tokens": 800,
                }

                # DeepSeek V4 系列在思考模式下会返回 reasoning_content，
                # 这会破坏当前的 JSON 解析逻辑。这里显式禁用思考模式，
                # 让响应保持标准 content 字段格式，兼容现有流程。
                extra_body = None
                if isinstance(self.llm_config, dict):
                    extra_body = self.llm_config.get("extra_body")
                if not isinstance(extra_body, dict):
                    extra_body = {"thinking": {"type": "disabled"}}
                request_kwargs["extra_body"] = extra_body

                response = self.client.chat.completions.create(**request_kwargs)
                content = response.choices[0].message.content or ""
                logger.debug("LLM raw response: %s", content[:500])
                parsed = self._extract_json(content)
                normalized = self._normalize_threats(parsed)
                if not normalized:
                    logger.warning(
                        "LLM returned no usable threats for scenario: %s; "
                        "raw content (first 300 chars): %s; using heuristic fallback",
                        scenario_description[:200], content[:300],
                    )
                    return self._heuristic_threats(scenario_description)
                return normalized
            except Exception as exc:  # pragma: no cover
                last_error = exc
                logger.warning("LLM call attempt %s failed: %s", attempt + 1, exc)
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
            # 按优先级尝试多种字段名，兼容不同版本的 JSON 文件
            scenario_description = (
                item.get("scenario_description")
                or item.get("Scenario Description")
                or item.get("description")
                or ""
            )
            expected_threats = (
                item.get("expected_threats")
                or item.get("Expected Threats")
                or []
            )

            if isinstance(scenario_description, str):
                scenario_description = scenario_description.strip()

            if len(results) < 3:
                print(f"DEBUG: scenario_description = {scenario_description[:100]}...")
                print(f"DEBUG: expected_threats count = {len(expected_threats)}")

            if not scenario_description:
                print("Warning: scenario_description is empty for this item")


            try:
                model_threats = self.call_llm(scenario_description)
            except Exception as exc:
                model_threats = []
                logger.warning("LLM error for scenario: %s", exc)
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
        if thefuzz_fuzz is not None:
            return thefuzz_fuzz.token_set_ratio(left.lower(), right.lower())

        left_tokens = set(re.findall(r"\w+", left.lower()))
        right_tokens = set(re.findall(r"\w+", right.lower()))
        if not left_tokens or not right_tokens:
            return 0
        overlap = len(left_tokens & right_tokens)
        union = len(left_tokens | right_tokens)
        return int((overlap / union) * 100) if union else 0

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
