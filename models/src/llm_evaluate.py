"""Evaluate LLM-generated threats against expert labels for UAVThreatBench.

Produces textual report, JSON metrics, and match-rate distribution plot.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
plt.rcParams["axes.unicode_minus"] = False
font_candidates = [
    "Microsoft YaHei",
    "SimHei",
    "STHeiti",
    "PingFang SC",
    "AppleGothic",
    "WenQuanYi Zen Hei",
    "DejaVu Sans"
]
for font_name in font_candidates:
    if font_manager.findfont(font_name, fallback_to_default=False):
        plt.rcParams["font.sans-serif"] = [font_name]
        break
import numpy as np
from thefuzz import fuzz

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(ROOT_DIR / "configs") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "configs"))

try:
    import configs.config as project_config
except Exception:  # pragma: no cover
    try:
        from config import config as project_config
    except Exception:  # pragma: no cover
        project_config = None  # type: ignore


def is_threat_match(generated: str, ground_truth: str, threshold: int = 80) -> Tuple[bool, int]:
    """使用 token_set_ratio 进行模糊匹配，返回是否匹配和匹配分数。"""
    score = fuzz.token_set_ratio(str(generated).lower(), str(ground_truth).lower())
    return (score >= threshold, int(score))


def load_responses(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Responses file not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data


def evaluate_responses(responses_json_path: Path, threshold: int = 80) -> Dict[str, Any]:
    responses = load_responses(responses_json_path)

    total = len(responses)
    if total == 0:
        raise RuntimeError("No responses to evaluate.")

    at_least_one_match = 0
    perfect_match = 0
    best_scores: List[int] = []
    missing_generated_category = 0
    category_counts = Counter()
    category_matches = Counter()

    examples_high: List[Dict[str, Any]] = []
    examples_low: List[Dict[str, Any]] = []

    for entry in responses:
        expected = entry.get("expected_threats", []) or []
        model_output = entry.get("model_output", []) or []

        # 标准化 expected 为描述列表和类别映射
        expected_descs = []
        expected_cats = []
        for e in expected:
            if isinstance(e, dict):
                expected_descs.append(str(e.get("description", "")).strip())
                expected_cats.append(str(e.get("category", "")).strip().lower())
            else:
                expected_descs.append(str(e).strip())

        # 标准化 model_output
        gen_items = []
        for g in model_output:
            if isinstance(g, dict):
                gen_items.append({
                    "description": str(g.get("description", "")).strip(),
                    "category": str(g.get("category", "")).strip().lower(),
                })
            else:
                gen_items.append({"description": str(g).strip(), "category": ""})

        # 记录各类别总数
        for c in expected_cats:
            if c in ("d", "e", "f"):
                category_counts[c] += 1

        if not expected_descs:
            # 无 ground truth，跳过评估，但仍收集示例
            continue

        scene_best_scores: List[int] = []
        matched_flags: List[bool] = []

        for gen in gen_items:
            best_for_gen = 0
            matched_any = False
            if not gen.get("category"):
                missing_generated_category += 1
            for exp in expected_descs:
                matched, score = is_threat_match(gen["description"], exp, threshold=threshold)
                if score > best_for_gen:
                    best_for_gen = score
                if matched:
                    matched_any = True
            scene_best_scores.append(best_for_gen)
            matched_flags.append(matched_any)

        # 对于没有生成任何威胁的场景，视为无响应
        if not gen_items:
            best_scores.append(0)
        else:
            # 场景最佳匹配分数取生成威胁的最大最佳分数
            best_scores.append(max(scene_best_scores) if scene_best_scores else 0)

        if any(matched_flags):
            at_least_one_match += 1

        # 完美匹配：所有 ground_truth 都被匹配到至少一次
        all_matched = True
        for exp in expected_descs:
            exp_matched = False
            for gen in gen_items:
                matched, score = is_threat_match(gen["description"], exp, threshold=threshold)
                if matched:
                    exp_matched = True
                    # 增加类别匹配计数
                    if gen.get("category") in ("d", "e", "f"):
                        category_matches[gen.get("category")] += 1
                    break
            if not exp_matched:
                all_matched = False
        if all_matched:
            perfect_match += 1

        # 记录示例用于报告
        avg_scene_score = int(sum(scene_best_scores) / len(scene_best_scores)) if scene_best_scores else 0
        if avg_scene_score >= 90:
            examples_high.append({"scenario": entry.get("scenario_description", ""), "score": avg_scene_score, "expected": expected_descs, "generated": gen_items})
        if avg_scene_score <= 60:
            examples_low.append({"scenario": entry.get("scenario_description", ""), "score": avg_scene_score, "expected": expected_descs, "generated": gen_items})

    total_with_gt = sum(1 for e in responses if e.get("expected_threats"))

    overall_match_rate = at_least_one_match / max(1, total_with_gt)
    perfect_match_rate = perfect_match / max(1, total_with_gt)
    avg_best_score = float(sum(best_scores) / max(1, len(best_scores)))

    category_rates = {}
    for cat in ("d", "e", "f"):
        category_rates[cat] = (category_matches.get(cat, 0) / category_counts.get(cat, 1)) if category_counts.get(cat, 0) else 0.0

    metrics = {
        "total_scenes": total,
        "scenes_with_ground_truth": total_with_gt,
        "overall_match_rate": overall_match_rate,
        "perfect_match_rate": perfect_match_rate,
        "average_best_score": avg_best_score,
        "category_match_rates": category_rates,
        "missing_generated_category_count": int(missing_generated_category),
    }

    outputs = {
        "metrics": metrics,
        "examples_high": examples_high[:5],
        "examples_low": examples_low[:5],
        "best_scores": best_scores,
    }

    return outputs


def plot_match_distribution(scores: List[int], output_path: Path) -> None:
    if not scores:
        return
    bins = [0, 60, 70, 80, 90, 100]
    labels = ["<60", "60-69", "70-79", "80-89", "90-100"]
    counts, _ = np.histogram(scores, bins=bins)
    plt.figure(figsize=(8, 4))
    plt.bar(labels, counts, color="#4C78A8")
    plt.ylabel("场景数")
    plt.title("匹配分数分布")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_report_text(outputs: Dict[str, Any], output_path: Path) -> None:
    metrics = outputs["metrics"]
    lines = [
        "========================================",
        "UAVThreatBench 大模型威胁识别评估报告",
        "========================================",
        f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"评估场景数: {metrics['total_scenes']}",
        f"包含 ground_truth 的场景数: {metrics['scenes_with_ground_truth']}",
        "",
        "=== 总体指标 ===",
        f"总匹配率: {metrics['overall_match_rate']:.2%} (至少一个威胁匹配)",
        f"完美匹配率: {metrics['perfect_match_rate']:.2%} (所有威胁匹配)",
        f"平均匹配分数: {metrics['average_best_score']:.1f}",
        "",
        "=== 各类别匹配率 ===",
    ]
    for k, v in metrics['category_match_rates'].items():
        label = {'d': '网络完整性', 'e': '个人数据/隐私', 'f': '欺诈/经济'}[k]
        lines.append(f"({k}) {label} 威胁: {v:.2%}")

    lines.append("")
    lines.append("=== 匹配分数分布 ===")

    # 构造分布明细（使用全部 best_scores 而非仅示例）
    scores = outputs.get("best_scores", [])

    # 写入示例
    lines.append("")
    lines.append("=== 典型场景示例（高分） ===")
    for ex in outputs.get('examples_high', []):
        lines.append(f"Score: {ex['score']}")
        lines.append(f"Scenario: {ex['scenario']}")
        lines.append(f"Expected: {ex['expected']}")
        lines.append(f"Generated: {ex['generated']}")
        lines.append("")

    lines.append("=== 典型场景示例（低分） ===")
    for ex in outputs.get('examples_low', []):
        lines.append(f"Score: {ex['score']}")
        lines.append(f"Scenario: {ex['scenario']}")
        lines.append(f"Expected: {ex['expected']}")
        lines.append(f"Generated: {ex['generated']}")
        lines.append("")

    try:
        output_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"Saved textual report to {output_path}")
    except Exception as exc:
        print(f"Failed to write report to {output_path}: {exc}")


def main() -> None:
    responses_path = Path(ROOT_DIR / "models" / "outputs" / "llm_evaluation" / "withThreats" / "responses.json")
    if not responses_path.exists():
        print("Responses file not found. Run llm_analyzer.py first to generate responses.")
        return

    outputs = evaluate_responses(responses_path)

    out_dir = responses_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    report_txt = out_dir / "evaluation_report.txt"
    metrics_json = out_dir / "evaluation_metrics.json"
    plot_png = out_dir / "match_rate_distribution.png"

    save_report_text(outputs, report_txt)
    # 保存 JSON 指标
    try:
        with open(metrics_json, "w", encoding="utf-8") as handle:
            json.dump(outputs["metrics"], handle, indent=2, ensure_ascii=False)
        print(f"Saved metrics JSON to {metrics_json}")
    except Exception as exc:
        print(f"Failed to write metrics JSON to {metrics_json}: {exc}")

    # 生成分布图：使用全部 best_scores 列表
    scores_list = outputs.get("best_scores", [])
    try:
        plot_match_distribution(scores_list, plot_png)
        print(f"Saved match rate distribution plot to {plot_png}")
    except Exception as exc:
        print(f"Failed to generate/save plot to {plot_png}: {exc}")


if __name__ == "__main__":
    main()
