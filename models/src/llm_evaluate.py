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

# 语义匹配模型（sentencetransformers，懒加载）
_SENTENCE_MODEL = None
_SENTENCE_MODEL_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer, util

    _SENTENCE_MODEL_AVAILABLE = True
except ImportError:  # pragma: no cover
    SentenceTransformer = None  # type: ignore
    util = None  # type: ignore


def _get_sentence_model():
    global _SENTENCE_MODEL
    if _SENTENCE_MODEL is None and _SENTENCE_MODEL_AVAILABLE:
        # all-mpnet-base-v2 (420M params) provides much stronger semantic
        # similarity than all-MiniLM-L6-v2 (22M), especially for domain-
        # specific cybersecurity terminology where token overlap alone fails.
        _SENTENCE_MODEL = SentenceTransformer("all-mpnet-base-v2")
    return _SENTENCE_MODEL


def _semantic_match(text1: str, text2: str, threshold: float = 0.7) -> bool:
    """使用 Sentence-BERT 计算语义相似度，作为模糊匹配的备选方案。"""
    model = _get_sentence_model()
    if model is None:
        return False
    emb1 = model.encode(str(text1), convert_to_tensor=True)
    emb2 = model.encode(str(text2), convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(emb1, emb2).item()
    return similarity >= threshold

def _rouge_l_f1(reference: str, candidate: str) -> float:
    """计算 ROUGE-L F1 分数（基于最长公共子序列）。

    作为 token_set_ratio 和语义相似度的补充指标，
    衡量生成文本与参考文本在词序列层面的覆盖程度。
    """
    ref_tokens = str(reference).lower().split()
    cand_tokens = str(candidate).lower().split()
    if not ref_tokens or not cand_tokens:
        return 0.0

    # LCS 长度（动态规划，O(m*n) 空间优化为 O(min(m,n))）
    if len(ref_tokens) < len(cand_tokens):
        shorter, longer = ref_tokens, cand_tokens
    else:
        shorter, longer = cand_tokens, ref_tokens

    prev = [0] * (len(shorter) + 1)
    for i in range(1, len(longer) + 1):
        curr = [0] * (len(shorter) + 1)
        for j in range(1, len(shorter) + 1):
            if longer[i - 1] == shorter[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    lcs_len = prev[-1]

    recall = lcs_len / len(ref_tokens)
    precision = lcs_len / len(cand_tokens)
    if recall + precision == 0:
        return 0.0
    return 2.0 * recall * precision / (recall + precision)


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


def is_threat_match(
    generated: str, ground_truth: str,
    threshold: int = 70, semantic_threshold: float = 0.65,
) -> Tuple[bool, int]:
    """多策略模糊匹配：token_set_ratio / token_sort_ratio / partial_ratio
    三路取最高分；若仍低于阈值，回退到 Sentence-BERT 语义匹配。

    token_set_ratio  — 忽略顺序的 token 集合重叠（对词序不敏感）
    token_sort_ratio — 排序后比较（对同义词组敏感）
    partial_ratio    — 子串匹配（对长短描述差异敏感）
    """
    text1 = str(generated).lower()
    text2 = str(ground_truth).lower()

    # 三路并行模糊匹配，取最高分
    scores = [
        fuzz.token_set_ratio(text1, text2),
        fuzz.token_sort_ratio(text1, text2),
        fuzz.partial_ratio(text1, text2),
    ]
    best_score = max(scores)

    if best_score >= threshold:
        return (True, int(best_score))

    # 语义匹配备选 — all-mpnet-base-v2 在 0.65 阈值下比 MiniLM 在 0.7 更可靠
    if _semantic_match(text1, text2, semantic_threshold):
        return (True, threshold)

    return (False, int(best_score))


def load_responses(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Responses file not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data


def _normalize_category(cat: str) -> str:
    """将类别统一为 (d)、(e)、(f) 格式。"""
    cat = str(cat or "").strip().lower()
    if cat in {"d", "(d)"}:
        return "(d)"
    if cat in {"e", "(e)"}:
        return "(e)"
    if cat in {"f", "(f)"}:
        return "(f)"
    return cat


def evaluate_responses(responses_json_path: Path, threshold: int = 70) -> Dict[str, Any]:
    responses = load_responses(responses_json_path)

    total = len(responses)
    if total == 0:
        raise RuntimeError("No responses to evaluate.")

    # 检测评分器类型，便于诊断评分差异
    _scorer_name = "thefuzz.token_set_ratio" if fuzz is not None else "jaccard_fallback"
    print(f"[评估] 使用评分器: {_scorer_name}，匹配阈值: {threshold}")

    at_least_one_match = 0
    perfect_match = 0
    relaxed_perfect_match = 0   # 宽松完美匹配（阈值60/0.55）
    best_scores: List[int] = []
    scene_weighted_rates: List[float] = []
    rouge_l_scores: List[float] = []          # ROUGE-L F1 per scene
    missing_generated_category = 0
    category_counts = Counter()          # 各类别专家威胁总数（per-threat）
    category_matches = Counter()         # 各类别描述匹配数（per-threat-pair）
    category_presence_expected = Counter()  # 各类别出现的场景数（per-scenario）
    category_presence_generated = Counter() # 各类别 LLM 覆盖的场景数（per-scenario）

    examples_high: List[Dict[str, Any]] = []
    examples_low: List[Dict[str, Any]] = []
    all_examples: List[Dict[str, Any]] = []  # 收集所有场景用于排序取极值

    for entry in responses:
        expected = entry.get("expected_threats", []) or []
        model_output = entry.get("model_output", []) or []

        # 标准化 expected 为描述列表和类别映射
        expected_descs = []
        expected_cats = []
        for e in expected:
            if isinstance(e, dict):
                threat_text = (
                    e.get("Threat")
                    or e.get("description")
                    or e.get("text")
                    or ""
                )
                expected_descs.append(str(threat_text).strip())
                expected_cats.append(_normalize_category(e.get("RED Article") or e.get("category") or ""))
            else:
                expected_descs.append(str(e).strip())

        # 标准化 model_output
        gen_items = []
        for g in model_output:
            if isinstance(g, dict):
                gen_items.append({
                    "description": str(g.get("description", "")).strip(),
                    "category": _normalize_category(g.get("category", "")),
                })
            else:
                gen_items.append({"description": str(g).strip(), "category": ""})

        # 记录各类别总数（per-threat）
        for c in expected_cats:
            normalized = _normalize_category(c)
            if normalized in ("(d)", "(e)", "(f)"):
                category_counts[normalized] += 1

        # ── 类别覆盖（per-scenario）：LLM 是否识别到该类别维度的威胁 ──
        expected_cats_set = {_normalize_category(c) for c in expected_cats} & {"(d)", "(e)", "(f)"}
        generated_cats_set = {g["category"] for g in gen_items} & {"(d)", "(e)", "(f)"}
        for cat in expected_cats_set:
            category_presence_expected[cat] += 1
            if cat in generated_cats_set:
                category_presence_generated[cat] += 1

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
                    if gen.get("category") in ("(d)", "(e)", "(f)"):
                        category_matches[gen.get("category")] += 1
                    break
            if not exp_matched:
                all_matched = False
        if all_matched:
            perfect_match += 1

        # ── 宽松完美匹配：阈值 60 / 语义 0.55 ──
        relaxed_all_matched = True
        for exp in expected_descs:
            relaxed_exp_matched = False
            for gen in gen_items:
                matched, _ = is_threat_match(
                    gen["description"], exp,
                    threshold=60, semantic_threshold=0.55,
                )
                if matched:
                    relaxed_exp_matched = True
                    break
            if not relaxed_exp_matched:
                relaxed_all_matched = False
                break
        if relaxed_all_matched:
            relaxed_perfect_match += 1

        # --- 加权匹配率：越靠前的专家威胁权重越高（假设按重要性排序）---
        num_exp = len(expected_descs)
        if num_exp > 0:
            weights = [max(1.0 - i * 0.1, 0.5) for i in range(num_exp)]
            total_weight = sum(weights)
            matched_weight = 0.0
            for i, exp in enumerate(expected_descs):
                for gen in gen_items:
                    matched, _ = is_threat_match(gen["description"], exp, threshold=threshold)
                    if matched:
                        matched_weight += weights[i]
                        break
            scene_weighted_rates.append(matched_weight / total_weight if total_weight > 0 else 0.0)
        else:
            scene_weighted_rates.append(1.0)

        # --- ROUGE-L：衡量生成文本与专家文本在词序列层面的覆盖 ---
        scene_rouge_vals = []
        for exp in expected_descs:
            best_rouge = max(
                (_rouge_l_f1(exp, gen.get("description", "")) for gen in gen_items),
                default=0.0,
            )
            scene_rouge_vals.append(best_rouge)
        rouge_l_scores.append(float(np.mean(scene_rouge_vals)) if scene_rouge_vals else 0.0)

        # 收集示例：记录 max（最佳单条匹配）和 avg（整体匹配质量）
        scene_max_score = max(scene_best_scores) if scene_best_scores else 0
        scene_avg_score = int(sum(scene_best_scores) / len(scene_best_scores)) if scene_best_scores else 0
        all_examples.append({
            "scenario": entry.get("scenario_description", ""),
            "max_score": scene_max_score,
            "avg_score": scene_avg_score,
            "expected": expected_descs,
            "generated": gen_items,
        })

    # ── 按 max_score 排序取 Top 5（高分示例）和 Bottom 5（低分示例）──
    all_examples.sort(key=lambda ex: ex["max_score"], reverse=True)
    examples_high = [
        {"scenario": ex["scenario"], "score": ex["max_score"],
         "avg_score": ex["avg_score"],
         "expected": ex["expected"], "generated": ex["generated"]}
        for ex in all_examples[:5]
    ]
    examples_low = [
        {"scenario": ex["scenario"], "score": ex["max_score"],
         "avg_score": ex["avg_score"],
         "expected": ex["expected"], "generated": ex["generated"]}
        for ex in all_examples[-5:]
    ]

    total_with_gt = sum(1 for e in responses if e.get("expected_threats"))
    avg_weighted_rate = float(np.mean(scene_weighted_rates)) if scene_weighted_rates else 0.0
    avg_rouge_l = float(np.mean(rouge_l_scores)) if rouge_l_scores else 0.0

    overall_match_rate = at_least_one_match / max(1, total_with_gt)
    perfect_match_rate = perfect_match / max(1, total_with_gt)
    relaxed_perfect_rate = relaxed_perfect_match / max(1, total_with_gt)
    avg_best_score = float(sum(best_scores) / max(1, len(best_scores)))

    # 基于描述匹配的各类别匹配率（per-threat-pair，需要措辞匹配）
    category_rates = {}
    for cat in ("(d)", "(e)", "(f)"):
        category_rates[cat] = (category_matches.get(cat, 0) / category_counts.get(cat, 1)) if category_counts.get(cat, 0) else 0.0

    # 基于类别覆盖的匹配率（per-scenario，只需要类别存在，不要求措辞匹配）
    category_presence_rates = {}
    for cat in ("(d)", "(e)", "(f)"):
        category_presence_rates[cat] = (
            category_presence_generated.get(cat, 0)
            / category_presence_expected.get(cat, 1)
        ) if category_presence_expected.get(cat, 0) else 0.0

    metrics = {
        "total_scenes": total,
        "scenes_with_ground_truth": total_with_gt,
        "match_threshold": threshold,
        "scorer": _scorer_name,
        "overall_match_rate": overall_match_rate,
        "weighted_match_rate": avg_weighted_rate,
        "perfect_match_rate": perfect_match_rate,
        "relaxed_perfect_match_rate": relaxed_perfect_rate,
        "average_best_score": avg_best_score,
        "average_rouge_l": avg_rouge_l,
        "category_match_rates": category_rates,
        "category_presence_rates": category_presence_rates,
        "missing_generated_category_count": int(missing_generated_category),
    }

    outputs = {
        "metrics": metrics,
        "examples_high": examples_high[:5],
        "examples_low": examples_low[:5],
        "best_scores": best_scores,
        "rouge_l_scores": rouge_l_scores,
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
    plt.title("匹配分数分布 (token_set_ratio)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_rouge_distribution(rouge_scores: List[float], output_path: Path) -> None:
    """生成 ROUGE-L F1 分布直方图。"""
    if not rouge_scores:
        return
    bins = [0, 0.10, 0.15, 0.20, 0.25, 1.0]
    labels = ["<0.10", "0.10-0.15", "0.15-0.20", "0.20-0.25", "≥0.25"]
    counts, _ = np.histogram(rouge_scores, bins=bins)
    plt.figure(figsize=(8, 4))
    plt.bar(labels, counts, color="#F58518")
    plt.ylabel("场景数")
    plt.title("ROUGE-L F1 分布")
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
        f"评分器: {metrics.get('scorer', 'unknown')}",
        f"匹配阈值 (token_set_ratio): {metrics.get('match_threshold', 75)}",
        f"语义匹配模型: all-mpnet-base-v2 (threshold=0.65)",
        f"模糊匹配策略: token_set_ratio | token_sort_ratio | partial_ratio → max",
        "",
        "=== 总体指标 ===",
        f"总匹配率: {metrics['overall_match_rate']:.2%} (至少一个威胁匹配)",
        f"加权匹配率: {metrics.get('weighted_match_rate', 0):.2%} (按重要性加权)",
        f"完美匹配率: {metrics['perfect_match_rate']:.2%} (所有威胁匹配, 阈值70/0.65)",
        f"宽松完美匹配率: {metrics.get('relaxed_perfect_match_rate', 0):.2%} (所有威胁匹配, 阈值60/0.55)",
        f"平均匹配分数: {metrics['average_best_score']:.1f} (三路模糊匹配取max)",
        f"平均 ROUGE-L F1: {metrics.get('average_rouge_l', 0):.3f} (词序列覆盖度)",
        "",
        "=== 各类别匹配率 ===",
    ]
    lines.append("=== 各类别描述匹配率（需措辞匹配） ===")
    for k, v in metrics['category_match_rates'].items():
        label = {'(d)': '网络完整性/设备保护', '(e)': '个人数据/隐私', '(f)': '欺诈/经济'}[k]
        lines.append(f"{k} {label}: {v:.2%}")

    lines.append("")
    lines.append("=== 各类别覆盖识别率（仅需类别存在，不要求措辞匹配） ===")
    presence_rates = metrics.get('category_presence_rates', {})
    for k in ("(d)", "(e)", "(f)"):
        v = presence_rates.get(k, 0)
        label = {'(d)': '网络完整性/设备保护', '(e)': '个人数据/隐私', '(f)': '欺诈/经济'}[k]
        lines.append(f"{k} {label}: {v:.2%}")

    # 构造匹配分数分布
    lines.append("")
    lines.append("=== 匹配分数分布 (token_set_ratio) ===")
    scores = outputs.get("best_scores", [])
    if scores:
        bins = [0, 60, 70, 80, 90, 100]
        labels = ["<60", "60-69", "70-79", "80-89", "90-100"]
        for lo, hi, label in [(0, 60, "<60"), (60, 70, "60-69"), (70, 80, "70-79"), (80, 90, "80-89"), (90, 101, "90-100")]:
            count = sum(1 for s in scores if lo <= s < hi)
            pct = count / len(scores) * 100
            lines.append(f"  {label}: {count} ({pct:.1f}%)")

    # ROUGE-L 分布
    lines.append("")
    lines.append("=== ROUGE-L F1 分布 ===")
    rouge_scores = outputs.get("rouge_l_scores", [])
    if rouge_scores:
        for lo, hi, label in [(0, 0.1, "<0.10"), (0.1, 0.15, "0.10-0.15"), (0.15, 0.20, "0.15-0.20"), (0.20, 0.25, "0.20-0.25"), (0.25, 1.01, "≥0.25")]:
            count = sum(1 for s in rouge_scores if lo <= s < hi)
            pct = count / len(rouge_scores) * 100
            lines.append(f"  {label}: {count} ({pct:.1f}%)")

    # 写入示例 —— 按 max_score 排序取极端值，确保始终有输出
    lines.append("")
    lines.append("=== 典型场景示例（高匹配 — 按 max_score Top 5） ===")
    for rank, ex in enumerate(outputs.get('examples_high', []), 1):
        lines.append(f"--- Top {rank} | max_score={ex['score']} | avg_score={ex.get('avg_score','?')} ---")
        lines.append(f"Scenario: {ex['scenario']}")
        lines.append(f"Expected: {ex['expected']}")
        lines.append(f"Generated: {ex['generated']}")
        lines.append("")

    lines.append("=== 典型场景示例（低匹配 — 按 max_score Bottom 5） ===")
    for rank, ex in enumerate(outputs.get('examples_low', []), 1):
        lines.append(f"--- Bottom {rank} | max_score={ex['score']} | avg_score={ex.get('avg_score','?')} ---")
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
    rouge_json = out_dir / "rouge_l_scores.json"
    plot_png = out_dir / "match_rate_distribution.png"
    plot_rouge_png = out_dir / "rouge_l_distribution.png"

    save_report_text(outputs, report_txt)
    # 保存 JSON 指标
    try:
        with open(metrics_json, "w", encoding="utf-8") as handle:
            json.dump(outputs["metrics"], handle, indent=2, ensure_ascii=False)
        print(f"Saved metrics JSON to {metrics_json}")
    except Exception as exc:
        print(f"Failed to write metrics JSON to {metrics_json}: {exc}")

    # 保存 ROUGE-L 分数（供后续分析）
    try:
        with open(rouge_json, "w", encoding="utf-8") as handle:
            json.dump(outputs.get("rouge_l_scores", []), handle, indent=2, ensure_ascii=False)
        print(f"Saved ROUGE-L scores to {rouge_json}")
    except Exception as exc:
        print(f"Failed to write ROUGE-L scores: {exc}")

    # 生成分布图
    scores_list = outputs.get("best_scores", [])
    try:
        plot_match_distribution(scores_list, plot_png)
        print(f"Saved match rate distribution plot to {plot_png}")
    except Exception as exc:
        print(f"Failed to generate/save plot to {plot_png}: {exc}")

    rouge_list = outputs.get("rouge_l_scores", [])
    try:
        plot_rouge_distribution(rouge_list, plot_rouge_png)
        print(f"Saved ROUGE-L distribution plot to {plot_rouge_png}")
    except Exception as exc:
        print(f"Failed to generate/save ROUGE-L plot to {plot_rouge_png}: {exc}")


if __name__ == "__main__":
    main()
