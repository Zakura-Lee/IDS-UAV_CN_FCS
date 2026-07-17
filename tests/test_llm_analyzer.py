"""Smoke test: validate prompt quality on 20-30 UAVThreatBench scenarios.

Usage:
    # dry-run only (validate prompt structure, no API calls)
    python tests/test_llm_analyzer.py

    # live test with API calls
    python tests/test_llm_analyzer.py --live
"""

import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.llm_analyzer import UAVThreatBenchAnalyzer
from thefuzz import fuzz

# ── config ───────────────────────────────────────────────────────────────
SAMPLE_SIZE = 25
RANDOM_SEED = 42

# ── helpers ──────────────────────────────────────────────────────────────

def token_score(left: str, right: str) -> int:
    return fuzz.token_set_ratio(left.lower(), right.lower())


def quick_evaluate(results: list) -> dict:
    """Compute match stats on a small set of results."""
    at_least_one = 0
    total_best_scores = []
    cat_hits = {"(d)": 0, "(e)": 0, "(f)": 0}
    cat_total = {"(d)": 0, "(e)": 0, "(f)": 0}

    for entry in results:
        expected_threats = entry.get("expected_threats", []) or []
        model_output = entry.get("model_output", []) or []

        expected_descs = []
        for e in expected_threats:
            if isinstance(e, dict):
                expected_descs.append(str(e.get("Threat", e.get("description", e))).strip())
                cat = str(e.get("RED Article", e.get("category", ""))).strip()
                if cat in ("(d)", "(e)", "(f)"):
                    cat_total[cat] += 1
            else:
                expected_descs.append(str(e).strip())

        gen_descs = []
        gen_cats = []
        for g in model_output:
            if isinstance(g, dict):
                gen_descs.append(str(g.get("description", "")).strip())
                gen_cats.append(str(g.get("category", "")).strip())

        if not expected_descs:
            continue

        scene_best = []
        matched_any = False
        for g_desc, g_cat in zip(gen_descs, gen_cats):
            best = max((token_score(g_desc, exp) for exp in expected_descs), default=0)
            scene_best.append(best)
            if best >= 70:
                matched_any = True

        if matched_any:
            at_least_one += 1

        total_best_scores.append(max(scene_best) if scene_best else 0)

        # category hits: check if LLM generated categories match expected
        for exp_desc, exp_cat in zip(expected_descs, expected_threats):
            if not isinstance(exp_cat, dict):
                continue
            e_cat = str(exp_cat.get("RED Article", exp_cat.get("category", ""))).strip()
            if e_cat not in ("(d)", "(e)", "(f)"):
                continue
            for g_desc, g_cat in zip(gen_descs, gen_cats):
                if token_score(g_desc, exp_desc) >= 70 and g_cat == e_cat:
                    cat_hits[e_cat] += 1
                    break

    n = max(len(results), 1)
    return {
        "scenarios": n,
        "overall_match_rate": at_least_one / n,
        "avg_best_score": sum(total_best_scores) / max(len(total_best_scores), 1),
        "(d)_desc_rate": cat_hits["(d)"] / max(cat_total["(d)"], 1),
        "(e)_desc_rate": cat_hits["(e)"] / max(cat_total["(e)"], 1),
        "(f)_desc_rate": cat_hits["(f)"] / max(cat_total["(f)"], 1),
    }


# ── main ─────────────────────────────────────────────────────────────────

def main():
    live_mode = "--live" in sys.argv

    print("=" * 60)
    print("UAVThreatBench Smoke Test")
    print(f"  Sample size:  {SAMPLE_SIZE}")
    print(f"  Live API:     {live_mode}")
    print("=" * 60)

    config = SimpleNamespace(
        RAW_DATA_DIR=ROOT_DIR / "data" / "raw",
        UAV_THREAT_BENCH_WITH=ROOT_DIR / "data" / "raw" / "plausible_uav_ot_cyber_scenarios_withThreats.json",
        UAV_THREAT_BENCH_WITHOUT=ROOT_DIR / "data" / "raw" / "plausible_uav_ot_cyber_scenarios_withoutThreats.json",
        LLM_CONFIG={},
    )

    analyzer = UAVThreatBenchAnalyzer(config_module=config)

    # 1. Load and sample ──────────────────────────────────────────────────
    all_items = analyzer.load_dataset("withThreats")
    rng = random.Random(RANDOM_SEED)
    sample = rng.sample(all_items, min(SAMPLE_SIZE, len(all_items)))
    print(f"\nLoaded {len(all_items)} scenarios, sampled {len(sample)}\n")

    # 2. Validate prompt structure ────────────────────────────────────────
    print("--- Prompt Structure Check ---")
    empty_count = 0
    for i, item in enumerate(sample[:3]):  # deep-check first 3
        desc = (item.get("scenario_description")
                or item.get("Scenario Description", ""))
        prompt = analyzer.build_prompt(desc)
        sys_len = len(prompt["system"])
        usr_len = len(prompt["user"])
        has_cats = all(cat in prompt["system"] for cat in ("(d)", "(e)", "(f)"))
        has_examples = "attacker could disrupt" in prompt["system"]
        print(f"  [{i}] sys={sys_len} chars  usr={usr_len} chars  "
              f"categories={'OK' if has_cats else 'MISSING'}  "
              f"examples={'OK' if has_examples else 'MISSING'}")
        if not desc.strip():
            empty_count += 1

    # Bulk check all sampled
    for item in sample:
        desc = (item.get("scenario_description")
                or item.get("Scenario Description", ""))
        if not desc.strip():
            empty_count += 1
        prompt = analyzer.build_prompt(desc)
        assert "description" in prompt["system"], "system prompt missing 'description' field"
        assert "category" in prompt["system"], "system prompt missing 'category' field"
        assert isinstance(prompt["user"], str) and len(prompt["user"]) > 50

    print(f"  All {len(sample)} prompts structurally valid")
    print(f"  Empty descriptions: {empty_count}\n")

    # 3. Quick evaluation (dry-run using existing responses if available) ──
    responses_path = (
        ROOT_DIR / "models" / "outputs" / "llm_evaluation"
        / "withThreats" / "responses.json"
    )
    if responses_path.exists():
        with open(responses_path, "r", encoding="utf-8") as f:
            all_responses = json.load(f)
        print("--- Quick Eval (existing responses.json) ---")
        # Sample the same number from responses
        resp_sample = rng.sample(all_responses, min(SAMPLE_SIZE, len(all_responses)))
        metrics = quick_evaluate(resp_sample)
        for k, v in metrics.items():
            if "rate" in k:
                print(f"  {k}: {v:.2%}")
            else:
                print(f"  {k}: {v:.1f}")
        print()
    else:
        print("--- No existing responses.json (run llm_analyzer.py first) ---\n")

    # 4. Live API test (optional) ─────────────────────────────────────────
    if live_mode and analyzer.client is not None:
        print("--- Live API Test ---")
        live_sample = sample[:5]  # only 5 to save cost
        results = []
        for i, item in enumerate(live_sample):
            desc = (item.get("scenario_description")
                    or item.get("Scenario Description", ""))
            print(f"  [{i+1}/{len(live_sample)}] calling LLM...", end=" ", flush=True)
            try:
                threats = analyzer.call_llm(desc)
                results.append({
                    "scenario_description": desc,
                    "model_output": threats,
                    "expected_threats": item.get("expected_threats",
                                                 item.get("Expected Threats", [])),
                })
                print(f"got {len(threats)} threats")
            except Exception as exc:
                print(f"ERROR: {exc}")
                results.append({
                    "scenario_description": desc,
                    "model_output": [],
                    "expected_threats": item.get("expected_threats",
                                                 item.get("Expected Threats", [])),
                })
            time.sleep(0.5)

        if results:
            metrics = quick_evaluate(results)
            print(f"\n  Live results ({len(results)} scenarios):")
            for k, v in metrics.items():
                if "rate" in k:
                    print(f"    {k}: {v:.2%}")
                else:
                    print(f"    {k}: {v:.1f}")

            # Heuristic fallback rate check
            fallback_count = sum(1 for r in results
                                 if not r.get("model_output")
                                 or all("Unauthorized access or attack via" in
                                        g.get("description", "")
                                        for g in r["model_output"]
                                        if isinstance(g, dict)))
            print(f"    heuristic_fallback_rate: {fallback_count / len(results):.1%}")
    elif live_mode:
        print("--- Live API: SKIPPED (no API key configured) ---")

    print("\nDone.")


if __name__ == "__main__":
    main()
