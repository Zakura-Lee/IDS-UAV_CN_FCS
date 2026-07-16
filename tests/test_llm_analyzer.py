from pathlib import Path
from types import SimpleNamespace

from app.llm_analyzer import UAVThreatBenchAnalyzer


ROOT_DIR = Path(__file__).resolve().parents[1]


def test_loads_with_threats_dataset_from_existing_json():
    config = SimpleNamespace(
        RAW_DATA_DIR=ROOT_DIR / "data" / "raw",
        UAV_THREAT_BENCH_WITH=ROOT_DIR / "data" / "raw" / "plausible_uav_ot_cyber_scenarios_withThreats.json",
        UAV_THREAT_BENCH_WITHOUT=ROOT_DIR / "data" / "raw" / "plausible_uav_ot_cyber_scenarios_withoutThreats.json",
        LLM_CONFIG={},
    )
    analyzer = UAVThreatBenchAnalyzer(config_module=config)

    items = analyzer.load_dataset("withThreats")

    assert isinstance(items, list)
    assert len(items) > 0
    assert "Scenario Description" in items[0]


def test_heuristic_fallback_returns_threats_without_api_key():
    config = SimpleNamespace(
        RAW_DATA_DIR=ROOT_DIR / "data" / "raw",
        UAV_THREAT_BENCH_WITH=ROOT_DIR / "data" / "raw" / "plausible_uav_ot_cyber_scenarios_withThreats.json",
        UAV_THREAT_BENCH_WITHOUT=ROOT_DIR / "data" / "raw" / "plausible_uav_ot_cyber_scenarios_withoutThreats.json",
        LLM_CONFIG={},
    )
    analyzer = UAVThreatBenchAnalyzer(config_module=config)

    threats = analyzer._heuristic_threats("A drone is connected to a Wi-Fi network and data is exposed")

    assert threats
    # 新启发式函数返回带括号的 RED 类别标记 "(d)" / "(e)" / "(f)"
    assert any(item["category"] in ("e", "(e)") for item in threats)
