"""Sklearn ensemble model factory for UAV-NIDD classification."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(ROOT_DIR / "configs") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "configs"))

try:
    import configs.config as project_config
except ImportError:  # pragma: no cover
    try:
        from config import config as project_config
    except ImportError:  # pragma: no cover
        project_config = None  # type: ignore

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None  # type: ignore


def _get_config_section(name: str, default: Dict[str, Any]) -> Dict[str, Any]:
    if project_config is None:
        return default
    section = getattr(project_config, name, None)
    if isinstance(section, dict):
        return {**default, **section}
    return default


def build_sklearn_ensemble(verbose: bool = False) -> VotingClassifier:
    """Build a VotingClassifier ensemble using configuration from configs.config."""
    rf_default = {
        "n_estimators": 150,
        "max_depth": 16,
        "random_state": 42,
        "n_jobs": -1,
    }
    xgb_default = {
        "n_estimators": 150,
        "learning_rate": 0.08,
        "max_depth": 8,
        "use_label_encoder": False,
        "eval_metric": "mlogloss",
        "random_state": 42,
        "verbosity": 1 if verbose else 0,
    }
    dt_default = {
        "max_depth": 12,
        "min_samples_split": 8,
        "random_state": 42,
    }

    rf_params = _get_config_section("SKLEARN_RF_CONFIG", rf_default)
    xgb_params = _get_config_section("SKLEARN_XGB_CONFIG", xgb_default)
    dt_params = _get_config_section("SKLEARN_DT_CONFIG", dt_default)

    rf_clf = RandomForestClassifier(**rf_params)
    dt_clf = DecisionTreeClassifier(**dt_params)

    estimators = [("rf", rf_clf), ("dt", dt_clf)]

    if XGBClassifier is not None:
        if verbose:
            xgb_params["verbosity"] = 1
        xgb_clf = XGBClassifier(**xgb_params)
        estimators.insert(1, ("xgb", xgb_clf))
    else:
        if verbose:
            print("Warning: xgboost is not installed; VotingClassifier will omit XGBoost.")

    voting = _get_config_section("SKLEARN_VOTING_CONFIG", {"voting": "soft", "weights": None})
    return VotingClassifier(estimators=estimators, **voting)


def get_default_model_name() -> str:
    return "uav_nidd_ensemble"


def get_feature_importance_names() -> list[str]:
    return ["random_forest", "xgboost", "decision_tree"]


if __name__ == "__main__":
    model = build_sklearn_ensemble()
    print("Built ensemble model:", model)
    print("Component estimators:", [name for name, _ in model.estimators])
