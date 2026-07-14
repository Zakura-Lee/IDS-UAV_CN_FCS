"""Train an sklearn ensemble model for UAV-NIDD intrusion detection."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

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
    from sklearn.preprocessing import StandardScaler
except ImportError:  # pragma: no cover
    StandardScaler = None  # type: ignore

from models.src.model_sklearn import build_sklearn_ensemble


@dataclass
class OutputPaths:
    checkpoint_dir: Path
    log_dir: Path
    model_path: Path
    config_path: Path
    history_path: Path


def build_output_paths() -> OutputPaths:
    checkpoint_dir = ROOT_DIR / "models" / "outputs" / "checkpoints"
    log_dir = ROOT_DIR / "models" / "outputs" / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    return OutputPaths(
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        model_path=checkpoint_dir / "model_sklearn.pkl",
        config_path=log_dir / "training_config.json",
        history_path=log_dir / "training_history.json",
    )


def load_splits(scenario: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if project_config is None:
        raise RuntimeError("Unable to load config module.")

    processed_dir = Path(getattr(project_config, "PROCESSED_DATA_DIR", ROOT_DIR / "data" / "processed"))
    scenario_dir = processed_dir / scenario
    split_names = ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]
    arrays: dict[str, np.ndarray] = {}

    for name in split_names:
        filename = f"{scenario}_{name}.npy"
        path = scenario_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Processed split not found: {path}")
        arrays[name] = np.load(path)

    return (
        arrays["X_train"],
        arrays["y_train"],
        arrays["X_val"],
        arrays["y_val"],
        arrays["X_test"],
        arrays["y_test"],
    )


def standardize_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    use_standardizer: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not use_standardizer or StandardScaler is None:
        return X_train, X_val, X_test
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled


def save_training_config(config: dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)


def save_history(history: list[dict[str, Any]], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2, ensure_ascii=False)


def train(
    scenario: str = "uav_case1",
    use_standardizer: bool = False,
    save_config: bool = True,
    verbose: bool = False,
) -> None:
    if project_config is None:
        raise RuntimeError("Unable to load configuration from configs.config.")

    paths = build_output_paths()
    X_train, y_train, X_val, y_val, X_test, y_test = load_splits(scenario)

    print(f"Loaded scenario '{scenario}' data:")
    print(f"  train: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}")
    X_train, X_val, X_test = standardize_features(X_train, X_val, X_test, use_standardizer)

    model = build_sklearn_ensemble(verbose=verbose)
    print("Training sklearn ensemble model...")
    start_time = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.1f} seconds")

    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    history = [
        {
            "scenario": scenario,
            "train_accuracy": float(accuracy_score(y_train, y_train_pred)),
            "val_accuracy": float(accuracy_score(y_val, y_val_pred)),
            "test_accuracy": float(accuracy_score(y_test, y_test_pred)),
            "train_f1_macro": float(f1_score(y_train, y_train_pred, average="macro")),
            "val_f1_macro": float(f1_score(y_val, y_val_pred, average="macro")),
            "test_f1_macro": float(f1_score(y_test, y_test_pred, average="macro")),
        }
    ]

    joblib.dump(model, paths.model_path)
    print(f"Saved sklearn model to {paths.model_path}")

    config_payload = {
        "scenario": scenario,
        "model_type": "VotingClassifier",
        "use_standardizer": use_standardizer,
        "sklearn_config": getattr(project_config, "SKLEARN_MODEL_CONFIG", {}),
        "rf_config": getattr(project_config, "SKLEARN_RF_CONFIG", {}),
        "xgb_config": getattr(project_config, "SKLEARN_XGB_CONFIG", {}),
        "dt_config": getattr(project_config, "SKLEARN_DT_CONFIG", {}),
        "voting_config": getattr(project_config, "SKLEARN_VOTING_CONFIG", {}),
    }
    if save_config:
        save_training_config(config_payload, paths.config_path)
        print(f"Saved training config to {paths.config_path}")

    save_history(history, paths.history_path)
    print(f"Saved training history to {paths.history_path}")

    print("Training results:")
    print(json.dumps(history[0], indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train sklearn ensemble model for UAV-NIDD.")
    parser.add_argument("--scenario", type=str, default="uav_case1", help="Scenario folder name in data/processed")
    parser.add_argument("--standardize", action="store_true", help="Apply StandardScaler to features before training")
    parser.add_argument("--no-save-config", dest="save_config", action="store_false", help="Do not save training config JSON")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose training output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        train(
            scenario=args.scenario,
            use_standardizer=args.standardize,
            save_config=args.save_config,
            verbose=args.verbose,
        )
    except Exception as exc:
        print(f"Training failed: {exc}")


if __name__ == "__main__":
    main()
