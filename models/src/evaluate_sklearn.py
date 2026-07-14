"""Evaluate a trained sklearn ensemble model on UAV-NIDD test data."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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


def load_test_data(scenario: str) -> tuple[np.ndarray, np.ndarray]:
    processed_dir = Path(getattr(project_config, "PROCESSED_DATA_DIR", ROOT_DIR / "data" / "processed"))
    scenario_dir = processed_dir / scenario
    x_path = scenario_dir / f"{scenario}_X_test.npy"
    y_path = scenario_dir / f"{scenario}_y_test.npy"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Missing test data files: {x_path}, {y_path}")
    X_test = np.load(x_path)
    y_test = np.load(y_path)
    return X_test, y_test


def load_model(model_path: Path) -> Any:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


def save_report(report_text: str, output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(report_text)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> None:
    matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def build_output_paths() -> dict[str, Path]:
    log_dir = ROOT_DIR / "models" / "outputs" / "logs"
    plot_dir = ROOT_DIR / "models" / "outputs" / "plots"
    log_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    return {
        "report_path": log_dir / "evaluation_report.txt",
        "confusion_path": plot_dir / "confusion_matrix.png",
    }


def evaluate(scenario: str = "uav_case1") -> None:
    if project_config is None:
        raise RuntimeError("Unable to load configuration from configs.config.")

    model_path = ROOT_DIR / "models" / "outputs" / "checkpoints" / "model_sklearn.pkl"
    model = load_model(model_path)
    X_test, y_test = load_test_data(scenario)

    if hasattr(model, "predict"):
        y_pred = model.predict(X_test)
    else:
        raise RuntimeError("Loaded model does not support predict().")

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    full_text = (
        f"UAV-NIDD sklearn ensemble evaluation report\n"
        f"Scenario: {scenario}\n"
        f"Model path: {model_path}\n"
        f"Accuracy: {acc:.4f}\n\n"
        f"Classification report:\n{report}\n"
    )

    paths = build_output_paths()
    save_report(full_text, paths["report_path"])
    plot_confusion_matrix(y_test, y_pred, paths["confusion_path"])

    print(full_text)
    print(f"Saved evaluation report to {paths['report_path']}")
    print(f"Saved confusion matrix to {paths['confusion_path']}")


def main() -> None:
    try:
        evaluate()
    except Exception as exc:
        print(f"Evaluation failed: {exc}")


if __name__ == "__main__":
    main()
