"""Evaluate a trained PyTorch CNN-BiLSTM model on UAV-NIDD test data.

本脚本读取配置中的路径与模型参数，加载训练好的权重文件，
在测试集上执行推理并生成评估报告、混淆矩阵热力图与 CSV 数据。
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

try:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover
    torch = None
    DataLoader = None
    TensorDataset = None

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


def _get_device() -> torch.device:
    if torch is None:
        raise RuntimeError("PyTorch is not installed. Install torch to run evaluate_pytorch.py.")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_test_paths(scenario: str) -> tuple[Path, Path]:
    if project_config is None:
        raise RuntimeError("Unable to load configuration from configs.config.")

    processed_dir = Path(getattr(project_config, "PROCESSED_DATA_DIR", ROOT_DIR / "data" / "processed"))
    scenario_dir = processed_dir / scenario
    candidates = [
        (scenario_dir / "X_test.npy", scenario_dir / "y_test.npy"),
        (scenario_dir / f"{scenario}_X_test.npy", scenario_dir / f"{scenario}_y_test.npy"),
    ]

    for x_path, y_path in candidates:
        if x_path.exists() and y_path.exists():
            return x_path, y_path

    raise FileNotFoundError(
        f"Missing test data files for scenario '{scenario}'. Tried paths:"
        f" {candidates[0][0]}, {candidates[0][1]}, {candidates[1][0]}, {candidates[1][1]}"
    )


def load_test_data(scenario: str = "uav_case1", batch_size: int = 128) -> tuple[Any, tuple[int, int]]:
    if torch is None or DataLoader is None or TensorDataset is None:
        raise RuntimeError("PyTorch is required to load test data. Install torch to run this script.")

    x_path, y_path = _resolve_test_paths(scenario)
    X_test = np.load(x_path)
    y_test = np.load(y_path)

    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)
    if X_test.ndim != 2:
        raise ValueError(f"X_test must be 2D, got shape {X_test.shape}")
    if y_test.ndim != 1:
        y_test = y_test.ravel()

    X_tensor = torch.from_numpy(X_test).float()
    y_tensor = torch.from_numpy(y_test).long()
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return loader, (X_test.shape[0], X_test.shape[1])


def build_output_paths() -> dict[str, Path]:
    if project_config is None:
        raise RuntimeError("Unable to load configuration from configs.config.")

    log_dir = Path(getattr(project_config, "LOG_DIR", ROOT_DIR / "models" / "outputs" / "logs"))
    plot_dir = Path(getattr(project_config, "PLOT_DIR", ROOT_DIR / "models" / "outputs" / "plots"))
    log_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    return {
        "report_path": log_dir / "evaluation_report_pytorch.txt",
        "confusion_image_path": plot_dir / "confusion_matrix_pytorch.png",
        "confusion_csv_path": plot_dir / "confusion_matrix_pytorch.csv",
    }


def load_model(model_path: Path, device: torch.device) -> Any:
    if torch is None:
        raise RuntimeError("PyTorch is required to load the model. Install torch to run this script.")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        import models.src.model_pytorch as model_module
    except ImportError as exc:
        raise RuntimeError("Unable to import models.src.model_pytorch. Ensure the package is available.") from exc

    model = model_module.UAVNIDD_CNNBiLSTM.from_config(config_module=project_config)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> None:
    matrix = confusion_matrix(y_true, y_pred)
    labels = np.arange(matrix.shape[0])
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("PyTorch Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_report(report_text: str, output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(report_text)


def save_confusion_csv(matrix: np.ndarray, output_path: Path) -> None:
    np.savetxt(output_path, matrix, delimiter=",", fmt="%d")


def build_report(
    scenario: str,
    model_path: Path,
    num_samples: int,
    num_params: int,
    accuracy: float,
    report_text: str,
    macro_f1: float,
    weighted_f1: float,
    sklearn_accuracy: str | None = None,
) -> str:
    header = [
        "========================================",
        "UAV-NIDD PyTorch CNN-BiLSTM 评估报告",
        "========================================",
        f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Scenario: {scenario}",
        f"测试集样本数: {num_samples}",
        f"模型参数总量: {num_params:,}",
        "",
        f"总体准确率: {accuracy:.4f}",
        "",
        "=== 分类报告（按类别） ===",
        report_text,
        "",
        f"宏平均 F1: {macro_f1:.4f}",
        f"加权平均 F1: {weighted_f1:.4f}",
        "",
        "=== 混淆矩阵 ===",
        "已保存为图片和 CSV 文件。",
        "",
        "=== 与 sklearn 方案对比 ===",
        f"sklearn 方案准确率: {sklearn_accuracy if sklearn_accuracy is not None else 'N/A'}",
        f"PyTorch 方案准确率: {accuracy:.4f}",
        "结论: 若已知 sklearn 方案的准确率，可通过上述对比进一步分析。",
    ]
    return "\n".join(header)


def evaluate(scenario: str = "uav_case1") -> None:
    if project_config is None:
        raise RuntimeError("Unable to load configuration from configs.config.")

    checkpoint_dir = Path(getattr(project_config, "MODEL_DIR", ROOT_DIR / "models" / "outputs" / "checkpoints")) / "pytorch"
    model_path = checkpoint_dir / "best_model.pth"
    device = _get_device()

    model = load_model(model_path, device)
    test_loader, data_shape = load_test_data(scenario)

    y_true_list: list[np.ndarray] = []
    y_pred_list: list[np.ndarray] = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            if x_batch.dim() == 2:
                x_batch = x_batch.unsqueeze(1)
            outputs = model(x_batch)
            preds = outputs.argmax(dim=1).cpu().numpy()
            y_pred_list.append(preds)
            y_true_list.append(y_batch.numpy())

    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)
    accuracy = accuracy_score(y_true, y_pred)
    report_text = classification_report(y_true, y_pred, digits=4)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    matrix = confusion_matrix(y_true, y_pred)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    sklearn_accuracy = getattr(project_config, "SKLEARN_BASELINE_ACCURACY", None)

    paths = build_output_paths()
    full_report = build_report(
        scenario=scenario,
        model_path=model_path,
        num_samples=y_true.shape[0],
        num_params=num_params,
        accuracy=accuracy,
        report_text=report_text,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        sklearn_accuracy=f"{sklearn_accuracy:.4f}" if isinstance(sklearn_accuracy, (int, float)) else None,
    )

    save_report(full_report, paths["report_path"])
    plot_confusion_matrix(y_true, y_pred, paths["confusion_image_path"])
    save_confusion_csv(matrix, paths["confusion_csv_path"])

    print(full_report)
    print(f"Saved evaluation report to {paths['report_path']}")
    print(f"Saved confusion matrix image to {paths['confusion_image_path']}")
    print(f"Saved confusion matrix CSV to {paths['confusion_csv_path']}")


def main() -> None:
    try:
        evaluate()
    except Exception as exc:
        print(f"Evaluation failed: {exc}")


if __name__ == "__main__":
    main()
