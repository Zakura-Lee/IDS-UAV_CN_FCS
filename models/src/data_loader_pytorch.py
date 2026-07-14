"""PyTorch 数据加载助手，用于读取 UAV-NIDD 预处理后的 numpy 数据集。"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Tuple

import numpy as np

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


def _resolve_processed_path(scenario_name: str, name: str) -> Path:
    """解析预处理数据文件路径并校验是否存在。"""
    processed_dir = Path(getattr(project_config, "PROCESSED_DATA_DIR", ROOT_DIR / "data" / "processed"))
    scenario_dir = processed_dir / scenario_name
    filename = f"{scenario_name}_{name}.npy"
    path = scenario_dir / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Processed file not found: {path}. "
            "Please run the preprocessing script to generate the dataset files."
        )
    return path


def _load_numpy_array(path: Path) -> np.ndarray:
    try:
        return np.load(path)
    except Exception as exc:
        raise RuntimeError(f"Unable to load numpy file {path}: {exc}") from exc


def get_data_shapes(scenario_name: str) -> Tuple[int, int]:
    """Return input_dim and num_classes for the given scenario."""
    x_train_path = _resolve_processed_path(scenario_name, "X_train")
    y_train_path = _resolve_processed_path(scenario_name, "y_train")

    X_train = _load_numpy_array(x_train_path)
    y_train = _load_numpy_array(y_train_path)

    if X_train.ndim != 2:
        raise ValueError(f"Expected X_train to be 2D, got shape {X_train.shape}")

    input_dim = int(X_train.shape[1])
    num_classes = int(np.unique(y_train).shape[0])

    model_config = getattr(project_config, "MODEL_CONFIG", {}) if project_config is not None else {}
    if model_config:
        num_classes = int(model_config.get("num_classes", num_classes))
        input_dim = int(model_config.get("input_dim", input_dim))

    return input_dim, num_classes


def create_dataloaders(scenario_name: str, batch_size: int | None = None) -> tuple[Any, Any, Any]:
    """创建训练、验证和测试 DataLoader。"""
    if torch is None or DataLoader is None or TensorDataset is None:
        raise RuntimeError("PyTorch is not installed. Install torch to use data_loader_pytorch.")
    if batch_size is None:
        batch_size = int(getattr(getattr(project_config, "TRAIN_CONFIG_PYTORCH", {}), "get", lambda *_: 64)("batch_size", 64))

    x_train = _load_numpy_array(_resolve_processed_path(scenario_name, "X_train"))
    y_train = _load_numpy_array(_resolve_processed_path(scenario_name, "y_train"))
    x_val = _load_numpy_array(_resolve_processed_path(scenario_name, "X_val"))
    y_val = _load_numpy_array(_resolve_processed_path(scenario_name, "y_val"))
    x_test = _load_numpy_array(_resolve_processed_path(scenario_name, "X_test"))
    y_test = _load_numpy_array(_resolve_processed_path(scenario_name, "y_test"))

    if x_train.ndim != 2 or x_val.ndim != 2 or x_test.ndim != 2:
        raise ValueError("Input arrays must be 2D arrays with shape [num_samples, features].")

    X_train = torch.from_numpy(x_train).float()
    X_val = torch.from_numpy(x_val).float()
    X_test = torch.from_numpy(x_test).float()
    y_train = torch.from_numpy(y_train).long()
    y_val = torch.from_numpy(y_val).long()
    y_test = torch.from_numpy(y_test).long()

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def main() -> None:
    """验证入口：检查预处理数据与 DataLoader 创建是否正常。"""
    try:
        input_dim, num_classes = get_data_shapes("uav_case1")
        print(f"Scenario 'uav_case1' input_dim={input_dim}, num_classes={num_classes}")

        if torch is not None:
            batch_size = int(getattr(getattr(project_config, "TRAIN_CONFIG_PYTORCH", {}), "get", lambda *_: 64)("batch_size", 64))
            loaders = create_dataloaders("uav_case1", batch_size=batch_size)
            print("DataLoaders created:")
            print(f"  train={len(loaders[0])}, val={len(loaders[1])}, test={len(loaders[2])}")
    except Exception as exc:
        print(f"data_loader_pytorch.py check failed: {exc}")


if __name__ == "__main__":
    main()
