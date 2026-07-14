"""PyTorch training script for UAV-NIDD CNN-BiLSTM model.

This script defines a full PyTorch training loop but does not run by default.
Set TRAIN_MODE = True to enable training execution.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import torch
    import torch.nn as nn
    from torch.optim import Adam
    from torch.optim.lr_scheduler import ReduceLROnPlateau
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    Adam = None
    ReduceLROnPlateau = None

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

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

from models.src.data_loader_pytorch import create_dataloaders

TRAIN_MODE = False


@dataclass
class OutputPaths:
    checkpoint_dir: Path
    log_dir: Path
    history_path: Path


def build_paths() -> OutputPaths:
    checkpoint_dir = ROOT_DIR / "models" / "outputs" / "checkpoints"
    log_dir = ROOT_DIR / "models" / "outputs" / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    return OutputPaths(
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        history_path=log_dir / "training_history.json",
    )


def _import_model_module() -> Any:
    try:
        import models.src.model_pytorch as model_module
        return model_module
    except ImportError as exc:
        raise RuntimeError("PyTorch model module could not be imported. Ensure PyTorch is installed and available.") from exc


def _get_device() -> torch.device:
    if torch is None:
        raise RuntimeError("PyTorch is not installed. Set TRAIN_MODE = False or install torch.")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def save_checkpoint(model: Any, epoch: int, path: Path) -> None:
    if torch is None:
        raise RuntimeError("Cannot save checkpoint without PyTorch.")
    file_path = path / f"best_model_epoch{epoch}.pth"
    torch.save(model.state_dict(), file_path)


def save_history(history: list[dict[str, float]], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2, ensure_ascii=False)


def _progress(iterable, **kwargs):
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)


def train_epoch(model: Any, loader: Any, criterion: Any, optimizer: Any, device: torch.device) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x_batch, y_batch in _progress(loader, desc="Train", leave=False):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        if x_batch.dim() == 2:
            x_batch = x_batch.unsqueeze(1)

        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x_batch.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


def validate_epoch(model: Any, loader: Any, criterion: Any, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x_batch, y_batch in _progress(loader, desc="Valid", leave=False):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            if x_batch.dim() == 2:
                x_batch = x_batch.unsqueeze(1)

            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * x_batch.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


def run_training(scenario_name: str = "uav_case1") -> None:
    if project_config is None:
        raise RuntimeError("Unable to load configs.config.")
    if torch is None or nn is None or Adam is None:
        raise RuntimeError("PyTorch is required for training. Install torch to enable training.")

    train_config = getattr(project_config, "TRAIN_CONFIG_PYTORCH", {})
    batch_size = int(train_config.get("batch_size", 64))
    learning_rate = float(train_config.get("learning_rate", 1e-3))
    num_epochs = int(train_config.get("num_epochs", 50))
    patience = int(train_config.get("early_stopping_patience", 5))

    model_module = _import_model_module()
    model = model_module.UAVNIDD_CNNBiLSTM.from_config(scenario_name).to(_get_device())
    train_loader, val_loader, _ = create_dataloaders(scenario_name, batch_size=batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, verbose=True)

    paths = build_paths()
    best_val_acc = 0.0
    best_epoch = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, _get_device())
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, _get_device())

        scheduler.step(val_loss)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }
        )

        print(
            f"  train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            save_checkpoint(model, epoch, paths.checkpoint_dir)
            print(f"  New best model saved at epoch {epoch}.")

        if epoch - best_epoch >= patience:
            print(f"Early stopping at epoch {epoch}. Best epoch was {best_epoch}.")
            break

    save_history(history, paths.history_path)
    print(f"Training history saved to {paths.history_path}")
    print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyTorch training wrapper for UAV-NIDD CNN-BiLSTM.")
    parser.add_argument("--scenario", type=str, default="uav_case1", help="Scenario folder under data/processed")
    parser.add_argument("--train", action="store_true", help="Enable training execution")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not TRAIN_MODE and not args.train:
        print("PyTorch training is disabled. Set TRAIN_MODE = True to run.")
        return
    print("PyTorch training is enabled.")
    run_training(scenario_name=args.scenario)


if __name__ == "__main__":
    main()
