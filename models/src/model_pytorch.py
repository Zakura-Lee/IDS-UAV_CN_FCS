"""PyTorch CNN-BiLSTM 模型定义，用于 UAV-NIDD 入侵检测。

此模块仅定义模型结构和配置读取，不执行训练。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

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


class UAVNIDD_CNNBiLSTM(nn.Module):
    """CNN-BiLSTM 混合网络模型，用于 UAV-NIDD 入侵检测。"""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        cnn_channels: int = 64,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.3,
        config_module: Any | None = None,
    ) -> None:
        super().__init__()
        self.config = config_module or project_config
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.cnn_channels = cnn_channels
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.dropout = dropout

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=self.cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.cnn_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(self.dropout),
        )

        self.bilstm = nn.LSTM(
            input_size=self.cnn_channels,
            hidden_size=self.lstm_hidden,
            num_layers=self.lstm_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.lstm_hidden * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(128, self.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播，返回未激活的 logits。"""
        # 数据输入可能是二维 [batch, input_dim]，模型需要三维 [batch, channels, input_dim]
        if x.dim() == 2:
            x = x.unsqueeze(1)

        if x.dim() != 3:
            raise ValueError("Input tensor must be 2D or 3D with shape [batch, features] or [batch, channels, features].")

        # CNN 提取空间特征
        x = self.cnn(x)
        # 变换为 LSTM 期望的 [batch, seq_len, features]
        x = x.permute(0, 2, 1)
        output, _ = self.bilstm(x)
        # 取最后一个时间步的输出用于分类
        output = output[:, -1, :]
        logits = self.classifier(output)
        return logits

    def get_feature_dim(self) -> int:
        return self.input_dim

    @classmethod
    def from_config(cls, scenario_name: str | None = None, config_module: Any | None = None) -> "UAVNIDD_CNNBiLSTM":
        config = config_module or project_config
        # 优先读取备用 PyTorch 配置，其次回退到通用 MODEL_CONFIG
        model_config = {}
        if config is not None:
            model_config = getattr(config, "MODEL_CONFIG_PYTORCH", {}) or getattr(config, "MODEL_CONFIG", {})

        input_dim = int(model_config.get("input_dim", 45))
        num_classes = int(model_config.get("num_classes", 11))
        cnn_channels = int(model_config.get("cnn_out_channels", 64))
        lstm_hidden = int(model_config.get("lstm_hidden_size", 128))
        lstm_layers = int(model_config.get("lstm_num_layers", 2))
        dropout = float(model_config.get("dropout_rate", 0.3))

        if scenario_name is not None:
            scenario_dims = {"uav_case1": 45, "ap_case2": 51, "gcs_case3": 85}
            input_dim = scenario_dims.get(scenario_name, input_dim)

        return cls(
            input_dim=input_dim,
            num_classes=num_classes,
            cnn_channels=cnn_channels,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            dropout=dropout,
            config_module=config,
        )


def model_summary(model: nn.Module) -> None:
    """打印模型参数量，用于验证模型定义是否正确。"""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model.__class__.__name__}")
    print(f"Total trainable parameters: {total_params:,}")


def _build_dummy_input(input_dim: int, batch_size: int = 2) -> torch.Tensor:
    return torch.randn(batch_size, 1, input_dim)


def main() -> None:
    model = UAVNIDD_CNNBiLSTM.from_config()
    model_summary(model)
    dummy_input = _build_dummy_input(model.get_feature_dim())
    logits = model(dummy_input)
    print(f"Dummy input shape: {dummy_input.shape}")
    print(f"Logits shape: {logits.shape}")


if __name__ == "__main__":
    main()
